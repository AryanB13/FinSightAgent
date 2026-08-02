"""
eval/retrieval_eval.py — Precision@5 and Recall@5 evaluation for the
Financial Research Agent retrieval pipeline.

Zero Gemini calls are made here — this is pure BM25 / Pinecone / RRF math.
The only external I/O is two Pinecone ANN queries and one Voyage embedding
call per sub-query spec (same path as the live pipeline).

Usage (from project root):
    python -c "
    from dotenv import load_dotenv; load_dotenv()
    from query.pipeline import init_pipeline_resources
    from eval.dataset import load_eval_dataset
    from eval.retrieval_eval import evaluate_retrieval, print_retrieval_report

    resources = init_pipeline_resources()
    queries = load_eval_dataset()
    results, agg = evaluate_retrieval(queries, resources)
    print_retrieval_report(results, agg)
    "
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from query.retrieval.hybrid_retriever import retrieve_for_subquery
from query.retrieval.reranker import rerank_chunks
from query.utils.schema import RouterOutput, SubQuery, RetrievedChunk
from eval.dataset import EvalQuery, SubQuerySpec

logger = logging.getLogger(__name__)


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """Precision@5 and Recall@5 scores for one EvalQuery."""
    query_id: str
    query: str
    category: str
    retrieved_chunk_ids: list[str]   # union of all sub-query top-5 results
    retrieved_chunk_texts: list[str] # .text of each retrieved chunk (same order as IDs)
    relevant_chunk_ids: list[str]    # ground truth from EvalQuery
    precision: float                 # Context Precision@K — rank-weighted, normalised by relevant found in top-K
    recall: float                    # |retrieved ∩ relevant| / |relevant|
    n_sub_queries: int               # number of sub-queries run
    elapsed_seconds: float


# ── Core metric functions ──────────────────────────────────────────────────────

def compute_precision(retrieved: list[str], relevant: set[str]) -> float:
    """
    Context Precision@K — rank-weighted precision normalised by relevant items
    found in the top-K results (not by K itself).

    Formula (from RAGAS):
        Context Precision@K = Σ_{k=1}^{K} (Precision@k × v_k)
                              ─────────────────────────────────
                              Total relevant items in top-K results

    Where:
        Precision@k = (# relevant in top k positions) / k
        v_k         = 1 if the chunk at rank k is relevant, else 0

    Compared to simple Precision@K (hits / K):
    - Simple P@K always divides by K=5, so a query with only 1 ground-truth
      chunk can never exceed 0.20 — even with perfect retrieval.
    - Context Precision@K normalises by relevant items *found*, so placing
      the single relevant chunk at rank 1 returns 1.0 (correct behaviour).

    Returns:
        1.0  when retrieved is empty (no false positives possible).
        0.0  when no relevant chunk appears anywhere in retrieved.
        Rank-weighted score in (0, 1] otherwise.
    """
    if not retrieved:
        return 1.0

    total_relevant_in_topk = sum(1 for cid in retrieved if cid in relevant)
    if total_relevant_in_topk == 0:
        return 0.0

    numerator: float = 0.0
    hits_so_far: int = 0
    for k, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            hits_so_far += 1
            precision_at_k = hits_so_far / k
            numerator += precision_at_k  # × v_k=1

    return numerator / total_relevant_in_topk


def compute_recall(retrieved: list[str], relevant: set[str]) -> float:
    """
    Recall = # relevant chunks in retrieved set / # total relevant chunks.

    Returns 1.0 when relevant is empty (adversarial queries with no ground truth).
    """
    if not relevant:
        return 1.0
    hits = sum(1 for cid in retrieved if cid in relevant)
    return hits / len(relevant)


# ── Sub-query builder ──────────────────────────────────────────────────────────

def _build_sub_query_and_router(
    spec: SubQuerySpec,
    route: str,
    all_companies: list[str],
    all_years: list[int],
) -> tuple[SubQuery, RouterOutput]:
    """Converts one SubQuerySpec into the SubQuery + RouterOutput the retriever expects."""
    sub_query = SubQuery(
        sub_query_id=1,
        text=spec.text,
        company=spec.company,
        fiscal_year=spec.year,
    )
    router_output = RouterOutput(
        route=route,
        companies=all_companies,
        years=all_years,
        needs_computation=False,
    )
    return sub_query, router_output


def _build_default_sub_query_and_router(
    eval_query: EvalQuery,
) -> tuple[SubQuery, RouterOutput]:
    """Fallback for direct_lookup queries that have no explicit sub_query_specs."""
    sub_query = SubQuery(
        sub_query_id=1,
        text=eval_query.query,
        company=eval_query.companies[0] if eval_query.companies else None,
        fiscal_year=eval_query.years[0] if eval_query.years else None,
    )
    router_output = RouterOutput(
        route=eval_query.expected_route,
        companies=eval_query.companies,
        years=eval_query.years,
        needs_computation=False,
    )
    return sub_query, router_output


# ── Single-query evaluator ─────────────────────────────────────────────────────

def evaluate_retrieval_single(
    eval_query: EvalQuery,
    resources: dict,
) -> RetrievalResult:
    """
    Runs retrieval for one EvalQuery and computes Precision and Recall.

    If eval_query.sub_query_specs is non-empty, one retrieval is run per spec
    and the top-5 result sets are unioned.  This correctly mirrors how the
    live pipeline handles single_hop and multi_hop queries.

    If sub_query_specs is empty (direct_lookup), a single retrieval is run
    using the full query text and the first company/year in the EvalQuery.

    Args:
        eval_query: The evaluation query to run.
        resources:  Dict from init_pipeline_resources().

    Returns:
        RetrievalResult with precision, recall, and retrieved chunk IDs.
    """
    t0 = time.perf_counter()
    all_retrieved_chunks: list[RetrievedChunk] = []  # collect full RetrievedChunk objects
    seen: set[str] = set()

    specs: list[SubQuerySpec] = eval_query.sub_query_specs

    if not specs:
        # direct_lookup — single sub-query from the full query text
        sub_query, router_output = _build_default_sub_query_and_router(eval_query)
        try:
            chunks = retrieve_for_subquery(sub_query, router_output, resources)
            for rc in chunks:
                cid = rc.chunk.chunk_id
                if cid not in seen:
                    all_retrieved_chunks.append(rc)
                    seen.add(cid)
        except Exception as exc:
            logger.warning("[%s] retrieval failed: %s", eval_query.id, exc)
        n_specs = 1
    else:
        # single_hop / multi_hop — one retrieval per spec, union results
        for i, spec in enumerate(specs):
            sub_query, router_output = _build_sub_query_and_router(
                spec,
                route=eval_query.expected_route,
                all_companies=eval_query.companies,
                all_years=eval_query.years,
            )
            sub_query.sub_query_id = i + 1
            try:
                chunks = retrieve_for_subquery(sub_query, router_output, resources)
                for rc in chunks:
                    cid = rc.chunk.chunk_id
                    if cid not in seen:
                        all_retrieved_chunks.append(rc)
                        seen.add(cid)
            except Exception as exc:
                logger.warning(
                    "[%s] sub-query %d/%d retrieval failed: %s",
                    eval_query.id, i + 1, len(specs), exc,
                )
        n_specs = len(specs)

        # Cross-query reranking: for multi-hop with >5 chunks, rerank all together
        # using the original query text to get a true top-5
        if all_retrieved_chunks and len(all_retrieved_chunks) > 5:
            pc_client = resources["pinecone_client"]
            reranked = rerank_chunks(eval_query.query, all_retrieved_chunks, pc_client, top_n=5)
            all_retrieved_chunks = reranked
            logger.debug(
                "[%s] cross-query rerank: %d → %d chunks",
                eval_query.id, len(seen), len(all_retrieved_chunks),
            )

    # Extract IDs and texts from final chunk list
    all_retrieved = [rc.chunk.chunk_id for rc in all_retrieved_chunks]
    all_texts = [rc.chunk.text for rc in all_retrieved_chunks]

    relevant_set = set(eval_query.relevant_chunk_ids)
    elapsed = time.perf_counter() - t0

    prec = compute_precision(all_retrieved, relevant_set)
    rec  = compute_recall(all_retrieved, relevant_set)

    logger.info(
        "[%s] %s | retrieved=%d unique | relevant=%d | P=%.3f | R=%.3f | %.1fs",
        eval_query.id, eval_query.category,
        len(all_retrieved), len(relevant_set),
        prec, rec, elapsed,
    )

    return RetrievalResult(
        query_id=eval_query.id,
        query=eval_query.query,
        category=eval_query.category,
        retrieved_chunk_ids=all_retrieved,
        retrieved_chunk_texts=all_texts,
        relevant_chunk_ids=eval_query.relevant_chunk_ids,
        precision=prec,
        recall=rec,
        n_sub_queries=n_specs,
        elapsed_seconds=elapsed,
    )


# ── Batch evaluator ────────────────────────────────────────────────────────────

def evaluate_retrieval(
    eval_queries: list[EvalQuery],
    resources: dict,
) -> tuple[list[RetrievalResult], dict]:
    """
    Runs retrieval evaluation for all queries.

    Args:
        eval_queries: List of EvalQuery objects (from load_eval_dataset).
        resources:    Dict from init_pipeline_resources().

    Returns:
        Tuple of:
          - list[RetrievalResult]: one result per query
          - dict: aggregate metrics with keys
              "mean_precision_at_5", "mean_recall_at_5",
              "by_category": {category: {"precision": float, "recall": float, "n": int}}
    """
    results: list[RetrievalResult] = []
    for i, eq in enumerate(eval_queries):
        logger.info("evaluate_retrieval: [%d/%d] %s", i + 1, len(eval_queries), eq.id)
        results.append(evaluate_retrieval_single(eq, resources))

    agg = _aggregate(results)
    return results, agg


def _aggregate(results: list[RetrievalResult]) -> dict:
    """Computes mean Precision and Recall overall and per category."""
    if not results:
        return {"mean_precision_at_5": 0.0, "mean_recall_at_5": 0.0, "by_category": {}}

    mean_p = sum(r.precision for r in results) / len(results)
    mean_r = sum(r.recall    for r in results) / len(results)

    by_cat: dict[str, list[RetrievalResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    cat_agg: dict[str, dict] = {}
    for cat, cat_results in by_cat.items():
        cat_agg[cat] = {
            "precision": sum(r.precision for r in cat_results) / len(cat_results),
            "recall":    sum(r.recall    for r in cat_results) / len(cat_results),
            "n":         len(cat_results),
        }

    return {
        "mean_context_precision": mean_p,
        "mean_recall_at_5":       mean_r,
        "by_category":            cat_agg,
    }


# ── Console reporter ───────────────────────────────────────────────────────────

_GREEN  = "\033[92m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_BOLD   = "\033[1m"
_RESET  = "\033[0m"

THRESHOLDS = {
    "precision": 0.70,   # Context Precision@K — higher bar than simple P@5 (was 0.55)
    "recall":    0.65,
}


def print_retrieval_report(
    results: list[RetrievalResult],
    agg: dict,
    verbose: bool = False,
) -> None:
    """
    Prints a formatted retrieval evaluation report to stdout.

    Args:
        results: Per-query results from evaluate_retrieval().
        agg:     Aggregate metrics dict from evaluate_retrieval().
        verbose: If True, prints per-query breakdown with retrieved/relevant IDs.
    """
    print()
    print(_BOLD + "=" * 68 + _RESET)
    print(_BOLD + "  RETRIEVAL EVALUATION REPORT" + _RESET)
    print(_BOLD + "=" * 68 + _RESET)
    print()

    # ── Aggregate ──────────────────────────────────────────────────────────
    def _fmt(label: str, score: float, threshold: float, thresh_label: str) -> str:
        tick  = (_GREEN + "✓" + _RESET) if score >= threshold else (_RED + "✗" + _RESET)
        color = _GREEN if score >= threshold else _RED
        return (
            f"  {label:<22} {color}{score:.4f}{_RESET}   "
            f"threshold {thresh_label}  {tick}"
        )

    p = agg["mean_context_precision"]
    r = agg["mean_recall_at_5"]
    print(_fmt("Context Precision@K (mean)",  p, THRESHOLDS["precision"], "≥ 0.70"))
    print(_fmt("Recall@5            (mean)",  r, THRESHOLDS["recall"],    "≥ 0.65"))
    print()

    # ── By category ────────────────────────────────────────────────────────
    print(_BOLD + "  By category:" + _RESET)
    cat_order = ["direct_lookup", "single_hop", "multi_hop", "adversarial"]
    by_cat = agg.get("by_category", {})
    for cat in cat_order:
        if cat not in by_cat:
            continue
        c = by_cat[cat]
        print(
            f"    {cat:<15}  n={c['n']:2d}  "
            f"P={c['precision']:.3f}  R={c['recall']:.3f}"
        )
    print()

    # ── Per-query table ─────────────────────────────────────────────────────
    print(_BOLD + "  Per-query results:" + _RESET)
    header = f"  {'ID':<8} {'Category':<14} {'P':>6} {'R':>6} {'Retrieved':>9} {'Relevant':>8}  Result"
    print(header)
    print("  " + "-" * 66)
    for r_ in results:
        passed = r_.precision >= THRESHOLDS["precision"] and r_.recall >= THRESHOLDS["recall"]
        status = (_GREEN + "PASS" + _RESET) if passed else (_RED + "FAIL" + _RESET)
        adv_note = " [no GT]" if not r_.relevant_chunk_ids else ""
        print(
            f"  {r_.query_id:<8} {r_.category:<14} "
            f"{r_.precision:>6.3f} {r_.recall:>6.3f} "
            f"{len(r_.retrieved_chunk_ids):>9} {len(r_.relevant_chunk_ids):>8}"
            f"  {status}{adv_note}"
        )

    total_elapsed = sum(r_.elapsed_seconds for r_ in results)
    print()
    print(f"  Total queries: {len(results)}  |  Total elapsed: {total_elapsed:.1f}s")
    print(_BOLD + "=" * 68 + _RESET)
    print()

    # ── Verbose: per-query chunk breakdown ────────────────────────────────
    if verbose:
        for r_ in results:
            relevant_set = set(r_.relevant_chunk_ids)
            hits    = [c for c in r_.retrieved_chunk_ids if c in relevant_set]
            misses  = [c for c in r_.retrieved_chunk_ids if c not in relevant_set]
            skipped = [c for c in r_.relevant_chunk_ids  if c not in r_.retrieved_chunk_ids]
            print(f"  {_BOLD}[{r_.query_id}]{_RESET} {r_.query[:70]}")
            print(f"    Hits    ({len(hits)}):    {hits}")
            print(f"    Noise   ({len(misses)}):  {misses}")
            print(f"    Missed  ({len(skipped)}): {skipped}")
            print()
