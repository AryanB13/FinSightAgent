"""
eval/generation_eval.py — Faithfulness and Answer Relevance evaluation
for the Financial Research Agent using LLM-as-Judge (Gemini).

2 Gemini calls per query:
  1. Faithfulness: decomposes the generated answer into atomic factual
     claims and checks each claim against the retrieved evidence chunks.
     Score = supported_claims / total_claims.
  2. Answer Relevance: scores how well the answer addresses the question
     on a 1–5 Likert scale, normalised to [0, 1] (1→0.0, 5→1.0).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from query.utils.gemini_client import call_structured, GeminiCallCounter
from eval.dataset import EvalQuery

logger = logging.getLogger(__name__)


# ── Result dataclass ────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Faithfulness and Answer Relevance scores for one EvalQuery."""
    query_id: str
    query: str
    category: str
    generated_answer: str
    faithfulness: float               # 0.0–1.0
    answer_relevance: float           # 0.0–1.0
    faithfulness_claims: list[dict]   # per-claim breakdown: [{claim, supported, evidence}]
    relevance_reasoning: str          # judge's explanation for the relevance score
    n_chunks_used: int                # number of evidence chunks passed to faithfulness judge
    elapsed_seconds: float


# ── Faithfulness judge ──────────────────────────────────────────────────────────

FAITHFULNESS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim":     {"type": "string"},
                    "supported": {"type": "boolean"},
                    "evidence":  {"type": "string"},
                },
                "required": ["claim", "supported"],
            },
        },
        "faithfulness_score": {"type": "number"},
    },
    "required": ["claims", "faithfulness_score"],
}

FAITHFULNESS_SYSTEM_PROMPT = """\
You are a strict financial fact-checker for a RAG system.

Given a generated answer and the retrieved evidence chunks that were provided
to the generator, your job is:

1. Break the answer into individual atomic factual claims.
   A claim = one specific number, percentage, company name + metric, or date.
   Do NOT include logical connectives, hedges, or identity facts
   (e.g. "Apple is a technology company") as claims.

2. For each claim, decide: is it DIRECTLY supported by a specific retrieved
   chunk? A claim is supported only if the exact figure or fact appears
   verbatim (or near-verbatim) in a chunk. Approximate values from memory
   ("about $383 billion") that differ from the precise chunk value are NOT
   supported.

3. Compute faithfulness_score = (# supported claims) / (# total claims).
   If there are zero claims, return faithfulness_score = 1.0.

Claims marked [UNVERIFIED] in the answer are unsupported by definition —
count them as not supported.

Special rule for derived computations (e.g. operating margin = 29.8%):
If the two input values (e.g. operating income $114,301M and net sales
$383,285M) are both supported by chunks, the derived percentage is
considered supported even if the exact percentage does not appear verbatim.
"""


def evaluate_faithfulness(
    answer: str,
    retrieved_chunk_texts: list[str],
    gemini_client,
    call_counter: GeminiCallCounter,
) -> tuple[float, list[dict]]:
    """
    Asks Gemini to decompose the answer into claims and check each against
    the retrieved chunks. Returns (faithfulness_score, claims list).
    Uses 1 Gemini call.

    Args:
        answer: Generated answer text from run_query().
        retrieved_chunk_texts: List of chunk .text strings (evidence context).
        gemini_client: Initialised google.genai.Client.
        call_counter: Shared GeminiCallCounter.

    Returns:
        (faithfulness_score [0,1], claims [list of {claim, supported, evidence}])
    """
    if not answer.strip():
        return 1.0, []

    chunks_block = "\n\n".join(
        f"[CHUNK {i + 1}]\n{text}" for i, text in enumerate(retrieved_chunk_texts)
    )
    user_content = (
        f"RETRIEVED EVIDENCE:\n{chunks_block}\n\n"
        f"GENERATED ANSWER:\n{answer}\n\n"
        "Evaluate faithfulness as described. Return the structured JSON."
    )

    try:
        response = call_structured(
            gemini_client,
            FAITHFULNESS_SYSTEM_PROMPT,
            user_content,
            FAITHFULNESS_SCHEMA,
            call_counter,
        )
        score  = float(response.get("faithfulness_score", 0.0))
        claims = response.get("claims", [])
    except Exception as exc:
        logger.warning("evaluate_faithfulness: Gemini call failed: %s", exc)
        return 0.0, []

    return max(0.0, min(1.0, score)), claims


# ── Answer Relevance judge ──────────────────────────────────────────────────────

RELEVANCE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "score":     {"type": "integer"},
        "reasoning": {"type": "string"},
    },
    "required": ["score", "reasoning"],
}

RELEVANCE_SYSTEM_PROMPT = """\
You are evaluating the answer relevance of a financial RAG system.

Score the answer on a scale of 1–5:
  5 = Completely answers the question with the specific metric, company,
      and time period requested. Includes the actual number if a number
      was asked for.
  4 = Mostly answers. Minor omission, one hedge, or slight incompleteness.
  3 = Partially answers. Mentions the right topic but lacks the specific
      number, or answers only part of a multi-part question.
  2 = Wrong metric, wrong year, or wrong company. Tangentially related.
  1 = Completely off-topic, refuses to answer, or says "I don't know"
      without providing any data.

Rules:
- Do NOT penalise for citing sources or [UNVERIFIED] annotations.
- Do NOT penalise for correct hedging when data is genuinely unavailable
  (e.g. asking for FY2025 data that does not exist in the corpus).
- DO penalise for excessive hedging when the answer IS in the corpus.
- The answer does not need to match the reference word-for-word.
"""


def evaluate_answer_relevance(
    query: str,
    answer: str,
    gemini_client,
    call_counter: GeminiCallCounter,
) -> tuple[float, str]:
    """
    Asks Gemini to score the answer on relevance 1–5 and normalises to [0, 1].
    Returns (relevance_score, reasoning_string). Uses 1 Gemini call.

    Normalisation: 1→0.00, 2→0.25, 3→0.50, 4→0.75, 5→1.00.

    Args:
        query: Original user query string.
        answer: Generated answer text from run_query().
        gemini_client: Initialised google.genai.Client.
        call_counter: Shared GeminiCallCounter.

    Returns:
        (normalised_score [0,1], reasoning string)
    """
    user_content = (
        f"QUESTION:\n{query}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Score the answer relevance as described. Return the structured JSON."
    )

    try:
        response = call_structured(
            gemini_client,
            RELEVANCE_SYSTEM_PROMPT,
            user_content,
            RELEVANCE_SCHEMA,
            call_counter,
        )
        raw_score = max(1, min(5, int(response.get("score", 1))))
        normalised = (raw_score - 1) / 4
        reasoning  = response.get("reasoning", "")
    except Exception as exc:
        logger.warning("evaluate_answer_relevance: Gemini call failed: %s", exc)
        return 0.0, f"[ERROR] {exc}"

    return normalised, reasoning


# ── Combined generation evaluator ───────────────────────────────────────────────

def evaluate_generation_single(
    eval_query: EvalQuery,
    pipeline_result: dict,
    retrieved_chunk_texts: list[str],
    gemini_client,
    call_counter: GeminiCallCounter,
) -> GenerationResult:
    """
    Runs Faithfulness and Answer Relevance judges for one query.
    2 Gemini calls total.

    Args:
        eval_query: The evaluation query with ground truth metadata.
        pipeline_result: Dict returned by run_query(), must have "final_answer" key.
        retrieved_chunk_texts: Chunk .text strings from the retrieval step
                               (used as evidence context for the faithfulness judge).
        gemini_client: Initialised google.genai.Client.
        call_counter: Shared GeminiCallCounter.

    Returns:
        GenerationResult with faithfulness score, answer relevance score,
        per-claim breakdown, and judge reasoning.
    """
    t0     = time.perf_counter()
    answer = pipeline_result.get("final_answer", "")

    faithfulness, claims = evaluate_faithfulness(
        answer, retrieved_chunk_texts, gemini_client, call_counter
    )
    relevance, reasoning = evaluate_answer_relevance(
        eval_query.query, answer, gemini_client, call_counter
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "[%s] %s | F=%.3f | AR=%.3f | %.1fs",
        eval_query.id, eval_query.category, faithfulness, relevance, elapsed,
    )

    return GenerationResult(
        query_id=eval_query.id,
        query=eval_query.query,
        category=eval_query.category,
        generated_answer=answer,
        faithfulness=faithfulness,
        answer_relevance=relevance,
        faithfulness_claims=claims,
        relevance_reasoning=reasoning,
        n_chunks_used=len(retrieved_chunk_texts),
        elapsed_seconds=elapsed,
    )


def evaluate_generation(
    eval_queries: list[EvalQuery],
    pipeline_results: list[dict],
    retrieved_chunks_per_query: list[list[str]],
    gemini_client,
    call_counter: GeminiCallCounter,
) -> tuple[list[GenerationResult], dict]:
    """
    Runs generation evaluation for all queries.

    Args:
        eval_queries: List of EvalQuery objects (same order as pipeline_results).
        pipeline_results: List of dicts from run_query(), one per query.
        retrieved_chunks_per_query: List of chunk-text lists (one list per query),
                                    sourced from RetrievalResult.retrieved_chunk_texts.
        gemini_client: Initialised google.genai.Client.
        call_counter: Shared GeminiCallCounter.

    Returns:
        Tuple of:
          - list[GenerationResult]: one result per query
          - dict: aggregate metrics:
              "mean_faithfulness", "mean_answer_relevance",
              "by_category": {category: {"faithfulness", "answer_relevance", "n"}}
    """
    results: list[GenerationResult] = []
    for i, (eq, pr, chunk_texts) in enumerate(
        zip(eval_queries, pipeline_results, retrieved_chunks_per_query)
    ):
        logger.info("evaluate_generation: [%d/%d] %s", i + 1, len(eval_queries), eq.id)
        results.append(
            evaluate_generation_single(eq, pr, chunk_texts, gemini_client, call_counter)
        )

    return results, _aggregate(results)


def _aggregate(results: list[GenerationResult]) -> dict:
    """Computes mean Faithfulness and Answer Relevance overall and per category."""
    if not results:
        return {"mean_faithfulness": 0.0, "mean_answer_relevance": 0.0, "by_category": {}}

    mean_f  = sum(r.faithfulness     for r in results) / len(results)
    mean_ar = sum(r.answer_relevance for r in results) / len(results)

    by_cat: dict[str, list[GenerationResult]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)

    cat_agg: dict[str, dict] = {}
    for cat, cat_results in by_cat.items():
        cat_agg[cat] = {
            "faithfulness":     sum(r.faithfulness     for r in cat_results) / len(cat_results),
            "answer_relevance": sum(r.answer_relevance for r in cat_results) / len(cat_results),
            "n":                len(cat_results),
        }

    return {
        "mean_faithfulness":     mean_f,
        "mean_answer_relevance": mean_ar,
        "by_category":           cat_agg,
    }


# ── Console reporter ─────────────────────────────────────────────────────────────

_GREEN = "\033[92m"
_RED   = "\033[91m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"

THRESHOLDS = {
    "faithfulness":     0.75,
    "answer_relevance": 0.70,
}


def print_generation_report(
    results: list[GenerationResult],
    agg: dict,
    verbose: bool = False,
) -> None:
    """
    Prints a formatted generation evaluation report to stdout.

    Args:
        results: Per-query results from evaluate_generation().
        agg: Aggregate dict from evaluate_generation().
        verbose: If True, prints per-claim faithfulness breakdown and
                 judge reasoning for every query.
    """
    print()
    print(_BOLD + "=" * 70 + _RESET)
    print(_BOLD + "  GENERATION EVALUATION REPORT" + _RESET)
    print(_BOLD + "=" * 70 + _RESET)
    print()

    def _fmt(label: str, score: float, threshold: float, thresh_label: str) -> str:
        color = _GREEN if score >= threshold else _RED
        tick  = (_GREEN + "✓" + _RESET) if score >= threshold else (_RED + "✗" + _RESET)
        return (
            f"  {label:<28} {color}{score:.4f}{_RESET}  "
            f"threshold {thresh_label}  {tick}"
        )

    f  = agg["mean_faithfulness"]
    ar = agg["mean_answer_relevance"]
    print(_fmt("Faithfulness     (mean)", f,  THRESHOLDS["faithfulness"],     "≥ 0.75"))
    print(_fmt("Answer Relevance (mean)", ar, THRESHOLDS["answer_relevance"], "≥ 0.70"))
    print()

    # ── By category ──────────────────────────────────────────────────────────
    print(_BOLD + "  By category:" + _RESET)
    for cat in ["direct_lookup", "single_hop", "multi_hop", "adversarial"]:
        c = agg.get("by_category", {}).get(cat)
        if c:
            print(
                f"    {cat:<15}  n={c['n']:2d}  "
                f"F={c['faithfulness']:.3f}  AR={c['answer_relevance']:.3f}"
            )
    print()

    # ── Per-query table ───────────────────────────────────────────────────────
    print(_BOLD + "  Per-query results:" + _RESET)
    print(f"  {'ID':<8} {'Category':<14} {'F':>6} {'AR':>6}  {'Result':<6}  Answer preview")
    print("  " + "-" * 74)
    for r in results:
        passed = (
            r.faithfulness     >= THRESHOLDS["faithfulness"] and
            r.answer_relevance >= THRESHOLDS["answer_relevance"]
        )
        status  = (_GREEN + "PASS" + _RESET) if passed else (_RED + "FAIL" + _RESET)
        preview = (r.generated_answer[:55].replace("\n", " ")) if r.generated_answer else "[empty]"
        print(
            f"  {r.query_id:<8} {r.category:<14} "
            f"{r.faithfulness:>6.3f} {r.answer_relevance:>6.3f}  {status}  {preview}"
        )

    total_elapsed = sum(r.elapsed_seconds for r in results)
    total_claims  = sum(len(r.faithfulness_claims) for r in results)
    total_unsup   = sum(
        sum(1 for c in r.faithfulness_claims if not c.get("supported", True))
        for r in results
    )
    print()
    print(
        f"  Queries: {len(results)}  |  Gemini calls: ~{len(results) * 2}  |"
        f"  Claims: {total_claims} ({total_unsup} unsupported)  |"
        f"  Elapsed: {total_elapsed:.1f}s"
    )
    print(_BOLD + "=" * 70 + _RESET)
    print()

    # ── Verbose: per-claim breakdown ──────────────────────────────────────────
    if verbose:
        for r in results:
            unsupported = [c for c in r.faithfulness_claims if not c.get("supported", True)]
            print(f"  {_BOLD}[{r.query_id}]{_RESET}  F={r.faithfulness:.3f}  AR={r.answer_relevance:.3f}")
            print(f"    Q: {r.query}")
            print(f"    Relevance reasoning: {r.relevance_reasoning[:120]}")
            if r.faithfulness_claims:
                print(f"    Claims ({len(r.faithfulness_claims)}, {len(unsupported)} unsupported):")
                for c in r.faithfulness_claims:
                    mark = "  ✗" if not c.get("supported", True) else "  ✓"
                    print(f"      {mark}  {c.get('claim','')}")
            print()
