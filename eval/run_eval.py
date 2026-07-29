"""
eval/run_eval.py — CLI entry point for the full evaluation pipeline.

Usage:
  python eval/run_eval.py                             # all 30 queries (~180 Gemini calls)
  python eval/run_eval.py --category direct_lookup    # cheapest: 10 queries, ~50 Gemini calls
  python eval/run_eval.py --retrieval-only            # 0 Gemini calls, instant retrieval check
  python eval/run_eval.py --output eval/results/v1.json
  python eval/run_eval.py --json                      # print full JSON to stdout
  python eval/run_eval.py --verbose                   # per-claim faithfulness breakdown

Pipeline for each query (unless --retrieval-only):
  1. Retrieval eval  — runs retrieve_for_subquery; 0 Gemini calls; yields P@5 and R@5.
  2. Full pipeline   — runs run_query(); ~5 Gemini calls (router/decomposer/generator/verifier).
  3. Generation eval — LLM-as-Judge; 2 Gemini calls (faithfulness + answer relevance).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Ensure project root is on sys.path when invoked as `python eval/run_eval.py`
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv()

from query.pipeline import init_pipeline_resources, run_query
from eval.dataset import load_eval_dataset
from eval.retrieval_eval import (
    RetrievalResult,
    evaluate_retrieval,
    print_retrieval_report,
)
from eval.generation_eval import (
    GenerationResult,
    evaluate_generation,
    print_generation_report,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Thresholds (mirrors EVALUATION_PIPELINE.md §7) ───────────────────────────

THRESHOLDS: dict[str, float] = {
    "mean_precision_at_5":   0.55,
    "mean_recall_at_5":      0.65,
    "mean_faithfulness":     0.75,
    "mean_answer_relevance": 0.70,
}

_GREEN = "\033[92m"
_RED   = "\033[91m"
_BOLD  = "\033[1m"
_RESET = "\033[0m"


# ── Summary table ─────────────────────────────────────────────────────────────

def _score_cell(score: float, threshold: float) -> str:
    color = _GREEN if score >= threshold else _RED
    tick  = "✓" if score >= threshold else "✗"
    return f"{color}{score:.4f}{_RESET}  {tick}"


def print_summary_table(
    ret_agg: dict,
    gen_agg: dict | None,
    elapsed: float,
) -> None:
    """
    Prints the combined box-drawing summary table:

    ╔══════════════════════════════════╦═══════════╦══════════════╗
    ║ METRIC                           ║  Score    ║  Threshold   ║
    ╠══════════════════════════════════╬═══════════╬══════════════╣
    ║ RETRIEVAL                        ║           ║              ║
    ║   Precision@5                    ║  0.6800   ║  ≥ 0.55  ✓  ║
    ║   Recall@5                       ║  0.7400   ║  ≥ 0.65  ✓  ║
    ║ GENERATION                       ║           ║              ║
    ║   Faithfulness                   ║  0.8600   ║  ≥ 0.75  ✓  ║
    ║   Answer Relevance               ║  0.8100   ║  ≥ 0.70  ✓  ║
    ╚══════════════════════════════════╩═══════════╩══════════════╝
      Total elapsed: 847.3s
    """
    SEP  = "╠══════════════════════════════════╬═══════════╦══════════════╣"
    TOP  = "╔══════════════════════════════════╦═══════════╦══════════════╗"
    BOT  = "╚══════════════════════════════════╩═══════════╩══════════════╝"
    HDR  = "║ METRIC                           ║  Score    ║  Threshold   ║"
    SECT = "║ {:<32} ║           ║              ║"
    ROW  = "║ {:<32} ║  {}  ║  {}  ║"

    def _row(label: str, score: float, thresh: float, thresh_str: str) -> str:
        color = _GREEN if score >= thresh else _RED
        tick  = (_GREEN + "✓" + _RESET) if score >= thresh else (_RED + "✗" + _RESET)
        score_str  = f"{color}{score:.4f}{_RESET}"
        thresh_cell = f"{thresh_str}  {tick} "
        return f"║ {label:<32} ║  {score_str}   ║  {thresh_cell}  ║"

    print()
    print(TOP)
    print(HDR)
    print(SEP)
    print(SECT.format("RETRIEVAL"))
    print(_row("  Precision@5", ret_agg["mean_precision_at_5"], THRESHOLDS["mean_precision_at_5"], "≥ 0.55"))
    print(_row("  Recall@5",    ret_agg["mean_recall_at_5"],    THRESHOLDS["mean_recall_at_5"],    "≥ 0.65"))

    if gen_agg is not None:
        print(SECT.format("GENERATION"))
        print(_row("  Faithfulness",     gen_agg["mean_faithfulness"],     THRESHOLDS["mean_faithfulness"],     "≥ 0.75"))
        print(_row("  Answer Relevance", gen_agg["mean_answer_relevance"], THRESHOLDS["mean_answer_relevance"], "≥ 0.70"))

    print(BOT)
    print(f"  Total elapsed: {elapsed:.1f}s")
    print()


# ── JSON serialisation ────────────────────────────────────────────────────────

def _ret_result_to_dict(r: RetrievalResult) -> dict:
    return {
        "query_id":             r.query_id,
        "query":                r.query,
        "category":             r.category,
        "precision_at_5":       round(r.precision, 4),
        "recall_at_5":          round(r.recall, 4),
        "retrieved_chunk_ids":  r.retrieved_chunk_ids,
        "relevant_chunk_ids":   r.relevant_chunk_ids,
        "n_sub_queries":        r.n_sub_queries,
        "elapsed_seconds":      round(r.elapsed_seconds, 2),
    }


def _gen_result_to_dict(r: GenerationResult) -> dict:
    return {
        "query_id":             r.query_id,
        "query":                r.query,
        "category":             r.category,
        "faithfulness":         round(r.faithfulness, 4),
        "answer_relevance":     round(r.answer_relevance, 4),
        "generated_answer":     r.generated_answer,
        "faithfulness_claims":  r.faithfulness_claims,
        "relevance_reasoning":  r.relevance_reasoning,
        "n_chunks_used":        r.n_chunks_used,
        "elapsed_seconds":      round(r.elapsed_seconds, 2),
    }


def build_json_report(
    ret_results: list[RetrievalResult],
    ret_agg: dict,
    gen_results: list[GenerationResult] | None,
    gen_agg: dict | None,
    elapsed: float,
) -> dict:
    """Builds the full JSON report dict for --output / --json."""
    aggregate: dict = {
        "mean_precision_at_5": round(ret_agg["mean_precision_at_5"], 4),
        "mean_recall_at_5":    round(ret_agg["mean_recall_at_5"],    4),
    }
    if gen_agg is not None:
        aggregate["mean_faithfulness"]     = round(gen_agg["mean_faithfulness"],     4)
        aggregate["mean_answer_relevance"] = round(gen_agg["mean_answer_relevance"], 4)

    report: dict = {
        "timestamp":        datetime.now(tz=timezone.utc).isoformat(),
        "elapsed_seconds":  round(elapsed, 1),
        "aggregate":        aggregate,
        "retrieval_by_category": {
            cat: {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in vals.items()}
            for cat, vals in ret_agg.get("by_category", {}).items()
        },
        "retrieval_results": [_ret_result_to_dict(r) for r in ret_results],
    }

    if gen_results is not None:
        report["generation_by_category"] = {
            cat: {k: round(v, 4) if isinstance(v, float) else v
                  for k, v in vals.items()}
            for cat, vals in (gen_agg or {}).get("by_category", {}).items()
        }
        report["generation_results"] = [_gen_result_to_dict(r) for r in gen_results]

    return report


# ── Full-pipeline runner ──────────────────────────────────────────────────────

def run_pipeline_for_query(query: str, resources: dict) -> dict:
    """
    Calls run_query and returns the pipeline result dict.
    Returns {"final_answer": "", ...} on failure so generation eval
    can still run (and will score the empty answer low).
    """
    try:
        return run_query(query, resources)
    except Exception as exc:
        logger.error("run_query failed for query %r: %s", query[:60], exc)
        return {"final_answer": "", "citations": [], "verdict": "partial",
                "flagged_claims": [], "cached_at": ""}


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python eval/run_eval.py",
        description="Financial Research Agent — evaluation pipeline runner.",
    )
    parser.add_argument(
        "--category",
        action="append",
        choices=["direct_lookup", "single_hop", "multi_hop", "adversarial"],
        default=None,
        help="Evaluate one or more categories. Repeatable (e.g. --category single_hop --category multi_hop). Default: all 30 queries.",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run only Precision@5 / Recall@5. Zero Gemini calls.",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Save full JSON report to FILE (e.g. eval/results/v1.json).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report to stdout after the human-readable report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query chunk hits/misses and per-claim faithfulness breakdown.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # When --json is set, route human-readable progress to stderr so stdout
    # contains only valid JSON (enabling: python eval/run_eval.py --json | jq .)
    progress = sys.stderr if args.json else sys.stdout

    # ── 1. Init resources ─────────────────────────────────────────────────────
    print("Initialising pipeline resources …", file=progress, flush=True)
    resources = init_pipeline_resources(no_cache_read=True)
    print("Resources ready.\n", file=progress, flush=True)

    # ── 2. Load dataset ───────────────────────────────────────────────────────
    categories = args.category if args.category else None
    eval_queries = load_eval_dataset(categories)
    n = len(eval_queries)
    cat_label = ", ".join(categories) if categories else "all categories"
    print(f"Loaded {n} queries ({cat_label}).", file=progress, flush=True)

    gemini_client = resources["gemini_client"]
    call_counter  = resources["call_counter"]

    wall_start = time.perf_counter()

    # ── 3. Retrieval evaluation ───────────────────────────────────────────────
    print(f"\n[1/2] Retrieval evaluation ({n} queries, 0 Gemini calls) …", file=progress, flush=True)
    ret_results, ret_agg = evaluate_retrieval(eval_queries, resources)

    if args.retrieval_only:
        elapsed = time.perf_counter() - wall_start
        if not args.json:
            print_retrieval_report(ret_results, ret_agg, verbose=args.verbose)
            print_summary_table(ret_agg, gen_agg=None, elapsed=elapsed)

        if args.output or args.json:
            report = build_json_report(ret_results, ret_agg, None, None, elapsed)
            _output_json(report, args.output, args.json)
        return

    # ── 4. Full pipeline — generate answers ───────────────────────────────────
    print(
        f"\n[2/2] Running full pipeline for {n} queries "
        f"(~{n * 5} Gemini calls for generation + ~{n * 2} for judges) …",
        file=progress, flush=True,
    )
    pipeline_results: list[dict] = []
    for i, eq in enumerate(eval_queries):
        print(f"  [{i + 1}/{n}] {eq.id}  {eq.query[:65]}", file=progress, flush=True)
        pipeline_results.append(run_pipeline_for_query(eq.query, resources))

    # ── 5. Generation evaluation ──────────────────────────────────────────────
    chunk_texts_per_query = [r.retrieved_chunk_texts for r in ret_results]
    gen_results, gen_agg  = evaluate_generation(
        eval_queries,
        pipeline_results,
        chunk_texts_per_query,
        gemini_client,
        call_counter,
    )

    # ── 6. Reports ────────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - wall_start

    if not args.json:
        print_retrieval_report(ret_results, ret_agg, verbose=args.verbose)
        print_generation_report(gen_results, gen_agg, verbose=args.verbose)
        print_summary_table(ret_agg, gen_agg, elapsed=elapsed)

    # ── 7. JSON output ────────────────────────────────────────────────────────
    if args.output or args.json:
        report = build_json_report(ret_results, ret_agg, gen_results, gen_agg, elapsed)
        _output_json(report, args.output, args.json)


def _output_json(report: dict, output_path: str | None, print_stdout: bool) -> None:
    json_str = json.dumps(report, indent=2, ensure_ascii=False)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(json_str)
        print(f"JSON report saved → {output_path}", flush=True)

    if print_stdout:
        print(json_str)


if __name__ == "__main__":
    main()
