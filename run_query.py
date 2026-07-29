"""
run_query.py — CLI entry point for the Financial Research Agent query pipeline.

Usage:
  python run_query.py "What was Apple's FY2023 revenue?"
  python run_query.py "Compare R&D as % of revenue for Apple, Microsoft, and NVIDIA, FY2022-FY2024" --json
  python run_query.py "..." --no-cache   # bypass cache READ (writes still happen)

Loads .env automatically so credentials don't need to be exported manually.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("run_query")


def _format_answer(payload: dict) -> str:
    """Pretty-prints the pipeline result for console output."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("ANSWER")
    lines.append("=" * 72)
    lines.append(payload.get("final_answer", "(no answer)"))
    lines.append("")

    verdict = payload.get("verdict", "")
    flagged = payload.get("flagged_claims", [])
    if verdict:
        lines.append(f"Verdict: {verdict.upper()}")
    if flagged:
        lines.append(f"Flagged claims ({len(flagged)}):")
        for claim in flagged:
            lines.append(f"  - {claim}")

    lines.append("=" * 72)
    return "\n".join(lines)


def main() -> None:
    """
    CLI entry point.

    Arguments:
      query         The user's financial research question (positional).
      --no-cache    Bypass cache READ — for testing pipeline changes.
                    Cache writes are still performed after this run.
      --json        Print the full ``final_answer_payload`` as JSON instead
                    of the formatted answer text.
    """
    parser = argparse.ArgumentParser(
        description="Financial Research Agent — Query Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_query.py \"What was Apple's FY2023 revenue?\"\n"
            "  python run_query.py \"Compare R&D % of revenue across Apple, Microsoft, NVIDIA FY2022-FY2024\" --json\n"
            "  python run_query.py \"What was NVIDIA's net income FY2024?\" --no-cache\n"
        ),
    )
    parser.add_argument("query", type=str, help="Financial research question")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass cache read (still writes result to cache after pipeline run)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output full JSON payload instead of formatted answer",
    )
    args = parser.parse_args()

    from query.pipeline import init_pipeline_resources, run_query

    logger.info("Initialising pipeline resources...")
    resources = init_pipeline_resources(no_cache_read=args.no_cache)

    logger.info("Running query: %r", args.query)
    result = run_query(args.query, resources)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(_format_answer(result))


if __name__ == "__main__":
    main()
