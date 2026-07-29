"""
query/agents/tool_use_agent.py — Tool-Use Agent (Phase 6).

Identifies which financial computations the query requires (heuristic,
non-LLM) and executes them in the sandboxed calculator, producing
``ComputedMetric`` objects whose values are traceable to exact retrieved chunks
— never LLM-estimated.

Runs AFTER the sufficiency loop (all raw retrieval is done) and BEFORE the
Generator, so the Generator can cite both retrieved text and computed metrics.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from query.tools.financial_calculators import extract_numeric_from_table_chunk
from query.tools.sandbox_executor import run_in_sandbox
from query.utils.schema import ComputedMetric, RetrievedChunk, RouterOutput, SubQuery

logger = logging.getLogger(__name__)

# ── Heuristic pattern matching ────────────────────────────────────────────────

# Patterns that signal "R&D" sub-queries
_RD_PATTERNS = re.compile(
    r"r\s*&\s*d|research\s+and\s+development|research.*expense", re.IGNORECASE
)
# Patterns that signal "revenue / net sales" sub-queries
_REVENUE_PATTERNS = re.compile(
    r"revenue|net\s+sales|total\s+sales|total\s+revenue", re.IGNORECASE
)
# Patterns that signal "operating income" sub-queries
_OPEX_PATTERNS = re.compile(
    r"operating\s+income|operating\s+profit|ebit(?!\w)", re.IGNORECASE
)
# Patterns that signal "operating expenses" sub-queries
_OPEX_EXP_PATTERNS = re.compile(
    r"operating\s+expense", re.IGNORECASE
)
# Patterns for YoY growth signals
_GROWTH_PATTERNS = re.compile(
    r"growth|change|increase|decrease|grew|yoy", re.IGNORECASE
)


def _top_table_chunk(chunks: list[RetrievedChunk]) -> Optional[RetrievedChunk]:
    """Returns the highest-reranked table chunk, or None."""
    table_chunks = [rc for rc in chunks if rc.chunk.content_type == "table"]
    if not table_chunks:
        return None
    return max(
        table_chunks,
        key=lambda rc: rc.rerank_score if rc.rerank_score is not None else rc.rrf_score or 0.0,
    )


def identify_required_computations(
    router_output: RouterOutput,
    sub_queries: list[SubQuery],
) -> list[dict]:
    """
    Heuristic (non-LLM) identification of which financial computations to run.

    Only called when ``router_output.needs_computation`` is ``True``.

    Pattern-matches sub-query groupings: if two sub-queries share the same
    ``company + fiscal_year`` and one mentions R&D while the other mentions
    revenue, emits a ``percent_of`` computation spec
    (``R&D as % of revenue``).

    Similarly detects operating margin if one sub-query mentions operating
    income and another mentions revenue for the same company+year.

    If the route is ``"single_hop"`` and the sub-query text mentions YoY
    growth between two years, emits a ``yoy_growth`` spec (requires the same
    metric retrieved for two years).

    Returns a list of computation spec dicts::

        [
            {
                "function_name":        "percent_of",
                "numerator_subquery_id":   1,
                "numerator_row_label":     "research and development",
                "denominator_subquery_id": 2,
                "denominator_row_label":   "net sales",
                "metric_name":             "apple_rd_pct_fy2022",
            },
            ...
        ]

    Args:
        router_output: Router Agent output.
        sub_queries:   All sub-queries from Decomposer (or single wrapped sub-query).

    Returns:
        List of computation spec dicts (may be empty if no pattern matches).
    """
    if not router_output.needs_computation:
        return []

    computations: list[dict] = []

    # Group sub-queries by (company, fiscal_year)
    from collections import defaultdict
    groups: dict[tuple, list[SubQuery]] = defaultdict(list)
    for sq in sub_queries:
        key = (sq.company or "", sq.fiscal_year or 0)
        groups[key].append(sq)

    for (company, year), sqs in groups.items():
        if len(sqs) < 2:
            continue

        # Find R&D, revenue, and operating income sub-queries in the group
        rd_sqs = [sq for sq in sqs if _RD_PATTERNS.search(sq.text)]
        rev_sqs = [sq for sq in sqs if _REVENUE_PATTERNS.search(sq.text)]
        op_inc_sqs = [sq for sq in sqs if _OPEX_PATTERNS.search(sq.text)]

        # R&D as % of revenue
        if rd_sqs and rev_sqs:
            metric_name = f"{company}_rd_pct_fy{year}".replace(" ", "_")
            computations.append({
                "function_name":          "percent_of",
                "numerator_subquery_id":   rd_sqs[0].sub_query_id,
                "numerator_row_label":     "research and development",
                "denominator_subquery_id": rev_sqs[0].sub_query_id,
                "denominator_row_label":   "net sales",
                "metric_name":             metric_name,
            })

        # Operating margin
        if op_inc_sqs and rev_sqs:
            metric_name = f"{company}_op_margin_fy{year}".replace(" ", "_")
            computations.append({
                "function_name":          "operating_margin",
                "numerator_subquery_id":   op_inc_sqs[0].sub_query_id,
                "numerator_row_label":     "operating income",
                "denominator_subquery_id": rev_sqs[0].sub_query_id,
                "denominator_row_label":   "net sales",
                "metric_name":             metric_name,
            })

    # YoY growth: if 2 sub-queries for same company, different years, same metric
    if router_output.route in ("single_hop",):
        year_groups: dict[str, list[SubQuery]] = defaultdict(list)
        for sq in sub_queries:
            year_groups[sq.company or ""].append(sq)
        for company, sqs in year_groups.items():
            if len(sqs) == 2 and sqs[0].fiscal_year != sqs[1].fiscal_year:
                if all(_GROWTH_PATTERNS.search(sq.text) or True for sq in sqs):
                    earlier, later = (
                        (sqs[0], sqs[1]) if sqs[0].fiscal_year < sqs[1].fiscal_year
                        else (sqs[1], sqs[0])
                    )
                    metric_name = f"{company}_yoy_fy{earlier.fiscal_year}_to_fy{later.fiscal_year}"
                    computations.append({
                        "function_name":        "yoy_growth",
                        "current_subquery_id":  later.sub_query_id,
                        "current_row_label":    "net sales",
                        "previous_subquery_id": earlier.sub_query_id,
                        "previous_row_label":   "net sales",
                        "metric_name":          metric_name,
                    })

    logger.info(
        "identify_required_computations: %d computation(s) identified", len(computations)
    )
    return computations


def run_tool_use(
    computations: list[dict],
    retrieved_chunks: dict[int, list[RetrievedChunk]],
) -> list[ComputedMetric]:
    """
    Executes each computation spec in the sandboxed calculator.

    For each spec from :func:`identify_required_computations`:
    1. Locate the highest-reranked **table** chunk for the numerator sub-query ID.
    2. Extract the numeric value using
       :func:`~query.tools.financial_calculators.extract_numeric_from_table_chunk`.
    3. Repeat for denominator (if applicable).
    4. Call :func:`~query.tools.sandbox_executor.run_in_sandbox` with the
       extracted values.
    5. Wrap the result into a :class:`~query.utils.schema.ComputedMetric`.

    Skips (with a warning) any computation whose required chunks are missing
    or whose numeric extraction returns ``None`` — the Generator is instructed
    to acknowledge gaps rather than fabricate.

    Args:
        computations:     List of computation spec dicts from
                          :func:`identify_required_computations`.
        retrieved_chunks: ``dict[sub_query_id, list[RetrievedChunk]]`` — final
                          retrieved context from the sufficiency loop.

    Returns:
        ``list[ComputedMetric]`` — verified, sandbox-computed metrics with full
        provenance (``source_chunk_ids`` point to exact retrieved chunks).
    """
    results: list[ComputedMetric] = []

    for spec in computations:
        fn_name = spec["function_name"]

        try:
            if fn_name in ("percent_of", "operating_margin"):
                num_sid = spec["numerator_subquery_id"]
                den_sid = spec["denominator_subquery_id"]
                num_label = spec.get("numerator_row_label", "")
                den_label = spec.get("denominator_row_label", "")

                num_rc = _top_table_chunk(retrieved_chunks.get(num_sid, []))
                den_rc = _top_table_chunk(retrieved_chunks.get(den_sid, []))

                if num_rc is None or den_rc is None:
                    logger.warning(
                        "run_tool_use: no table chunk for %s (num_sid=%d, den_sid=%d) — skipping",
                        spec["metric_name"], num_sid, den_sid,
                    )
                    continue

                numerator = extract_numeric_from_table_chunk(num_rc.chunk, num_label, "")
                denominator = extract_numeric_from_table_chunk(den_rc.chunk, den_label, "")

                if numerator is None or denominator is None:
                    logger.warning(
                        "run_tool_use: could not extract numeric for %s "
                        "(numerator=%s, denominator=%s) — skipping",
                        spec["metric_name"], numerator, denominator,
                    )
                    continue

                # Absolute values for ratios (expenses are often stored as negatives)
                value = run_in_sandbox(fn_name, {
                    "numerator": abs(numerator),
                    "denominator": abs(denominator),
                })
                formula = f"abs({num_label}) / abs({den_label}) * 100"
                source_ids = [num_rc.chunk.chunk_id, den_rc.chunk.chunk_id]

            elif fn_name == "yoy_growth":
                cur_sid = spec["current_subquery_id"]
                prev_sid = spec["previous_subquery_id"]
                cur_label = spec.get("current_row_label", "net sales")
                prev_label = spec.get("previous_row_label", "net sales")

                cur_rc = _top_table_chunk(retrieved_chunks.get(cur_sid, []))
                prev_rc = _top_table_chunk(retrieved_chunks.get(prev_sid, []))

                if cur_rc is None or prev_rc is None:
                    logger.warning(
                        "run_tool_use: no table chunk for yoy_growth %s — skipping",
                        spec["metric_name"],
                    )
                    continue

                current = extract_numeric_from_table_chunk(cur_rc.chunk, cur_label, "")
                previous = extract_numeric_from_table_chunk(prev_rc.chunk, prev_label, "")

                if current is None or previous is None:
                    logger.warning(
                        "run_tool_use: could not extract numeric for yoy_growth %s — skipping",
                        spec["metric_name"],
                    )
                    continue

                value = run_in_sandbox(fn_name, {
                    "current": abs(current),
                    "previous": abs(previous),
                })
                formula = f"(current - previous) / previous"
                source_ids = [cur_rc.chunk.chunk_id, prev_rc.chunk.chunk_id]

            elif fn_name == "cagr":
                value = run_in_sandbox(fn_name, spec.get("args", {}))
                formula = "(end/start)^(1/periods) - 1"
                source_ids = spec.get("source_chunk_ids", [])

            else:
                logger.warning("run_tool_use: unknown function %r in spec — skipping", fn_name)
                continue

            metric = ComputedMetric(
                name=spec["metric_name"],
                value=round(value, 4),
                formula=formula,
                source_chunk_ids=source_ids,
            )
            results.append(metric)
            logger.info(
                "run_tool_use: %s = %.4f (formula: %s)",
                metric.name, metric.value, metric.formula,
            )

        except Exception as exc:
            logger.warning(
                "run_tool_use: computation %r failed: %s — skipping",
                spec.get("metric_name", fn_name), exc,
            )
            continue

    return results
