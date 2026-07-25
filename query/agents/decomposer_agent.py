"""
query/agents/decomposer_agent.py — Decomposer Agent: breaks a multi-hop query into
atomic, independently-retrievable sub-questions.

Only called when ``router_output.route == "multi_hop"``. For ``direct_lookup``
and ``single_hop`` routes, ``router_agent.build_direct_lookup_subquery`` is used
instead (saves one Gemini call per query).

Example: "R&D as % of revenue, Apple vs Microsoft vs NVIDIA, FY2022–FY2024"
→ 18 sub-queries (3 companies × 3 years × 2 raw metrics: R&D expense + Revenue).
The Tool-Use Agent then computes the ratio from the retrieved raw figures.
"""

from __future__ import annotations

import logging

from query.utils.gemini_client import GeminiCallCounter, call_structured
from query.utils.prompts import DECOMPOSER_SYSTEM_PROMPT, build_decomposer_prompt
from query.utils.schema import RouterOutput, SubQuery

logger = logging.getLogger(__name__)

# ── Gemini response schema ────────────────────────────────────────────────────

DECOMPOSER_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text":        {"type": "string"},
                    "company":     {"type": "string"},
                    "fiscal_year": {"type": "integer"},
                    "section_hint": {"type": "string"},
                },
                "required": ["text", "company", "fiscal_year"],
            },
        },
    },
    "required": ["sub_queries"],
}


# ── Public API ────────────────────────────────────────────────────────────────

def decompose_query(
    query: str,
    router_output: RouterOutput,
    gemini_client,
    call_counter: GeminiCallCounter,
) -> list[SubQuery]:
    """
    Decomposes a multi-hop query into ordered atomic sub-queries.

    Steps:
    1. Build the decomposer user prompt via
       :func:`~query.utils.prompts.build_decomposer_prompt`.
    2. Call Gemini with structured JSON output constrained by
       ``DECOMPOSER_RESPONSE_SCHEMA``.
    3. Convert the raw dict list to :class:`~query.utils.schema.SubQuery` objects
       via :func:`assign_subquery_ids`, preserving Gemini's ordering (dependent
       sub-queries before the computations that need them).

    Only called when ``router_output.route == "multi_hop"``.

    Example: ``"R&D as % of revenue, Apple vs Microsoft vs NVIDIA, FY2022–FY2024"``
    → 18 sub-queries: for each of ``[apple, microsoft, nvidia]`` × ``[2022, 2023, 2024]``:

    - ``"What was {company}'s R&D expense in FY{year}?"``
    - ``"What was {company}'s total revenue in FY{year}?"``

    The Tool-Use Agent then computes R&D/Revenue × 100 for each pair.

    Args:
        query:         Raw user query string.
        router_output: Validated output of the Router Agent.
        gemini_client: Initialised ``google.genai.Client``.
        call_counter:  Shared :class:`~query.utils.gemini_client.GeminiCallCounter`.

    Returns:
        Ordered ``list[SubQuery]`` with sequential ``sub_query_id`` values (1..N).
    """
    prompt = build_decomposer_prompt(query, router_output)
    result = call_structured(
        gemini_client,
        DECOMPOSER_SYSTEM_PROMPT,
        prompt,
        DECOMPOSER_RESPONSE_SCHEMA,
        call_counter,
    )
    raw_list: list[dict] = result.get("sub_queries", [])
    sub_queries = assign_subquery_ids(raw_list)
    logger.info(
        "decompose_query: %d sub-queries from Gemini for query=%r",
        len(sub_queries), query[:80],
    )
    return sub_queries


def assign_subquery_ids(raw_sub_queries: list[dict]) -> list[SubQuery]:
    """
    Converts raw dicts from Gemini's JSON output into
    :class:`~query.utils.schema.SubQuery` dataclasses.

    Assigns sequential ``sub_query_id`` values starting at 1, preserving the
    order Gemini returned (dependent sub-queries come before the computations
    that need them, as instructed in ``DECOMPOSER_SYSTEM_PROMPT``).

    Missing optional fields (``section_hint``) default to ``None``.

    Args:
        raw_sub_queries: List of dicts from Gemini's structured JSON response.

    Returns:
        ``list[SubQuery]`` with ``sub_query_id`` 1..N.
    """
    sub_queries: list[SubQuery] = []
    for idx, raw in enumerate(raw_sub_queries, start=1):
        company_raw = raw.get("company", "")
        sub_queries.append(
            SubQuery(
                sub_query_id=idx,
                text=raw.get("text", ""),
                company=company_raw.lower() if company_raw else None,
                fiscal_year=int(raw["fiscal_year"]) if raw.get("fiscal_year") else None,
                section_hint=raw.get("section_hint") or None,
                retry_count=0,
            )
        )
    return sub_queries
