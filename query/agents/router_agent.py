"""
query/agents/router_agent.py — Router Agent: classifies a query and extracts entities.

This is the first Gemini call in the query pipeline and the most consequential:
- ``direct_lookup`` skips the Decomposer AND the sufficiency retry loop (saves 4+ calls)
- ``single_hop`` skips the Decomposer but keeps one sufficiency check
- ``multi_hop`` triggers full decomposition into N atomic sub-queries

Per system_design.md §5's quota-mitigation strategy, every shortcut here
directly reduces Gemini API calls for the day.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from query.config import PINECONE_NAMESPACES, ROUTE_DIRECT_LOOKUP, ROUTE_SINGLE_HOP, ROUTE_MULTI_HOP
from query.utils.gemini_client import GeminiCallCounter, call_structured
from query.utils.prompts import ROUTER_SYSTEM_PROMPT, build_router_prompt
from query.utils.schema import RouterOutput, SubQuery

logger = logging.getLogger(__name__)

# ── Valid entity sets (from ingestion corpus) ─────────────────────────────────
_VALID_COMPANIES: set[str] = set(PINECONE_NAMESPACES)   # {"apple", "microsoft", "nvidia"}
_VALID_YEARS: set[int] = {2022, 2023, 2024}

# ── Gemini response schema ────────────────────────────────────────────────────

ROUTER_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "route": {
            "type": "string",
            "enum": ["direct_lookup", "single_hop", "multi_hop"],
        },
        "companies": {
            "type": "array",
            "items": {"type": "string"},
        },
        "years": {
            "type": "array",
            "items": {"type": "integer"},
        },
        "needs_computation": {"type": "boolean"},
    },
    "required": ["route", "companies", "years", "needs_computation"],
}


# ── Public API ────────────────────────────────────────────────────────────────

def classify_query(
    query: str,
    gemini_client,
    call_counter: GeminiCallCounter,
) -> RouterOutput:
    """
    Classifies a financial research query and extracts company/year entities.

    Steps:
    1. Build the router user prompt via :func:`~query.utils.prompts.build_router_prompt`.
    2. Call Gemini with structured JSON output constrained by ``ROUTER_RESPONSE_SCHEMA``.
    3. Validate ``companies`` — drop any values not in ``{"apple","microsoft","nvidia"}``;
       default to all three if none remain after filtering.
    4. Validate ``years`` — drop any values not in ``{2022, 2023, 2024}``; default to
       all three if none remain.
    5. Return a :class:`~query.utils.schema.RouterOutput`.

    This ONE Gemini call determines the entire downstream path: ``direct_lookup``
    skips Decomposer + sufficiency retry loop, saving 4+ calls per query
    (per system_design.md §5's quota-mitigation strategy).

    Args:
        query:        Raw user query string.
        gemini_client: Initialised ``google.genai.Client``.
        call_counter: Shared :class:`~query.utils.gemini_client.GeminiCallCounter`.

    Returns:
        :class:`~query.utils.schema.RouterOutput` with validated entities.
    """
    prompt = build_router_prompt(query)
    result = call_structured(
        gemini_client,
        ROUTER_SYSTEM_PROMPT,
        prompt,
        ROUTER_RESPONSE_SCHEMA,
        call_counter,
    )

    # Validate companies
    raw_companies: list[str] = [c.lower() for c in result.get("companies", [])]
    valid_companies = [c for c in raw_companies if c in _VALID_COMPANIES]
    if len(valid_companies) < len(raw_companies):
        dropped = set(raw_companies) - _VALID_COMPANIES
        logger.warning("classify_query: dropped unrecognised companies: %s", dropped)
    if not valid_companies:
        logger.warning("classify_query: no valid companies extracted — defaulting to all three")
        valid_companies = list(_VALID_COMPANIES)

    # Validate years
    raw_years: list[int] = [int(y) for y in result.get("years", [])]
    valid_years = [y for y in raw_years if y in _VALID_YEARS]
    if len(valid_years) < len(raw_years):
        dropped_y = set(raw_years) - _VALID_YEARS
        logger.warning("classify_query: dropped out-of-range years: %s", dropped_y)
    if not valid_years:
        logger.warning("classify_query: no valid years extracted — defaulting to all three")
        valid_years = sorted(_VALID_YEARS)

    route: str = result.get("route", "single_hop")
    needs_computation: bool = bool(result.get("needs_computation", False))

    # Programmatic override: queries spanning multiple companies always require
    # separate retrievals per company — force multi_hop so the Decomposer Agent
    # builds one sub-query per company×year, preventing all companies from
    # competing for the same top-5 rerank slots.
    if len(valid_companies) > 1 and route != ROUTE_MULTI_HOP:
        logger.info(
            "classify_query: upgrading route %s → multi_hop "
            "(multiple companies detected: %s)",
            route, valid_companies,
        )
        route = ROUTE_MULTI_HOP

    logger.info(
        "classify_query: route=%s | companies=%s | years=%s | needs_computation=%s",
        route, valid_companies, valid_years, needs_computation,
    )

    return RouterOutput(
        route=route,
        companies=valid_companies,
        years=valid_years,
        needs_computation=needs_computation,
    )


def build_direct_lookup_subquery(query: str, router_output: RouterOutput) -> SubQuery:
    """
    Wraps the original query as a single :class:`~query.utils.schema.SubQuery`
    for ``direct_lookup`` and ``single_hop`` routes (no Decomposer needed).

    Sets ``company`` to the single extracted company (or ``None`` if multiple),
    and ``fiscal_year`` to the single extracted year (or ``None`` if multiple).

    Args:
        query:         Raw user query string.
        router_output: Output of :func:`classify_query`.

    Returns:
        A :class:`~query.utils.schema.SubQuery` with ``sub_query_id=1``.
    """
    company: Optional[str] = (
        router_output.companies[0] if len(router_output.companies) == 1 else None
    )
    fiscal_year: Optional[int] = (
        router_output.years[0] if len(router_output.years) == 1 else None
    )
    return SubQuery(
        sub_query_id=1,
        text=query,
        company=company,
        fiscal_year=fiscal_year,
        section_hint=None,
        retry_count=0,
    )
