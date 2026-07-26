"""
query/retrieval/metadata_filter.py — Metadata-driven filtering for hybrid retrieval.

Two distinct filter applications:
1. ``build_pinecone_native_filter`` → applied at Pinecone query time (ANN-side filtering,
   cheaper than post-filtering for the semantic side).
2. ``apply_metadata_filter`` → applied post-fusion to BM25 results (BM25 has no native
   metadata filtering), ensuring both result sets are filtered by the same criteria
   before reranking.
"""

from __future__ import annotations

import logging

from query.utils.schema import RouterOutput, SubQuery, RetrievedChunk

logger = logging.getLogger(__name__)

# Section hints that strongly suggest financial table content
_FINANCIAL_SECTION_HINTS: frozenset[str] = frozenset({
    "income_statement",
    "balance_sheet",
    "cash_flow",
    "financial_statements",
    "selected_financial_data",
    "consolidated_statements",
})

# Lowercase → display-case mapping for company names (matches Pinecone metadata)
_COMPANY_DISPLAY: dict[str, str] = {
    "apple":     "Apple",
    "microsoft": "Microsoft",
    "nvidia":    "NVIDIA",
}


def build_filter_from_router(
    router_output: RouterOutput,
    sub_query: SubQuery,
) -> dict:
    """
    Builds a generic filter dict from the Router Agent's output and the
    sub-query's extracted entities.

    Returned dict shape (omitting ``None``-valued keys)::

        {
            "company":      "Apple",       # if sub_query.company is set
            "fiscal_year":  2022,          # if sub_query.fiscal_year is set
            "content_type": "table",       # only if needs_computation AND
                                           # section_hint implies financials
            "section":      "income_stmt", # if sub_query.section_hint is set
        }

    An empty/partial filter means "no constraint on this field."

    Args:
        router_output: Validated :class:`~query.utils.schema.RouterOutput`.
        sub_query:     The specific sub-query being filtered for.

    Returns:
        Filter dict (may be empty if no constraints apply).
    """
    filters: dict = {}

    if sub_query.company:
        # Store display-case company for metadata comparison consistency
        filters["company"] = _COMPANY_DISPLAY.get(
            sub_query.company.lower(), sub_query.company
        )

    if sub_query.fiscal_year:
        filters["fiscal_year"] = sub_query.fiscal_year

    if sub_query.section_hint:
        filters["section_hint"] = sub_query.section_hint
        # Prefer table chunks for financial computation sub-queries
        if (
            router_output.needs_computation
            and sub_query.section_hint.lower() in _FINANCIAL_SECTION_HINTS
        ):
            filters["content_type"] = "table"

    return filters


def apply_metadata_filter(
    chunks: list[RetrievedChunk],
    filters: dict,
) -> list[RetrievedChunk]:
    """
    Post-filters a fused chunk list by the given filter dict.

    Needed because BM25 results have no native metadata filtering (unlike
    Pinecone's query-time filter). Applied **after** RRF fusion so both
    BM25-only and semantic-only matches are filtered consistently before
    reranking.

    Comparisons are case-insensitive for ``company`` to handle "Apple" vs
    "apple" mismatches between the router (lowercase) and stored metadata
    (display case).

    Args:
        chunks:  Fused ``list[RetrievedChunk]`` from RRF.
        filters: Filter dict from :func:`build_filter_from_router`.

    Returns:
        Filtered subset in the same order.
    """
    if not filters:
        return chunks

    filtered = chunks
    if "company" in filters:
        target = filters["company"].lower()
        filtered = [
            rc for rc in filtered
            if rc.chunk.company.lower() == target
        ]

    if "fiscal_year" in filters:
        filtered = [
            rc for rc in filtered
            if rc.chunk.fiscal_year == filters["fiscal_year"]
        ]

    if "content_type" in filters:
        filtered = [
            rc for rc in filtered
            if rc.chunk.content_type == filters["content_type"]
        ]

    logger.debug(
        "apply_metadata_filter: %d → %d chunks after filters=%s",
        len(chunks), len(filtered), filters,
    )
    return filtered


def build_pinecone_native_filter(filters: dict) -> dict:
    """
    Converts a generic filter dict into Pinecone's query-time filter syntax.

    Example output::

        {
            "company":     {"$eq": "Apple"},
            "fiscal_year": {"$eq": 2022},
        }

    Applied at Pinecone ANN query time (cheaper than post-filtering since
    Pinecone skips non-matching vectors before scoring).

    ``section_hint`` is not a stored Pinecone metadata field so it is
    silently ignored here (it is used only in :func:`apply_metadata_filter`
    for BM25 post-filtering).

    Args:
        filters: Generic filter dict from :func:`build_filter_from_router`.

    Returns:
        Pinecone-compatible filter dict (empty dict if no applicable fields).
    """
    pinecone_filter: dict = {}

    if "company" in filters:
        pinecone_filter["company"] = {"$eq": filters["company"]}

    if "fiscal_year" in filters:
        pinecone_filter["fiscal_year"] = {"$eq": filters["fiscal_year"]}

    if "content_type" in filters:
        pinecone_filter["content_type"] = {"$eq": filters["content_type"]}

    return pinecone_filter
