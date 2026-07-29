"""
query/agents/generator_agent.py — Generator Agent (Phase 7).

Produces a cited draft answer from retrieved chunks and computed metrics.
ONE Gemini call regardless of query complexity or number of sub-queries
— this is the single Generator call in system_design.md §5's budget math.
"""

from __future__ import annotations

import logging

from query.utils.gemini_client import GeminiCallCounter, call_freeform
from query.utils.prompts import GENERATOR_SYSTEM_PROMPT, build_generator_prompt
from query.utils.schema import ComputedMetric, RetrievedChunk, SubQuery

logger = logging.getLogger(__name__)


def build_citation_context(
    retrieved_chunks: dict[int, list[RetrievedChunk]],
) -> str:
    """
    Serialises all retrieved chunks (across all sub-queries) into a single
    text block, each entry tagged with its ``chunk_id`` and a human-readable
    citation header.

    Example output::

        [AAPL-FY2023-10K-item8-table-0002]
        Company: Apple, FY2023, Item 8, page 41
        <chunk.text>

    Deduplicates chunks that appear under multiple sub-queries (a chunk
    retrieved for both sub-query 1 and sub-query 3 appears only once).

    Args:
        retrieved_chunks: ``dict[sub_query_id, list[RetrievedChunk]]`` from the
                          sufficiency loop.

    Returns:
        Formatted citation context string.
    """
    seen_ids: set[str] = set()
    lines: list[str] = []

    for chunks in retrieved_chunks.values():
        for rc in chunks:
            cid = rc.chunk.chunk_id
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            c = rc.chunk
            lines.append(f"[{cid}]")
            lines.append(
                f"Company: {c.company}, FY{c.fiscal_year}, "
                f"{c.section}, page {c.page_number}"
            )
            lines.append(c.text)
            lines.append("")

    return "\n".join(lines)


def generate_answer(
    original_query: str,
    sub_queries: list[SubQuery],
    retrieved_chunks: dict[int, list[RetrievedChunk]],
    computed_metrics: list[ComputedMetric],
    gemini_client,
    call_counter: GeminiCallCounter,
) -> str:
    """
    Generates a cited draft answer from retrieved evidence and computed metrics.

    Steps:
    1. Build the grounding context via :func:`build_citation_context`.
    2. Build the full generator prompt via
       :func:`~query.utils.prompts.build_generator_prompt`.
    3. Call Gemini in freeform mode (no ``response_schema``) to produce a
       prose answer with inline citations.

    ONE call regardless of how many sub-queries, chunks, or metrics were
    involved — the single Generator call in §5's budget math.

    Args:
        original_query:   Raw user query string.
        sub_queries:      All sub-queries resolved by the pipeline.
        retrieved_chunks: Final retrieved context per sub-query.
        computed_metrics: Computed financial metrics from the Tool-Use Agent.
        gemini_client:    Initialised ``google.genai.Client``.
        call_counter:     Shared :class:`~query.utils.gemini_client.GeminiCallCounter`.

    Returns:
        Draft answer text with inline citations.
    """
    prompt = build_generator_prompt(
        original_query, sub_queries, retrieved_chunks, computed_metrics
    )
    draft = call_freeform(
        gemini_client, GENERATOR_SYSTEM_PROMPT, prompt, call_counter
    )
    logger.info(
        "generate_answer: draft produced (%d chars, %d citations context)",
        len(draft),
        sum(len(chunks) for chunks in retrieved_chunks.values()),
    )
    return draft
