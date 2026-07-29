"""
query/agents/sufficiency_agent.py — Sufficiency Check & Retry Loop (Phase 5).

Key quota design: ALL sub-queries are evaluated in ONE Gemini call
(``check_sufficiency``), regardless of count. This collapses N calls into 1
as described in system_design.md §5's budget math.

Reformulation is deliberately rule-based (no Gemini call) so each retry
does not burn additional quota.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from query.config import MAX_SUFFICIENCY_RETRIES, ROUTE_DIRECT_LOOKUP
from query.retrieval.hybrid_retriever import retrieve_for_subquery
from query.utils.gemini_client import GeminiCallCounter, call_structured
from query.utils.prompts import SUFFICIENCY_SYSTEM_PROMPT, build_sufficiency_prompt
from query.utils.schema import (
    RouterOutput, SubQuery, RetrievedChunk, SufficiencyResult
)

logger = logging.getLogger(__name__)

# ── Synonym map for attempt-2 reformulation ───────────────────────────────────
_SYNONYMS: dict[str, str] = {
    "r&d":                     "research and development",
    "research and development": "r&d",
    "revenue":                 "net sales",
    "net sales":               "revenue",
    "profit":                  "net income",
    "net income":              "profit",
    "earnings":                "net income",
    "operating expenses":      "opex",
    "opex":                    "operating expenses",
    "capex":                   "capital expenditure",
    "capital expenditure":     "capex",
    "gross margin":            "gross profit margin",
    "gross profit margin":     "gross margin",
}

# ── Gemini response schema ────────────────────────────────────────────────────

SUFFICIENCY_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sub_query_id": {"type": "integer"},
                    "sufficient":   {"type": "boolean"},
                    "missing": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["sub_query_id", "sufficient", "missing"],
            },
        },
    },
    "required": ["results"],
}


# ── Public API ────────────────────────────────────────────────────────────────

def check_sufficiency(
    sub_queries_with_chunks: list[tuple[SubQuery, list[RetrievedChunk]]],
    gemini_client,
    call_counter: GeminiCallCounter,
) -> list[SufficiencyResult]:
    """
    Batch-evaluates retrieved chunks for ALL sub-queries in ONE Gemini call.

    This "1 call for N sub-queries" pattern is the core quota mitigation from
    system_design.md §5 — a pipeline with 18 sub-queries still spends exactly
    one Gemini call here.

    Steps:
    1. Build the batched prompt via
       :func:`~query.utils.prompts.build_sufficiency_prompt`.
    2. Call Gemini with structured JSON output constrained by
       ``SUFFICIENCY_RESPONSE_SCHEMA``.
    3. Convert raw result dicts to :class:`~query.utils.schema.SufficiencyResult`.

    Args:
        sub_queries_with_chunks: List of ``(SubQuery, list[RetrievedChunk])`` pairs.
        gemini_client:           Initialised ``google.genai.Client``.
        call_counter:            Shared :class:`~query.utils.gemini_client.GeminiCallCounter`.

    Returns:
        One :class:`~query.utils.schema.SufficiencyResult` per sub-query.
    """
    prompt = build_sufficiency_prompt(sub_queries_with_chunks)
    result = call_structured(
        gemini_client,
        SUFFICIENCY_SYSTEM_PROMPT,
        prompt,
        SUFFICIENCY_RESPONSE_SCHEMA,
        call_counter,
    )
    results = [
        SufficiencyResult(
            sub_query_id=r["sub_query_id"],
            sufficient=bool(r["sufficient"]),
            missing=list(r.get("missing", [])),
        )
        for r in result.get("results", [])
    ]
    sufficient_count = sum(1 for r in results if r.sufficient)
    logger.info(
        "check_sufficiency: %d/%d sub-queries sufficient (1 Gemini call)",
        sufficient_count, len(results),
    )
    return results


def reformulate_subquery(
    sub_query: SubQuery,
    missing: list[str],
    retry_attempt: int,
) -> SubQuery:
    """
    Rule-based reformulation of an insufficient sub-query — deliberately NOT
    a Gemini call to preserve quota.

    Strategy by ``retry_attempt``:
    - **attempt 1**: Broaden scope — drop ``section_hint`` so
      :func:`~query.retrieval.metadata_filter.apply_metadata_filter` is less
      restrictive. This surfaces chunks from all sections rather than the
      hinted one.
    - **attempt 2**: Rephrase with synonyms — apply the ``_SYNONYMS`` map to
      common financial terms (e.g. ``"R&D"`` ↔ ``"research and development"``,
      ``"revenue"`` ↔ ``"net sales"``). Also clears ``section_hint`` if still
      set from attempt 1.

    Returns a new :class:`~query.utils.schema.SubQuery` with
    ``retry_count`` incremented; does **not** mutate the input.

    Args:
        sub_query:     The insufficient sub-query to reformulate.
        missing:       List of missing info strings from :func:`check_sufficiency`.
        retry_attempt: 1 or 2 (``retry_count + 1``).

    Returns:
        New :class:`~query.utils.schema.SubQuery` with updated text/section_hint
        and incremented ``retry_count``.
    """
    new_text = sub_query.text
    new_section_hint = sub_query.section_hint

    if retry_attempt == 1:
        # Broaden: drop section_hint only
        new_section_hint = None
        logger.debug(
            "reformulate_subquery [id=%d, attempt=1]: dropped section_hint=%r",
            sub_query.sub_query_id, sub_query.section_hint,
        )
    elif retry_attempt >= 2:
        # Rephrase: apply synonym substitution + clear all metadata hints
        new_section_hint = None
        for term, replacement in _SYNONYMS.items():
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(new_text):
                new_text = pattern.sub(replacement, new_text)
                logger.debug(
                    "reformulate_subquery [id=%d, attempt=2]: %r → %r",
                    sub_query.sub_query_id, term, replacement,
                )
                break  # apply one synonym at a time to avoid double substitution

    return SubQuery(
        sub_query_id=sub_query.sub_query_id,
        text=new_text,
        company=sub_query.company,
        fiscal_year=sub_query.fiscal_year,
        section_hint=new_section_hint,
        retry_count=retry_attempt,
    )


def run_sufficiency_loop(
    sub_queries: list[SubQuery],
    router_output: RouterOutput,
    resources: dict,
    gemini_client,
    call_counter: GeminiCallCounter,
) -> dict[int, list[RetrievedChunk]]:
    """
    Full orchestration: initial retrieval → sufficiency check → retry loop.

    Steps:
    1. Initial retrieval: ``retrieve_for_subquery`` for every sub-query.
    2. Batch sufficiency check (ONE Gemini call) across all sub-queries.
    3. Collect insufficient sub-queries (``retry_count < MAX_SUFFICIENCY_RETRIES``).
    4. For each retry round (up to ``MAX_SUFFICIENCY_RETRIES``):
       a. Reformulate insufficient sub-queries.
       b. Re-retrieve for each reformulated sub-query.
       c. Batch sufficiency check for the reformulated ones only (ONE more call).
    5. After max retries, accept whatever context was retrieved ("partial
       context" per system_design.md §2 diagram).

    Logs a warning listing any sub-queries that remained insufficient after
    all retries — these are surfaced as implicit caveats in the final answer.

    Returns ``dict[sub_query_id, list[RetrievedChunk]]`` — final retrieved
    context per sub-query, ready for the Tool-Use Agent and Generator.

    Args:
        sub_queries:   All sub-queries for this pipeline run.
        router_output: Router Agent output (needed by
                       :func:`~query.retrieval.hybrid_retriever.retrieve_for_subquery`).
        resources:     Pipeline resource dict (bm25_index, pinecone_index, etc.).
        gemini_client: Initialised ``google.genai.Client``.
        call_counter:  Shared :class:`~query.utils.gemini_client.GeminiCallCounter`.

    Returns:
        ``dict[int, list[RetrievedChunk]]`` keyed by ``sub_query_id``.
    """
    # ── Step 1: initial retrieval ─────────────────────────────────────────────
    active_queries: dict[int, SubQuery] = {sq.sub_query_id: sq for sq in sub_queries}
    retrieved: dict[int, list[RetrievedChunk]] = {}

    for sq in sub_queries:
        retrieved[sq.sub_query_id] = retrieve_for_subquery(sq, router_output, resources)

    # direct_lookup: skip the Gemini sufficiency check entirely — retrieval alone
    # is sufficient and this saves 1 Gemini call (≤3 total for the fast path,
    # per system_design.md §5's budget math).
    if router_output.route == ROUTE_DIRECT_LOOKUP:
        logger.info(
            "run_sufficiency_loop: direct_lookup route — skipping sufficiency check "
            "(retrieved %d chunks for %d sub-queries)",
            sum(len(v) for v in retrieved.values()), len(sub_queries),
        )
        return retrieved

    # ── Steps 2–4: sufficiency + retry loop ───────────────────────────────────
    for round_idx in range(MAX_SUFFICIENCY_RETRIES + 1):
        pairs = [
            (active_queries[sid], retrieved[sid])
            for sid in active_queries
        ]
        suf_results = check_sufficiency(pairs, gemini_client, call_counter)

        # Identify sub-queries that are insufficient and can still be retried
        to_retry: list[tuple[SubQuery, list[str]]] = []
        for sr in suf_results:
            sq = active_queries.get(sr.sub_query_id)
            if sq is None:
                continue
            if not sr.sufficient and sq.retry_count < MAX_SUFFICIENCY_RETRIES:
                to_retry.append((sq, sr.missing))
            elif not sr.sufficient:
                logger.warning(
                    "run_sufficiency_loop: sub_query_id=%d still insufficient "
                    "after %d retries — accepting partial context",
                    sr.sub_query_id, MAX_SUFFICIENCY_RETRIES,
                )

        if not to_retry or round_idx == MAX_SUFFICIENCY_RETRIES:
            break

        # Reformulate, re-retrieve, update active_queries + retrieved
        for sq, missing in to_retry:
            new_sq = reformulate_subquery(sq, missing, retry_attempt=sq.retry_count + 1)
            active_queries[new_sq.sub_query_id] = new_sq
            retrieved[new_sq.sub_query_id] = retrieve_for_subquery(
                new_sq, router_output, resources
            )

    return retrieved
