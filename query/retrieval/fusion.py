"""
query/retrieval/fusion.py — Reciprocal Rank Fusion (RRF) for hybrid retrieval.

RRF score: ``score(chunk) = Σ 1 / (rank_i + k)``

Where ``rank_i`` is the 1-indexed position of the chunk in ranked list ``i``
and ``k = RRF_K`` (60) is the standard damping constant that limits the
maximum score contribution of any single match (prevents top-1 results from
dominating when the two lists strongly agree).

A chunk appearing in BOTH BM25 and semantic lists gets contributions from
both terms — this is the core of hybrid retrieval.
"""

from __future__ import annotations

import logging

from query.config import RRF_K
from query.utils.schema import RetrievedChunk

logger = logging.getLogger(__name__)


def dedupe_by_chunk_id(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """
    Merges duplicate :class:`~query.utils.schema.RetrievedChunk` entries that
    share the same ``chunk.chunk_id`` into a single entry.

    When a chunk appears in both BM25 and semantic result sets, it generates
    two ``RetrievedChunk`` objects — one with ``bm25_score`` set and one with
    ``semantic_score`` set. This function merges them so the final entry has
    all non-``None`` scores preserved.

    Preserves the order of first occurrence.

    Args:
        chunks: Possibly-duplicated list of ``RetrievedChunk`` objects.

    Returns:
        Deduplicated list with all score fields consolidated.
    """
    seen: dict[str, RetrievedChunk] = {}
    for rc in chunks:
        cid = rc.chunk.chunk_id
        if cid not in seen:
            seen[cid] = rc
        else:
            # Merge non-None scores into existing entry
            existing = seen[cid]
            if rc.bm25_score is not None:
                existing.bm25_score = rc.bm25_score
            if rc.semantic_score is not None:
                existing.semantic_score = rc.semantic_score
            if rc.rrf_score is not None:
                existing.rrf_score = rc.rrf_score
            if rc.rerank_score is not None:
                existing.rerank_score = rc.rerank_score
    return list(seen.values())


def reciprocal_rank_fusion(
    bm25_results: list[RetrievedChunk],
    semantic_results: list[RetrievedChunk],
    k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Fuses BM25 and semantic results using Reciprocal Rank Fusion.

    RRF formula: ``score(chunk) = Σ 1 / (rank_i + k)``

    Steps:
    1. Build rank maps ``{chunk_id: 1-indexed-rank}`` for each list.
    2. Compute ``union`` of all chunk IDs across both lists.
    3. For each chunk ID, compute ``rrf_score = 1/(bm25_rank+k) [if present]
       + 1/(sem_rank+k) [if present]``.
    4. Deduplicate via :func:`dedupe_by_chunk_id`, preserving both score fields.
    5. Set ``rrf_score`` on each entry and sort descending.

    Chunks appearing in both lists receive a higher combined score than those
    in only one list, naturally surfacing the most cross-retrieved results.

    Args:
        bm25_results:     Results from BM25, ordered by descending ``bm25_score``.
        semantic_results: Results from Pinecone, ordered by descending ``semantic_score``.
        k:                RRF damping constant; defaults to ``RRF_K`` (60).

    Returns:
        Fused, deduplicated ``list[RetrievedChunk]`` sorted by descending
        ``rrf_score``.
    """
    _k = k if k is not None else RRF_K

    # Build rank maps (1-indexed)
    bm25_rank: dict[str, int] = {
        rc.chunk.chunk_id: i + 1 for i, rc in enumerate(bm25_results)
    }
    sem_rank: dict[str, int] = {
        rc.chunk.chunk_id: i + 1 for i, rc in enumerate(semantic_results)
    }

    all_chunk_ids = set(bm25_rank) | set(sem_rank)

    # Compute RRF scores
    rrf_scores: dict[str, float] = {}
    for cid in all_chunk_ids:
        score = 0.0
        if cid in bm25_rank:
            score += 1.0 / (bm25_rank[cid] + _k)
        if cid in sem_rank:
            score += 1.0 / (sem_rank[cid] + _k)
        rrf_scores[cid] = score

    # Merge both lists and deduplicate
    fused = dedupe_by_chunk_id(bm25_results + semantic_results)

    # Attach rrf_score to each entry
    for rc in fused:
        rc.rrf_score = rrf_scores.get(rc.chunk.chunk_id, 0.0)

    # Sort by rrf_score descending
    fused.sort(key=lambda rc: rc.rrf_score or 0.0, reverse=True)

    logger.debug(
        "reciprocal_rank_fusion: %d BM25 + %d semantic → %d fused (k=%d)",
        len(bm25_results), len(semantic_results), len(fused), _k,
    )
    return fused
