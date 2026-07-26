"""
query/retrieval/bm25_retriever.py — BM25 keyword retrieval wrapper.

Thin wrappers around ``ingestion.indexing.bm25_indexer`` — no logic is
duplicated, only the interface is adapted to the query pipeline's
``RetrievedChunk`` schema.
"""

from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi

from ingestion.indexing.bm25_indexer import load_index, search_bm25
from ingestion.utils.schema import Chunk
from query.config import BM25_INDEX_PATH, BM25_CHUNKS_PATH, BM25_TOP_K
from query.utils.schema import RetrievedChunk, SubQuery

logger = logging.getLogger(__name__)


def load_bm25_resources() -> tuple[BM25Okapi, list[Chunk]]:
    """
    Loads the BM25 index and parallel chunk list from disk.

    Thin wrapper around
    :func:`ingestion.indexing.bm25_indexer.load_index`.
    Called **once** at pipeline startup (``init_pipeline_resources``) and
    the returned objects are held in memory for the whole process lifetime
    — never reloaded per query.

    Returns:
        ``(bm25_index, bm25_chunks)`` tuple ready to pass into
        :func:`retrieve_bm25`.
    """
    logger.info("load_bm25_resources: loading BM25 index from %s", BM25_INDEX_PATH)
    bm25_index, bm25_chunks = load_index(BM25_INDEX_PATH, BM25_CHUNKS_PATH)
    logger.info("load_bm25_resources: loaded %d chunks into BM25 index", len(bm25_chunks))
    return bm25_index, bm25_chunks


def retrieve_bm25(
    sub_query: SubQuery,
    bm25_index: BM25Okapi,
    bm25_chunks: list[Chunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Runs a BM25 keyword search for a sub-query.

    Calls :func:`ingestion.indexing.bm25_indexer.search_bm25` and wraps
    each ``(Chunk, score)`` pair into a
    :class:`~query.utils.schema.RetrievedChunk` with ``bm25_score`` set.

    Args:
        sub_query:   Sub-query to retrieve for.
        bm25_index:  Loaded ``BM25Okapi`` object from :func:`load_bm25_resources`.
        bm25_chunks: Parallel chunk list (same order as BM25 corpus).
        top_k:       Number of results to return; defaults to ``BM25_TOP_K`` (20).

    Returns:
        ``list[RetrievedChunk]`` ordered by descending ``bm25_score``.
        Empty list if no query token appears in the index vocabulary.
    """
    k = top_k or BM25_TOP_K
    raw_results: list[tuple[Chunk, float]] = search_bm25(
        sub_query.text, bm25_index, bm25_chunks, top_k=k
    )
    retrieved = [
        RetrievedChunk(chunk=chunk, bm25_score=float(score))
        for chunk, score in raw_results
    ]
    logger.debug(
        "retrieve_bm25: sub_query_id=%d → %d results (top score=%.4f)",
        sub_query.sub_query_id,
        len(retrieved),
        retrieved[0].bm25_score if retrieved else 0.0,
    )
    return retrieved
