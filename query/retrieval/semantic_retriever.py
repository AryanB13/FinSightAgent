"""
query/retrieval/semantic_retriever.py — Pinecone ANN semantic retrieval.

Embeds the query with the same ``voyage-multimodal-3`` model used at ingestion
so the query vector and all chunk vectors share the same 1024-dim space,
enabling cross-modal retrieval (a text query can surface chart chunks).

Pinecone metadata does NOT include ``image_bytes`` or ``table_data`` (too large
/ not JSON-serialisable). :func:`hydrate_chunk_from_bm25_list` re-attaches
these from the BM25 pickle after retrieval.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import voyageai

from ingestion.indexing.pinecone_upserter import init_pinecone, get_or_create_index
from ingestion.indexing.voyage_embedder import embed_query
from ingestion.utils.schema import Chunk
from query.config import (
    PINECONE_NAMESPACES,
    SEMANTIC_TOP_K,
    VOYAGE_TEXT_MODEL,
)
from query.utils.schema import RetrievedChunk, SubQuery

logger = logging.getLogger(__name__)

# Module-level chunk_id → Chunk lookup dict (built lazily from BM25 list)
_BM25_CHUNK_MAP: dict[str, Chunk] = {}


def init_pinecone_query_resources():
    """
    Initialises the Pinecone client and index handle for the query pipeline.

    Thin wrapper around :func:`ingestion.indexing.pinecone_upserter.init_pinecone`
    and :func:`~ingestion.indexing.pinecone_upserter.get_or_create_index`.
    Called **once** at startup; the index handle is reused for every query.

    Returns:
        ``(pinecone_client, pinecone_index)`` tuple.
    """
    api_key = os.environ.get("PINECONE_API_KEY", "")
    pc = init_pinecone(api_key)
    index = get_or_create_index(pc)
    logger.info("init_pinecone_query_resources: Pinecone index ready")
    return pc, index


def _metadata_to_chunk(match_id: str, metadata: dict) -> Chunk:
    """
    Reconstructs a partial :class:`~ingestion.utils.schema.Chunk` from
    Pinecone match metadata.

    ``image_bytes`` and ``table_data`` are always ``None`` here — call
    :func:`hydrate_chunk_from_bm25_list` afterwards for chart/table chunks.
    """
    return Chunk(
        chunk_id=match_id,
        company=metadata.get("company", ""),
        ticker=metadata.get("ticker", ""),
        fiscal_year=int(metadata.get("fiscal_year", 0)),
        filing_type=metadata.get("filing_type", ""),
        section=metadata.get("section", ""),
        content_type=metadata.get("content_type", "text"),
        page_number=int(metadata.get("page_number", 0)),
        text=metadata.get("text", ""),
        source_file=metadata.get("source_file", ""),
        corpus_version=metadata.get("corpus_version", ""),
        embedding_model=metadata.get("embedding_model", ""),
        table_data=None,
        image_bytes=None,
    )


def retrieve_semantic(
    sub_query: SubQuery,
    pinecone_index,
    voyage_client: voyageai.Client,
    namespaces: list[str] | None = None,
    top_k: int | None = None,
    pinecone_filter: dict | None = None,
) -> list[RetrievedChunk]:
    """
    Runs Pinecone ANN search for a sub-query.

    Steps:
    1. Embed ``sub_query.text`` via ``embed_query`` (``voyage-multimodal-3``).
    2. Determine target namespaces: ``[sub_query.company]`` if set, else all
       ``PINECONE_NAMESPACES`` (cross-company sub-query).
    3. For each namespace, query Pinecone with ``top_k`` and optional filter.
    4. Merge results across namespaces, deduplicating by ``chunk_id``.
    5. Wrap each match into a :class:`~query.utils.schema.RetrievedChunk`
       with ``semantic_score`` set.

    Note: returned ``Chunk`` objects have ``image_bytes=None`` and
    ``table_data=None``; call :func:`hydrate_chunk_from_bm25_list` if the
    downstream step needs full objects.

    Args:
        sub_query:        Sub-query to retrieve for.
        pinecone_index:   Pinecone ``Index`` handle from :func:`init_pinecone_query_resources`.
        voyage_client:    Initialised ``voyageai.Client``.
        namespaces:       Override target namespaces; defaults to ``[sub_query.company]``
                          if set, else all of ``PINECONE_NAMESPACES``.
        top_k:            Results per namespace; defaults to ``SEMANTIC_TOP_K`` (20).
        pinecone_filter:  Pre-built Pinecone native filter dict (from
                          :func:`~query.retrieval.metadata_filter.build_pinecone_native_filter`).

    Returns:
        ``list[RetrievedChunk]`` ordered by descending ``semantic_score``.
    """
    k = top_k or SEMANTIC_TOP_K

    # Embed the query
    query_vector = embed_query(sub_query.text, voyage_client, model=VOYAGE_TEXT_MODEL)

    # Determine namespaces
    if namespaces is not None:
        target_ns = namespaces
    elif sub_query.company:
        target_ns = [sub_query.company]
    else:
        target_ns = PINECONE_NAMESPACES

    # Query each namespace and merge
    seen_ids: set[str] = set()
    all_results: list[RetrievedChunk] = []

    for ns in target_ns:
        query_kwargs: dict = dict(
            vector=query_vector,
            top_k=k,
            namespace=ns,
            include_metadata=True,
        )
        if pinecone_filter:
            query_kwargs["filter"] = pinecone_filter

        try:
            response = pinecone_index.query(**query_kwargs)
            for match in response.matches:
                if match.id in seen_ids:
                    continue
                seen_ids.add(match.id)
                chunk = _metadata_to_chunk(match.id, match.metadata or {})
                all_results.append(
                    RetrievedChunk(chunk=chunk, semantic_score=float(match.score))
                )
        except Exception as exc:
            logger.warning(
                "retrieve_semantic: namespace=%s query failed: %s", ns, exc
            )

    # Sort by descending semantic score
    all_results.sort(key=lambda rc: rc.semantic_score or 0.0, reverse=True)

    logger.debug(
        "retrieve_semantic: sub_query_id=%d → %d results across namespaces=%s",
        sub_query.sub_query_id, len(all_results), target_ns,
    )
    return all_results


def hydrate_chunk_from_bm25_list(
    chunk_id: str,
    bm25_chunks: list[Chunk],
) -> Optional[Chunk]:
    """
    Retrieves the full :class:`~ingestion.utils.schema.Chunk` object from
    the BM25 pickle by ``chunk_id``.

    Builds a module-level ``chunk_id → Chunk`` dict on first call for O(1)
    lookups on all subsequent calls. This is cheap (≤ ~700 chunks) but avoids
    linear scans when called repeatedly inside :func:`retrieve_for_subquery`.

    Returns ``None`` if not found — indicates index drift between Pinecone
    and the BM25 pickle (logs a warning).

    Args:
        chunk_id:    ID to look up (matches ``Chunk.chunk_id``).
        bm25_chunks: Full chunk list from :func:`~query.retrieval.bm25_retriever.load_bm25_resources`.

    Returns:
        Full ``Chunk`` with ``image_bytes`` / ``table_data`` populated,
        or ``None`` if not found.
    """
    global _BM25_CHUNK_MAP
    if not _BM25_CHUNK_MAP and bm25_chunks:
        _BM25_CHUNK_MAP = {c.chunk_id: c for c in bm25_chunks}

    chunk = _BM25_CHUNK_MAP.get(chunk_id)
    if chunk is None:
        logger.warning(
            "hydrate_chunk_from_bm25_list: chunk_id=%r not found in BM25 pickle "
            "(index drift?)", chunk_id,
        )
    return chunk
