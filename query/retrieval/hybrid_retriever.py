"""
query/retrieval/hybrid_retriever.py — Top-level orchestrator for hybrid retrieval.

``retrieve_for_subquery`` is the single entry point called by the retrieval
node in the LangGraph pipeline. It coordinates:

  BM25 → Semantic ANN → RRF Fusion → Metadata Filter → Hydration → Reranking

Each sub-query is independent, so callers can dispatch concurrent sub-queries
via ``asyncio.gather`` or a ``ThreadPoolExecutor`` for multi-hop queries.

Zero Gemini calls are made in this phase.
"""

from __future__ import annotations

import logging

from query.retrieval.bm25_retriever import retrieve_bm25
from query.retrieval.semantic_retriever import retrieve_semantic, hydrate_chunk_from_bm25_list
from query.retrieval.fusion import reciprocal_rank_fusion
from query.retrieval.metadata_filter import (
    build_filter_from_router,
    apply_metadata_filter,
    build_pinecone_native_filter,
)
from query.retrieval.reranker import rerank_chunks
from query.utils.schema import RetrievedChunk, RouterOutput, SubQuery

logger = logging.getLogger(__name__)


def retrieve_for_subquery(
    sub_query: SubQuery,
    router_output: RouterOutput,
    resources: dict,
) -> list[RetrievedChunk]:
    """
    Full hybrid retrieval pipeline for a single sub-query.

    ``resources`` is the dict built by ``pipeline.init_pipeline_resources()``
    and must contain:

    - ``"bm25_index"``    — loaded ``BM25Okapi`` object
    - ``"bm25_chunks"``   — parallel ``list[Chunk]``
    - ``"pinecone_index"``— Pinecone ``Index`` handle
    - ``"voyage_client"`` — initialised ``voyageai.Client``
    - ``"pinecone_client"``— initialised ``pinecone.Pinecone`` client

    Steps:
    1. BM25 keyword retrieval.
    2. Build metadata filters from router + sub-query entities.
    3. Semantic ANN retrieval with native Pinecone filter.
    4. Reciprocal Rank Fusion.
    5. Post-fusion metadata filter (normalises BM25 results to same constraints).
    6. Hydrate ``image_bytes`` / ``table_data`` for chart/table chunks from BM25 pickle.
    7. Cross-encoder reranking → final ``list[RetrievedChunk]`` (≤ ``RERANK_TOP_N``).

    Independent sub-queries can be dispatched concurrently (none of steps
    1–7 depend on another sub-query's results).

    Args:
        sub_query:     The sub-query to retrieve chunks for.
        router_output: Router Agent's output (needed for filter construction).
        resources:     Pipeline resource dict from ``init_pipeline_resources``.

    Returns:
        Final ``list[RetrievedChunk]`` sorted by descending ``rerank_score``,
        length ≤ ``RERANK_TOP_N``.
    """
    sid = sub_query.sub_query_id
    logger.info(
        "retrieve_for_subquery: [%d] %r (company=%s, year=%s)",
        sid, sub_query.text[:70], sub_query.company, sub_query.fiscal_year,
    )

    bm25_index   = resources["bm25_index"]
    bm25_chunks  = resources["bm25_chunks"]
    pc_index     = resources["pinecone_index"]
    voyage_cl    = resources["voyage_client"]
    pc_client    = resources["pinecone_client"]

    # Step 1 — BM25
    bm25_results = retrieve_bm25(sub_query, bm25_index, bm25_chunks)
    logger.debug("[%d] BM25: %d results", sid, len(bm25_results))

    # Step 2 — build filters
    filters = build_filter_from_router(router_output, sub_query)
    pc_filter = build_pinecone_native_filter(filters)

    # Step 3 — semantic (with native Pinecone filter for ANN-side efficiency)
    semantic_results = retrieve_semantic(
        sub_query, pc_index, voyage_cl,
        pinecone_filter=pc_filter if pc_filter else None,
    )
    logger.debug("[%d] Semantic: %d results", sid, len(semantic_results))

    # Step 4 — RRF fusion
    fused = reciprocal_rank_fusion(bm25_results, semantic_results)
    logger.debug("[%d] RRF: %d fused chunks", sid, len(fused))

    # Step 5 — post-fusion metadata filter (normalises BM25 results)
    filtered = apply_metadata_filter(fused, filters)
    logger.debug("[%d] Filtered: %d chunks after metadata filter", sid, len(filtered))

    # If filtering removed everything, fall back to unfiltered fused list
    if not filtered and fused:
        logger.warning(
            "[%d] metadata filter left 0 results; falling back to unfiltered fused list",
            sid,
        )
        filtered = fused

    # Step 6 — hydrate chart/table chunks missing image_bytes/table_data
    for rc in filtered:
        if rc.chunk.content_type in ("table", "chart") and (
            rc.chunk.image_bytes is None and rc.chunk.table_data is None
        ):
            full = hydrate_chunk_from_bm25_list(rc.chunk.chunk_id, bm25_chunks)
            if full is not None:
                rc.chunk = full

    # Step 7 — cross-encoder reranking
    reranked = rerank_chunks(sub_query.text, filtered, pc_client)
    logger.info(
        "[%d] Final: %d chunks after reranking (top score=%.4f)",
        sid, len(reranked),
        reranked[0].rerank_score if reranked else 0.0,
    )

    return reranked
