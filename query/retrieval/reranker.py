"""
query/retrieval/reranker.py — Pinecone inference cross-encoder reranking.

Uses Pinecone's hosted ``bge-reranker-v2-m3`` model to jointly encode
(query, chunk) pairs and produce a more accurate relevance score than the
RRF-fused bi-encoder ranking alone.

No separate SDK or auth needed — reuses the same Pinecone client already
instantiated for retrieval (per system_design.md §3.4's rationale).
"""

from __future__ import annotations

import logging

from query.config import RERANK_TOP_N
from query.utils.schema import RetrievedChunk

logger = logging.getLogger(__name__)

_RERANK_MODEL = "bge-reranker-v2-m3"


def init_reranker_client(pinecone_client):
    """
    Returns the Pinecone client's inference handle for reranking.

    No separate instantiation needed — the inference API is accessed through
    the same ``Pinecone`` client used for ANN queries.

    Args:
        pinecone_client: Initialised ``pinecone.Pinecone`` client.

    Returns:
        The ``pinecone_client.inference`` handle (passed through for clarity).
    """
    return pinecone_client.inference


def rerank_chunks(
    query_text: str,
    chunks: list[RetrievedChunk],
    pinecone_client,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """
    Re-scores a list of chunks with the cross-encoder reranker.

    Steps:
    1. Extract ``documents = [rc.chunk.text for rc in chunks]``.
    2. Call ``pinecone_client.inference.rerank(model, query, documents,
       top_n, return_documents=False)``.
    3. Map each ``(index, relevance_score)`` in the response back to the
       original ``RetrievedChunk``, setting ``rerank_score``.
    4. Sort by ``rerank_score`` descending and return top ``top_n``.

    If ``chunks`` is empty or the reranker call fails, returns the input
    list unchanged (up to ``top_n`` items) with a warning — retrieval
    still works via RRF scores.

    Args:
        query_text:      The sub-query text (used as the reranker query).
        chunks:          RRF-fused list of :class:`~query.utils.schema.RetrievedChunk`.
        pinecone_client: Initialised ``pinecone.Pinecone`` client.
        top_n:           Final number of chunks to keep; defaults to ``RERANK_TOP_N`` (5).

    Returns:
        Top ``top_n`` :class:`~query.utils.schema.RetrievedChunk` objects
        sorted by descending ``rerank_score``.
    """
    _top_n = top_n if top_n is not None else RERANK_TOP_N

    if not chunks:
        logger.debug("rerank_chunks: empty input list, skipping reranking")
        return []

    # Reranker requires non-empty text; filter out blank chunks first
    valid_chunks = [rc for rc in chunks if rc.chunk.text and rc.chunk.text.strip()]
    if not valid_chunks:
        logger.warning("rerank_chunks: all chunks have empty text, returning first %d by RRF", _top_n)
        return chunks[:_top_n]

    documents = [rc.chunk.text for rc in valid_chunks]

    try:
        response = pinecone_client.inference.rerank(
            model=_RERANK_MODEL,
            query=query_text,
            documents=documents,
            top_n=min(_top_n, len(valid_chunks)),
            return_documents=False,
            parameters={"truncate": "END"},
        )

        # Map (index, score) back to original RetrievedChunk objects
        reranked: list[RetrievedChunk] = []
        for item in response.data:
            rc = valid_chunks[item.index]
            rc.rerank_score = float(item.score)
            reranked.append(rc)

        reranked.sort(key=lambda rc: rc.rerank_score or 0.0, reverse=True)

        logger.debug(
            "rerank_chunks: %d chunks → top %d after reranking (top score=%.4f)",
            len(valid_chunks), len(reranked),
            reranked[0].rerank_score if reranked else 0.0,
        )
        return reranked

    except Exception as exc:
        logger.warning(
            "rerank_chunks: reranker call failed (%s), falling back to RRF order", exc
        )
        # Fallback: return by RRF score
        return sorted(
            valid_chunks,
            key=lambda rc: rc.rrf_score or 0.0,
            reverse=True,
        )[:_top_n]
