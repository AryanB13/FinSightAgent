"""
query/pipeline.py — Pipeline initialisation and single-query entry point.

``init_pipeline_resources`` is called ONCE at process startup and returns a
dict of all long-lived clients. ``run_query`` is called per user query and
invokes the compiled LangGraph graph.
"""

from __future__ import annotations

import logging
import os

from query.cache.redis_client import init_redis_client
from query.graph.state_graph import build_query_graph
from query.retrieval.bm25_retriever import load_bm25_resources
from query.retrieval.semantic_retriever import init_pinecone_query_resources
from query.utils.gemini_client import GeminiCallCounter, init_gemini_client
from ingestion.indexing.voyage_embedder import init_voyage_client

logger = logging.getLogger(__name__)


def init_pipeline_resources(no_cache_read: bool = False) -> dict:
    """
    Instantiates every long-lived client ONCE for the process lifetime.

    Returns a ``resources`` dict containing:

    - ``"gemini_client"``   — initialised ``google.genai.Client``
    - ``"voyage_client"``   — initialised ``voyageai.Client``
    - ``"pinecone_client"`` — ``pinecone.Pinecone`` client
    - ``"pinecone_index"``  — Pinecone ``Index`` handle
    - ``"bm25_index"``      — loaded ``BM25Okapi`` object
    - ``"bm25_chunks"``     — parallel ``list[Chunk]``
    - ``"redis_client"``    — Upstash Redis REST client
    - ``"call_counter"``    — ``GeminiCallCounter`` backed by Redis
    - ``"graph"``           — compiled LangGraph ``CompiledStateGraph``
    - ``"no_cache_read"``   — bool flag for ``--no-cache`` CLI option

    Called once at process/CLI startup — never per query.

    Args:
        no_cache_read: If ``True``, ``cache_check_node`` always treats reads
                       as misses (writes are still performed). Used for the
                       ``--no-cache`` CLI flag.

    Returns:
        Populated resources dict.
    """
    logger.info("init_pipeline_resources: initialising all clients...")

    gemini_client  = init_gemini_client(os.environ["GEMINI_API_KEY"])
    voyage_client  = init_voyage_client(os.environ["VOYAGE_API_KEY"])
    pinecone_client, pinecone_index = init_pinecone_query_resources()
    bm25_index, bm25_chunks = load_bm25_resources()
    redis_client   = init_redis_client(
        os.environ["UPSTASH_REDIS_REST_URL"],
        os.environ["UPSTASH_REDIS_REST_TOKEN"],
    )
    call_counter   = GeminiCallCounter(redis_client)

    resources: dict = {
        "gemini_client":   gemini_client,
        "voyage_client":   voyage_client,
        "pinecone_client": pinecone_client,
        "pinecone_index":  pinecone_index,
        "bm25_index":      bm25_index,
        "bm25_chunks":     bm25_chunks,
        "redis_client":    redis_client,
        "call_counter":    call_counter,
        "no_cache_read":   no_cache_read,
    }

    resources["graph"] = build_query_graph(resources)
    logger.info("init_pipeline_resources: all clients ready.")
    return resources


def run_query(user_query: str, resources: dict) -> dict:
    """
    Executes the full query pipeline for a single user query.

    Steps:
    1. Build the initial LangGraph state dict.
    2. Invoke the compiled graph.
    3. Return ``final_answer_payload`` on a full pipeline run, or
       ``cached_answer`` on a cache hit.

    Args:
        user_query: Raw user question string.
        resources:  Dict from :func:`init_pipeline_resources`.

    Returns:
        ``final_answer_payload`` dict::

            {
                "final_answer": str,
                "citations":    list[str],
                "verdict":      "verified" | "partial",
                "flagged_claims": list[str],
                "cached_at":    "<ISO-8601 timestamp>"
            }
    """
    initial_state: dict = {
        "user_query":         user_query,
        "normalized_query":   "",
        "cache_hit":          False,
        "cached_answer":      None,
        "router_output":      None,
        "sub_queries":        [],
        "retrieved_chunks":   {},
        "sufficiency_results": {},
        "computed_metrics":   [],
        "draft_answer":       "",
        "verifier_result":    None,
        "final_answer_payload": None,
        "gemini_call_count":  0,
        "warnings":           [],
    }

    final_state = resources["graph"].invoke(initial_state)

    if final_state.get("cache_hit") and final_state.get("cached_answer"):
        logger.info("run_query: cache hit — returned cached answer")
        return final_state["cached_answer"]

    payload = final_state.get("final_answer_payload")
    if payload is None:
        logger.warning("run_query: no final_answer_payload in final state")
        payload = {"final_answer": "", "citations": [], "verdict": "partial",
                   "flagged_claims": [], "cached_at": ""}
    return payload
