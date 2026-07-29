"""
query/graph/state_graph.py — LangGraph StateGraph wiring all pipeline phases.

Full flow (system_design.md §2 and §8 Part B):

  cache_check → [cache hit? → END]
               → router → [multi_hop? → decomposer → retrieval]
                         [else       → retrieval (skip decomposer)]
               → retrieval_and_sufficiency → tool_use
               → generator → verifier → cache_write → END

Every node is a pure function ``(state: dict) -> dict`` wrapped with
``functools.partial`` to bind the ``resources`` dict of long-lived clients.
"""

from __future__ import annotations

import functools
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

from query.agents.decomposer_agent import decompose_query
from query.agents.generator_agent import generate_answer
from query.agents.router_agent import classify_query, build_direct_lookup_subquery
from query.agents.sufficiency_agent import run_sufficiency_loop
from query.agents.tool_use_agent import identify_required_computations, run_tool_use
from query.agents.verifier_agent import verify_answer, strip_flagged_claims
from query.cache.exact_cache import get_exact_cache, set_exact_cache
from query.cache.semantic_cache import get_semantic_cache, add_semantic_cache_entry
from query.config import CORPUS_VERSION, ROUTE_MULTI_HOP
from query.utils.schema import RouterOutput, VerifierResult
from ingestion.indexing.voyage_embedder import embed_query

logger = logging.getLogger(__name__)


# ── LangGraph state schema ────────────────────────────────────────────────────

class GraphState(TypedDict, total=False):
    """TypedDict schema for the LangGraph StateGraph. All fields are optional
    (``total=False``) so each node can return a partial update that LangGraph
    merges into the accumulated state."""
    user_query: str
    normalized_query: str
    cache_hit: bool
    cached_answer: Optional[dict]
    router_output: Any          # RouterOutput
    sub_queries: Any            # list[SubQuery]
    retrieved_chunks: Any       # dict[int, list[RetrievedChunk]]
    sufficiency_results: Any    # dict[int, SufficiencyResult]
    computed_metrics: Any       # list[ComputedMetric]
    draft_answer: str
    verifier_result: Any        # VerifierResult
    final_answer_payload: Optional[dict]
    gemini_call_count: int
    warnings: Any               # list[str]


# ── LangGraph node functions ──────────────────────────────────────────────────

def cache_check_node(state: dict, resources: dict) -> dict:
    """
    Cache check: exact match first, then semantic similarity on miss.

    Returns partial state update:
    ``{"cache_hit": bool, "cached_answer": dict|None, "normalized_query": str}``
    """
    from query.cache.redis_client import normalize_query

    user_query = state.get("user_query", "")
    redis_client = resources["redis_client"]
    voyage_client = resources["voyage_client"]
    no_cache = resources.get("no_cache_read", False)

    normalized = normalize_query(user_query)

    if no_cache:
        logger.info("cache_check_node: --no-cache flag set, bypassing cache read")
        return {"normalized_query": normalized, "cache_hit": False, "cached_answer": None}

    # 1. Exact match
    cached = get_exact_cache(redis_client, user_query, CORPUS_VERSION)
    if cached is not None:
        logger.info("cache_check_node: exact cache HIT")
        return {"normalized_query": normalized, "cache_hit": True, "cached_answer": cached}

    # 2. Semantic match
    try:
        query_emb = embed_query(user_query, voyage_client)
        cached = get_semantic_cache(redis_client, query_emb, CORPUS_VERSION)
        if cached is not None:
            logger.info("cache_check_node: semantic cache HIT")
            return {"normalized_query": normalized, "cache_hit": True, "cached_answer": cached}
    except Exception as exc:
        logger.warning("cache_check_node: semantic cache lookup failed: %s", exc)

    return {"normalized_query": normalized, "cache_hit": False, "cached_answer": None}


def router_node(state: dict, resources: dict) -> dict:
    """
    Router: classify query into route + entities. For non-multi_hop routes,
    also populate ``sub_queries`` directly (skipping decomposer_node).

    Returns partial state update:
    ``{"router_output": RouterOutput, "sub_queries": list[SubQuery]}``
    """
    user_query = state.get("user_query", "")
    gemini_client = resources["gemini_client"]
    call_counter = resources["call_counter"]

    router_output = classify_query(user_query, gemini_client, call_counter)
    update: dict = {"router_output": router_output}

    # For non-multi_hop, build the single SubQuery here (decomposer is skipped)
    if router_output.route != ROUTE_MULTI_HOP:
        sq = build_direct_lookup_subquery(user_query, router_output)
        update["sub_queries"] = [sq]

    return update


def decomposer_node(state: dict, resources: dict) -> dict:
    """
    Decomposer: only reached when route == "multi_hop".
    Returns ``{"sub_queries": list[SubQuery]}``.
    """
    user_query = state.get("user_query", "")
    router_output: RouterOutput = state["router_output"]
    gemini_client = resources["gemini_client"]
    call_counter = resources["call_counter"]

    sub_queries = decompose_query(user_query, router_output, gemini_client, call_counter)
    return {"sub_queries": sub_queries}


def retrieval_and_sufficiency_node(state: dict, resources: dict) -> dict:
    """
    Hybrid retrieval + batched sufficiency check + retry loop.
    Returns ``{"retrieved_chunks": dict, "sufficiency_results": dict}``.
    """
    sub_queries = state.get("sub_queries", [])
    router_output: RouterOutput = state["router_output"]
    gemini_client = resources["gemini_client"]
    call_counter = resources["call_counter"]

    retrieved = run_sufficiency_loop(
        sub_queries, router_output, resources, gemini_client, call_counter
    )
    return {"retrieved_chunks": retrieved, "sufficiency_results": {}}


def tool_use_node(state: dict, resources: dict) -> dict:
    """
    Tool-Use Agent: runs sandboxed financial computations.
    Returns ``{"computed_metrics": list[ComputedMetric]}``.
    """
    router_output: RouterOutput = state.get("router_output")
    if router_output is None or not router_output.needs_computation:
        return {"computed_metrics": []}

    sub_queries = state.get("sub_queries", [])
    retrieved_chunks = state.get("retrieved_chunks", {})

    computations = identify_required_computations(router_output, sub_queries)
    metrics = run_tool_use(computations, retrieved_chunks)
    return {"computed_metrics": metrics}


def generator_node(state: dict, resources: dict) -> dict:
    """
    Generator Agent: drafts a cited answer.
    Returns ``{"draft_answer": str}``.
    """
    user_query = state.get("user_query", "")
    sub_queries = state.get("sub_queries", [])
    retrieved_chunks = state.get("retrieved_chunks", {})
    computed_metrics = state.get("computed_metrics", [])
    gemini_client = resources["gemini_client"]
    call_counter = resources["call_counter"]

    draft = generate_answer(
        user_query, sub_queries, retrieved_chunks, computed_metrics,
        gemini_client, call_counter,
    )
    return {"draft_answer": draft}


def verifier_node(state: dict, resources: dict) -> dict:
    """
    Verifier Agent: fact-checks the draft and produces the final answer.
    Returns ``{"verifier_result": VerifierResult, "final_answer_payload": dict}``.
    """
    draft_answer = state.get("draft_answer", "")
    retrieved_chunks = state.get("retrieved_chunks", {})
    gemini_client = resources["gemini_client"]
    call_counter = resources["call_counter"]

    verifier_result: VerifierResult = verify_answer(
        draft_answer, retrieved_chunks, gemini_client, call_counter
    )

    # Use Gemini's cleaned final_answer; fall back to strip_flagged_claims if empty
    final_answer = verifier_result.final_answer
    if not final_answer or not final_answer.strip():
        final_answer = strip_flagged_claims(draft_answer, verifier_result.flagged_claims)

    # Extract citation chunk_ids from all retrieved chunks for the payload
    citation_ids: list[str] = list({
        rc.chunk.chunk_id
        for chunks in retrieved_chunks.values()
        for rc in chunks
    })

    final_answer_payload = {
        "final_answer": final_answer,
        "citations": citation_ids,
        "verdict": verifier_result.verdict,
        "flagged_claims": verifier_result.flagged_claims,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "verifier_result": verifier_result,
        "final_answer_payload": final_answer_payload,
    }


def cache_write_node(state: dict, resources: dict) -> dict:
    """
    Cache write: persists final_answer_payload to exact + semantic caches.
    Terminal node — write-only side effect, returns state unchanged.
    """
    final_answer_payload = state.get("final_answer_payload")
    if not final_answer_payload:
        logger.warning("cache_write_node: no final_answer_payload to cache")
        return {}

    user_query = state.get("user_query", "")
    redis_client = resources["redis_client"]
    voyage_client = resources["voyage_client"]

    try:
        set_exact_cache(redis_client, user_query, final_answer_payload, CORPUS_VERSION)
    except Exception as exc:
        logger.warning("cache_write_node: exact cache write failed: %s", exc)

    try:
        query_emb = embed_query(user_query, voyage_client)
        add_semantic_cache_entry(redis_client, query_emb, final_answer_payload, CORPUS_VERSION)
    except Exception as exc:
        logger.warning("cache_write_node: semantic cache write failed: %s", exc)

    return {}


# ── Conditional edge functions ────────────────────────────────────────────────

def route_after_cache_check(state: dict) -> str:
    """Returns ``"end"`` on cache hit, ``"router"`` on miss."""
    return "end" if state.get("cache_hit") else "router"


def route_after_router(state: dict) -> str:
    """
    Returns ``"decomposer"`` for multi_hop, else ``"retrieval"`` (skips
    decomposer — the quota-saving short-circuit from system_design.md §5).
    """
    router_output: RouterOutput = state.get("router_output")
    if router_output and router_output.route == ROUTE_MULTI_HOP:
        return "decomposer"
    return "retrieval"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_query_graph(resources: dict):
    """
    Constructs and compiles the LangGraph ``StateGraph``.

    All 7 nodes are wrapped with ``functools.partial`` to bind ``resources``
    (the dict of long-lived clients), giving each node the expected
    ``(state: dict) -> dict`` signature.

    Graph topology::

        cache_check → [hit? → END]
                    → router → [multi_hop? → decomposer → retrieval]
                               [else       → retrieval (skip decomposer)]
                    → tool_use → generator → verifier → cache_write → END

    Args:
        resources: Pipeline resource dict from
                   :func:`~query.pipeline.init_pipeline_resources`.

    Returns:
        Compiled LangGraph graph, ready for ``.invoke(initial_state)``.
    """
    graph = StateGraph(GraphState)

    # Bind resources to every node
    def _wrap(fn):
        return functools.partial(fn, resources=resources)

    graph.add_node("cache_check",  _wrap(cache_check_node))
    graph.add_node("router",       _wrap(router_node))
    graph.add_node("decomposer",   _wrap(decomposer_node))
    graph.add_node("retrieval",    _wrap(retrieval_and_sufficiency_node))
    graph.add_node("tool_use",     _wrap(tool_use_node))
    graph.add_node("generator",    _wrap(generator_node))
    graph.add_node("verifier",     _wrap(verifier_node))
    graph.add_node("cache_write",  _wrap(cache_write_node))

    graph.set_entry_point("cache_check")

    # Conditional: cache_check → END (hit) or router (miss)
    graph.add_conditional_edges(
        "cache_check",
        route_after_cache_check,
        {"end": END, "router": "router"},
    )

    # Conditional: router → decomposer (multi_hop) or retrieval (direct/single)
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {"decomposer": "decomposer", "retrieval": "retrieval"},
    )

    # Unconditional edges
    graph.add_edge("decomposer", "retrieval")
    graph.add_edge("retrieval",  "tool_use")
    graph.add_edge("tool_use",   "generator")
    graph.add_edge("generator",  "verifier")
    graph.add_edge("verifier",   "cache_write")
    graph.add_edge("cache_write", END)

    return graph.compile()
