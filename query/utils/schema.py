"""
query/utils/schema.py — All typed dataclasses that flow through the
LangGraph query-pipeline state machine.

``QueryState`` is the single object passed between all graph nodes —
the query-pipeline equivalent of ``ingestion.utils.schema.Chunk``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ingestion.utils.schema import Chunk


# ── Router Agent output ────────────────────────────────────────────────────────

@dataclass
class RouterOutput:
    """Structured output of the Router Agent (Gemini, response_schema-constrained)."""
    route: str                    # "direct_lookup" | "single_hop" | "multi_hop"
    companies: list[str]          # e.g. ["apple", "microsoft", "nvidia"]
    years: list[int]              # e.g. [2022, 2023, 2024]
    needs_computation: bool       # triggers Tool-Use Agent downstream


# ── Decomposer Agent output ────────────────────────────────────────────────────

@dataclass
class SubQuery:
    """One decomposed sub-question. For direct_lookup/single_hop, exactly one exists."""
    sub_query_id: int
    text: str                           # e.g. "What was Apple's R&D expense in FY2022?"
    company: Optional[str] = None       # lowercase slug, e.g. "apple"
    fiscal_year: Optional[int] = None
    section_hint: Optional[str] = None  # e.g. "income_statement" — guides metadata_filter
    retry_count: int = 0                # incremented by the sufficiency retry loop


# ── Retrieval layer ────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A chunk plus every retrieval-stage score it accumulated."""
    chunk: Chunk
    bm25_score: Optional[float] = None
    semantic_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


# ── Sufficiency Check Agent output ────────────────────────────────────────────

@dataclass
class SufficiencyResult:
    """One entry of the Sufficiency Check Agent's batched structured output."""
    sub_query_id: int
    sufficient: bool
    missing: list[str] = field(default_factory=list)  # e.g. ["FY2022 R&D figure"]


# ── Tool-Use Agent output ─────────────────────────────────────────────────────

@dataclass
class ComputedMetric:
    """One derived number computed by the sandboxed Tool-Use Agent."""
    name: str                       # e.g. "aapl_rd_pct_fy22"
    value: float
    formula: str                    # e.g. "R&D_expense / Revenue * 100"
    source_chunk_ids: list[str]     # provenance — never LLM-estimated


# ── Verifier Agent output ─────────────────────────────────────────────────────

@dataclass
class VerifierResult:
    """Structured output of the Verifier Agent."""
    verdict: str                    # "verified" | "partial"
    flagged_claims: list[str]
    final_answer: str


# ── Top-level LangGraph state ─────────────────────────────────────────────────

@dataclass
class QueryState:
    """
    The single state object threaded through every LangGraph node.
    Each node reads fields it needs and returns a partial update merged
    into this state by LangGraph's reducer.
    """
    user_query: str
    normalized_query: str = ""
    cache_hit: bool = False
    cached_answer: Optional[dict] = None

    router_output: Optional[RouterOutput] = None
    sub_queries: list[SubQuery] = field(default_factory=list)

    # keyed by sub_query_id
    retrieved_chunks: dict[int, list[RetrievedChunk]] = field(default_factory=dict)
    sufficiency_results: dict[int, SufficiencyResult] = field(default_factory=dict)

    computed_metrics: list[ComputedMetric] = field(default_factory=list)
    draft_answer: str = ""
    verifier_result: Optional[VerifierResult] = None
    final_answer_payload: Optional[dict] = None

    gemini_call_count: int = 0      # incremented by gemini_client on every call
    warnings: list[str] = field(default_factory=list)
