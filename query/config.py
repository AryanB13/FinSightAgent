"""
query/config.py — All constants for the online query pipeline (Part B).

Mirrors ingestion/config.py's role for Part A. Changing a model name,
threshold, or top-k should require editing only this file.

Also re-exports the ingestion-side constants needed by query modules so
query/ code has one consistent import surface.
"""

from ingestion.config import (
    CORPUS_VERSION,
    BM25_INDEX_PATH,
    BM25_CHUNKS_PATH,
    PINECONE_INDEX_NAME,
    VOYAGE_MULTIMODAL_MODEL,
)

# Re-export under the query-pipeline canonical name
VOYAGE_TEXT_MODEL = VOYAGE_MULTIMODAL_MODEL  # unified vector space (all use multimodal)

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-3.1-flash-lite"

GEMINI_MAX_RETRIES: int = 3
GEMINI_RETRY_BACKOFF_BASE: float = 1.0      # seconds; doubles per retry (1/2/4s)

# Log a WARNING once today's call counter crosses this value.
# 250 is the actual free-tier ceiling per system_design.md §5.
GEMINI_DAILY_QUOTA_WARNING: int = 230

# ── Redis / Cache ─────────────────────────────────────────────────────────────
REDIS_CACHE_TTL_SECONDS: int = 2_592_000    # 30 days

# Semantic cache: minimum cosine similarity to count as a hit
SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.95

# LTRIM cap so the Redis list doesn't grow unbounded
SEMANTIC_CACHE_MAX_ENTRIES: int = 2_000

# ── Retrieval ─────────────────────────────────────────────────────────────────
BM25_TOP_K: int = 20           # candidates pulled from BM25 before fusion
SEMANTIC_TOP_K: int = 20       # candidates pulled from Pinecone before fusion
RRF_K: int = 60                # damping constant: score = Σ 1/(rank_i + RRF_K)
RERANK_TOP_N: int = 5          # final chunks per sub-query after reranking

# ── Sufficiency retry ─────────────────────────────────────────────────────────
MAX_SUFFICIENCY_RETRIES: int = 2    # from system_design.md §2 diagram

# ── Pinecone namespaces ───────────────────────────────────────────────────────
PINECONE_NAMESPACES: list[str] = ["apple", "microsoft", "nvidia"]

# ── Sandbox ───────────────────────────────────────────────────────────────────
SANDBOX_TIMEOUT_SECONDS: int = 5    # kill Tool-Use subprocess if it hangs

# ── Router route constants ────────────────────────────────────────────────────
ROUTE_DIRECT_LOOKUP: str = "direct_lookup"
ROUTE_SINGLE_HOP: str = "single_hop"
ROUTE_MULTI_HOP: str = "multi_hop"
