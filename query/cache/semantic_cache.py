"""
query/cache/semantic_cache.py — Embedding-similarity query cache backed by Upstash Redis.

Stores ``(query_embedding, answer_payload)`` pairs in a Redis list. On lookup,
fetches all entries and finds the one whose embedding has the highest cosine
similarity to the incoming query embedding. Returns the cached answer if
similarity ≥ ``SEMANTIC_CACHE_SIMILARITY_THRESHOLD`` (0.95).

Called only on an exact-cache MISS. Embedding the query costs one Voyage API
call but prevents multiple Gemini calls, making it cost-effective.

List key format: ``"semantic:{corpus_version}"``
List is kept bounded by an LTRIM to ``SEMANTIC_CACHE_MAX_ENTRIES`` entries.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timezone
from typing import Optional

from query.config import (
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD,
    SEMANTIC_CACHE_MAX_ENTRIES,
)

logger = logging.getLogger(__name__)


def semantic_cache_list_key(corpus_version: str) -> str:
    """
    Returns the Redis list key that holds all semantic cache entries.

    Format: ``"semantic:{corpus_version}"``

    Args:
        corpus_version: e.g. ``"v1"`` from :data:`~query.config.CORPUS_VERSION`.

    Returns:
        Redis key string.
    """
    return f"semantic:{corpus_version}"


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Computes the cosine similarity between two vectors.

    Formula: ``dot(v1, v2) / (norm(v1) * norm(v2))``

    Pure-Python/math implementation — no numpy required. Efficient enough for
    1024-dim vectors at cache scale (hundreds of entries, not millions).

    Args:
        vec1: First embedding vector.
        vec2: Second embedding vector (must be same length as ``vec1``).

    Returns:
        Cosine similarity in ``[-1.0, 1.0]``. Returns ``0.0`` if either
        vector has zero norm (avoids division by zero).
    """
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)


def get_semantic_cache(
    redis_client,
    query_embedding: list[float],
    corpus_version: str,
    threshold: Optional[float] = None,
) -> Optional[dict]:
    """
    Searches the semantic cache for a sufficiently similar past query.

    Steps:
    1. Fetch all entries from the Redis list (``LRANGE 0 -1``).
    2. For each entry: deserialise and compute cosine similarity with
       ``query_embedding``.
    3. Track the best-scoring entry.
    4. If ``best_similarity >= threshold`` (default:
       ``SEMANTIC_CACHE_SIMILARITY_THRESHOLD`` = 0.95), return that entry's
       ``answer_payload``.
    5. Otherwise return ``None``.

    Args:
        redis_client:    Upstash Redis client.
        query_embedding: 1024-dim embedding of the incoming query.
        corpus_version:  Must match the version used when entries were written.
        threshold:       Similarity threshold override; defaults to
                         ``SEMANTIC_CACHE_SIMILARITY_THRESHOLD``.

    Returns:
        Cached ``answer_payload`` dict on a hit, ``None`` on a miss.
    """
    _threshold = threshold if threshold is not None else SEMANTIC_CACHE_SIMILARITY_THRESHOLD
    key = semantic_cache_list_key(corpus_version)
    raw_entries = redis_client.lrange(key, 0, -1)

    if not raw_entries:
        logger.debug("get_semantic_cache: list empty for key=%s", key)
        return None

    best_score: float = -1.0
    best_payload: Optional[dict] = None

    for raw in raw_entries:
        try:
            entry = json.loads(raw)
            cached_embedding: list[float] = entry["embedding"]
            sim = cosine_similarity(query_embedding, cached_embedding)
            if sim > best_score:
                best_score = sim
                best_payload = entry["answer_payload"]
        except (KeyError, json.JSONDecodeError, TypeError):
            logger.warning("get_semantic_cache: malformed entry skipped")
            continue

    if best_score >= _threshold:
        logger.info(
            "get_semantic_cache: HIT (similarity=%.4f >= threshold=%.2f)",
            best_score, _threshold,
        )
        return best_payload

    logger.debug(
        "get_semantic_cache: MISS (best_similarity=%.4f < threshold=%.2f)",
        best_score, _threshold,
    )
    return None


def add_semantic_cache_entry(
    redis_client,
    query_embedding: list[float],
    answer_payload: dict,
    corpus_version: str,
) -> None:
    """
    Adds a new entry to the semantic cache list and trims to the size cap.

    Steps:
    1. Build the list key from ``corpus_version``.
    2. Serialise ``{"embedding": query_embedding, "answer_payload": answer_payload,
       "cached_at": <ISO-8601 UTC timestamp>}`` to JSON.
    3. ``LPUSH`` the entry to the head of the list.
    4. ``LTRIM`` the list to ``[0, SEMANTIC_CACHE_MAX_ENTRIES - 1]`` so it
       never exceeds the configured cap.

    Called by ``cache_write_node`` alongside :func:`~query.cache.exact_cache.set_exact_cache`.

    Args:
        redis_client:    Upstash Redis client.
        query_embedding: 1024-dim embedding vector to store.
        answer_payload:  Answer dict to return on future cache hits.
        corpus_version:  e.g. ``"v1"``.
    """
    key = semantic_cache_list_key(corpus_version)
    entry = {
        "embedding": query_embedding,
        "answer_payload": answer_payload,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    redis_client.lpush(key, json.dumps(entry))
    redis_client.ltrim(key, 0, SEMANTIC_CACHE_MAX_ENTRIES - 1)
    logger.info(
        "add_semantic_cache_entry: pushed to key=%s (cap=%d)",
        key, SEMANTIC_CACHE_MAX_ENTRIES,
    )
