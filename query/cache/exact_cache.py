"""
query/cache/exact_cache.py — Exact-match query cache backed by Upstash Redis.

Keys are namespaced by ``corpus_version`` so bumping the version after
re-ingestion automatically orphans all old cache entries — no bulk delete
needed (per system_design.md §3.6's invalidation design).

Key format: ``"exact:{corpus_version}:{sha256_of_normalized_query}"``
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from query.cache.redis_client import normalize_query, hash_query
from query.config import REDIS_CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)


def build_exact_cache_key(query_hash: str, corpus_version: str) -> str:
    """
    Returns the Redis key for an exact-match cache entry.

    Format: ``"exact:{corpus_version}:{query_hash}"``

    Namespacing by ``corpus_version`` means bumping ``CORPUS_VERSION`` after
    re-ingestion automatically orphans all old keys — no bulk delete needed.

    Args:
        query_hash:     64-char hex string from :func:`~query.cache.redis_client.hash_query`.
        corpus_version: e.g. ``"v1"`` from :data:`~query.config.CORPUS_VERSION`.

    Returns:
        Redis key string.
    """
    return f"exact:{corpus_version}:{query_hash}"


def get_exact_cache(
    redis_client,
    query: str,
    corpus_version: str,
) -> Optional[dict]:
    """
    Looks up a query in the exact-match cache.

    Steps:
    1. Normalise ``query`` via :func:`~query.cache.redis_client.normalize_query`.
    2. Build the cache key via :func:`build_exact_cache_key`.
    3. Fetch the raw JSON string from Redis.
    4. Return ``json.loads(raw)`` on a hit, ``None`` on a miss.

    Expected stored shape::

        {
            "final_answer": str,
            "citations":    list,
            "cached_at":    "<ISO-8601 timestamp>"
        }

    Called as the very first step of the query graph (``cache_check_node``).

    Args:
        redis_client:   Upstash Redis client from :func:`~query.cache.redis_client.init_redis_client`.
        query:          Raw user query string.
        corpus_version: Must match the version used when the entry was written.

    Returns:
        Cached answer payload dict, or ``None`` on a miss.
    """
    normalized = normalize_query(query)
    key = build_exact_cache_key(hash_query(normalized), corpus_version)
    raw = redis_client.get(key)
    if raw is None:
        logger.debug("get_exact_cache: MISS key=%s", key)
        return None
    logger.info("get_exact_cache: HIT key=%s", key)
    return json.loads(raw)


def set_exact_cache(
    redis_client,
    query: str,
    answer_payload: dict,
    corpus_version: str,
    ttl_seconds: Optional[int] = None,
) -> None:
    """
    Writes an answer payload to the exact-match cache.

    Steps:
    1. Normalise ``query``.
    2. Build the cache key.
    3. ``redis_client.set(key, json.dumps(answer_payload), ex=ttl_seconds)``.

    Called by ``cache_write_node`` after the Verifier Agent produces
    ``final_answer_payload``.

    Args:
        redis_client:    Upstash Redis client.
        query:           Raw user query string.
        answer_payload:  Dict to cache (must be JSON-serialisable).
        corpus_version:  e.g. ``"v1"``.
        ttl_seconds:     Override TTL; defaults to ``REDIS_CACHE_TTL_SECONDS`` (30 days).
    """
    normalized = normalize_query(query)
    key = build_exact_cache_key(hash_query(normalized), corpus_version)
    ttl = ttl_seconds if ttl_seconds is not None else REDIS_CACHE_TTL_SECONDS
    redis_client.set(key, json.dumps(answer_payload), ex=ttl)
    logger.info("set_exact_cache: wrote key=%s (ttl=%ds)", key, ttl)
