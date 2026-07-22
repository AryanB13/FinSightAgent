"""
query/cache/redis_client.py — Upstash Redis REST client init and shared key helpers.

HTTP-based client (no persistent TCP connection) matching Upstash's serverless
design. All cache modules import from here rather than instantiating their own
connections.
"""

from __future__ import annotations

import hashlib
import logging
import re

from upstash_redis import Redis

logger = logging.getLogger(__name__)


def init_redis_client(url: str, token: str) -> Redis:
    """
    Instantiates an Upstash Redis REST client.

    Uses the ``upstash-redis`` Python package which communicates over HTTPS —
    no persistent TCP connection needed, matching Upstash's serverless design.

    Args:
        url:   Upstash Redis REST URL (``UPSTASH_REDIS_REST_URL`` env var).
        token: Upstash Redis REST token (``UPSTASH_REDIS_REST_TOKEN`` env var).

    Raises:
        EnvironmentError: If ``url`` or ``token`` is empty or None.

    Returns:
        Configured ``upstash_redis.Redis`` client instance.
    """
    if not url:
        raise EnvironmentError(
            "UPSTASH_REDIS_REST_URL is not set. "
            "Add it to your .env file from https://console.upstash.com/"
        )
    if not token:
        raise EnvironmentError(
            "UPSTASH_REDIS_REST_TOKEN is not set. "
            "Add it to your .env file from https://console.upstash.com/"
        )
    client = Redis(url=url, token=token)
    logger.info("init_redis_client: connected to Upstash Redis at %s", url)
    return client


def normalize_query(query: str) -> str:
    """
    Normalises a query string before hashing or embedding for cache lookup.

    Transformation steps:
    1. Lowercase.
    2. Strip leading/trailing whitespace.
    3. Collapse internal whitespace runs to single spaces.
    4. Strip trailing punctuation (``?``, ``.``, ``!``).

    This ensures ``"What was Apple's revenue?"`` and
    ``"what was apple's revenue"`` hash to the same exact-cache key.

    Args:
        query: Raw user query string.

    Returns:
        Normalised query string.
    """
    q = query.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = q.rstrip("?!.")
    return q


def hash_query(normalized_query: str) -> str:
    """
    Returns ``SHA256(normalized_query)`` as a lowercase hex string.

    Used as the suffix of every exact-match cache key so that two queries
    normalising to the same string share the same cache entry.

    Args:
        normalized_query: Output of :func:`normalize_query`.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()
