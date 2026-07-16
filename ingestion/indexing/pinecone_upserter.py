"""
pinecone_upserter.py — All Pinecone SDK interactions in one module.

Responsibilities:
  - Idempotent index creation / connection
  - Chunk → Pinecone record conversion
  - Batched namespace upserts with rate-limit sleep
  - Namespace deletion for full re-ingestion
  - Namespace-routing orchestrator (company.lower() → namespace)
  - Index stats helper for post-upsert assertions

Design notes:
  - image_bytes and table_data are NOT stored in Pinecone metadata.
    Binary data is too large and the nested CleanTable object is not
    JSON-serialisable. The query pipeline re-loads the BM25 chunk list
    (which has both fields) and does a chunk_id lookup after Pinecone
    returns its top-k ids.
  - Namespace = company.lower() so multi-company queries can target a
    single namespace for cheaper Pinecone reads.
"""

from __future__ import annotations

import time
from typing import Any, Union

from pinecone import Pinecone, ServerlessSpec

from ingestion.config import (
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_REGION,
    PINECONE_UPSERT_BATCH_SIZE,
    VOYAGE_EMBEDDING_DIM,
)
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import Chunk

logger = get_logger(__name__)

_INDEX_READY_TIMEOUT_S: int = 60
_INDEX_POLL_INTERVAL_S: int = 5
_UPSERT_BATCH_SLEEP_S: float = 0.2   # between batches (Pinecone write-unit rate limit)


# ── Client init ───────────────────────────────────────────────────────────────

def init_pinecone(api_key: str) -> Pinecone:
    """
    Instantiates and returns a :class:`~pinecone.Pinecone` client.

    Args:
        api_key: Pinecone API key (from ``PINECONE_API_KEY`` env var).

    Returns:
        Authenticated :class:`~pinecone.Pinecone` client.

    Raises:
        EnvironmentError: If *api_key* is empty or ``None``.
    """
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    pc = Pinecone(api_key=api_key)
    logger.debug("Pinecone client initialised.")
    return pc


# ── Index management ──────────────────────────────────────────────────────────

def get_or_create_index(
    pc: Pinecone,
    index_name: str | None = None,
    dimension: int | None = None,
):
    """
    Returns a Pinecone Index handle for *index_name*.

    If the index does **not** exist, creates it with:

    - ``dimension``: :data:`~ingestion.config.VOYAGE_EMBEDDING_DIM` (1024)
    - ``metric``: ``"cosine"``
    - ``spec``: :class:`~pinecone.ServerlessSpec` (cloud=``PINECONE_CLOUD``,
      region=``PINECONE_REGION``)

    If it **already** exists, connects directly (idempotent — safe to call on
    every pipeline run).

    Polls up to :data:`_INDEX_READY_TIMEOUT_S` seconds (every
    :data:`_INDEX_POLL_INTERVAL_S` s) for the index to reach ``READY`` state.

    Args:
        pc:         Initialised :class:`~pinecone.Pinecone` client.
        index_name: Override the default :data:`~ingestion.config.PINECONE_INDEX_NAME`.
        dimension:  Override the default :data:`~ingestion.config.VOYAGE_EMBEDDING_DIM`.

    Returns:
        Pinecone Index handle (``pc.Index(index_name)``).

    Raises:
        TimeoutError: If the index does not reach READY within
                      :data:`_INDEX_READY_TIMEOUT_S` seconds.
    """
    name = index_name or PINECONE_INDEX_NAME
    dim  = dimension  or VOYAGE_EMBEDDING_DIM

    existing = [idx["name"] for idx in pc.list_indexes()]

    if name not in existing:
        logger.info("get_or_create_index: creating index '%s' (dim=%d).", name, dim)
        pc.create_index(
            name=name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    else:
        logger.info("get_or_create_index: index '%s' already exists — connecting.", name)

    # Poll until READY
    elapsed = 0
    while elapsed < _INDEX_READY_TIMEOUT_S:
        desc = pc.describe_index(name)
        # SDK returns IndexModel; status is a nested object with a .ready bool
        try:
            status = desc.status
            ready = status.ready if hasattr(status, "ready") else status.get("ready", False)
        except Exception:
            ready = False
        if ready:
            logger.debug("get_or_create_index: '%s' is READY.", name)
            break
        logger.debug(
            "get_or_create_index: waiting for '%s' to be READY (%ds elapsed).", name, elapsed
        )
        time.sleep(_INDEX_POLL_INTERVAL_S)
        elapsed += _INDEX_POLL_INTERVAL_S
    else:
        raise TimeoutError(
            f"Pinecone index '{name}' did not reach READY state within "
            f"{_INDEX_READY_TIMEOUT_S} seconds."
        )

    return pc.Index(name)


# ── Record conversion ─────────────────────────────────────────────────────────

def chunk_to_pinecone_record(chunk: Chunk) -> dict:
    """
    Converts a :class:`~ingestion.utils.schema.Chunk` to Pinecone's upsert
    record format::

        {
            "id":     chunk.chunk_id,
            "values": chunk.embedding,          # list[float], len=1024
            "metadata": {
                "company", "ticker", "fiscal_year", "filing_type",
                "section", "content_type", "page_number",
                "text",           # stored for display in query results
                "source_file", "embedding_model", "corpus_version"
            }
        }

    Fields intentionally **not** stored in Pinecone metadata:

    - ``image_bytes`` — binary, too large; lives in the BM25 pickle.
    - ``table_data``  — nested object; query pipeline re-loads from BM25.
    - ``embedding``   — Pinecone stores it in its own vector storage.

    Chart and table chunks are retrieved by ``chunk_id`` from Pinecone,
    then the full :class:`~ingestion.utils.schema.Chunk` (with
    ``image_bytes`` / ``table_data``) is looked up from the BM25 chunk
    list in the query pipeline.

    Args:
        chunk: A :class:`~ingestion.utils.schema.Chunk` with a
               non-``None`` ``embedding``.

    Returns:
        ``dict`` ready to pass to ``index.upsert(vectors=[...])``.

    Raises:
        ValueError: If ``chunk.embedding`` is ``None``.
    """
    if chunk.embedding is None:
        raise ValueError(
            f"chunk '{chunk.chunk_id}' has embedding=None. "
            "Call embed_text_chunks / embed_multimodal_chunks first."
        )

    return {
        "id":     chunk.chunk_id,
        "values": chunk.embedding,
        "metadata": {
            "company":         chunk.company,
            "ticker":          chunk.ticker,
            "fiscal_year":     chunk.fiscal_year,
            "filing_type":     chunk.filing_type,
            "section":         chunk.section,
            "content_type":    chunk.content_type,
            "page_number":     chunk.page_number,
            "text":            chunk.text,
            "source_file":     chunk.source_file,
            "embedding_model": chunk.embedding_model,
            "corpus_version":  chunk.corpus_version,
        },
    }


# ── Upsert ────────────────────────────────────────────────────────────────────

def upsert_chunks(
    index,
    chunks: list[Chunk],
    namespace: str,
    batch_size: int | None = None,
) -> None:
    """
    Upserts *chunks* into the given Pinecone *namespace* in batches.

    Steps:

    1. :func:`chunk_to_pinecone_record` for each chunk.
    2. Split into batches of *batch_size* (default
       :data:`~ingestion.config.PINECONE_UPSERT_BATCH_SIZE`).
    3. ``index.upsert(vectors=batch, namespace=namespace)``
    4. Sleep :data:`_UPSERT_BATCH_SLEEP_S` between batches.
    5. Log ``"Upserted batch N/M (K vectors total)"``.

    Additive — does **not** delete existing vectors. Call
    :func:`delete_namespace` first if you need to re-index from scratch.

    Args:
        index:      Pinecone Index handle.
        chunks:     List of embedded chunks.
        namespace:  Target namespace string (e.g. ``"apple"``).
        batch_size: Override default batch size.

    Raises:
        ValueError: If any chunk has ``embedding=None``.
    """
    bs = batch_size or PINECONE_UPSERT_BATCH_SIZE
    records = [chunk_to_pinecone_record(c) for c in chunks]   # raises if any embedding=None

    batches = [records[i:i + bs] for i in range(0, len(records), bs)]
    total   = len(records)

    logger.info(
        "upsert_chunks: upserting %d vectors into namespace='%s', %d batches.",
        total, namespace, len(batches),
    )

    for batch_idx, batch in enumerate(batches):
        index.upsert(vectors=batch, namespace=namespace)
        logger.info(
            "Upserted batch %d/%d (%d vectors total) → namespace='%s'.",
            batch_idx + 1, len(batches), (batch_idx + 1) * len(batch), namespace,
        )
        if batch_idx < len(batches) - 1:
            time.sleep(_UPSERT_BATCH_SLEEP_S)


# ── Namespace deletion ────────────────────────────────────────────────────────

def delete_namespace(index, namespace: str) -> None:
    """
    Deletes **all** vectors in *namespace*.

    Used when re-ingesting a company's filings after a corpus update.
    Combined with a :data:`~ingestion.config.CORPUS_VERSION` bump and
    Upstash Redis cache invalidation, this is the complete re-ingestion
    procedure for one company.

    Args:
        index:     Pinecone Index handle.
        namespace: Namespace to wipe (e.g. ``"apple"``).
    """
    try:
        stats_before = get_index_stats(index)
        before_count = stats_before.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
    except Exception:
        before_count = "unknown"

    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as exc:
        # Namespace does not exist yet — nothing to delete
        if "not found" in str(exc).lower() or "404" in str(exc):
            logger.debug("delete_namespace: namespace '%s' not found — skipping.", namespace)
        else:
            raise
    logger.info(
        "delete_namespace: deleted ~%s vectors from namespace='%s'.",
        before_count, namespace,
    )


# ── Orchestrator ──────────────────────────────────────────────────────────────

def upsert_all_by_namespace(index, chunks: list[Chunk]) -> None:
    """
    Routes *chunks* to the correct Pinecone namespace by company name
    (``chunk.company.lower()``) then calls :func:`upsert_chunks` for
    each namespace group.

    Namespace mapping examples::

        "Apple"     → "apple"
        "Microsoft" → "microsoft"
        "NVIDIA"    → "nvidia"
        "TestCo"    → "testco"

    Logs per-namespace vector counts before and after upsert.

    Args:
        index:  Pinecone Index handle.
        chunks: All embedded chunks for a pipeline run (mixed companies
                are supported — each routed to its own namespace).
    """
    # Group by namespace
    ns_map: dict[str, list[Chunk]] = {}
    for c in chunks:
        ns = c.company.lower()
        ns_map.setdefault(ns, []).append(c)

    stats_before = get_index_stats(index)
    ns_before = stats_before.get("namespaces", {})

    for ns, ns_chunks in ns_map.items():
        count_before = ns_before.get(ns, {}).get("vector_count", 0)
        logger.info(
            "upsert_all_by_namespace: namespace='%s' — %d chunks to upsert "
            "(currently %d vectors in index).",
            ns, len(ns_chunks), count_before,
        )
        upsert_chunks(index, ns_chunks, namespace=ns)

    stats_after = get_index_stats(index)
    ns_after = stats_after.get("namespaces", {})
    for ns in ns_map:
        count_after = ns_after.get(ns, {}).get("vector_count", 0)
        logger.info(
            "upsert_all_by_namespace: namespace='%s' — now %d vectors in index.",
            ns, count_after,
        )


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_index_stats(index) -> dict:
    """
    Returns ``index.describe_index_stats()`` as a plain ``dict``.

    Logged by ``pipeline.py`` after upsert to confirm all vectors are
    stored.  Also used in verification steps to assert
    ``total_vector_count == len(chunks)``.

    Args:
        index: Pinecone Index handle.

    Returns:
        Stats dict with at minimum ``"total_vector_count"`` and
        ``"namespaces"`` keys.
    """
    raw = index.describe_index_stats()
    # Pinecone SDK returns a DescribeIndexStatsResponse object.
    # Normalise to a plain dict with guaranteed keys for assertion use.
    if hasattr(raw, "to_dict"):
        return raw.to_dict()

    # Manual extraction for SDK versions that don't expose to_dict
    result: dict[str, Any] = {}
    result["total_vector_count"] = getattr(raw, "total_vector_count", 0)

    # Namespaces: dict[str, NamespaceSummary] → dict[str, {"vector_count": int}]
    raw_ns = getattr(raw, "namespaces", {}) or {}
    ns_out: dict[str, dict] = {}
    for ns_name, ns_val in raw_ns.items():
        if hasattr(ns_val, "vector_count"):
            ns_out[ns_name] = {"vector_count": ns_val.vector_count}
        elif isinstance(ns_val, dict):
            ns_out[ns_name] = ns_val
        else:
            ns_out[ns_name] = {"vector_count": 0}
    result["namespaces"] = ns_out

    result["dimension"]           = getattr(raw, "dimension", None)
    result["index_fullness"]      = getattr(raw, "index_fullness", None)
    return result
