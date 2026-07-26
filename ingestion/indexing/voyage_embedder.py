"""
voyage_embedder.py — All Voyage AI API calls in one module.

Handles:
  - Client initialisation
  - Batched text embedding (voyage-3.5)
  - Batched multimodal embedding (voyage-multimodal-3)
  - Single-query embedding for the query pipeline
  - Rate-limit retries with exponential backoff
  - Pre-flight cost estimation

ALL chunks use VOYAGE_MULTIMODAL_MODEL to share the same vector space:
  Text chunks  → VOYAGE_MULTIMODAL_MODEL (text-only input)
  Table chunks → VOYAGE_MULTIMODAL_MODEL (text-only input)
  Chart chunks → VOYAGE_MULTIMODAL_MODEL (text + image input)
  Queries      → VOYAGE_MULTIMODAL_MODEL (text-only input)

This ensures cross-modal retrieval works: a text query can retrieve
table chunks and chart chunks with valid cosine similarity.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import voyageai
from PIL import Image

from ingestion.config import (
    APPROX_CHARS_PER_TOKEN,
    VOYAGE_EMBEDDING_DIM,
    VOYAGE_MULTIMODAL_BATCH_SIZE,
    VOYAGE_MULTIMODAL_MODEL,
    VOYAGE_TEXT_BATCH_SIZE,
    VOYAGE_TEXT_MODEL,
)
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import Chunk

logger = get_logger(__name__)

# Voyage free-tier caps (used by estimate_embedding_cost)
_FREE_TIER_TOKEN_CAP: int = 200_000_000   # 200M tokens / month
_FREE_TIER_PIXEL_CAP: int = 150_000_000_000  # 150B pixels / month

_MAX_RETRIES: int = 3
_RETRY_BASE_SLEEP: float = 22.0  # seconds; doubles each retry (22/44/88s for 3 RPM free tier)


# ── Client init ───────────────────────────────────────────────────────────────

def init_voyage_client(api_key: str) -> voyageai.Client:
    """
    Instantiates and returns a :class:`voyageai.Client`.

    Args:
        api_key: Voyage AI secret key (from ``VOYAGE_API_KEY`` env var).

    Returns:
        Authenticated :class:`voyageai.Client` ready for embedding calls.

    Raises:
        EnvironmentError: If *api_key* is empty or ``None``.
    """
    if not api_key:
        raise EnvironmentError(
            "VOYAGE_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    client = voyageai.Client(api_key=api_key)
    logger.debug("Voyage AI client initialised.")
    return client


# ── Text embedding ─────────────────────────────────────────────────────────────

def embed_text_chunks(
    chunks: list[Chunk],
    client: voyageai.Client,
    batch_size: Optional[int] = None,
) -> list[Chunk]:
    """
    Embeds all ``content_type="text"`` chunks using :data:`VOYAGE_MULTIMODAL_MODEL`
    with text-only input.

    **Design note**: Text chunks use the multimodal model (not the text-only
    ``voyage-3.5``) so they share the same 1024-dim vector space as tables
    and charts. This enables cross-modal retrieval: a text query can retrieve
    table chunks and chart chunks with semantically valid cosine similarity.

    Table and chart chunks are silently skipped (handled by
    :func:`embed_multimodal_chunks`).

    Steps:

    1. Filter to ``text`` chunks only.
    2. Split into batches of *batch_size* (default :data:`VOYAGE_MULTIMODAL_BATCH_SIZE`).
    3. For each batch: build text-only input lists ``[[chunk.text]]`` →
       :func:`_batch_embed_multimodal` → assign vectors **in-place**.
    4. Sleep 25 s before each batch (after the first) to respect free-tier
       3 RPM rate limit (1 request per 20 seconds + 5s safety buffer).

    Modifies *chunks* in place and also returns *chunks* for chaining.

    Logs: batch count, estimated tokens consumed, elapsed time.

    Args:
        chunks:     Full chunk list (mixed content types — non-text skipped).
        client:     Initialised :class:`voyageai.Client`.
        batch_size: Override default batch size (useful for testing).

    Returns:
        The same *chunks* list with ``embedding`` filled for all text chunks.
    """
    bs = batch_size or VOYAGE_MULTIMODAL_BATCH_SIZE
    text_chunks = [c for c in chunks if c.content_type == "text"]

    if not text_chunks:
        logger.debug("embed_text_chunks: no text chunks to embed.")
        return chunks

    t0 = time.time()
    total_chars = sum(len(c.text) for c in text_chunks)
    est_tokens  = round(total_chars / APPROX_CHARS_PER_TOKEN)

    # Build text-only inputs for multimodal model: [[text], [text], ...]
    inputs = [[c.text] for c in text_chunks]

    batches = [
        (text_chunks[i:i + bs], inputs[i:i + bs])
        for i in range(0, len(text_chunks), bs)
    ]
    logger.info(
        "embed_text_chunks: %d text chunks, %d batches, ~%d estimated tokens.",
        len(text_chunks), len(batches), est_tokens,
    )

    for batch_idx, (batch_chunks, batch_inputs) in enumerate(batches):
        # Sleep 22s before EVERY batch (including first) to respect 3 RPM free tier
        logger.info("embed_text_chunks: sleeping 22s (batch %d/%d)...", batch_idx + 1, len(batches))
        time.sleep(22)
        vectors = _batch_embed_multimodal(batch_inputs, client, VOYAGE_MULTIMODAL_MODEL)
        for chunk, vec in zip(batch_chunks, vectors):
            chunk.embedding = vec
        logger.debug("embed_text_chunks: batch %d/%d done.", batch_idx + 1, len(batches))

    elapsed = time.time() - t0
    logger.info(
        "embed_text_chunks: complete — %d chunks embedded in %.1fs.",
        len(text_chunks), elapsed,
    )
    return chunks


# ── Multimodal embedding ───────────────────────────────────────────────────────

def embed_multimodal_chunks(
    chunks: list[Chunk],
    client: voyageai.Client,
    batch_size: Optional[int] = None,
) -> list[Chunk]:
    """
    Embeds all ``content_type in {"table", "chart"}`` chunks using
    :data:`VOYAGE_MULTIMODAL_MODEL`.

    **Table chunks** → text-only input block::

        [{"type": "text", "text": chunk.text}]

    This embeds tables through the multimodal model so they share the same
    1024-dimensional vector space as charts — the key design decision that
    enables cross-modal nearest-neighbour retrieval.

    **Chart chunks with caption** → mixed input::

        [{"type": "text",        "text":        chunk.text},
         {"type": "image_bytes", "image_bytes": chunk.image_bytes}]

    **Chart chunks without caption** (``chunk.text`` ends in
    ``"no caption extracted."``\ ) → image-only input::

        [{"type": "image_bytes", "image_bytes": chunk.image_bytes}]

    Steps:

    1. Filter to ``table`` / ``chart`` chunks only.
    2. Build per-chunk input list.
    3. Batches of *batch_size* (default :data:`VOYAGE_MULTIMODAL_BATCH_SIZE`).
    4. :func:`_batch_embed_multimodal` per batch → assign embeddings in-place.
    5. Sleep 25 s before each batch (after the first) to respect free-tier
       3 RPM rate limit (1 request per 20 seconds + 5s safety buffer).

    Modifies *chunks* in place and returns *chunks* for chaining.

    Args:
        chunks:     Full chunk list.
        client:     Initialised :class:`voyageai.Client`.
        batch_size: Override default batch size.

    Returns:
        The same *chunks* list with ``embedding`` filled for multimodal chunks.
    """
    bs = batch_size or VOYAGE_MULTIMODAL_BATCH_SIZE
    mm_chunks = [c for c in chunks if c.content_type in {"table", "chart"}]

    if not mm_chunks:
        logger.debug("embed_multimodal_chunks: no multimodal chunks to embed.")
        return chunks

    # Build per-chunk input lists (str | PIL.Image per SDK requirements)
    inputs: list[list] = []
    for chunk in mm_chunks:
        if chunk.content_type == "table":
            # Text-only block — plain string in a list
            inputs.append([chunk.text])
        else:  # chart
            pil_img = Image.open(io.BytesIO(chunk.image_bytes))
            no_caption = "no caption extracted" in chunk.text.lower()
            if no_caption:
                # Image-only block
                inputs.append([pil_img])
            else:
                # Text + image block
                inputs.append([chunk.text, pil_img])

    t0 = time.time()
    batches = [
        (mm_chunks[i:i + bs], inputs[i:i + bs])
        for i in range(0, len(mm_chunks), bs)
    ]
    logger.info(
        "embed_multimodal_chunks: %d multimodal chunks, %d batches.",
        len(mm_chunks), len(batches),
    )

    for batch_idx, (batch_chunks, batch_inputs) in enumerate(batches):
        # Sleep 22s before EVERY batch (including first) to respect 3 RPM free tier
        logger.info("embed_multimodal_chunks: sleeping 22s (batch %d/%d)...", batch_idx + 1, len(batches))
        time.sleep(22)
        vectors = _batch_embed_multimodal(batch_inputs, client, VOYAGE_MULTIMODAL_MODEL)
        for chunk, vec in zip(batch_chunks, vectors):
            chunk.embedding = vec
        logger.debug(
            "embed_multimodal_chunks: batch %d/%d done.", batch_idx + 1, len(batches)
        )

    elapsed = time.time() - t0
    logger.info(
        "embed_multimodal_chunks: complete — %d chunks embedded in %.1fs.",
        len(mm_chunks), elapsed,
    )
    return chunks


# ── Query embedding ────────────────────────────────────────────────────────────

def embed_query(
    query: str,
    client: voyageai.Client,
    model: Optional[str] = None,
) -> list[float]:
    """
    Embeds a single query string for retrieval use.

    Uses :data:`VOYAGE_MULTIMODAL_MODEL` with text-only input so the query
    vector shares the same 1024-dim space as all chunks (text, table, chart).
    This enables cross-modal retrieval: a text query can retrieve chart
    chunks with valid cosine similarity.

    Called by the query pipeline (Part B), not during ingestion.  Lives
    here so the same client and model constants are reused in both phases.

    Args:
        query:  Raw query string (e.g. ``"Apple net sales FY2023"``).
        client: Initialised :class:`voyageai.Client`.
        model:  Override the default :data:`VOYAGE_MULTIMODAL_MODEL`.

    Returns:
        ``list[float]`` of length :data:`VOYAGE_EMBEDDING_DIM` (1024).
    """
    m = model or VOYAGE_MULTIMODAL_MODEL
    vectors = _batch_embed_multimodal([[query]], client, m)
    return vectors[0]


# ── Cost estimator ─────────────────────────────────────────────────────────────

def estimate_embedding_cost(chunks: list[Chunk]) -> dict:
    """
    Pre-flight estimate of Voyage AI usage before making any API calls.

    Estimates:

    - **text_tokens_estimated**: sum of ``len(c.text) / APPROX_CHARS_PER_TOKEN``
      for all ``content_type="text"`` chunks.
    - **multimodal_text_tokens_estimated**: same for ``table`` / ``chart``
      chunk text portions.
    - **multimodal_pixels_estimated**: sum of ``chunk image_bytes`` decoded
      pixel count — approximated as ``len(image_bytes) * 2`` (conservative
      estimate assuming ~50% JPEG/PNG compression ratio); uses exact
      ``pixel_count`` where available via metadata heuristic.

    Logs a ``WARNING`` if any estimate exceeds the free-tier cap.

    Args:
        chunks: All chunks for a pipeline run.

    Returns:
        ::

            {
                "text_tokens_estimated":              int,
                "multimodal_text_tokens_estimated":   int,
                "multimodal_pixels_estimated":        int,
                "within_free_tier":                   bool,
            }
    """
    text_tokens = 0
    mm_text_tokens = 0
    mm_pixels = 0

    for c in chunks:
        tok = round(len(c.text) / APPROX_CHARS_PER_TOKEN)
        if c.content_type == "text":
            text_tokens += tok
        else:
            mm_text_tokens += tok
            if c.image_bytes:
                # Conservative: assume 50% compression → raw pixels ≈ 2× bytes
                mm_pixels += len(c.image_bytes) * 2

    within = (
        text_tokens    < _FREE_TIER_TOKEN_CAP
        and mm_text_tokens < _FREE_TIER_TOKEN_CAP
        and mm_pixels  < _FREE_TIER_PIXEL_CAP
    )

    if not within:
        logger.warning(
            "estimate_embedding_cost: EXCEEDS free-tier limits! "
            "text_tokens=%d, mm_text_tokens=%d, mm_pixels=%d",
            text_tokens, mm_text_tokens, mm_pixels,
        )
    else:
        logger.info(
            "estimate_embedding_cost: within free tier — "
            "text_tokens=%d, mm_text_tokens=%d, mm_pixels=%d.",
            text_tokens, mm_text_tokens, mm_pixels,
        )

    return {
        "text_tokens_estimated":            text_tokens,
        "multimodal_text_tokens_estimated": mm_text_tokens,
        "multimodal_pixels_estimated":      mm_pixels,
        "within_free_tier":                 within,
    }


# ── Internal batch helpers ─────────────────────────────────────────────────────

def _batch_embed_multimodal(
    inputs: list[list[dict]],
    client: voyageai.Client,
    model: str,
) -> list[list[float]]:
    """
    Executes one Voyage multimodal embedding API call for a batch of *inputs*.

    Each element of *inputs* is a list of content blocks (text + optional
    image_bytes dicts) for one chunk. 
    Args:
        inputs: ``list[list[dict]]`` — one inner list per chunk.
        client: Initialised :class:`voyageai.Client`.
        model:  Voyage multimodal model name.

    Returns:
        ``list[list[float]]`` — one 1024-dim vector per input.
    """
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            result = client.multimodal_embed(inputs, model=model)
            return result.embeddings
        except Exception as exc:
            exc_name = type(exc).__name__
            if "RateLimit" in exc_name or "rate_limit" in str(exc).lower():
                sleep_s = _RETRY_BASE_SLEEP * (2 ** attempt)
                logger.warning(
                    "_batch_embed_multimodal: rate-limited (attempt %d/%d), sleeping %.1fs.",
                    attempt + 1, _MAX_RETRIES, sleep_s,
                )
                time.sleep(sleep_s)
                last_exc = exc
            else:
                raise
    raise RuntimeError(
        f"_batch_embed_multimodal: failed after {_MAX_RETRIES} retries."
    ) from last_exc
