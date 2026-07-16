"""
bm25_indexer.py — Build, persist, load, and query a BM25 index.

Wraps rank_bm25.BM25Okapi.  Kept fully in-process (no Elasticsearch) as
the corpus is 2 000–5 000 chunks — well within BM25Okapi's in-RAM sweet
spot.  Both index and chunk list are pickled together so the parallel-list
relationship index.get_scores(query)[i] ↔ chunks[i] is always preserved.

The same tokenize_chunk() function is used at index time AND query time,
which makes tokenization mismatch impossible.
"""

from __future__ import annotations

import pickle
import re
import time
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from ingestion.config import (
    BM25_B,
    BM25_CHUNKS_PATH,
    BM25_INDEX_PATH,
    BM25_K1,
)
from ingestion.utils.file_utils import ensure_dir
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import Chunk

logger = get_logger(__name__)

# ── Stop words ────────────────────────────────────────────────────────────────
# Financial-domain stop words: common English function words that carry no
# retrieval signal in an earnings/10-K corpus.  Financial acronyms like
# "eps", "ebitda", "cagr", "aapl", "nvda" are intentionally preserved.
_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "for",
    "is", "was", "are", "were", "has", "have", "been",
    "with", "that", "this", "its", "on", "at", "by", "as",
    "from", "our", "we", "their", "which", "will", "may",
    "can", "not", "such",
})

# Replace every character that is not alphanumeric or a hyphen with a space.
# This turns "R&D" → "r d", "10-K" → "10-k", "$383.3B" → " 383 3b".
_NONALNUM_PATTERN = re.compile(r"[^a-z0-9\-]")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def tokenize_chunk(text: str) -> list[str]:
    """
    Tokenizes *text* for BM25 indexing **and** query-time search.

    This function must be called identically at both index time and query
    time — using it in both places makes tokenization mismatch impossible.

    Steps:

    1. **Lowercase** the entire string.
    2. **Replace** every character that is not alphanumeric or a hyphen with
       a space (``"R&D"`` → ``"r d"``, ``"10-K"`` → ``"10-k"``,
       ``"EBITDA"`` → ``"ebitda"``).
    3. **Split** on whitespace.
    4. **Remove** tokens in ``_STOP_WORDS`` (common English function words).
    5. **Filter** tokens shorter than 2 characters.

    Financial acronyms — ``"aapl"``, ``"nvda"``, ``"eps"``, ``"cagr"``,
    ``"ebitda"`` — survive all steps and are the primary advantage of BM25
    over semantic search for exact ticker/metric queries.

    Args:
        text: Raw chunk text (may be the BM25-indexed description of a
              chart or markdown table — not just narrative prose).

    Returns:
        ``list[str]`` of tokens.  May be empty for very short or
        symbol-only inputs.
    """
    lowered   = text.lower()
    cleaned   = _NONALNUM_PATTERN.sub(" ", lowered)
    tokens    = cleaned.split()
    filtered  = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 2]
    return filtered


# ── Index builder ─────────────────────────────────────────────────────────────

def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    """
    Builds a :class:`~rank_bm25.BM25Okapi` index from the tokenized text
    of every chunk.

    The returned index and the input *chunks* list are **always used as a
    pair**: ``index.get_scores(query_tokens)[i]`` is the BM25 score for
    ``chunks[i]``.  This parallel-list relationship is preserved by
    :func:`save_index` / :func:`load_index` (both are pickled together).

    Logs: number of chunks indexed, vocabulary size, average document
    length in tokens.

    Args:
        chunks: Flat list of all :class:`~ingestion.utils.schema.Chunk`
                objects for one ingestion run (text + table + chart).

    Returns:
        :class:`~rank_bm25.BM25Okapi` instance ready for querying.
    """
    t0 = time.time()
    corpus: list[list[str]] = [tokenize_chunk(c.text) for c in chunks]

    index = BM25Okapi(corpus, k1=BM25_K1, b=BM25_B)

    # Diagnostics
    vocab_size  = len(set(tok for doc in corpus for tok in doc))
    avg_doc_len = sum(len(doc) for doc in corpus) / max(len(corpus), 1)
    elapsed     = time.time() - t0

    logger.info(
        "build_bm25_index: indexed %d chunks | vocab=%d | avg_doc_len=%.1f tokens | %.2fs",
        len(chunks), vocab_size, avg_doc_len, elapsed,
    )
    return index


# ── Persistence ───────────────────────────────────────────────────────────────

def save_index(
    index: BM25Okapi,
    chunks: list[Chunk],
    index_path: str = BM25_INDEX_PATH,
    chunks_path: str = BM25_CHUNKS_PATH,
) -> None:
    """
    Persists the BM25 index and chunk list to disk via :mod:`pickle`.

    Both files are always saved / loaded together to preserve the
    ``index ↔ chunk`` parallel-list alignment.

    - *index_path*  → pickled :class:`~rank_bm25.BM25Okapi` object.
    - *chunks_path* → pickled ``list[Chunk]``.

    Calls :func:`~ingestion.utils.file_utils.ensure_dir` before writing so
    the target directory is created if it does not exist.  Overwrites any
    existing files.  Logs the file sizes written.

    Args:
        index:       Built BM25 index (output of :func:`build_bm25_index`).
        chunks:      Parallel chunk list (same order as index corpus).
        index_path:  Destination path for the pickled index.
        chunks_path: Destination path for the pickled chunk list.
    """
    ensure_dir(str(index_path).rsplit("/", 1)[0])
    ensure_dir(str(chunks_path).rsplit("/", 1)[0])

    with open(index_path, "wb") as fh:
        pickle.dump(index, fh)
    idx_size = _file_kb(index_path)

    with open(chunks_path, "wb") as fh:
        pickle.dump(chunks, fh)
    chunks_size = _file_kb(chunks_path)

    logger.info(
        "save_index: wrote index (%.1f KB) → %s | chunks (%.1f KB) → %s",
        idx_size, index_path, chunks_size, chunks_path,
    )


def load_index(
    index_path: str = BM25_INDEX_PATH,
    chunks_path: str = BM25_CHUNKS_PATH,
) -> tuple[BM25Okapi, list[Chunk]]:
    """
    Loads a persisted BM25 index and chunk list from disk.

    Called once at query-pipeline startup; the returned objects are held
    in memory for the session duration.

    Args:
        index_path:  Path to the pickled :class:`~rank_bm25.BM25Okapi`.
        chunks_path: Path to the pickled ``list[Chunk]``.

    Returns:
        ``(BM25Okapi, list[Chunk])`` tuple.

    Raises:
        FileNotFoundError: If either *index_path* or *chunks_path* is
                           missing.
    """
    for path in (index_path, chunks_path):
        if not __import__("os").path.isfile(path):
            raise FileNotFoundError(
                f"BM25 index file not found: '{path}'. "
                "Run the ingestion pipeline first."
            )

    with open(index_path, "rb") as fh:
        index: BM25Okapi = pickle.load(fh)

    with open(chunks_path, "rb") as fh:
        chunks: list[Chunk] = pickle.load(fh)

    logger.info(
        "load_index: loaded %d chunks from '%s'.",
        len(chunks), chunks_path,
    )
    return index, chunks


# ── Search ────────────────────────────────────────────────────────────────────

def search_bm25(
    query: str,
    index: BM25Okapi,
    chunks: list[Chunk],
    top_k: int = 20,
) -> list[tuple[Chunk, float]]:
    """
    Runs a BM25 query against a loaded index.

    Steps:

    1. :func:`tokenize_chunk` (query) — same tokenizer as index time.
    2. ``index.get_scores(query_tokens)`` → score array aligned with *chunks*.
    3. ``argsort`` descending → top-*k* indices.
    4. Return ``[(chunks[i], float(scores[i])), ...]`` for the top-*k*
       non-zero-score results.

    Returns ``[]`` if all scores are zero (i.e. no query token appears in
    the index vocabulary), so callers can detect a zero-recall query
    without inspecting score arrays.

    Args:
        query:  Raw query string (e.g. ``"Apple total revenue FY2023"``).
        index:  Loaded :class:`~rank_bm25.BM25Okapi`.
        chunks: Parallel chunk list (same order as the index corpus).
        top_k:  Maximum number of results to return.

    Returns:
        ``list[tuple[Chunk, float]]`` sorted by descending BM25 score.
        Length ≤ *top_k*.
    """
    query_tokens = tokenize_chunk(query)
    if not query_tokens:
        logger.debug("search_bm25: query produced no tokens after tokenization.")
        return []

    scores: np.ndarray = index.get_scores(query_tokens)

    if scores.max() == 0.0:
        logger.debug("search_bm25: all scores zero for query %r.", query)
        return []

    # argsort descending, slice top_k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[tuple[Chunk, float]] = [
        (chunks[i], float(scores[i]))
        for i in top_indices
        if scores[i] > 0.0
    ]
    return results


# ── Internal helpers ──────────────────────────────────────────────────────────

def _file_kb(path: str) -> float:
    """Returns file size in kilobytes."""
    return __import__("os").path.getsize(path) / 1024
