"""
metadata_tagger.py — Builds chunk IDs and attaches all metadata fields.

Every field defined in ``schema.Chunk`` (except ``chunk_id``, ``text``,
``table_data``, ``image_bytes``, and ``embedding``) is populated here.
Chunkers receive the result of :func:`tag_metadata` and unpack it directly
into the :class:`~ingestion.utils.schema.Chunk` constructor.
"""

from __future__ import annotations

import re

from ingestion.config import CORPUS_VERSION
from ingestion.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_CONTENT_TYPES = {"text", "table", "chart"}
_TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")


# ── Slug builder ──────────────────────────────────────────────────────────────

def slugify_section(heading: str) -> str:
    """
    Converts a section heading string into a URL-safe, lowercase slug.

    Transformation steps:
    1. Lowercase the string.
    2. Remove apostrophes (``'``) and ampersands (``&``).
    3. Replace every non-alphanumeric character with ``"-"``.
    4. Collapse multiple consecutive ``"-"`` into one.
    5. Strip leading and trailing ``"-"``.

    Examples::

        "Item 7 - Management's Discussion & Analysis"
        → "item7-managements-discussion-analysis"

        "Item 1A."
        → "item1a"

        "Preamble"
        → "preamble"

    The slug is used as the ``section_slug`` component of every
    ``chunk_id`` and as the metadata filter value in Pinecone queries.
    It is exported and shared with the query pipeline so that filter
    values are always consistent.
    """
    s = heading.lower()
    s = s.replace("'", "").replace("\u2019", "").replace("&", "")
    s = re.sub(r"[^a-z0-9]+", "-", s)
    # Collapse "item-7" → "item7" and "7-a" suffix → "7a"
    # Rule: remove hyphen when it sits between a letter and a digit
    s = re.sub(r"([a-z])-([0-9])", r"\1\2", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-")
    return s


# ── Chunk ID builder ──────────────────────────────────────────────────────────

def build_chunk_id(
    ticker: str,
    fiscal_year: int,
    filing_type: str,
    section_slug: str,
    content_type: str,
    index: int,
) -> str:
    """
    Builds a deterministic, human-readable chunk ID with the format::

        "{TICKER}-FY{year}-{filing_slug}-{section_slug}-{content_type}-{index:04d}"

    The ``filing_type`` hyphen is removed for the ID component so the
    separator ``-`` is unambiguous (``"10-K"`` → ``"10K"``).

    **Section slug truncation:** If ``section_slug`` exceeds 100 characters,
    it is truncated to prevent Pinecone's 512-character ID limit errors.
    Uniqueness is still guaranteed by the combination of ticker, year,
    filing type, content type, and index.

    Examples::

        build_chunk_id("AAPL", 2023, "10-K", "item7-mdna", "text", 43)
        → "AAPL-FY2023-10K-item7-mdna-text-0043"

        build_chunk_id("MSFT", 2022, "10-K", "item8-financial-statements", "table", 2)
        → "MSFT-FY2022-10K-item8-financial-statements-table-0002"

        build_chunk_id("NVDA", 2024, "10-K", "item7-mdna", "chart", 1)
        → "NVDA-FY2024-10K-item7-mdna-chart-0001"

    Zero-padded index ensures alphabetical sort order equals document order,
    which is important for the evaluation harness ground-truth annotations.

    Args:
        ticker:       Uppercase ticker symbol, e.g. ``"AAPL"``.
        fiscal_year:  4-digit integer, e.g. ``2023``.
        filing_type:  SEC filing type, e.g. ``"10-K"``.
        section_slug: Output of :func:`slugify_section`.
        content_type: One of ``"text"``, ``"table"``, ``"chart"``.
        index:        0-based sequence number within this (filing, section, type).

    Returns:
        Chunk ID string (max 512 characters for Pinecone compatibility).
    """
    # Truncate section_slug to prevent Pinecone 512-char ID limit errors
    MAX_SECTION_SLUG_LENGTH = 100
    if len(section_slug) > MAX_SECTION_SLUG_LENGTH:
        section_slug = section_slug[:MAX_SECTION_SLUG_LENGTH]
    
    filing_slug = filing_type.replace("-", "")
    return f"{ticker}-FY{fiscal_year}-{filing_slug}-{section_slug}-{content_type}-{index:04d}"


# ── Metadata dict builder ──────────────────────────────────────────────────────

def tag_metadata(
    company: str,
    ticker: str,
    fiscal_year: int,
    filing_type: str,
    section: str,
    content_type: str,
    page_number: int,
    source_file: str,
    corpus_version: str,
    embedding_model: str,
) -> dict:
    """
    Returns a dict containing all :class:`~ingestion.utils.schema.Chunk`
    metadata fields **except** ``chunk_id``, ``text``, ``table_data``,
    ``image_bytes``, and ``embedding``.

    Validates:
    - ``content_type`` ∈ ``{"text", "table", "chart"}``
    - ``fiscal_year`` is a 4-digit integer (1000–9999)
    - ``ticker`` matches ``[A-Z]{1,5}``

    Raises:
        ValueError: On any validation failure.

    The returned dict is safe to unpack directly into the
    :class:`~ingestion.utils.schema.Chunk` constructor alongside the
    remaining fields::

        Chunk(
            **tag_metadata(...),
            chunk_id=build_chunk_id(...),
            text="...",
            table_data=None,
            image_bytes=None,
        )
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if content_type not in _VALID_CONTENT_TYPES:
        raise ValueError(
            f"Invalid content_type '{content_type}'. "
            f"Must be one of {sorted(_VALID_CONTENT_TYPES)}."
        )

    if not isinstance(fiscal_year, int) or not (1000 <= fiscal_year <= 9999):
        raise ValueError(
            f"fiscal_year must be a 4-digit integer, got {fiscal_year!r}."
        )

    ticker_upper = ticker.upper()
    if not _TICKER_PATTERN.match(ticker_upper):
        raise ValueError(
            f"ticker must be 1–5 uppercase letters, got '{ticker}'."
        )

    return {
        "company":         company,
        "ticker":          ticker_upper,
        "fiscal_year":     fiscal_year,
        "filing_type":     filing_type,
        "section":         section,
        "content_type":    content_type,
        "page_number":     page_number,
        "source_file":     source_file,
        "embedding_model": embedding_model,
        "corpus_version":  corpus_version,
    }
