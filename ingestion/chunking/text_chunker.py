"""
text_chunker.py — Produces content_type="text" Chunk objects.

Respects SEC Item section boundaries (never splits a chunk across two Items),
then applies sliding-window chunking within each section for long sections
like Risk Factors or MD&A.
"""

from __future__ import annotations

import re
from typing import Optional

from ingestion.config import (
    APPROX_CHARS_PER_TOKEN,
    NARRATIVE_CHUNK_OVERLAP_TOKENS,
    NARRATIVE_CHUNK_SIZE_TOKENS,
    VOYAGE_TEXT_MODEL,
)
from ingestion.preprocessing.metadata_tagger import (
    build_chunk_id,
    slugify_section,
    tag_metadata,
)
from ingestion.preprocessing.text_cleaner import clean_text
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import Chunk, Section

logger = get_logger(__name__)

# ── Abbreviations whose trailing period must NOT trigger a sentence split ──────
# (used in split_sentences as a post-split repair step)
_ABBREV_SET = {
    "u.s", "u.k", "u.a.e", "e.u",
    "inc", "corp", "co", "ltd", "llc", "l.p",
    "no", "vol", "fig", "vs", "et al",
    "mr", "mrs", "ms", "dr", "prof",
    "approx", "est", "etc", "e.g", "i.e",
    "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "oct", "nov", "dec",
}

# Split on ". " / "! " / "? " followed by a capital letter or quote.
# Abbreviation false-positives are repaired in split_sentences() below.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")


# ── Public helpers ─────────────────────────────────────────────────────────────

def estimate_token_count(text: str) -> int:
    """
    Fast estimate: ``round(len(text) / APPROX_CHARS_PER_TOKEN)``.
    Accurate within ~10% for English financial prose.
    Used for buffer-size checks inside :func:`sliding_window_chunk`.
    """
    return round(len(text) / APPROX_CHARS_PER_TOKEN)


def split_sentences(text: str) -> list[str]:
    """
    Regex-based sentence splitter that handles common abbreviations
    containing periods (``"U.S."``, ``"Inc."``, ``"Corp."``, etc.).

    Strategy:
    1. Split on newlines first (paragraph boundaries are reliable splits).
    2. Within each paragraph, apply ``_SENTENCE_SPLIT`` (split on ``. /
       ! / ?`` + whitespace + capital letter).
    3. Repair false splits caused by abbreviations: if the last token
       of segment *i* (stripped of its trailing period) is in
       ``_ABBREV_SET``, merge segment *i* back with segment *i+1*.

    Returns a list of non-empty sentence strings.
    """
    sentences: list[str] = []
    for paragraph in text.split("\n"):
        para = paragraph.strip()
        if not para:
            continue

        raw_parts = _SENTENCE_SPLIT.split(para)

        # Repair abbreviation false-positives
        merged: list[str] = []
        for part in raw_parts:
            if not part.strip():
                continue
            if merged:
                # Check whether the previous segment ended with an abbreviation
                prev = merged[-1].rstrip()
                last_token = prev.split()[-1].rstrip(".").lower() if prev.split() else ""
                if last_token in _ABBREV_SET:
                    merged[-1] = prev + " " + part.lstrip()
                    continue
            merged.append(part)

        for s in merged:
            s = s.strip()
            if s:
                sentences.append(s)

    return sentences


def sliding_window_chunk(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Splits *text* into overlapping chunks of approximately *chunk_size_tokens*.

    Algorithm:
    1. :func:`split_sentences` → sentence list.
    2. Greedily accumulate sentences into a buffer until adding the next
       sentence would exceed *chunk_size_tokens*.
    3. On overflow: save buffer as a chunk, then seed the next buffer with
       the trailing sentences whose combined token count ≤ *overlap_tokens*.
    4. Save the final buffer as the last chunk.

    Edge cases:
    - A single sentence that already exceeds *chunk_size_tokens* is emitted
      as its own chunk (minimum chunk size = 1 sentence).
    - If *text* has no sentences after splitting, returns ``[]``.

    Args:
        text:              Cleaned text to chunk.
        chunk_size_tokens: Target max tokens per chunk.
        overlap_tokens:    How many tokens of the previous chunk to repeat
                           at the start of the next chunk.

    Returns:
        ``list[str]`` of chunk texts.  May be empty.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    buffer: list[str] = []
    buffer_tokens: int = 0

    for sentence in sentences:
        sent_tokens = estimate_token_count(sentence)

        if buffer_tokens + sent_tokens > chunk_size_tokens and buffer:
            # Save current buffer as a chunk
            chunks.append(" ".join(buffer))

            # Build overlap seed from the tail of the buffer
            overlap_buffer: list[str] = []
            overlap_count: int = 0
            for s in reversed(buffer):
                s_tok = estimate_token_count(s)
                if overlap_count + s_tok > overlap_tokens:
                    break
                overlap_buffer.insert(0, s)
                overlap_count += s_tok

            buffer = overlap_buffer
            buffer_tokens = overlap_count

        buffer.append(sentence)
        buffer_tokens += sent_tokens

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks


# ── Main chunking function ─────────────────────────────────────────────────────

def chunk_narrative_sections(
    sections: list[Section],
    metadata_base: dict,
    chunk_size_tokens: int = NARRATIVE_CHUNK_SIZE_TOKENS,
    overlap_tokens: int = NARRATIVE_CHUNK_OVERLAP_TOKENS,
) -> list[Chunk]:
    """
    Top-level function — converts all :class:`~ingestion.utils.schema.Section`
    objects into a flat list of ``content_type="text"``
    :class:`~ingestion.utils.schema.Chunk` objects.

    For each section:
    1. Concatenates all ``TextBlock.text`` values with newline separator.
    2. Applies :func:`~ingestion.preprocessing.text_cleaner.clean_text`.
    3. Calls :func:`sliding_window_chunk` to produce text fragments.
    4. Wraps each fragment in a :class:`~ingestion.utils.schema.Chunk` with
       a deterministic ``chunk_id`` from
       :func:`~ingestion.preprocessing.metadata_tagger.build_chunk_id` and
       full metadata from
       :func:`~ingestion.preprocessing.metadata_tagger.tag_metadata`.

    Sections whose text is empty after ``clean_text()`` (e.g. *"Item 4 —
    Mine Safety: None."*) are silently skipped.

    Args:
        sections:          List of :class:`~ingestion.utils.schema.Section`
                           objects (after ``merge_short_sections``).
        metadata_base:     Dict with keys ``company``, ``ticker``,
                           ``fiscal_year``, ``filing_type``, ``source_file``,
                           ``corpus_version``.  All other metadata is
                           inferred per section/chunk.
        chunk_size_tokens: Target token count per chunk (default 512).
        overlap_tokens:    Overlap in tokens between adjacent chunks (default 64).

    Returns:
        Flat ``list[Chunk]`` across all sections, ordered by document position.
    """
    chunks: list[Chunk] = []

    ticker       = metadata_base["ticker"]
    fiscal_year  = metadata_base["fiscal_year"]
    filing_type  = metadata_base["filing_type"]
    source_file  = metadata_base["source_file"]
    corpus_version = metadata_base["corpus_version"]
    company      = metadata_base["company"]

    # Global chunk index across all text sections (for unique IDs)
    chunk_index = 0

    for section in sections:
        raw_text = "\n".join(tb.text for tb in section.text_blocks)
        cleaned  = clean_text(raw_text)

        if not cleaned:
            logger.debug("Section '%s': empty after clean_text — skipping.", section.heading)
            continue

        fragments = sliding_window_chunk(cleaned, chunk_size_tokens, overlap_tokens)

        for frag in fragments:
            if not frag.strip():
                continue

            section_slug = slugify_section(section.full_heading)
            meta = tag_metadata(
                company=company,
                ticker=ticker,
                fiscal_year=fiscal_year,
                filing_type=filing_type,
                section=section.full_heading,
                content_type="text",
                page_number=section.page_start,
                source_file=source_file,
                corpus_version=corpus_version,
                embedding_model=VOYAGE_TEXT_MODEL,
            )
            cid = build_chunk_id(ticker, fiscal_year, filing_type, section_slug, "text", chunk_index)

            chunks.append(
                Chunk(
                    **meta,
                    chunk_id=cid,
                    text=frag,
                    table_data=None,
                    image_bytes=None,
                )
            )
            chunk_index += 1

    logger.info(
        "chunk_narrative_sections: %d text chunks from %d sections.",
        len(chunks), len(sections),
    )
    return chunks
