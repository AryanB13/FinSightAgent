"""
text_cleaner.py — Cleans raw text extracted by PyMuPDF before chunking.

10-K PDFs contain hyphenated line-breaks, running page headers, repeated
legal boilerplate, and inconsistent whitespace.  None of those artefacts
should appear in an indexed chunk.  Each sub-cleaner handles one class of
noise and can be tested independently.
"""

from __future__ import annotations

import re

from ingestion.utils.logger import get_logger

logger = get_logger(__name__)

# ── Compiled patterns (module-level for performance) ──────────────────────────

# Running header/footer patterns — whole-line matches
_HEADER_FOOTER_PATTERNS: list[re.Pattern] = [
    # "APPLE INC. | 2023 FORM 10-K" and variants
    re.compile(
        r"^[A-Z][A-Z\s\.,&]+\|\s*\d{4}\s+FORM\s+10-K\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # "Table of Contents" standalone line
    re.compile(r"^\s*TABLE\s+OF\s+CONTENTS\s*$", re.MULTILINE | re.IGNORECASE),
    # Standalone page numbers: "42" or "Page 42" or "Page 42 of 100"
    re.compile(r"^\s*(?:Page\s+)?\d+(?:\s+of\s+\d+)?\s*$", re.MULTILINE | re.IGNORECASE),
    # "See accompanying notes to consolidated financial statements."
    re.compile(
        r"^\s*See accompanying notes to (?:consolidated |condensed )?financial statements\.?\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
    # "The accompanying notes are an integral part of these financial statements."
    re.compile(
        r"^\s*The accompanying notes are an integral part of (?:these|the) (?:condensed )?(?:consolidated )?financial statements\.?\s*$",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Soft-hyphen line-break: "manage-\nment" → "management"
# Keep uppercase-after-hyphen ("COVID-\n19" → "COVID-19")
_DEHYPHENATE_PATTERN = re.compile(r"([a-z])-\n([a-z])")

# Legal boilerplate patterns — multi-line paragraph removal
_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Forward-looking statements (paragraph starting with this phrase)
    re.compile(
        r"(?:This (?:Annual )?Report|These statements?|The following)?\s*"
        r"(?:contains?|includes?|may contain)?\s*"
        r"forward[- ]looking statements?[^.]*\.[^\n]*(?:\n(?!\n).+)*",
        re.IGNORECASE,
    ),
    # Safe harbor — "Private Securities Litigation Reform Act" paragraph
    re.compile(
        r"(?:pursuant to|under|within the meaning of)\s+(?:the\s+)?"
        r"Private Securities Litigation Reform Act[^.]*\.[^\n]*(?:\n(?!\n).+)*",
        re.IGNORECASE,
    ),
    # "Unless otherwise indicated, all references to..."
    re.compile(
        r"Unless otherwise (?:indicated|noted|stated)[^.]*\.[^\n]*(?:\n(?!\n).+)*",
        re.IGNORECASE,
    ),
]

_MIN_CLEAN_LENGTH = 20  # chunks shorter than this after cleaning are discarded


# ── Master cleaner ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Master cleaner — applies sub-cleaners in order:

    1. :func:`remove_headers_footers`
    2. :func:`dehyphenate`
    3. :func:`strip_legal_boilerplate`
    4. :func:`normalize_whitespace`

    Returns ``""`` if the result is shorter than 20 characters so that
    callers can simply skip the chunk without further checks.
    """
    text = remove_headers_footers(text)
    text = dehyphenate(text)
    text = strip_legal_boilerplate(text)
    text = normalize_whitespace(text)
    return text if len(text) >= _MIN_CLEAN_LENGTH else ""


# ── Sub-cleaners ──────────────────────────────────────────────────────────────

def remove_headers_footers(text: str) -> str:
    """
    Removes SEC 10-K running header and footer lines:

    - ``"APPLE INC. | 2023 FORM 10-K"`` style lines
    - ``"Table of Contents"`` standalone lines
    - Standalone page numbers or ``"Page N of M"``
    - ``"See accompanying notes to consolidated financial statements."``
    - ``"The accompanying notes are an integral part of..."``

    Each matching *entire line* is removed (not replaced with a space).
    Non-matching lines are preserved verbatim.
    """
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        if any(pat.fullmatch(line.strip()) or pat.match(line) for pat in _HEADER_FOOTER_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept)


def dehyphenate(text: str) -> str:
    """
    Rejoins words split across a line-break by a soft hyphen.

    Pattern: ``word-\\nword`` where the post-newline character is **lowercase**
    → the hyphen and newline are removed (``"man-\\nagement"`` → ``"management"``).

    Compound words where the character after the newline is **uppercase or a
    digit** are kept unchanged (``"COVID-\\n19"`` → ``"COVID-19"``).
    """
    return _DEHYPHENATE_PATTERN.sub(r"\1\2", text)


def strip_legal_boilerplate(text: str) -> str:
    """
    Removes standard legal disclaimer paragraphs repeated verbatim in most
    10-K filings:

    - Forward-looking statements disclaimers
    - Safe harbor paragraphs (identified by
      *"Private Securities Litigation Reform Act"*)
    - ``"Unless otherwise indicated, all references to..."`` boilerplate

    Each matched span is deleted entirely.  The number of removed blocks is
    logged at DEBUG level so they can be audited.
    """
    removed = 0
    for pat in _BOILERPLATE_PATTERNS:
        new_text, n = pat.subn("", text)
        if n:
            removed += n
            text = new_text
    if removed:
        logger.debug("strip_legal_boilerplate: removed %d boilerplate block(s).", removed)
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalises whitespace in *text*:

    1. Collapses multiple consecutive spaces (or tabs) into a single space.
    2. Collapses three or more consecutive newlines into exactly two
       (preserves paragraph breaks, removes excessive blank lines).
    3. Strips leading and trailing whitespace from the whole string.
    """
    # Collapse horizontal whitespace (spaces + tabs) — preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
