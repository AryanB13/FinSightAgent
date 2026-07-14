"""
content_separator.py — Bridges Phase 2 parser outputs to Phase 4 chunkers.

Assigns each CleanTable and FigureImage to the Section it belongs to by
spatial page position, merges near-empty sections to avoid wasted embedding
calls, and bundles everything into a single ``separated_content`` dict that
is the sole input to all three chunkers.
"""

from __future__ import annotations

from ingestion.config import APPROX_CHARS_PER_TOKEN
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import CleanTable, FigureImage, Section

logger = get_logger(__name__)


# ── Table assignment ──────────────────────────────────────────────────────────

def assign_tables_to_sections(
    tables: list[CleanTable],
    sections: list[Section],
) -> dict[str, list[CleanTable]]:
    """
    Maps each :class:`~ingestion.utils.schema.CleanTable` to the
    :class:`~ingestion.utils.schema.Section` whose page range contains
    ``table.page_number``.

    When a table's page falls inside multiple overlapping section ranges
    (rare in well-formed 10-Ks), the section whose last ``text_blocks``
    entry on or before the table's page has the largest ``y0`` value
    (i.e. nearest-above) wins.

    Tables that cannot be assigned to any section are filed under the
    ``"Unknown"`` key.

    Returns:
        ``dict[str, list[CleanTable]]`` keyed by ``section.heading``.
        Sections with no tables are omitted from the dict.
    """
    result: dict[str, list[CleanTable]] = {}

    for table in tables:
        best_section = _find_section_for_page(table.page_number, sections)
        key = best_section.heading if best_section is not None else "Unknown"
        result.setdefault(key, []).append(table)

    logger.debug(
        "assign_tables_to_sections: %d tables → %d sections (+Unknown=%d).",
        len(tables),
        sum(1 for k in result if k != "Unknown"),
        len(result.get("Unknown", [])),
    )
    return result


# ── Figure assignment ─────────────────────────────────────────────────────────

def assign_figures_to_sections(
    figures: list[FigureImage],
    sections: list[Section],
) -> dict[str, list[FigureImage]]:
    """
    Same spatial logic as :func:`assign_tables_to_sections` but for
    :class:`~ingestion.utils.schema.FigureImage` objects.

    Returns:
        ``dict[str, list[FigureImage]]`` keyed by ``section.heading``.
    """
    result: dict[str, list[FigureImage]] = {}

    for figure in figures:
        best_section = _find_section_for_page(figure.page_number, sections)
        key = best_section.heading if best_section is not None else "Unknown"
        result.setdefault(key, []).append(figure)

    logger.debug(
        "assign_figures_to_sections: %d figures → %d sections (+Unknown=%d).",
        len(figures),
        sum(1 for k in result if k != "Unknown"),
        len(result.get("Unknown", [])),
    )
    return result


def _find_section_for_page(page_number: int, sections: list[Section]) -> Section | None:
    """
    Returns the best-matching :class:`~ingestion.utils.schema.Section` for
    *page_number* using the following priority:

    1. Sections whose ``page_start <= page_number <= page_end``.
    2. Among candidates, prefer the one that starts latest (most specific).
    3. If no section covers the page, fall back to the section with the
       largest ``page_start`` that is still <= *page_number*.
    4. Returns ``None`` if no section can be matched at all.
    """
    # Primary: sections that contain the page
    candidates = [
        s for s in sections
        if s.page_start <= page_number <= s.page_end
    ]
    if candidates:
        # Latest-starting section wins (most specific coverage)
        return max(candidates, key=lambda s: s.page_start)

    # Fallback: nearest section that starts before the page
    before = [s for s in sections if s.page_start <= page_number]
    if before:
        return max(before, key=lambda s: s.page_start)

    return None


# ── Section merging ───────────────────────────────────────────────────────────

def merge_short_sections(
    sections: list[Section],
    min_tokens: int = 50,
) -> list[Section]:
    """
    Merges sections whose total text is shorter than *min_tokens* into the
    **preceding** section.

    This prevents near-empty chunks like *"Item 4 — Mine Safety: None."*
    from wasting an embedding call and polluting retrieval results.

    Rules:
    - Token count is estimated as ``total_chars / APPROX_CHARS_PER_TOKEN``.
    - The first section (usually ``"Preamble"``) is always kept as-is even
      if it is short (there is no preceding section to merge into).
    - Returns a **new** list; input is not modified.

    Args:
        sections:   Ordered list of :class:`~ingestion.utils.schema.Section`.
        min_tokens: Sections with fewer estimated tokens are merged.

    Returns:
        New ``list[Section]`` after merging.
    """
    if not sections:
        return []

    merged: list[Section] = [_copy_section(sections[0])]

    for section in sections[1:]:
        total_chars = sum(len(tb.text) for tb in section.text_blocks)
        estimated_tokens = total_chars / APPROX_CHARS_PER_TOKEN

        if estimated_tokens < min_tokens:
            # Merge into the last section in `merged`
            prev = merged[-1]
            merged[-1] = Section(
                heading=prev.heading,
                full_heading=prev.full_heading,
                page_start=prev.page_start,
                page_end=max(prev.page_end, section.page_end),
                text_blocks=prev.text_blocks + section.text_blocks,
            )
            logger.debug(
                "Merged short section '%s' (%d estimated tokens) into '%s'.",
                section.heading, int(estimated_tokens), prev.heading,
            )
        else:
            merged.append(_copy_section(section))

    logger.debug(
        "merge_short_sections: %d sections → %d after merging (min_tokens=%d).",
        len(sections), len(merged), min_tokens,
    )
    return merged


def _copy_section(s: Section) -> Section:
    """Returns a shallow copy of *s* (text_blocks list is a new list, blocks themselves shared)."""
    return Section(
        heading=s.heading,
        full_heading=s.full_heading,
        page_start=s.page_start,
        page_end=s.page_end,
        text_blocks=list(s.text_blocks),
    )


# ── Top-level builder ─────────────────────────────────────────────────────────

def build_separated_content(
    sections: list[Section],
    tables: list[CleanTable],
    figures: list[FigureImage],
) -> dict:
    """
    Orchestrates the full content-separation step and returns the canonical
    ``separated_content`` dict consumed by all Phase 4 chunkers::

        {
            "sections":           list[Section],           # after merge_short_sections
            "tables_by_section":  dict[str, list[CleanTable]],
            "figures_by_section": dict[str, list[FigureImage]],
        }

    Steps:
    1. :func:`merge_short_sections` — collapse near-empty sections.
    2. :func:`assign_tables_to_sections` — spatial page assignment.
    3. :func:`assign_figures_to_sections` — spatial page assignment.

    This dict is the **sole** input to all three chunkers in Phase 4.
    """
    merged_sections = merge_short_sections(sections)
    tables_by_section  = assign_tables_to_sections(tables, merged_sections)
    figures_by_section = assign_figures_to_sections(figures, merged_sections)

    total_tables  = sum(len(v) for v in tables_by_section.values())
    total_figures = sum(len(v) for v in figures_by_section.values())
    logger.info(
        "build_separated_content: %d sections, %d tables, %d figures assigned.",
        len(merged_sections), total_tables, total_figures,
    )

    return {
        "sections":           merged_sections,
        "tables_by_section":  tables_by_section,
        "figures_by_section": figures_by_section,
    }
