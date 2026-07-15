"""
chart_chunker.py — Produces content_type="chart" Chunk objects.

chunk.text  = natural-language description for BM25 retrieval.
chunk.image_bytes = raw image passed to voyage-multimodal-3.

This is the only content type where the embedded modality (image) differs
from the text stored in chunk.text.
"""

from __future__ import annotations

from ingestion.config import VOYAGE_MULTIMODAL_MODEL
from ingestion.parsers.figure_extractor import resize_figure_if_needed
from ingestion.preprocessing.metadata_tagger import build_chunk_id, slugify_section, tag_metadata
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import Chunk, FigureImage

logger = get_logger(__name__)


# ── Text builder ──────────────────────────────────────────────────────────────

def build_chart_chunk_text(
    caption: str,
    section: str,
    company: str,
    fiscal_year: int,
    page_number: int,
) -> str:
    """
    Builds the BM25-indexed text description for a chart chunk.

    Format with caption::

        "Chart — {company}, FY{year}, {section}, page {page}.
        {caption}"

    Format without caption (empty or whitespace-only)::

        "Chart — {company}, FY{year}, {section}, page {page}.
        Financial chart or figure (no caption extracted)."

    The company/year/section prefix guarantees BM25 can retrieve the chart
    by company name, fiscal year, or section even when the caption is sparse.

    Args:
        caption:      Text near the image (may be ``""``).
        section:      Section full heading, e.g. ``"Item 7 - MD&A"``.
        company:      Company name, e.g. ``"Apple"``.
        fiscal_year:  4-digit integer, e.g. ``2023``.
        page_number:  1-indexed page where the figure appears.

    Returns:
        Non-empty string suitable for BM25 indexing.
    """
    prefix = f"Chart — {company}, FY{fiscal_year}, {section}, page {page_number}."
    body   = caption.strip() if caption and caption.strip() else \
             "Financial chart or figure (no caption extracted)."
    return f"{prefix}\n{body}"


# ── Per-section chunker ────────────────────────────────────────────────────────

def chunk_charts(
    figures_by_section: dict[str, list[FigureImage]],
    metadata_base: dict,
    section_name: str,
) -> list[Chunk]:
    """
    Converts all :class:`~ingestion.utils.schema.FigureImage` objects for
    *section_name* into ``content_type="chart"`` chunks.

    For each ``FigureImage``:

    1. :func:`build_chart_chunk_text` → ``chunk.text`` (BM25-indexed).
    2. :func:`~ingestion.parsers.figure_extractor.resize_figure_if_needed` →
       ensure image is within Voyage pixel billing window.
    3. :func:`~ingestion.preprocessing.metadata_tagger.build_chunk_id` →
       ``chunk.chunk_id``.
    4. ``chunk.image_bytes = resized_figure.image_bytes``.
    5. ``embedding_model = VOYAGE_MULTIMODAL_MODEL``.

    Args:
        figures_by_section: Mapping from section heading to list of FigureImage.
        metadata_base:      Dict with ``company``, ``ticker``, ``fiscal_year``,
                            ``filing_type``, ``source_file``, ``corpus_version``.
        section_name:       Key into *figures_by_section* to process.

    Returns:
        List of chart :class:`~ingestion.utils.schema.Chunk` objects.
        Empty list if *section_name* not in *figures_by_section*.
    """
    figures = figures_by_section.get(section_name, [])
    if not figures:
        return []

    ticker        = metadata_base["ticker"]
    fiscal_year   = metadata_base["fiscal_year"]
    filing_type   = metadata_base["filing_type"]
    source_file   = metadata_base["source_file"]
    corpus_version = metadata_base["corpus_version"]
    company       = metadata_base["company"]
    section_slug  = slugify_section(section_name)

    chunks: list[Chunk] = []

    for idx, figure in enumerate(figures):
        resized = resize_figure_if_needed(figure)

        text = build_chart_chunk_text(
            caption=resized.caption,
            section=section_name,
            company=company,
            fiscal_year=fiscal_year,
            page_number=resized.page_number,
        )

        meta = tag_metadata(
            company=company,
            ticker=ticker,
            fiscal_year=fiscal_year,
            filing_type=filing_type,
            section=section_name,
            content_type="chart",
            page_number=resized.page_number,
            source_file=source_file,
            corpus_version=corpus_version,
            embedding_model=VOYAGE_MULTIMODAL_MODEL,
        )
        cid = build_chunk_id(
            ticker, fiscal_year, filing_type, section_slug, "chart", idx
        )

        chunks.append(
            Chunk(
                **meta,
                chunk_id=cid,
                text=text,
                table_data=None,
                image_bytes=resized.image_bytes,
            )
        )

    logger.debug(
        "chunk_charts: section='%s' → %d chart chunks.",
        section_name, len(chunks),
    )
    return chunks


# ── All-sections orchestrator ─────────────────────────────────────────────────

def chunk_all_charts(
    figures_by_section: dict[str, list[FigureImage]],
    metadata_base: dict,
) -> list[Chunk]:
    """
    Calls :func:`chunk_charts` for every key in *figures_by_section* and
    returns the combined flat list of chart chunks.

    Also logs the total pixel count across all chart chunks for pre-flight
    Voyage API billing estimation.

    Args:
        figures_by_section: Full output of
            :func:`~ingestion.preprocessing.content_separator.assign_figures_to_sections`.
        metadata_base:      Shared filing metadata dict.

    Returns:
        Flat ``list[Chunk]`` of all chart chunks across all sections.
    """
    all_chunks: list[Chunk] = []

    for section_name in figures_by_section:
        all_chunks.extend(chunk_charts(figures_by_section, metadata_base, section_name))

    total_pixels = sum(
        f.pixel_count
        for figs in figures_by_section.values()
        for f in figs
    )
    logger.info(
        "chunk_all_charts: %d chart chunks, total pixel count = %d (%.1fM).",
        len(all_chunks),
        total_pixels,
        total_pixels / 1_000_000,
    )
    return all_chunks
