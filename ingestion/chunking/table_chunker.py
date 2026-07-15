"""
table_chunker.py — Produces content_type="table" Chunk objects.

One CleanTable → one Chunk (or more if split by split_large_table).
The text field = summary prefix + markdown table so both BM25 and
semantic search can find the chunk.  table_data stores the CleanTable
for the Tool-Use Agent to extract raw cell values at query time.
"""

from __future__ import annotations

from ingestion.config import TABLE_MAX_ROWS_PER_CHUNK, VOYAGE_MULTIMODAL_MODEL
from ingestion.parsers.table_extractor import generate_table_summary, split_large_table
from ingestion.preprocessing.metadata_tagger import build_chunk_id, slugify_section, tag_metadata
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import Chunk, CleanTable

logger = get_logger(__name__)


# ── Text builder ──────────────────────────────────────────────────────────────

def build_table_chunk_text(summary: str, table_text_repr: str) -> str:
    """
    Concatenates the natural-language summary prefix and the markdown
    pipe-table representation into a single string stored in ``chunk.text``.

    The summary prefix is what semantic search latches on to for conceptual
    questions; the table body is what BM25 matches for exact financial
    line-item queries.  Having both in one field means a single embedding
    covers both retrieval paths.

    Format::

        "{summary}\\n\\n{table_text_repr}"

    Example output::

        "10-K Income Statement excerpt — Apple, FY2023, page 41

        | | FY2023 | FY2022 | FY2021 |
        | --- | --- | --- | --- |
        | Net sales | 383285 | 394328 | 365817 |
        | Cost of sales | 214137 | 223546 | 212981 |"
    """
    return f"{summary}\n\n{table_text_repr}"


# ── Per-section chunker ────────────────────────────────────────────────────────

def chunk_tables(
    tables_by_section: dict[str, list[CleanTable]],
    metadata_base: dict,
    section_name: str,
) -> list[Chunk]:
    """
    Converts all :class:`~ingestion.utils.schema.CleanTable` objects for
    *section_name* into ``content_type="table"`` chunks.

    For each ``CleanTable`` (after optional splitting via
    :func:`~ingestion.parsers.table_extractor.split_large_table`):

    1. :func:`~ingestion.parsers.table_extractor.generate_table_summary` →
       natural-language summary prefix.
    2. :func:`build_table_chunk_text` → ``chunk.text``.
    3. :func:`~ingestion.preprocessing.metadata_tagger.build_chunk_id` →
       ``chunk.chunk_id``.
    4. ``chunk.table_data = CleanTable`` (raw cell data for Tool-Use Agent).
    5. ``embedding_model = VOYAGE_MULTIMODAL_MODEL`` — tables share the
       multimodal vector space with charts so both can be retrieved together.

    Args:
        tables_by_section: Mapping from section heading to list of CleanTable.
        metadata_base:     Dict with ``company``, ``ticker``, ``fiscal_year``,
                           ``filing_type``, ``source_file``, ``corpus_version``.
        section_name:      Key into *tables_by_section* to process.

    Returns:
        List of table :class:`~ingestion.utils.schema.Chunk` objects for this
        section.  Empty list if *section_name* not in *tables_by_section*.
    """
    tables = tables_by_section.get(section_name, [])
    if not tables:
        return []

    ticker        = metadata_base["ticker"]
    fiscal_year   = metadata_base["fiscal_year"]
    filing_type   = metadata_base["filing_type"]
    source_file   = metadata_base["source_file"]
    corpus_version = metadata_base["corpus_version"]
    company       = metadata_base["company"]
    section_slug  = slugify_section(section_name)

    chunks: list[Chunk] = []

    for table_idx, table in enumerate(tables):
        parts = split_large_table(table, max_rows=TABLE_MAX_ROWS_PER_CHUNK)

        for part_idx, part in enumerate(parts):
            summary  = generate_table_summary(part, metadata_base)
            text     = build_table_chunk_text(summary, part.text_repr)

            # Encode large-table part number in index to keep IDs unique
            global_idx = table_idx * 1000 + part_idx

            meta = tag_metadata(
                company=company,
                ticker=ticker,
                fiscal_year=fiscal_year,
                filing_type=filing_type,
                section=section_name,
                content_type="table",
                page_number=part.page_number,
                source_file=source_file,
                corpus_version=corpus_version,
                embedding_model=VOYAGE_MULTIMODAL_MODEL,
            )
            cid = build_chunk_id(
                ticker, fiscal_year, filing_type, section_slug, "table", global_idx
            )

            chunks.append(
                Chunk(
                    **meta,
                    chunk_id=cid,
                    text=text,
                    table_data=part,
                    image_bytes=None,
                )
            )

    logger.debug(
        "chunk_tables: section='%s' → %d table chunks from %d tables.",
        section_name, len(chunks), len(tables),
    )
    return chunks


# ── All-sections orchestrator ─────────────────────────────────────────────────

def chunk_all_tables(
    tables_by_section: dict[str, list[CleanTable]],
    metadata_base: dict,
) -> list[Chunk]:
    """
    Calls :func:`chunk_tables` for every key in *tables_by_section* and
    returns the combined flat list of table chunks.

    Args:
        tables_by_section: Full output of
            :func:`~ingestion.preprocessing.content_separator.assign_tables_to_sections`.
        metadata_base:     Shared filing metadata dict.

    Returns:
        Flat ``list[Chunk]`` of all table chunks across all sections.
    """
    all_chunks: list[Chunk] = []

    for section_name in tables_by_section:
        all_chunks.extend(chunk_tables(tables_by_section, metadata_base, section_name))

    logger.info(
        "chunk_all_tables: %d table chunks from %d sections.",
        len(all_chunks), len(tables_by_section),
    )
    return all_chunks
