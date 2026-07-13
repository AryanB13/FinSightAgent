"""
table_extractor.py — pdfplumber-based table extraction and cleaning.

pdfplumber uses the PDF's vector-graphic line/rectangle geometry to infer
cell boundaries — far more accurate than text-position heuristics for the
dense financial statement tables found in 10-K filings.
"""

from __future__ import annotations

import re
from typing import Optional

import pdfplumber

from ingestion.config import MIN_TABLE_COLS, MIN_TABLE_ROWS, TABLE_MAX_ROWS_PER_CHUNK
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import CleanTable, RawTable

logger = get_logger(__name__)


# ── Table extraction ──────────────────────────────────────────────────────────

def extract_tables_from_pdf(pdf_path: str) -> list[RawTable]:
    """
    Opens *pdf_path* with ``pdfplumber.open()`` and calls
    ``page.extract_tables()`` on every page.

    For each detected table:
    - Records raw rows, page_number (1-indexed), and bounding box.
    - Sets ``header_row`` to the first row if *all* cells in that row are
      non-numeric (i.e. it looks like a header, not data).
    - Filters tables with fewer than ``MIN_TABLE_ROWS`` rows or fewer than
      ``MIN_TABLE_COLS`` columns.

    Returns:
        ``list[RawTable]`` ordered by ``(page_number, bbox y0)``.
    """
    raw_tables: list[RawTable] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            try:
                tables = page.extract_tables()
            except Exception as exc:
                logger.warning("Page %d: extract_tables failed — %s. Skipping.", page_number, exc)
                continue

            if not tables:
                continue

            # Get per-table bboxes if pdfplumber supports it
            try:
                table_objs = page.find_tables()
                bboxes = [t.bbox for t in table_objs]
            except Exception:
                bboxes = [None] * len(tables)

            for i, rows in enumerate(tables):
                if not rows:
                    continue

                # Filter by minimum size (before None replacement)
                n_rows = len(rows)
                n_cols = max((len(r) for r in rows), default=0)
                if n_rows < MIN_TABLE_ROWS or n_cols < MIN_TABLE_COLS:
                    continue

                # Determine header_row: first row where every non-None cell is non-numeric
                first_row: list[Optional[str]] = rows[0]
                header_row: Optional[list[str]] = None
                if _row_is_header(first_row):
                    header_row = [str(c) if c is not None else "" for c in first_row]

                bbox_raw = bboxes[i] if i < len(bboxes) and bboxes[i] is not None else (0.0, 0.0, 0.0, 0.0)
                bbox: tuple[float, float, float, float] = (
                    float(bbox_raw[0]), float(bbox_raw[1]),
                    float(bbox_raw[2]), float(bbox_raw[3]),
                )

                raw_tables.append(
                    RawTable(
                        rows=rows,
                        page_number=page_number,
                        bbox=bbox,
                        header_row=header_row,
                    )
                )

    raw_tables.sort(key=lambda t: (t.page_number, t.bbox[1]))
    logger.debug("Extracted %d raw tables from '%s'.", len(raw_tables), pdf_path)
    return raw_tables


def _row_is_header(row: list[Optional[str]]) -> bool:
    """Returns True if every non-None, non-empty cell in *row* is non-numeric."""
    cells = [c for c in row if c is not None and str(c).strip()]
    if not cells:
        return False
    return all(not _is_numeric_cell(str(c)) for c in cells)


def _is_numeric_cell(value: str) -> bool:
    """Returns True if *value* looks like a number (possibly with $, %, commas, parens)."""
    cleaned = value.strip().lstrip("$").replace(",", "").replace("%", "")
    # Handle parenthesised negatives like "(1,234)"
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = cleaned[1:-1]
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


# ── Table cleaning ─────────────────────────────────────────────────────────────

def clean_table(raw: RawTable) -> CleanTable:
    """
    Transforms a :class:`~ingestion.utils.schema.RawTable` into a
    :class:`~ingestion.utils.schema.CleanTable`:

    1. Replace ``None`` cells with ``""``.
    2. Strip whitespace from every cell.
    3. Normalise numerics — remove ``$``, ``,``, convert ``(1,234)`` → ``-1234``.
    4. Remove all-empty rows and duplicate header rows.
    5. Build a markdown ``text_repr`` pipe table.

    Raises:
        ValueError: If zero data rows remain after cleaning (caller skips it).
    """
    cleaned_rows: list[list[str]] = []

    for row in raw.rows:
        cleaned = [_clean_cell(c) for c in row]
        # Skip entirely-empty rows
        if not any(c for c in cleaned):
            continue
        cleaned_rows.append(cleaned)

    if not cleaned_rows:
        raise ValueError("Table is empty after cleaning.")

    # Determine header_row from cleaned data
    header_row: list[str]
    data_rows: list[list[str]]

    if raw.header_row is not None:
        # Re-clean the header (it came from raw.rows[0])
        header_row = [_clean_cell(c) for c in raw.header_row]
        # Drop the first cleaned row if it duplicates the header
        if cleaned_rows and cleaned_rows[0] == header_row:
            data_rows = cleaned_rows[1:]
        else:
            data_rows = cleaned_rows
    else:
        # Use first cleaned row as header
        header_row = cleaned_rows[0]
        data_rows = cleaned_rows[1:]

    if not data_rows:
        raise ValueError("Table has no data rows after cleaning.")

    all_rows = [header_row] + data_rows
    text_repr = _build_markdown_table(all_rows)

    return CleanTable(
        rows=data_rows,
        page_number=raw.page_number,
        bbox=raw.bbox,
        header_row=header_row,
        text_repr=text_repr,
        row_count=len(data_rows),
        col_count=max(len(r) for r in all_rows),
    )


def _clean_cell(value: Optional[str]) -> str:
    """Strips whitespace; normalises numeric formatting."""
    if value is None:
        return ""
    text = str(value).strip()
    # Normalise parenthesised negatives: (1,234) → -1234
    paren_match = re.fullmatch(r"\(([0-9,]+)\)", text)
    if paren_match:
        return "-" + paren_match.group(1).replace(",", "")
    # Remove currency and thousand-separator characters for pure numeric cells
    # but preserve the original text for descriptive cells
    stripped = text.lstrip("$").replace(",", "")
    try:
        float(stripped)
        return stripped  # keep normalised numeric form
    except ValueError:
        return text  # keep original descriptive text


def _build_markdown_table(rows: list[list[str]]) -> str:
    """
    Builds a GitHub-Flavored Markdown pipe table from *rows*.
    The first row is treated as the header.
    """
    if not rows:
        return ""

    max_cols = max(len(r) for r in rows)

    def _pad(row: list[str]) -> list[str]:
        return row + [""] * (max_cols - len(row))

    lines: list[str] = []
    header = _pad(rows[0])
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")

    for row in rows[1:]:
        lines.append("| " + " | ".join(_pad(row)) + " |")

    return "\n".join(lines)


# ── Table summary ──────────────────────────────────────────────────────────────

_INCOME_KEYWORDS   = {"revenue", "net income", "eps", "earnings per share", "sales", "gross profit", "operating income"}
_BALANCE_KEYWORDS  = {"assets", "liabilities", "equity", "stockholders", "shareholders", "debt", "goodwill"}
_CASHFLOW_KEYWORDS = {"operating activities", "investing activities", "financing activities", "cash", "capital expenditures", "capex"}


def generate_table_summary(table: CleanTable, metadata: dict) -> str:
    """
    Produces a one-sentence natural-language label for a table chunk,
    used as the BM25-searchable prefix prepended to ``text_repr``.

    Section inference is based on keywords in the header row:

    - ``"Revenue"`` / ``"Net income"`` / ``"EPS"``  → ``"Income Statement"``
    - ``"Assets"`` / ``"Liabilities"`` / ``"Equity"`` → ``"Balance Sheet"``
    - ``"Operating activities"`` / ``"Cash"``         → ``"Cash Flow Statement"``
    - No keyword match                                 → ``"Financial Table"``

    Output format::

        "10-K Income Statement excerpt — Apple, FY2023, page 41"

    Args:
        table:    The :class:`~ingestion.utils.schema.CleanTable` to summarise.
        metadata: Dict with keys ``company``, ``fiscal_year``, ``filing_type``.
    """
    header_text = " ".join(table.header_row).lower()

    if any(kw in header_text for kw in _INCOME_KEYWORDS):
        section_guess = "Income Statement"
    elif any(kw in header_text for kw in _BALANCE_KEYWORDS):
        section_guess = "Balance Sheet"
    elif any(kw in header_text for kw in _CASHFLOW_KEYWORDS):
        section_guess = "Cash Flow Statement"
    else:
        # Fall back: scan first-column data cells for keywords
        first_col = " ".join(row[0] for row in table.rows if row).lower()
        if any(kw in first_col for kw in _INCOME_KEYWORDS):
            section_guess = "Income Statement"
        elif any(kw in first_col for kw in _BALANCE_KEYWORDS):
            section_guess = "Balance Sheet"
        elif any(kw in first_col for kw in _CASHFLOW_KEYWORDS):
            section_guess = "Cash Flow Statement"
        else:
            section_guess = "Financial Table"

    company     = metadata.get("company", "Unknown Company")
    fiscal_year = metadata.get("fiscal_year", "XXXX")
    filing_type = metadata.get("filing_type", "10-K")
    page        = table.page_number

    return f"{filing_type} {section_guess} excerpt — {company}, FY{fiscal_year}, page {page}"


# ── Large table splitting ──────────────────────────────────────────────────────

def split_large_table(table: CleanTable, max_rows: int = TABLE_MAX_ROWS_PER_CHUNK) -> list[CleanTable]:
    """
    Splits a :class:`~ingestion.utils.schema.CleanTable` with more than
    *max_rows* data rows into multiple ``CleanTable`` objects, each sharing
    the same ``header_row``.

    Returns ``[table]`` unchanged if ``table.row_count <= max_rows``.

    Note:
        The ``-partN`` suffix on ``chunk_id`` is applied by ``table_chunker``
        when it iterates the returned list — not here.
    """
    if table.row_count <= max_rows:
        return [table]

    parts: list[CleanTable] = []
    data_rows = table.rows
    i = 0
    part_num = 0

    while i < len(data_rows):
        chunk_rows = data_rows[i : i + max_rows]
        all_rows_for_repr = [table.header_row] + chunk_rows
        text_repr = _build_markdown_table(all_rows_for_repr)

        parts.append(
            CleanTable(
                rows=chunk_rows,
                page_number=table.page_number,
                bbox=table.bbox,
                header_row=table.header_row,
                text_repr=text_repr,
                row_count=len(chunk_rows),
                col_count=table.col_count,
            )
        )
        i += max_rows
        part_num += 1

    logger.debug(
        "Split large table (page %d, %d rows) into %d parts.",
        table.page_number,
        table.row_count,
        len(parts),
    )
    return parts
