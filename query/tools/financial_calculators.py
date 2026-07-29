"""
query/tools/financial_calculators.py — Pure financial math functions.

All functions are pure (no I/O, no network, no file access) and are the only
functions permitted in the sandbox. Called via ``sandbox_executor.run_in_sandbox``
(never imported directly by the Tool-Use Agent at call time — that would bypass
the sandbox isolation).

``extract_numeric_from_table_chunk`` is the exception: it runs in-process (the
data has already been parsed and sanitised by the ingestion pipeline) and is
the bridge between retrieved table chunks and the numeric inputs these
calculator functions expect.
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ingestion.utils.schema import Chunk


# ── Financial math ────────────────────────────────────────────────────────────

def yoy_growth(current: float, previous: float) -> float:
    """
    Year-over-year growth rate.

    Formula: ``(current - previous) / previous``

    Args:
        current:  Metric value in the later period.
        previous: Metric value in the earlier period.

    Returns:
        Growth rate (e.g. ``0.12`` = 12% growth).

    Raises:
        ZeroDivisionError: If ``previous == 0``.
    """
    if previous == 0:
        raise ZeroDivisionError("yoy_growth: previous value is zero")
    return (current - previous) / previous


def cagr(start_value: float, end_value: float, periods: int) -> float:
    """
    Compound Annual Growth Rate.

    Formula: ``(end_value / start_value) ** (1 / periods) - 1``

    Args:
        start_value: Value at the start of the period.
        end_value:   Value at the end of the period.
        periods:     Number of periods (years). Must be > 0.

    Returns:
        CAGR as a decimal (e.g. ``0.15`` = 15% CAGR).

    Raises:
        ValueError:      If ``periods <= 0`` or ``start_value <= 0``.
        ZeroDivisionError: If ``start_value == 0``.
    """
    if periods <= 0:
        raise ValueError(f"cagr: periods must be > 0, got {periods}")
    if start_value == 0:
        raise ZeroDivisionError("cagr: start_value is zero")
    if start_value < 0 or end_value < 0:
        raise ValueError("cagr: start_value and end_value must be non-negative")
    return (end_value / start_value) ** (1.0 / periods) - 1.0


def percent_of(numerator: float, denominator: float) -> float:
    """
    Percent-of (ratio × 100).

    Used for R&D%, operating margin%, gross margin%, etc.

    Formula: ``numerator / denominator * 100``

    Args:
        numerator:   The part (e.g. R&D expense).
        denominator: The whole (e.g. total revenue).

    Returns:
        Percentage value (e.g. ``6.85`` = 6.85%).

    Raises:
        ZeroDivisionError: If ``denominator == 0``.
    """
    if denominator == 0:
        raise ZeroDivisionError("percent_of: denominator is zero")
    return numerator / denominator * 100.0


def operating_margin(operating_income: float, revenue: float) -> float:
    """
    Operating margin as a percentage.

    Formula: ``percent_of(operating_income, revenue)``

    Args:
        operating_income: Operating income (EBIT).
        revenue:          Total net revenue/sales.

    Returns:
        Operating margin percentage.

    Raises:
        ZeroDivisionError: If ``revenue == 0``.
    """
    return percent_of(operating_income, revenue)


def debt_to_equity(total_debt: float, total_equity: float) -> float:
    """
    Debt-to-equity ratio.

    Formula: ``total_debt / total_equity``

    Args:
        total_debt:   Total interest-bearing debt.
        total_equity: Total shareholders' equity.

    Returns:
        D/E ratio (e.g. ``1.5`` = 150% leverage).

    Raises:
        ZeroDivisionError: If ``total_equity == 0``.
    """
    if total_equity == 0:
        raise ZeroDivisionError("debt_to_equity: total_equity is zero")
    return total_debt / total_equity


# ── Table chunk numeric extraction ───────────────────────────────────────────

_NUMERIC_STRIP_RE = re.compile(r"[\$,\s]")
_NEGATIVE_PAREN_RE = re.compile(r"^\(([0-9,.]+)\)$")


def _parse_cell_value(cell: str) -> Optional[float]:
    """
    Normalises a table cell string to a float.

    Handles:
    - Dollar signs and commas: ``"$1,234"`` → ``1234.0``
    - Parenthetical negatives: ``"(1,234)"`` → ``-1234.0``
    - Leading minus: ``"-26251"`` → ``-26251.0``
    - Empty / dash / N/A → ``None``
    """
    if not cell:
        return None
    cell = cell.strip()
    if not cell or cell in ("-", "—", "N/A", "n/a", "*"):
        return None

    # Parenthetical negative: (1,234) → -1234
    m = _NEGATIVE_PAREN_RE.match(cell)
    if m:
        cleaned = _NUMERIC_STRIP_RE.sub("", m.group(1))
        try:
            return -float(cleaned)
        except ValueError:
            return None

    # Strip dollar signs, commas, and spaces then parse
    cleaned = _NUMERIC_STRIP_RE.sub("", cell)
    try:
        return float(cleaned)
    except ValueError:
        return None


def extract_numeric_from_table_chunk(
    chunk: "Chunk",
    row_label: str,
    column_label: str,
) -> Optional[float]:
    """
    Extracts a single numeric value from a table chunk by row + column label.

    Uses fuzzy case-insensitive **substring matching** for both labels so that
    ``row_label="R&D"`` matches the stored cell ``"Research and development expense"``
    and ``column_label="2022"`` matches ``"FY2022"`` or ``"2022"``.

    Applies the same numeric normalisation as the ingestion pipeline's
    ``table_extractor`` (handles ``"$1,234"``, ``"(1,234)"`` negative notation,
    and leading-minus negatives from balance sheets).

    Args:
        chunk:        A :class:`~ingestion.utils.schema.Chunk` with
                      ``content_type == "table"`` and ``table_data`` populated.
        row_label:    Substring to match against the first cell of each row
                      (case-insensitive).
        column_label: Substring to match against the header row cells
                      (case-insensitive). Pass ``""`` to use the second column.

    Returns:
        Extracted numeric value, or ``None`` if no matching row/column found.
        The caller (Tool-Use Agent) treats ``None`` as "insufficient data for
        computation" and logs a warning rather than guessing.
    """
    if chunk.table_data is None:
        return None

    header = chunk.table_data.header_row or []
    rows = chunk.table_data.rows or []

    if not rows:
        return None

    # Find target column index from header (0-indexed)
    col_idx: Optional[int] = None
    if column_label and header:
        col_lower = column_label.lower()
        for i, h in enumerate(header):
            if h and col_lower in h.lower():
                col_idx = i
                break

    # Fallback: use second column (index 1) if header match fails
    if col_idx is None:
        col_idx = 1

    # Find matching row by first-cell substring match
    row_lower = row_label.lower()
    for row in rows:
        if not row:
            continue
        first_cell = str(row[0]) if row[0] is not None else ""
        if row_lower in first_cell.lower():
            # Found matching row — extract the value at col_idx
            if col_idx < len(row):
                raw_val = str(row[col_idx]) if row[col_idx] is not None else ""
                value = _parse_cell_value(raw_val)
                if value is None and len(row) > 1:
                    # Try next cell if current is empty
                    for fallback_idx in range(1, len(row)):
                        value = _parse_cell_value(str(row[fallback_idx] or ""))
                        if value is not None:
                            break
                return value

    return None
