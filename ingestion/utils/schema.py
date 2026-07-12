"""
schema.py — Shared dataclasses for the ingestion pipeline.

One Chunk = one Pinecone vector = one BM25 document.
All pipeline stages produce, consume, or transform these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TextBlock:
    """
    Raw output from PyMuPDF before cleaning or chunking.
    Represents one contiguous block of text on a page with layout metadata.
    """

    text: str
    page_number: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1) in PDF points
    font_size: float                          # largest font size across spans
    font_name: str                            # dominant font name (heading detection)
    block_type: int                           # 0=text, 1=image (fitz block type codes)


@dataclass
class RawTable:
    """
    Output from pdfplumber before cleaning.
    rows is row-major; cells may be None for merged/empty cells.
    """

    rows: list[list[Optional[str]]]
    page_number: int
    bbox: tuple[float, float, float, float]
    header_row: Optional[list[str]]           # first row if it looks non-numeric


@dataclass
class CleanTable:
    """
    Post-cleaning table ready for chunking.
    text_repr is the markdown pipe-table representation used for BM25 + text prefix.
    """

    rows: list[list[str]]
    page_number: int
    bbox: tuple[float, float, float, float]
    header_row: list[str]
    text_repr: str       # "| Revenue | 2023 | 2022 |\n| 383,285 | 394,328 |"
    row_count: int
    col_count: int


@dataclass
class FigureImage:
    """
    A chart or figure extracted from the PDF as raw image bytes.
    caption is the nearest text block below (or above) the figure bounding box.
    """

    image_bytes: bytes
    image_ext: str                            # "png" | "jpeg"
    page_number: int
    bbox: tuple[float, float, float, float]
    caption: str                              # nearest text near figure
    pixel_count: int                          # width × height; checked against Voyage limits
    xref: int                                 # fitz internal reference number


@dataclass
class Section:
    """
    A logical section of a 10-K filing (e.g. "Item 7 - MD&A").
    Contains all TextBlocks assigned to it after section boundary detection.
    """

    heading: str                              # e.g. "Item 7"
    full_heading: str                         # e.g. "Item 7 - Management's Discussion"
    page_start: int
    page_end: int
    text_blocks: list[TextBlock] = field(default_factory=list)


@dataclass
class Chunk:
    """
    The atomic unit of the entire ingestion pipeline.
    One Chunk = one Pinecone vector = one BM25 document.
    """

    chunk_id: str             # e.g. "AAPL-FY2023-10K-item7-mdna-text-0043"
    company: str              # e.g. "Apple"
    ticker: str               # e.g. "AAPL"
    fiscal_year: int          # e.g. 2023
    filing_type: str          # "10-K"
    section: str              # e.g. "Item 7 - MD&A"
    content_type: str         # "text" | "table" | "chart"
    page_number: int
    text: str                 # always present; table text_repr or chart caption for non-text
    table_data: Optional[CleanTable]   # only when content_type == "table"
    image_bytes: Optional[bytes]       # only when content_type == "chart"
    source_file: str                   # original PDF filename
    embedding_model: str               # "voyage-3.5" | "voyage-multimodal-3"
    corpus_version: str                # from config.CORPUS_VERSION
    embedding: Optional[list[float]] = None   # filled by voyage_embedder.py
