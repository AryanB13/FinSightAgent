"""
pdf_parser.py — PyMuPDF-based PDF loader, text-block extractor, and section detector.

PyMuPDF (fitz) exposes per-span font metadata and bounding boxes needed for
section-boundary detection; pdfplumber does not provide this.
"""

from __future__ import annotations

import statistics
from typing import Optional

import fitz  # PyMuPDF

from ingestion.config import SEC_SECTION_HEADINGS
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import Section, TextBlock

logger = get_logger(__name__)


# ── PDF loading ────────────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> fitz.Document:
    """
    Opens the PDF at *pdf_path* with ``fitz.open()``.

    Raises:
        FileNotFoundError: If *pdf_path* does not exist on disk.
        fitz.FileDataError: If the file exists but is not a valid PDF.

    Note:
        Caller is responsible for calling ``doc.close()`` when done to free
        memory.  Use as a context manager where possible::

            with load_pdf(path) as doc:
                ...
    """
    import os
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF not found: '{pdf_path}'")

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise fitz.FileDataError(f"Failed to open PDF '{pdf_path}': {exc}") from exc

    if doc.is_encrypted:
        raise ValueError(f"PDF is encrypted and cannot be parsed: '{pdf_path}'")

    logger.debug("Opened PDF '%s' — %d pages.", pdf_path, len(doc))
    return doc


# ── Text extraction ────────────────────────────────────────────────────────────

def extract_text_blocks(doc: fitz.Document) -> list[TextBlock]:
    """
    Extracts all text blocks from *doc* using ``page.get_text("dict")``.

    For each type-0 (text) block on every page:
    - Concatenates all span texts across all lines into a single string.
    - Records the bounding box, 1-indexed page number.
    - Tracks the *maximum* font size across all spans (used for heading detection).
    - Tracks the *most common* font name (dominant font per block).

    Blocks whose concatenated text is entirely whitespace are skipped.

    Returns:
        Flat list of :class:`~ingestion.utils.schema.TextBlock` ordered by
        ``(page_number, y0)`` — top-to-bottom reading order.
    """
    blocks: list[TextBlock] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1  # 1-indexed

        try:
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        except Exception as exc:
            logger.warning("Page %d: get_text failed — %s. Skipping.", page_number, exc)
            continue

        for block in page_dict.get("blocks", []):
            if block.get("type") != 0:  # type 1 = image block; skip
                continue

            # Accumulate text and font metrics across all lines and spans
            texts: list[str] = []
            font_sizes: list[float] = []
            font_name_counts: dict[str, int] = {}

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    if span_text:
                        texts.append(span_text)
                    size = span.get("size", 0.0)
                    if size > 0:
                        font_sizes.append(size)
                    fname = span.get("font", "")
                    if fname:
                        font_name_counts[fname] = font_name_counts.get(fname, 0) + 1

            combined_text = "".join(texts)
            if not combined_text.strip():
                continue  # skip whitespace-only blocks

            max_font_size = max(font_sizes) if font_sizes else 0.0
            dominant_font = (
                max(font_name_counts, key=font_name_counts.get)
                if font_name_counts
                else ""
            )

            bbox_raw = block.get("bbox", (0.0, 0.0, 0.0, 0.0))
            bbox: tuple[float, float, float, float] = (
                bbox_raw[0], bbox_raw[1], bbox_raw[2], bbox_raw[3]
            )

            blocks.append(
                TextBlock(
                    text=combined_text,
                    page_number=page_number,
                    bbox=bbox,
                    font_size=max_font_size,
                    font_name=dominant_font,
                    block_type=0,
                )
            )

    # Sort by (page_number, y0) for deterministic reading order
    blocks.sort(key=lambda b: (b.page_number, b.bbox[1]))
    logger.debug("Extracted %d text blocks from '%s'.", len(blocks), doc.name)
    return blocks


# ── Section boundary detection ────────────────────────────────────────────────

def detect_section_boundaries(
    text_blocks: list[TextBlock],
    known_headings: list[str],
) -> list[Section]:
    """
    Scans *text_blocks* for SEC section headings and groups subsequent blocks
    into that :class:`~ingestion.utils.schema.Section`.

    Heading detection (first match wins):
    1. ``block.text`` starts with a *known_heading* **and**
       ``block.font_size > median_font_size`` of all blocks.
    2. ``block.text`` starts with a *known_heading* (font-size fallback,
       handles PDFs that do not use larger fonts for headings).

    Blocks before the first heading are placed into a ``"Preamble"`` section.

    Returns:
        ``list[Section]`` in document order.
    """
    if not text_blocks:
        return []

    # Compute median font size for heading detection heuristic
    font_sizes = [b.font_size for b in text_blocks if b.font_size > 0]
    median_font = statistics.median(font_sizes) if font_sizes else 0.0

    def _is_heading(block: TextBlock) -> Optional[str]:
        """Returns the matched heading string if *block* is a section heading, else None."""
        stripped = block.text.strip()
        for heading in known_headings:
            if stripped.lower().startswith(heading.lower()):
                # Prefer font-size confirmation but fall back if absent
                if block.font_size > median_font or block.font_size == 0:
                    return heading
                # Font-size fallback: still accept if the text starts with "Item N"
                if stripped.lower().startswith("item "):
                    return heading
        return None

    sections: list[Section] = []
    current_heading = "Preamble"
    current_full_heading = "Preamble"
    current_page_start = text_blocks[0].page_number if text_blocks else 1
    current_blocks: list[TextBlock] = []

    for block in text_blocks:
        matched_heading = _is_heading(block)
        if matched_heading is not None:
            # Save the previous section
            if current_blocks or current_heading == "Preamble":
                page_end = current_blocks[-1].page_number if current_blocks else current_page_start
                sections.append(
                    Section(
                        heading=current_heading,
                        full_heading=current_full_heading,
                        page_start=current_page_start,
                        page_end=page_end,
                        text_blocks=current_blocks,
                    )
                )
            # Start a new section
            current_heading = matched_heading.rstrip(".")
            current_full_heading = block.text.strip()
            current_page_start = block.page_number
            current_blocks = [block]
        else:
            current_blocks.append(block)

    # Flush the final section
    if current_blocks:
        sections.append(
            Section(
                heading=current_heading,
                full_heading=current_full_heading,
                page_start=current_page_start,
                page_end=current_blocks[-1].page_number,
                text_blocks=current_blocks,
            )
        )

    logger.debug(
        "Detected %d sections. Headings: %s",
        len(sections),
        [s.heading for s in sections],
    )
    return sections


# ── Image xref discovery ──────────────────────────────────────────────────────

def get_image_xrefs(doc: fitz.Document) -> list[dict]:
    """
    Returns metadata for every embedded image in *doc* that passes the minimum
    area threshold (>= 10,000 px²) used to filter logos and decorative icons.

    Each entry is::

        {
            "xref":        int,   # fitz internal reference number
            "page_number": int,   # 1-indexed page where the image appears
            "bbox":        tuple, # (x0, y0, x1, y1) in PDF points
        }

    Images appearing on multiple pages are deduplicated by ``xref`` — only
    the first occurrence (lowest page number) is kept.

    Returns:
        List of image metadata dicts ordered by ``(page_number, y0)``.
    """
    MIN_AREA_PX2 = 10_000

    seen_xrefs: set[int] = set()
    results: list[dict] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1

        try:
            image_list = page.get_images(full=True)
        except Exception as exc:
            logger.warning("Page %d: get_images failed — %s. Skipping.", page_number, exc)
            continue

        for img_info in image_list:
            xref = img_info[0]  # first element is always the xref

            if xref in seen_xrefs:
                continue  # deduplicate across pages

            # Try to get image dimensions from the xref info
            try:
                img_dict = doc.extract_image(xref)
                width = img_dict.get("width", 0)
                height = img_dict.get("height", 0)
            except Exception:
                width, height = 0, 0

            if width * height < MIN_AREA_PX2:
                continue  # skip logos / icons

            # Find the bounding box on this page by searching image placements
            bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
            try:
                for item in page.get_image_rects(xref):
                    # get_image_rects returns fitz.Rect objects
                    r = item if isinstance(item, fitz.Rect) else fitz.Rect(item)
                    bbox = (r.x0, r.y0, r.x1, r.y1)
                    break  # take the first placement
            except Exception:
                pass  # bbox stays (0, 0, 0, 0) — still included

            seen_xrefs.add(xref)
            results.append(
                {
                    "xref": xref,
                    "page_number": page_number,
                    "bbox": bbox,
                }
            )

    results.sort(key=lambda d: (d["page_number"], d["bbox"][1]))
    logger.debug("Found %d image xrefs above size threshold.", len(results))
    return results
