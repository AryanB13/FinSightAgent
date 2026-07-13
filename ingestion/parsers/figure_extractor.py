"""
figure_extractor.py — Chart/figure extraction from PDF using PyMuPDF.

Extracts embedded images as raw bytes so they can be embedded by
voyage-multimodal-3 without OCR — a query about "revenue trend" can
retrieve a bar chart as a nearest neighbour in the vector space.
"""

from __future__ import annotations

import io
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

from ingestion.config import MAX_FIGURE_PIXELS, MIN_FIGURE_PIXELS
from ingestion.utils.logger import get_logger
from ingestion.utils.schema import FigureImage, TextBlock

logger = get_logger(__name__)

_FINANCIAL_CAPTION_KEYWORDS = [
    "revenue", "income", "growth", "margin", "eps", "earnings",
    "quarter", "annual", "fiscal", "billion", "million", "sales",
    "profit", "loss", "cash", "debt", "equity", "return",
]

_MIN_PIXEL_COUNT = 10_000   # hard filter: logos / decorative icons


# ── Main extraction ───────────────────────────────────────────────────────────

def extract_figures(
    doc: fitz.Document,
    image_xrefs: list[dict],
    text_blocks: list[TextBlock],
) -> list[FigureImage]:
    """
    Extracts chart/figure images from *doc* using the pre-computed
    *image_xrefs* list (produced by :func:`~ingestion.parsers.pdf_parser.get_image_xrefs`).

    For each xref entry:
    1. Calls ``doc.extract_image(xref)`` to get raw bytes and extension.
    2. Computes ``pixel_count = width × height``.
    3. Finds the nearest ``TextBlock`` *below* the image bbox (within 60 PDF
       points) as the caption; looks *above* (within 40 pt) if nothing is
       found below.
    4. Filters images with ``pixel_count < 10_000`` (logos, icons).

    Returns:
        ``list[FigureImage]`` ordered by ``(page_number, y0)``.
    """
    figures: list[FigureImage] = []

    for xref_info in image_xrefs:
        xref        = xref_info["xref"]
        page_number = xref_info["page_number"]
        bbox        = xref_info["bbox"]   # (x0, y0, x1, y1) in PDF points

        try:
            img_data = doc.extract_image(xref)
        except Exception as exc:
            logger.debug("xref %d: extract_image failed — %s. Skipping.", xref, exc)
            continue

        image_bytes: bytes = img_data.get("image", b"")
        image_ext: str     = img_data.get("ext", "png").lower()
        width: int         = img_data.get("width", 0)
        height: int        = img_data.get("height", 0)
        pixel_count: int   = width * height

        if pixel_count < _MIN_PIXEL_COUNT:
            logger.debug("xref %d: pixel_count=%d < threshold — skipping.", xref, pixel_count)
            continue

        if not image_bytes:
            logger.debug("xref %d: empty image bytes — skipping.", xref)
            continue

        caption = _find_caption(bbox, page_number, text_blocks)

        figures.append(
            FigureImage(
                image_bytes=image_bytes,
                image_ext=image_ext,
                page_number=page_number,
                bbox=bbox,
                caption=caption,
                pixel_count=pixel_count,
                xref=xref,
            )
        )

    figures.sort(key=lambda f: (f.page_number, f.bbox[1]))
    logger.debug("Extracted %d figures above pixel threshold.", len(figures))
    return figures


def _find_caption(
    img_bbox: tuple[float, float, float, float],
    page_number: int,
    text_blocks: list[TextBlock],
    below_gap: float = 60.0,
    above_gap: float = 40.0,
) -> str:
    """
    Finds the nearest text block below (or above) the image bounding box.

    Strategy:
    1. Collect all ``TextBlock`` objects on *page_number* whose ``y0``
       is between ``img_bbox[3]`` and ``img_bbox[3] + below_gap``
       (i.e. immediately below the image).
    2. If none found, look *above*: blocks whose ``y1`` is between
       ``img_bbox[1] - above_gap`` and ``img_bbox[1]``.
    3. Return the text of the closest block, or ``""`` if nothing found.
    """
    img_y0, img_y1 = img_bbox[1], img_bbox[3]

    candidates_below: list[tuple[float, str]] = []
    candidates_above: list[tuple[float, str]] = []

    for block in text_blocks:
        if block.page_number != page_number:
            continue
        bx0, by0, bx1, by1 = block.bbox
        text = block.text.strip()
        if not text:
            continue

        # Below image
        if img_y1 <= by0 <= img_y1 + below_gap:
            candidates_below.append((by0 - img_y1, text))

        # Above image (fallback)
        elif img_y1 - above_gap <= by1 <= img_y0:
            candidates_above.append((img_y0 - by1, text))

    if candidates_below:
        return min(candidates_below, key=lambda x: x[0])[1]
    if candidates_above:
        return min(candidates_above, key=lambda x: x[0])[1]
    return ""


# ── Resize ────────────────────────────────────────────────────────────────────

def resize_figure_if_needed(figure: FigureImage) -> FigureImage:
    """
    Ensures the figure's pixel count is within Voyage AI's billing window.

    - If ``pixel_count > MAX_FIGURE_PIXELS`` (2M): downsample via Pillow,
      preserving aspect ratio.
    - If ``pixel_count < MIN_FIGURE_PIXELS`` (50K): upsample to exactly
      ``MIN_FIGURE_PIXELS`` pixels (preserving aspect ratio).
    - Returns the original *figure* unchanged if already in range.

    The re-encoded bytes replace ``figure.image_bytes``; the extension is
    preserved where possible (JPEG stays JPEG, PNG stays PNG).
    """
    if MIN_FIGURE_PIXELS <= figure.pixel_count <= MAX_FIGURE_PIXELS:
        return figure  # already within Voyage billing window

    try:
        img = Image.open(io.BytesIO(figure.image_bytes))
    except Exception as exc:
        logger.warning(
            "xref %d: cannot open image bytes for resizing — %s. Returning original.",
            figure.xref, exc,
        )
        return figure

    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        return figure

    if figure.pixel_count > MAX_FIGURE_PIXELS:
        # Downsample
        scale = (MAX_FIGURE_PIXELS / figure.pixel_count) ** 0.5
        new_w = max(1, int(orig_w * scale))
        new_h = max(1, int(orig_h * scale))
        # Guarantee we are at or below the cap
        while new_w * new_h > MAX_FIGURE_PIXELS and new_w > 1 and new_h > 1:
            new_w -= 1
            new_h = max(1, int(new_w * orig_h / orig_w))
    else:
        # Upsample to MIN_FIGURE_PIXELS — use ceiling to avoid integer-truncation shortfall
        import math
        scale = (MIN_FIGURE_PIXELS / figure.pixel_count) ** 0.5
        new_w = math.ceil(orig_w * scale)
        new_h = math.ceil(orig_h * scale)
        # Guarantee we are at or above the floor
        while new_w * new_h < MIN_FIGURE_PIXELS:
            new_w += 1

    img_resized = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    fmt = figure.image_ext.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in ("JPEG", "PNG", "WEBP", "GIF", "BMP"):
        fmt = "PNG"
    img_resized.save(buf, format=fmt)
    new_bytes = buf.getvalue()
    new_pixel_count = new_w * new_h

    logger.debug(
        "xref %d: resized %dx%d (%d px) → %dx%d (%d px).",
        figure.xref, orig_w, orig_h, figure.pixel_count, new_w, new_h, new_pixel_count,
    )

    return FigureImage(
        image_bytes=new_bytes,
        image_ext=figure.image_ext,
        page_number=figure.page_number,
        bbox=figure.bbox,
        caption=figure.caption,
        pixel_count=new_pixel_count,
        xref=figure.xref,
    )


# ── Financial figure filter ───────────────────────────────────────────────────

def filter_financial_figures(figures: list[FigureImage]) -> list[FigureImage]:
    """
    Heuristic filter — keeps a figure if **any** of the following is true:

    1. ``pixel_count > 40_000`` **and** ``0.5 < aspect_ratio < 4.0``
       (charts are wider-than-tall or roughly square; excludes narrow
       decorative rule images).
    2. ``caption`` contains at least one financial keyword from the list:
       ``["revenue", "income", "growth", "margin", "eps", "earnings",
       "quarter", "annual", "fiscal", "billion", "million"]``.

    Logs the number of figures kept vs. skipped at DEBUG level.
    """
    kept: list[FigureImage] = []
    skipped = 0

    for fig in figures:
        x0, y0, x1, y1 = fig.bbox
        width  = x1 - x0
        height = y1 - y0
        aspect = (width / height) if height > 0 else 0.0

        passes_size  = fig.pixel_count > 40_000 and 0.5 < aspect < 4.0
        caption_lower = fig.caption.lower()
        passes_caption = any(kw in caption_lower for kw in _FINANCIAL_CAPTION_KEYWORDS)

        if passes_size or passes_caption:
            kept.append(fig)
        else:
            skipped += 1

    logger.debug(
        "filter_financial_figures: kept %d, skipped %d.",
        len(kept), skipped,
    )
    return kept
