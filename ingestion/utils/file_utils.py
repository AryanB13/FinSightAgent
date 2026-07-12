"""
file_utils.py — All filesystem I/O for the ingestion pipeline.

The rest of the pipeline never calls open(), os.path, or json directly;
it calls these helpers instead. This makes swapping local-disk I/O for
cloud storage later a single-file change.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from ingestion.utils.logger import get_logger
from ingestion.utils.schema import (
    Chunk,
    CleanTable,
    FigureImage,
    RawTable,
    Section,
    TextBlock,
)

logger = get_logger(__name__)


# ── Project root ──────────────────────────────────────────────────────────────

def get_project_root() -> str:
    """
    Returns the absolute path to the project root — the directory that
    contains the ``ingestion/`` package folder.

    Works regardless of the current working directory.
    """
    return str(Path(__file__).resolve().parent.parent.parent)


# ── Directory helpers ─────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """
    Creates the directory at *path* (and all parents) if it does not exist.
    Equivalent to ``mkdir -p``.  Silently succeeds if the directory already
    exists.

    Args:
        path: Absolute or relative directory path to create.
    """
    os.makedirs(path, exist_ok=True)


# ── PDF discovery ─────────────────────────────────────────────────────────────

_FILENAME_PATTERN = re.compile(
    r"^nasdaq-(?P<ticker>[a-z]+)-(?P<year>\d{4})-(?P<filing>.+)-\d+\.pdf$",
    re.IGNORECASE,
)


def parse_filename_metadata(filename: str) -> dict[str, Any]:
    """
    Parses structured metadata from a filename following the convention::

        nasdaq-{ticker}-{year}-{filing_type}-{accession}.pdf

    Example::

        "nasdaq-aapl-2023-10K-231373899.pdf"
        → {"ticker": "AAPL", "fiscal_year": 2023, "filing_type": "10-K"}

    Args:
        filename: The bare filename (not a full path).

    Returns:
        Dict with keys ``ticker`` (str, uppercase), ``fiscal_year`` (int),
        ``filing_type`` (str, e.g. ``"10-K"``).

    Raises:
        ValueError: If *filename* does not match the expected pattern.
    """
    match = _FILENAME_PATTERN.match(filename)
    if not match:
        raise ValueError(
            f"Filename '{filename}' does not match expected pattern "
            "'nasdaq-{{ticker}}-{{year}}-{{filing_type}}-{{accession}}.pdf'"
        )
    raw_filing = match.group("filing").upper()
    # Normalise common SEC filing type abbreviations that may omit the hyphen
    # in filenames (e.g. "10K" → "10-K", "20F" → "20-F").
    _FILING_NORMALISE = {
        "10K": "10-K",
        "10KSB": "10-KSB",
        "20F": "20-F",
        "10Q": "10-Q",
    }
    filing_type = _FILING_NORMALISE.get(raw_filing, raw_filing)

    return {
        "ticker": match.group("ticker").upper(),
        "fiscal_year": int(match.group("year")),
        "filing_type": filing_type,
    }


def discover_pdfs(reports_dir: str) -> list[dict[str, Any]]:
    """
    Walks *reports_dir* and returns one metadata dict per ``.pdf`` file found.

    Expected directory layout::

        reports_dir/
          Apple/
            nasdaq-aapl-2022-10K-*.pdf
            ...
          Microsoft/
            ...
          NVIDIA/
            ...

    The **company name** is inferred from the immediate parent folder name.
    All other metadata is parsed from the filename via
    :func:`parse_filename_metadata`.

    Args:
        reports_dir: Absolute path to the root ``Annual Reports/`` directory.

    Returns:
        List of dicts, each with keys:
        ``path``, ``company``, ``ticker``, ``fiscal_year``,
        ``filing_type``, ``filename``.
        Sorted by (company, fiscal_year) for deterministic processing order.

    Raises:
        FileNotFoundError: If *reports_dir* does not exist.
    """
    if not os.path.isdir(reports_dir):
        raise FileNotFoundError(
            f"Annual Reports directory not found: '{reports_dir}'"
        )

    results: list[dict[str, Any]] = []

    for company_folder in sorted(os.listdir(reports_dir)):
        company_path = os.path.join(reports_dir, company_folder)
        if not os.path.isdir(company_path):
            continue  # skip stray files at the top level

        for filename in sorted(os.listdir(company_path)):
            if not filename.lower().endswith(".pdf"):
                continue

            full_path = os.path.join(company_path, filename)
            try:
                meta = parse_filename_metadata(filename)
            except ValueError as exc:
                logger.warning("Skipping '%s': %s", filename, exc)
                continue

            results.append(
                {
                    "path": full_path,
                    "company": company_folder,        # folder name is the company name
                    "ticker": meta["ticker"],
                    "fiscal_year": meta["fiscal_year"],
                    "filing_type": meta["filing_type"],
                    "filename": filename,
                }
            )

    results.sort(key=lambda d: (d["company"], d["fiscal_year"]))
    logger.debug("Discovered %d PDFs under '%s'.", len(results), reports_dir)
    return results


# ── Chunk serialisation helpers ───────────────────────────────────────────────

def _chunk_to_dict(chunk: Chunk) -> dict[str, Any]:
    """
    Converts a :class:`~ingestion.utils.schema.Chunk` to a JSON-serialisable
    dict.  ``image_bytes`` is base64-encoded because JSON cannot hold raw bytes.
    Nested dataclasses are converted recursively.
    """
    d: dict[str, Any] = {}
    for f in dataclasses.fields(chunk):
        value = getattr(chunk, f.name)

        if value is None:
            d[f.name] = None

        elif f.name == "image_bytes" and isinstance(value, bytes):
            d[f.name] = base64.b64encode(value).decode("ascii")

        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            d[f.name] = _dataclass_to_dict(value)

        else:
            d[f.name] = value

    return d


def _dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively converts a dataclass instance to a plain dict, encoding any
    ``bytes`` fields as base64 strings.
    """
    if not dataclasses.is_dataclass(obj) or isinstance(obj, type):
        return obj

    out: dict[str, Any] = {}
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if isinstance(value, bytes):
            out[f.name] = base64.b64encode(value).decode("ascii")
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            out[f.name] = _dataclass_to_dict(value)
        elif isinstance(value, list):
            out[f.name] = [
                _dataclass_to_dict(i) if (dataclasses.is_dataclass(i) and not isinstance(i, type)) else i
                for i in value
            ]
        else:
            out[f.name] = value
    return out


def _dict_to_chunk(d: dict[str, Any]) -> Chunk:
    """
    Reconstructs a :class:`~ingestion.utils.schema.Chunk` from a plain dict
    (the inverse of :func:`_chunk_to_dict`).

    Decodes base64 ``image_bytes``.  Reconstructs nested ``CleanTable``
    objects from their serialised dicts.
    """
    # Decode image_bytes if present
    if d.get("image_bytes") is not None:
        d["image_bytes"] = base64.b64decode(d["image_bytes"])

    # Reconstruct CleanTable from nested dict if present
    if d.get("table_data") is not None and isinstance(d["table_data"], dict):
        td = d["table_data"]
        d["table_data"] = CleanTable(
            rows=td["rows"],
            page_number=td["page_number"],
            bbox=tuple(td["bbox"]),
            header_row=td["header_row"],
            text_repr=td["text_repr"],
            row_count=td["row_count"],
            col_count=td["col_count"],
        )

    # bbox stored as list → convert to tuple
    # (Chunk itself doesn't have a bbox but nested types do; handled above)

    return Chunk(
        chunk_id=d["chunk_id"],
        company=d["company"],
        ticker=d["ticker"],
        fiscal_year=d["fiscal_year"],
        filing_type=d["filing_type"],
        section=d["section"],
        content_type=d["content_type"],
        page_number=d["page_number"],
        text=d["text"],
        table_data=d.get("table_data"),
        image_bytes=d.get("image_bytes"),
        source_file=d["source_file"],
        embedding_model=d["embedding_model"],
        corpus_version=d["corpus_version"],
        embedding=d.get("embedding"),
    )


def save_chunks_jsonl(chunks: list[Chunk], path: str) -> None:
    """
    Serialises *chunks* to a ``.jsonl`` file — one JSON object per line.

    ``image_bytes`` fields are base64-encoded.  ``embedding`` fields are
    included if present (useful for inspection / offline debugging).
    Creates parent directories if they do not exist.  Overwrites the file if
    it already exists.

    Args:
        chunks: List of :class:`~ingestion.utils.schema.Chunk` objects.
        path:   Absolute path for the output ``.jsonl`` file.
    """
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(_chunk_to_dict(chunk), ensure_ascii=False) + "\n")
    logger.info("Saved %d chunks to '%s'.", len(chunks), path)


def load_chunks_jsonl(path: str) -> list[Chunk]:
    """
    Deserialises a ``.jsonl`` file produced by :func:`save_chunks_jsonl` back
    into a list of :class:`~ingestion.utils.schema.Chunk` instances.

    Decodes base64 ``image_bytes``.  Returns an empty list if the file does
    not exist (graceful for a first-run where no prior checkpoint exists).

    Args:
        path: Absolute path to the ``.jsonl`` file.

    Returns:
        List of reconstructed :class:`~ingestion.utils.schema.Chunk` objects.
    """
    if not os.path.isfile(path):
        logger.debug("Chunks file not found at '%s' — returning empty list.", path)
        return []

    chunks: list[Chunk] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                chunks.append(_dict_to_chunk(d))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed line %d in '%s': %s", line_num, path, exc)

    logger.info("Loaded %d chunks from '%s'.", len(chunks), path)
    return chunks
