"""
config.py — All constants and settings for the ingestion pipeline.

Changing a batch size, model name, or path should require editing only this file.
All paths that begin with "data/" are relative to the project root returned
by file_utils.get_project_root().
"""

import os
from pathlib import Path

# ── Project root (resolves regardless of where the script is run from) ────────
_PROJECT_ROOT: str = str(Path(__file__).resolve().parent.parent)

# ── Corpus ────────────────────────────────────────────────────────────────────
ANNUAL_REPORTS_DIR: str = os.path.join(_PROJECT_ROOT, "Annual Reports")

COMPANIES: dict[str, str] = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
}

FISCAL_YEARS: list[int] = [2022, 2023, 2024]

FILING_TYPE: str = "10-K"

# Bump this string whenever the corpus is re-ingested so that all
# Upstash Redis cache keys using it are automatically orphaned.
CORPUS_VERSION: str = "v1"

# ── PDF Section Headings ──────────────────────────────────────────────────────
# Known 10-K Item markers used for section boundary detection.
SEC_SECTION_HEADINGS: list[str] = [
    "Item 1.",
    "Item 1A.",
    "Item 1B.",
    "Item 2.",
    "Item 3.",
    "Item 4.",
    "Item 5.",
    "Item 6.",
    "Item 7.",
    "Item 7A.",
    "Item 8.",
    "Item 9.",
    "Item 9A.",
    "Item 9B.",
    "Item 10.",
    "Item 11.",
    "Item 12.",
    "Item 13.",
    "Item 14.",
    "Item 15.",
]

# ── Parsing ───────────────────────────────────────────────────────────────────
MIN_TABLE_ROWS: int = 2        # ignore single-row "tables" (usually just headers)
MIN_TABLE_COLS: int = 2        # ignore single-column pseudo-tables

# Voyage AI charges per pixel; images outside this range are resized
# in figure_extractor.resize_figure_if_needed() before embedding.
MAX_FIGURE_PIXELS: int = 2_000_000   # Voyage's billing cap
MIN_FIGURE_PIXELS: int = 50_000      # Voyage's minimum billable size

# ── Chunking ──────────────────────────────────────────────────────────────────
NARRATIVE_CHUNK_SIZE_TOKENS: int = 512
NARRATIVE_CHUNK_OVERLAP_TOKENS: int = 64

# Split tables with more data rows than this into multiple chunks.
TABLE_MAX_ROWS_PER_CHUNK: int = 50

# Fast token-count estimate: characters ÷ this constant ≈ token count.
# Used by text_chunker.estimate_token_count() to avoid calling a full
# tokenizer on every chunk.  Accurate within ~10% for English financial prose.
APPROX_CHARS_PER_TOKEN: float = 4.0

# ── Embedding ─────────────────────────────────────────────────────────────────
VOYAGE_TEXT_MODEL: str = "voyage-3.5"
VOYAGE_MULTIMODAL_MODEL: str = "voyage-multimodal-3"
VOYAGE_EMBEDDING_DIM: int = 1024          # output dimension for both models

VOYAGE_TEXT_BATCH_SIZE: int = 128         # max texts per API call
VOYAGE_MULTIMODAL_BATCH_SIZE: int = 6     # free-tier: 10K TPM / ~3 RPM → max ~3.3K tokens/batch

# ── BM25 ──────────────────────────────────────────────────────────────────────
BM25_INDEX_PATH: str = os.path.join(_PROJECT_ROOT, "data", "bm25_index", "bm25_index.pkl")
BM25_CHUNKS_PATH: str = os.path.join(_PROJECT_ROOT, "data", "bm25_index", "bm25_chunks.pkl")

BM25_K1: float = 1.5    # term-frequency saturation parameter
BM25_B: float = 0.75    # document-length normalisation parameter

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_INDEX_NAME: str = "financial-research-agent"
PINECONE_REGION: str = "us-east-1"
PINECONE_CLOUD: str = "aws"
PINECONE_UPSERT_BATCH_SIZE: int = 100

# ── Output ────────────────────────────────────────────────────────────────────
CHUNKS_JSONL_PATH: str = os.path.join(
    _PROJECT_ROOT, "data", "extracted_chunks", "chunks_v1.jsonl"
)
