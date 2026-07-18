"""
pipeline.py — Orchestrator for the full offline ingestion pipeline.

Wires Phases 2-7 together for a single PDF (ingest_pdf) and for
the full corpus batch (ingest_all).  Contains no new logic — only
sequential calls to modules built in earlier phases.

BM25 index and JSONL dump are written ONCE after ALL PDFs are processed
so BM25 scores are computed across the whole corpus (not per-filing).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from ingestion.chunking.chart_chunker import chunk_all_charts
from ingestion.chunking.table_chunker import chunk_all_tables
from ingestion.chunking.text_chunker import chunk_narrative_sections
from ingestion.config import (
    ANNUAL_REPORTS_DIR,
    BM25_CHUNKS_PATH,
    BM25_INDEX_PATH,
    CHUNKS_JSONL_PATH,
    CORPUS_VERSION,
    SEC_SECTION_HEADINGS,
)
from ingestion.indexing.bm25_indexer import build_bm25_index, save_index
from ingestion.indexing.pinecone_upserter import (
    get_or_create_index,
    init_pinecone,
    upsert_all_by_namespace,
)
from ingestion.indexing.voyage_embedder import (
    embed_multimodal_chunks,
    embed_text_chunks,
    estimate_embedding_cost,
    init_voyage_client,
)
from ingestion.parsers.figure_extractor import extract_figures
from ingestion.parsers.pdf_parser import (
    detect_section_boundaries,
    extract_text_blocks,
    get_image_xrefs,
    load_pdf,
)
from ingestion.parsers.table_extractor import clean_table, extract_tables_from_pdf
from ingestion.preprocessing.content_separator import build_separated_content
from ingestion.utils.file_utils import discover_pdfs, save_chunks_jsonl
from ingestion.utils.logger import get_logger, reset_pipeline_timer
from ingestion.utils.schema import Chunk

logger = get_logger(__name__)


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class IngestionResult:
    """Return value of :func:`ingest_pdf`. Used for logging and verification."""
    pdf_path:          str
    company:           str
    fiscal_year:       int
    text_chunk_count:  int
    table_chunk_count: int
    chart_chunk_count: int
    total_chunk_count: int
    elapsed_seconds:   float
    warnings:          list[str] = field(default_factory=list)


# ── Single-PDF pipeline ───────────────────────────────────────────────────────

def ingest_pdf(
    pdf_info: dict,
    voyage_client,
    pinecone_index,
    dry_run: bool = False,
) -> tuple[IngestionResult, list[Chunk]]:
    """
    Runs the full ingestion pipeline for a single PDF.

    Args:
        pdf_info:       One dict from :func:`~ingestion.utils.file_utils.discover_pdfs`
                        with keys ``path``, ``company``, ``ticker``,
                        ``fiscal_year``, ``filing_type``, ``filename``.
        voyage_client:  Initialised :class:`voyageai.Client` (or ``None`` in
                        dry-run mode).
        pinecone_index: Initialised Pinecone Index handle (or ``None`` in
                        dry-run mode).
        dry_run:        If ``True``, skip embedding and upsert steps.  Useful
                        for testing Phases 2–4 without making API calls.

    Pipeline steps:

    1. **[Phase 2]** ``load_pdf`` → ``extract_text_blocks`` →
       ``detect_section_boundaries``
    2. **[Phase 2]** ``extract_tables_from_pdf`` → ``clean_table``
    3. **[Phase 2]** ``get_image_xrefs`` → ``extract_figures``
    4. **[Phase 3]** ``build_separated_content``
    5. **[Phase 4]** ``chunk_narrative_sections`` + ``chunk_all_tables``
       + ``chunk_all_charts``
    6. **[Phase 6]** ``estimate_embedding_cost`` (abort if over free tier)
    7. **[Phase 6]** ``embed_text_chunks`` + ``embed_multimodal_chunks``
       *(skipped in dry-run)*
    8. **[Phase 7]** ``upsert_all_by_namespace``
       *(skipped in dry-run)*

    BM25 index is **not** built here — that is done once over all chunks
    in :func:`ingest_all` after every PDF has been processed.

    Returns:
        ``(IngestionResult, list[Chunk])`` — result metadata and the flat
        chunk list (needed by :func:`ingest_all` to accumulate the corpus).
    """
    t0 = time.time()
    warnings: list[str] = []

    pdf_path    = pdf_info["path"]
    company     = pdf_info["company"]
    ticker      = pdf_info["ticker"]
    fiscal_year = pdf_info["fiscal_year"]
    filing_type = pdf_info["filing_type"]
    filename    = pdf_info["filename"]

    metadata_base = {
        "company":        company,
        "ticker":         ticker,
        "fiscal_year":    fiscal_year,
        "filing_type":    filing_type,
        "source_file":    filename,
        "corpus_version": CORPUS_VERSION,
    }

    logger.info("[%s FY%d] Step 1/8 — Loading PDF", company, fiscal_year)

    # ── Step 1: Parse PDF ─────────────────────────────────────────────────────
    doc         = load_pdf(pdf_path)
    text_blocks = extract_text_blocks(doc)
    sections    = detect_section_boundaries(text_blocks, SEC_SECTION_HEADINGS)
    logger.info("[%s FY%d] Step 1 done — %d sections detected", company, fiscal_year, len(sections))

    # ── Step 2: Extract tables ────────────────────────────────────────────────
    logger.info("[%s FY%d] Step 2/8 — Extracting tables", company, fiscal_year)
    raw_tables   = extract_tables_from_pdf(pdf_path)
    clean_tables = []
    skipped      = 0
    for rt in raw_tables:
        try:
            clean_tables.append(clean_table(rt))
        except ValueError as exc:
            skipped += 1
            logger.debug("Skipping table on page %d: %s", rt.page_number, exc)
    if skipped:
        warnings.append(f"{skipped} tables skipped (0 rows after cleaning)")
    logger.info("[%s FY%d] Step 2 done — %d tables", company, fiscal_year, len(clean_tables))

    # ── Step 3: Extract figures ───────────────────────────────────────────────
    logger.info("[%s FY%d] Step 3/8 — Extracting figures", company, fiscal_year)
    xrefs   = get_image_xrefs(doc)
    figures = extract_figures(doc, xrefs, text_blocks)
    doc.close()
    logger.info("[%s FY%d] Step 3 done — %d figures", company, fiscal_year, len(figures))

    # ── Step 4: Separate & preprocess content ─────────────────────────────────
    logger.info("[%s FY%d] Step 4/8 — Content separation", company, fiscal_year)
    sc = build_separated_content(sections, clean_tables, figures)

    # ── Step 5: Chunk ─────────────────────────────────────────────────────────
    logger.info("[%s FY%d] Step 5/8 — Chunking", company, fiscal_year)
    text_chunks  = chunk_narrative_sections(sc["sections"],          metadata_base)
    table_chunks = chunk_all_tables(sc["tables_by_section"],         metadata_base)
    chart_chunks = chunk_all_charts(sc["figures_by_section"],        metadata_base)
    all_chunks   = text_chunks + table_chunks + chart_chunks
    logger.info(
        "[%s FY%d] Step 5 done — %d chunks (text=%d, table=%d, chart=%d)",
        company, fiscal_year,
        len(all_chunks), len(text_chunks), len(table_chunks), len(chart_chunks),
    )

    if dry_run:
        logger.info("[%s FY%d] dry-run mode — skipping embed + upsert.", company, fiscal_year)
        elapsed = time.time() - t0
        return (
            IngestionResult(
                pdf_path=pdf_path,
                company=company,
                fiscal_year=fiscal_year,
                text_chunk_count=len(text_chunks),
                table_chunk_count=len(table_chunks),
                chart_chunk_count=len(chart_chunks),
                total_chunk_count=len(all_chunks),
                elapsed_seconds=round(elapsed, 2),
                warnings=warnings,
            ),
            all_chunks,
        )

    # ── Step 6: Pre-flight cost estimate ─────────────────────────────────────
    logger.info("[%s FY%d] Step 6/8 — Cost estimate", company, fiscal_year)
    cost = estimate_embedding_cost(all_chunks)
    if not cost["within_free_tier"]:
        msg = (
            f"[{company} FY{fiscal_year}] Estimated cost exceeds free-tier limits: {cost}. "
            "Aborting embedding."
        )
        logger.error(msg)
        warnings.append(msg)
        elapsed = time.time() - t0
        return (
            IngestionResult(
                pdf_path=pdf_path,
                company=company,
                fiscal_year=fiscal_year,
                text_chunk_count=len(text_chunks),
                table_chunk_count=len(table_chunks),
                chart_chunk_count=len(chart_chunks),
                total_chunk_count=len(all_chunks),
                elapsed_seconds=round(elapsed, 2),
                warnings=warnings,
            ),
            [],  # return no chunks to exclude from BM25 corpus
        )

    # ── Step 7: Embed ─────────────────────────────────────────────────────────
    logger.info("[%s FY%d] Step 7/8 — Embedding", company, fiscal_year)
    embed_text_chunks(all_chunks, voyage_client)
    # Sleep between text and multimodal embedding to respect rate limit
    logger.info("[%s FY%d] Sleeping 25s before multimodal embedding...", company, fiscal_year)
    time.sleep(25)
    embed_multimodal_chunks(all_chunks, voyage_client)
    logger.info("[%s FY%d] Step 7 done", company, fiscal_year)

    # ── Step 8: Upsert to Pinecone ────────────────────────────────────────────
    logger.info("[%s FY%d] Step 8/8 — Pinecone upsert", company, fiscal_year)
    upsert_all_by_namespace(pinecone_index, all_chunks)
    logger.info("[%s FY%d] Step 8 done", company, fiscal_year)

    elapsed = time.time() - t0
    logger.info(
        "[%s FY%d] Ingestion complete — %d chunks in %.1fs.",
        company, fiscal_year, len(all_chunks), elapsed,
    )

    return (
        IngestionResult(
            pdf_path=pdf_path,
            company=company,
            fiscal_year=fiscal_year,
            text_chunk_count=len(text_chunks),
            table_chunk_count=len(table_chunks),
            chart_chunk_count=len(chart_chunks),
            total_chunk_count=len(all_chunks),
            elapsed_seconds=round(elapsed, 2),
            warnings=warnings,
        ),
        all_chunks,
    )


# ── Batch pipeline ────────────────────────────────────────────────────────────

def ingest_all(
    reports_dir: str = ANNUAL_REPORTS_DIR,
    company_filter: str | None = None,
    year_filter: int | None = None,
    dry_run: bool = False,
    reset_namespaces: list[str] | None = None,
    corpus_version: str | None = None,
) -> list[IngestionResult]:
    """
    Full batch ingestion for every PDF under *reports_dir*.

    Steps:

    1. :func:`~ingestion.utils.file_utils.discover_pdfs` → sorted PDF list.
    2. Apply optional ``company_filter`` / ``year_filter``.
    3. If ``reset_namespaces`` provided, call
       :func:`~ingestion.indexing.pinecone_upserter.delete_namespace` first.
    4. For each PDF: :func:`ingest_pdf` → accumulate all chunks.
    5. :func:`~ingestion.indexing.bm25_indexer.build_bm25_index` over ALL
       chunks → :func:`~ingestion.indexing.bm25_indexer.save_index`.
    6. :func:`~ingestion.utils.file_utils.save_chunks_jsonl` → JSONL dump.
    7. Log final summary.

    BM25 index is built **once** over the entire corpus so cross-company
    and cross-year queries are scored correctly.

    Args:
        reports_dir:       Root directory containing company sub-folders.
        company_filter:    If set, only ingest PDFs whose ``company`` matches
                           this string (case-insensitive).
        year_filter:       If set, only ingest PDFs for this fiscal year.
        dry_run:           Skip embedding and upsert (Phases 6-7).
        reset_namespaces:  List of company names whose Pinecone namespaces
                           should be wiped before re-ingesting.
        corpus_version:    Override :data:`~ingestion.config.CORPUS_VERSION`.

    Returns:
        List of :class:`IngestionResult` — one per processed PDF.
    """
    from ingestion.indexing.pinecone_upserter import delete_namespace  # avoid circular
    reset_pipeline_timer()
    t_all = time.time()

    pdf_infos = discover_pdfs(reports_dir)

    # Apply filters
    if company_filter:
        pdf_infos = [p for p in pdf_infos if p["company"].lower() == company_filter.lower()]
    if year_filter:
        pdf_infos = [p for p in pdf_infos if p["fiscal_year"] == year_filter]

    if not pdf_infos:
        logger.warning("ingest_all: no PDFs found matching filters.")
        return []

    logger.info(
        "ingest_all: processing %d PDFs (dry_run=%s).", len(pdf_infos), dry_run
    )

    # Initialise API clients (skip in dry-run)
    voyage_client   = None
    pinecone_index  = None

    if not dry_run:
        voyage_key   = os.environ.get("VOYAGE_API_KEY", "")
        pinecone_key = os.environ.get("PINECONE_API_KEY", "")
        voyage_client  = init_voyage_client(voyage_key)
        pc             = init_pinecone(pinecone_key)
        pinecone_index = get_or_create_index(pc)

        # Reset namespaces before upserting if requested
        if reset_namespaces:
            for ns_company in reset_namespaces:
                delete_namespace(pinecone_index, ns_company.lower())
                logger.info("ingest_all: reset namespace '%s'.", ns_company.lower())

    all_chunks: list[Chunk] = []
    results: list[IngestionResult] = []

    for i, pdf_info in enumerate(pdf_infos, start=1):
        logger.info(
            "ingest_all: [%d/%d] %s FY%d",
            i, len(pdf_infos), pdf_info["company"], pdf_info["fiscal_year"],
        )
        try:
            result, chunks = ingest_pdf(
                pdf_info, voyage_client, pinecone_index, dry_run=dry_run
            )
            results.append(result)
            all_chunks.extend(chunks)
            
            # Sleep between PDFs to respect 3 RPM rate limit (skip after last PDF)
            if not dry_run and i < len(pdf_infos):
                logger.info("ingest_all: sleeping 25s before next PDF...")
                time.sleep(25)
        except Exception as exc:
            logger.error(
                "ingest_all: FAILED for %s FY%d — %s",
                pdf_info["company"], pdf_info["fiscal_year"], exc,
                exc_info=True,
            )
            results.append(
                IngestionResult(
                    pdf_path=pdf_info["path"],
                    company=pdf_info["company"],
                    fiscal_year=pdf_info["fiscal_year"],
                    text_chunk_count=0,
                    table_chunk_count=0,
                    chart_chunk_count=0,
                    total_chunk_count=0,
                    elapsed_seconds=0.0,
                    warnings=[f"FAILED: {exc}"],
                )
            )
            # Sleep after failed PDF too to avoid cascading rate limit errors
            if not dry_run and i < len(pdf_infos):
                logger.info("ingest_all: sleeping 25s after failed PDF...")
                time.sleep(25)

    # ── Post-batch: BM25 + JSONL ──────────────────────────────────────────────
    if all_chunks:
        logger.info("ingest_all: building BM25 index over %d chunks …", len(all_chunks))
        bm25_index = build_bm25_index(all_chunks)
        save_index(bm25_index, all_chunks, BM25_INDEX_PATH, BM25_CHUNKS_PATH)

        jsonl_path = CHUNKS_JSONL_PATH
        save_chunks_jsonl(all_chunks, jsonl_path)
    else:
        logger.warning("ingest_all: no chunks produced — skipping BM25 / JSONL write.")

    total_elapsed = time.time() - t_all
    total_chunks  = sum(r.total_chunk_count for r in results)
    logger.info(
        "ingest_all: DONE — %d PDFs, %d total chunks in %.1fs.",
        len(results), total_chunks, total_elapsed,
    )

    for r in results:
        status = "WARN" if r.warnings else "OK"
        logger.info(
            "  [%s] %s FY%d — %d chunks (%.1fs)%s",
            status, r.company, r.fiscal_year, r.total_chunk_count, r.elapsed_seconds,
            f"  warnings={r.warnings}" if r.warnings else "",
        )

    return results
