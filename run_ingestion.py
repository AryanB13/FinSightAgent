"""
run_ingestion.py — CLI entry point for the offline ingestion pipeline.

Usage examples:
  python run_ingestion.py                           # full ingest all PDFs
  python run_ingestion.py --dry-run                 # parse + chunk only, no API
  python run_ingestion.py --company Apple           # ingest only Apple PDFs
  python run_ingestion.py --company Apple --year 2023
  python run_ingestion.py --reset-namespace Apple   # delete + re-ingest Apple
  python run_ingestion.py --reports-dir /my/path    # override reports dir
  python run_ingestion.py --corpus-version v2       # trigger cache invalidation
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()   # load .env before any import that reads os.environ


def main() -> None:
    """
    CLI entry point. Parses arguments and calls
    :func:`~ingestion.pipeline.ingest_all`.

    Arguments (all optional, use config defaults if not provided):

    - ``--reports-dir PATH``       Override ``ANNUAL_REPORTS_DIR``
    - ``--company COMPANY``        Ingest only this company (incremental runs)
    - ``--year YEAR``              Ingest only this fiscal year (with --company)
    - ``--dry-run``                Parse + chunk, skip embedding + upsert
    - ``--reset-namespace COMPANY`` Delete and re-ingest one company's namespace
    - ``--corpus-version VERSION`` Override ``CORPUS_VERSION``
    """
    parser = argparse.ArgumentParser(
        description="Financial Research Agent — Offline Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--reports-dir",
        default=None,
        help="Override ANNUAL_REPORTS_DIR from config.py",
    )
    parser.add_argument(
        "--company",
        default=None,
        help='Ingest only this company, e.g. "Apple"',
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Ingest only this fiscal year, e.g. 2023",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Parse + chunk only — skip embedding and Pinecone upsert",
    )
    parser.add_argument(
        "--reset-namespace",
        metavar="COMPANY",
        default=None,
        help='Delete and re-ingest one company namespace, e.g. "Apple"',
    )
    parser.add_argument(
        "--corpus-version",
        default=None,
        help='Override CORPUS_VERSION in config, e.g. "v2"',
    )

    args = parser.parse_args()

    # Optional corpus-version override (must happen before importing config constants)
    if args.corpus_version:
        os.environ["CORPUS_VERSION_OVERRIDE"] = args.corpus_version

    from ingestion.config import ANNUAL_REPORTS_DIR
    from ingestion.pipeline import ingest_all

    reports_dir = args.reports_dir or ANNUAL_REPORTS_DIR

    reset_namespaces = [args.reset_namespace] if args.reset_namespace else None

    results = ingest_all(
        reports_dir=reports_dir,
        company_filter=args.company,
        year_filter=args.year,
        dry_run=args.dry_run,
        reset_namespaces=reset_namespaces,
    )

    # Exit with non-zero code if any PDF failed
    failed = [r for r in results if any("FAILED" in w for w in r.warnings)]
    if failed:
        print(f"\n{len(failed)} PDF(s) failed ingestion:", file=sys.stderr)
        for r in failed:
            print(f"  {r.company} FY{r.fiscal_year}: {r.warnings}", file=sys.stderr)
        sys.exit(1)

    total = sum(r.total_chunk_count for r in results)
    print(f"\nIngestion complete: {len(results)} PDF(s), {total} total chunks.")


if __name__ == "__main__":
    main()
