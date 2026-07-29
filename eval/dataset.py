"""
eval/dataset.py — Ground truth dataset for the Financial Research Agent
evaluation pipeline.

Each EvalQuery contains:
  - query           : exact user query string
  - reference_answer: human-verified answer from the real 10-K filings
  - reference_number: primary numeric value (for sanity-checking answer text)
  - relevant_chunk_ids: chunk IDs that MUST appear in retrieved results
                        (ground truth for Precision@5 / Recall@5)
  - sub_query_specs : how to decompose the query for retrieval evaluation.
                      For direct_lookup queries this list is empty and the
                      evaluator uses (query, companies[0], years[0]).
                      For multi-hop / single-hop queries it carries one spec
                      per retrieval sub-query so each company-year pair is
                      fetched independently and the retrieved sets are unioned.

All chunk IDs were verified by querying the Pinecone index directly with
voyage-multimodal-3 embeddings.  Apple and Microsoft chunks are Pinecone-only
(not in the BM25 pickle).  NVIDIA chunks appear in both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ── Sub-query spec (for retrieval evaluation of multi-sub-query routes) ────────

@dataclass
class SubQuerySpec:
    """Defines one retrieval sub-query inside a multi-part EvalQuery."""
    text: str          # query text sent to the retriever
    company: str       # lowercase company name for the namespace / filter
    year: int          # fiscal year for the metadata filter


# ── Main evaluation query ─────────────────────────────────────────────────────

@dataclass
class EvalQuery:
    """One evaluation query with full ground truth."""
    id: str                            # e.g. "DL-001"
    category: str                      # "direct_lookup" | "single_hop" | "multi_hop" | "adversarial"
    query: str                         # exact user query string
    reference_answer: str              # human-verified ground truth answer
    reference_number: Optional[float]  # primary numeric value in millions USD (or %)
    relevant_chunk_ids: list[str]      # chunk IDs that contain the answer
    companies: list[str]               # e.g. ["apple"]
    years: list[int]                   # e.g. [2022, 2023]
    expected_route: str                # expected Router Agent output
    sub_query_specs: list[SubQuerySpec] = field(default_factory=list)
    notes: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH DATASET  (30 queries)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_DATASET: list[EvalQuery] = [

    # ── DIRECT LOOKUP (10) ───────────────────────────────────────────────────
    # Single company, single year, single metric.
    # sub_query_specs is empty → evaluator uses (query, companies[0], years[0]).

    EvalQuery(
        id="DL-001",
        category="direct_lookup",
        query="What was Apple's total net sales in FY2023?",
        reference_answer="Apple's total net sales for FY2023 were $383,285 million.",
        reference_number=383285.0,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-1000",     # income stmt: Total net sales 383285
            "AAPL-FY2023-10K-item8-table-33000",    # geographic breakdown: Total net sales 383285
        ],
        companies=["apple"],
        years=[2023],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-002",
        category="direct_lookup",
        query="What was Apple's total net sales in FY2022?",
        reference_answer="Apple's total net sales for FY2022 were $394,328 million.",
        reference_number=394328.0,
        relevant_chunk_ids=[
            "AAPL-FY2022-10K-item8-table-1000",     # income stmt: Total net sales 394328
            "AAPL-FY2022-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0020",  # text: Total net sales$394,328
        ],
        companies=["apple"],
        years=[2022],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-003",
        category="direct_lookup",
        query="What was Apple's net income in FY2022?",
        reference_answer="Apple's net income for FY2022 was $99,803 million.",
        reference_number=99803.0,
        relevant_chunk_ids=[
            "AAPL-FY2022-10K-item8-table-1000",     # income stmt: contains net income line
        ],
        companies=["apple"],
        years=[2022],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-004",
        category="direct_lookup",
        query="What was Apple's research and development expense in FY2023?",
        reference_answer="Apple's research and development expense for FY2023 was $29,915 million.",
        reference_number=29915.0,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-32000",
            "AAPL-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0023",
            "AAPL-FY2023-10K-item8-table-1000",
        ],
        companies=["apple"],
        years=[2023],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-005",
        category="direct_lookup",
        query="What was Apple's operating income in FY2023?",
        reference_answer="Apple's operating income for FY2023 was $114,301 million.",
        reference_number=114301.0,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-1000",     # income stmt: Operating income 114301
        ],
        companies=["apple"],
        years=[2023],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-006",
        category="direct_lookup",
        query="What was Microsoft's total revenue in FY2023?",
        reference_answer="Microsoft's total revenue for FY2023 was $211,915 million.",
        reference_number=211915.0,
        relevant_chunk_ids=[
            "MSFT-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0096",   # MD&A summary: Revenue $211,915
            "MSFT-FY2023-10K-item8-financial-statements-and-supplementary-data-text-0172",  # revenue breakdown by product/service
        ],
        companies=["microsoft"],
        years=[2023],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-007",
        category="direct_lookup",
        query="What was Microsoft's operating income in FY2022?",
        reference_answer="Microsoft's operating income for FY2022 was $83,383 million.",
        reference_number=83383.0,
        relevant_chunk_ids=[
            "MSFT-FY2022-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0101",  # operating income 83383, revenue 198270
        ],
        companies=["microsoft"],
        years=[2022],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-008",
        category="direct_lookup",
        query="What was Microsoft's net income in FY2023?",
        reference_answer="Microsoft's net income for FY2023 was $72,361 million.",
        reference_number=72361.0,
        relevant_chunk_ids=[
            "MSFT-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0096",   # MD&A summary: Net income 72,361
        ],
        companies=["microsoft"],
        years=[2023],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-009",
        category="direct_lookup",
        query="What was NVIDIA's total revenue in FY2024?",
        reference_answer="NVIDIA's total revenue for FY2024 was $60,922 million.",
        reference_number=60922.0,
        relevant_chunk_ids=[
            "NVDA-FY2024-10K-item7-table-0000",     # income stmt: Revenue $60,922 Up 126%
            "NVDA-FY2024-10K-item15-table-0000",    # full income stmt: Revenue 60922, net income 29760
        ],
        companies=["nvidia"],
        years=[2024],
        expected_route="direct_lookup",
    ),

    EvalQuery(
        id="DL-010",
        category="direct_lookup",
        query="What was NVIDIA's net income in FY2024?",
        reference_answer="NVIDIA's net income for FY2024 was $29,760 million.",
        reference_number=29760.0,
        relevant_chunk_ids=[
            "NVDA-FY2024-10K-item7-table-0000",     # income stmt: Revenue 60922, Net income 29760
            "NVDA-FY2024-10K-item15-table-0000",    # full income stmt: Revenue 60922, net income 29760
        ],
        companies=["nvidia"],
        years=[2024],
        expected_route="direct_lookup",
    ),

    # ── SINGLE HOP — YoY comparisons (8) ────────────────────────────────────
    # Requires chunks from two fiscal years.
    # sub_query_specs carries one spec per year so the evaluator fetches
    # each year independently and unions the results.

    EvalQuery(
        id="SH-001",
        category="single_hop",
        query="How did Apple's gross margin change from FY2022 to FY2023?",
        reference_answer=(
            "Apple's gross margin improved from 43.3% in FY2022 to 44.1% in FY2023, "
            "a 0.8 percentage point improvement despite lower total net sales."
        ),
        reference_number=None,
        relevant_chunk_ids=[
            "AAPL-FY2022-10K-item8-table-1000",     # FY2022 income stmt: gross margin line
            "AAPL-FY2023-10K-item8-table-1000",     # FY2023 income stmt: gross margin line
        ],
        companies=["apple"],
        years=[2022, 2023],
        expected_route="single_hop",
        sub_query_specs=[
            SubQuerySpec("Apple gross margin FY2022", "apple", 2022),
            SubQuerySpec("Apple gross margin FY2023", "apple", 2023),
        ],
    ),

    EvalQuery(
        id="SH-002",
        category="single_hop",
        query="How did Apple's research and development spending change from FY2022 to FY2023?",
        reference_answer=(
            "Apple's R&D expense increased from $26,251 million in FY2022 to $29,915 million "
            "in FY2023, a 14.0% year-over-year increase."
        ),
        reference_number=14.0,
        relevant_chunk_ids=[
            "AAPL-FY2022-10K-item8-table-30000",    # FY2022 R&D table: 26251
            "AAPL-FY2022-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0023",  # FY2022 MD&A: R&D figure
            "AAPL-FY2023-10K-item8-table-32000",    # FY2023 R&D table: 29915
            "AAPL-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0023",  # FY2023 MD&A: R&D figure
        ],
        companies=["apple"],
        years=[2022, 2023],
        expected_route="single_hop",
        sub_query_specs=[
            SubQuerySpec("Apple R&D research development expense FY2022", "apple", 2022),
            SubQuerySpec("Apple R&D research development expense FY2023", "apple", 2023),
        ],
    ),

    EvalQuery(
        id="SH-003",
        category="single_hop",
        query="How did Apple's net income change from FY2022 to FY2023?",
        reference_answer=(
            "Apple's net income declined from $99,803 million in FY2022 to $96,995 million "
            "in FY2023, a decrease of approximately 2.8%."
        ),
        reference_number=-2.8,
        relevant_chunk_ids=[
            "AAPL-FY2022-10K-item8-table-1000",     # FY2022 income stmt: net income
            "AAPL-FY2022-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0020",  # FY2022 net sales summary text
            "AAPL-FY2023-10K-item8-table-1000",     # FY2023 income stmt: net income
        ],
        companies=["apple"],
        years=[2022, 2023],
        expected_route="single_hop",
        sub_query_specs=[
            SubQuerySpec("Apple net income FY2022", "apple", 2022),
            SubQuerySpec("Apple net income FY2023", "apple", 2023),
        ],
    ),

    EvalQuery(
        id="SH-004",
        category="single_hop",
        query="How did Microsoft's revenue change from FY2022 to FY2023?",
        reference_answer=(
            "Microsoft's revenue grew from $198,270 million in FY2022 to $211,915 million "
            "in FY2023, a 6.9% year-over-year increase."
        ),
        reference_number=6.9,
        relevant_chunk_ids=[
            "MSFT-FY2022-10K-item8-financial-statements-and-supplementary-data-text-0174",  # FY2022 geographic revenue: Total $198,270
            "MSFT-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0096",  # FY2023 MD&A: Revenue $211,915 $198,270 7%
        ],
        companies=["microsoft"],
        years=[2022, 2023],
        expected_route="single_hop",
        sub_query_specs=[
            SubQuerySpec("Microsoft total revenue FY2022", "microsoft", 2022),
            SubQuerySpec("Microsoft total revenue FY2023", "microsoft", 2023),
        ],
    ),

    EvalQuery(
        id="SH-005",
        category="single_hop",
        query="How did Microsoft's net income change from FY2022 to FY2023?",
        reference_answer=(
            "Microsoft's net income was $72,738 million in FY2022 and $72,361 million in FY2023, "
            "roughly flat with a slight decrease of 0.5%."
        ),
        reference_number=-0.5,
        relevant_chunk_ids=[
            "MSFT-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0096",  # FY2023 MD&A: Net income 72,361 72,738
        ],
        companies=["microsoft"],
        years=[2022, 2023],
        expected_route="single_hop",
        sub_query_specs=[
            SubQuerySpec("Microsoft net income FY2022", "microsoft", 2022),
            SubQuerySpec("Microsoft net income FY2023", "microsoft", 2023),
        ],
    ),

    EvalQuery(
        id="SH-006",
        category="single_hop",
        query="How did NVIDIA's revenue change from FY2023 to FY2024?",
        reference_answer=(
            "NVIDIA's revenue grew from $26,974 million in FY2023 to $60,922 million in FY2024, "
            "an increase of approximately 125.9%, driven primarily by data center GPU demand."
        ),
        reference_number=125.9,
        relevant_chunk_ids=[
            "NVDA-FY2023-10K-item7-table-0000",     # FY2023 income stmt: Revenue 26974
            "NVDA-FY2024-10K-item7-table-0000",     # FY2024 income stmt: Revenue 60922 Up 126%
            "NVDA-FY2024-10K-item15-table-0000",    # FY2024 full income stmt: Revenue 60922
        ],
        companies=["nvidia"],
        years=[2023, 2024],
        expected_route="single_hop",
        sub_query_specs=[
            SubQuerySpec("NVIDIA total revenue FY2023", "nvidia", 2023),
            SubQuerySpec("NVIDIA total revenue FY2024", "nvidia", 2024),
        ],
    ),

    EvalQuery(
        id="SH-007",
        category="single_hop",
        query="How did NVIDIA's net income change from FY2023 to FY2024?",
        reference_answer=(
            "NVIDIA's net income increased from $4,368 million in FY2023 to $29,760 million "
            "in FY2024, a 581% increase."
        ),
        reference_number=581.0,
        relevant_chunk_ids=[
            "NVDA-FY2023-10K-item7-table-0000",     # FY2023 income stmt: Revenue 26974, net income 4368
            "NVDA-FY2024-10K-item7-table-0000",     # FY2024 income stmt: Revenue 60922, net income 29760
            "NVDA-FY2024-10K-item15-table-0000",    # FY2024 full income stmt: Revenue 60922, net income 29760
        ],
        companies=["nvidia"],
        years=[2023, 2024],
        expected_route="single_hop",
        sub_query_specs=[
            SubQuerySpec("NVIDIA net income FY2023", "nvidia", 2023),
            SubQuerySpec("NVIDIA net income FY2024", "nvidia", 2024),
        ],
    ),

    EvalQuery(
        id="SH-008",
        category="single_hop",
        query="How did Apple's selling general and administrative expenses change from FY2022 to FY2023?",
        reference_answer=(
            "Apple's SG&A expenses decreased slightly from $25,094 million in FY2022 "
            "to $24,932 million in FY2023."
        ),
        reference_number=-162.0,
        relevant_chunk_ids=[
            "AAPL-FY2022-10K-item8-table-1000",     # FY2022 income stmt: SG&A 25094
            "AAPL-FY2023-10K-item8-table-1000",     # FY2023 income stmt: SG&A 24932
        ],
        companies=["apple"],
        years=[2022, 2023],
        expected_route="single_hop",
        sub_query_specs=[
            SubQuerySpec("Apple SG&A selling general administrative expenses FY2022", "apple", 2022),
            SubQuerySpec("Apple SG&A selling general administrative expenses FY2023", "apple", 2023),
        ],
    ),

    # ── MULTI-HOP + COMPUTATION (7) ──────────────────────────────────────────
    # Requires retrieval from multiple sections, Tool-Use computation.

    EvalQuery(
        id="MH-001",
        category="multi_hop",
        query="What was Apple's R&D expense as a percentage of revenue in FY2022?",
        reference_answer=(
            "Apple's R&D expense was $26,251 million in FY2022, representing approximately "
            "6.7% of net sales of $394,328 million."
        ),
        reference_number=6.66,
        relevant_chunk_ids=[
            "AAPL-FY2022-10K-item8-table-30000",    # R&D table: 26251
            "AAPL-FY2022-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0023",  # MD&A: R&D discussion
            "AAPL-FY2022-10K-item8-table-1000",     # income stmt: Total net sales 394328
        ],
        companies=["apple"],
        years=[2022],
        expected_route="multi_hop",
        sub_query_specs=[
            SubQuerySpec("Apple R&D research development expense FY2022", "apple", 2022),
            SubQuerySpec("Apple total net sales revenue FY2022", "apple", 2022),
        ],
        notes="percent_of(26251, 394328) = 6.66%",
    ),

    EvalQuery(
        id="MH-002",
        category="multi_hop",
        query="What was Apple's R&D expense as a percentage of revenue in FY2023?",
        reference_answer=(
            "Apple's R&D expense was $29,915 million in FY2023, representing approximately "
            "7.8% of net sales of $383,285 million."
        ),
        reference_number=7.80,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-32000",    # R&D table: 29915
            "AAPL-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0023",  # MD&A: R&D discussion
            "AAPL-FY2023-10K-item8-table-1000",     # income stmt: Total net sales 383285
        ],
        companies=["apple"],
        years=[2023],
        expected_route="multi_hop",
        sub_query_specs=[
            SubQuerySpec("Apple R&D research development expense FY2023", "apple", 2023),
            SubQuerySpec("Apple total net sales revenue FY2023", "apple", 2023),
        ],
        notes="percent_of(29915, 383285) = 7.80%",
    ),

    EvalQuery(
        id="MH-003",
        category="multi_hop",
        query="What was Apple's operating margin in FY2023?",
        reference_answer=(
            "Apple's operating margin for FY2023 was approximately 29.8%, "
            "with operating income of $114,301 million on net sales of $383,285 million."
        ),
        reference_number=29.82,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-1000",     # income stmt: Operating income 114301, net sales 383285
        ],
        companies=["apple"],
        years=[2023],
        expected_route="multi_hop",
        sub_query_specs=[
            SubQuerySpec("Apple operating income FY2023", "apple", 2023),
            SubQuerySpec("Apple total net sales revenue FY2023", "apple", 2023),
        ],
        notes="operating_margin(114301, 383285) = 29.82%",
    ),

    EvalQuery(
        id="MH-004",
        category="multi_hop",
        query="Compare Apple and Microsoft's research and development spending in FY2023.",
        reference_answer=(
            "In FY2023, Apple spent $29,915 million on R&D (7.8% of $383,285M revenue) while "
            "Microsoft spent $27,195 million (12.8% of $211,915M revenue). Microsoft had higher "
            "R&D intensity despite Apple's larger absolute R&D spend."
        ),
        reference_number=None,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-32000",    # Apple R&D table: 29915
            "AAPL-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0023",  # Apple MD&A: R&D 29915
            "MSFT-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0101",  # Microsoft MD&A: R&D 27195
        ],
        companies=["apple", "microsoft"],
        years=[2023],
        expected_route="multi_hop",
        sub_query_specs=[
            SubQuerySpec("Apple R&D research development expense FY2023", "apple", 2023),
            SubQuerySpec("Microsoft R&D research development expense FY2023", "microsoft", 2023),
        ],
        notes="Two companies — retrieval must surface chunks from both namespaces.",
    ),

    EvalQuery(
        id="MH-005",
        category="multi_hop",
        query="What was Microsoft's R&D expense as a percentage of revenue in FY2023?",
        reference_answer=(
            "Microsoft's R&D expense was $27,195 million in FY2023, representing 12.8% "
            "of total revenue of $211,915 million."
        ),
        reference_number=12.83,
        relevant_chunk_ids=[
            "MSFT-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0101",  # MD&A: R&D 27195
            "MSFT-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0096",  # MD&A summary: Revenue 211915, Net income 72361
        ],
        companies=["microsoft"],
        years=[2023],
        expected_route="multi_hop",
        sub_query_specs=[
            SubQuerySpec("Microsoft R&D research development expense FY2023", "microsoft", 2023),
            SubQuerySpec("Microsoft total revenue FY2023", "microsoft", 2023),
        ],
        notes="percent_of(27195, 211915) = 12.83%",
    ),

    EvalQuery(
        id="MH-006",
        category="multi_hop",
        query="Compare Apple and NVIDIA's operating margins in FY2023.",
        reference_answer=(
            "Apple's operating margin in FY2023 was approximately 29.8% ($114,301M / $383,285M). "
            "NVIDIA's operating margin in FY2023 was approximately 16.2%. "
            "Apple had the significantly higher operating margin."
        ),
        reference_number=None,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-1000",     # Apple income stmt: Operating income 114301
            "NVDA-FY2023-10K-item7-table-0000",     # NVIDIA income stmt: Operating income FY2023
        ],
        companies=["apple", "nvidia"],
        years=[2023],
        expected_route="multi_hop",
        sub_query_specs=[
            SubQuerySpec("Apple operating income net sales FY2023", "apple", 2023),
            SubQuerySpec("NVIDIA operating income revenue FY2023", "nvidia", 2023),
        ],
        notes="Two operating_margin computations.",
    ),

    EvalQuery(
        id="MH-007",
        category="multi_hop",
        query="How did NVIDIA's revenue grow from FY2022 to FY2024?",
        reference_answer=(
            "NVIDIA's revenue grew from $26,914 million in FY2022 to $60,922 million in FY2024, "
            "approximately 126% cumulative growth over two years."
        ),
        reference_number=126.3,
        relevant_chunk_ids=[
            "NVDA-FY2022-10K-item7-table-0000",     # FY2022 income stmt: Revenue $26,914
            "NVDA-FY2024-10K-item7-table-0000",     # FY2024 income stmt: Revenue $60,922 Up 126%
        ],
        companies=["nvidia"],
        years=[2022, 2024],
        expected_route="multi_hop",
        sub_query_specs=[
            SubQuerySpec("NVIDIA total revenue FY2022", "nvidia", 2022),
            SubQuerySpec("NVIDIA total revenue FY2024", "nvidia", 2024),
        ],
        notes="Two-year span; cumulative growth not CAGR.",
    ),

    # ── ADVERSARIAL / EDGE CASES (5) ─────────────────────────────────────────

    EvalQuery(
        id="ADV-001",
        category="adversarial",
        query="What was Apple's revenue in FY2025?",
        reference_answer=(
            "Apple's FY2025 annual report data is not available in this corpus. "
            "The most recent Apple data available is FY2023."
        ),
        reference_number=None,
        relevant_chunk_ids=[],
        companies=["apple"],
        years=[2025],
        expected_route="direct_lookup",
        notes="Out-of-corpus year. Retrieval recall trivially 1.0 (no ground truth chunks). "
              "Faithfulness must be 1.0 — no hallucination of a fabricated revenue figure.",
    ),

    EvalQuery(
        id="ADV-002",
        category="adversarial",
        query="What was Tesla's revenue in FY2023?",
        reference_answer="Tesla is not covered in this corpus. Only Apple, Microsoft, and NVIDIA data is available.",
        reference_number=None,
        relevant_chunk_ids=[],
        companies=["tesla"],
        years=[2023],
        expected_route="direct_lookup",
        notes="Out-of-corpus company. No retrieval possible. Faithfulness must be 1.0.",
    ),

    EvalQuery(
        id="ADV-003",
        category="adversarial",
        query="What was Apple's revenue in FY2023 and its total employee headcount?",
        reference_answer=(
            "Apple's total net sales in FY2023 were $383,285 million. "
            "Employee headcount data is not present in the financial statements in this corpus."
        ),
        reference_number=383285.0,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-1000",     # income stmt: Total net sales 383285
            "AAPL-FY2023-10K-item8-table-33000",    # geographic breakdown: Total net sales 383285
        ],
        companies=["apple"],
        years=[2023],
        expected_route="direct_lookup",
        notes="Mixed: one answerable sub-question (revenue), one unanswerable (headcount).",
    ),

    EvalQuery(
        id="ADV-004",
        category="adversarial",
        query="What was Apple's revenue?",
        reference_answer=(
            "Apple's most recent available annual revenue (FY2023) was $383,285 million in net sales."
        ),
        reference_number=383285.0,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-1000",     # income stmt: Total net sales 383285
            "AAPL-FY2023-10K-item8-table-33000",    # geographic: Total net sales 383285
        ],
        companies=["apple"],
        years=[2023],
        expected_route="direct_lookup",
        notes="No year specified. Router should default to most recent (FY2023). "
              "Must not hallucinate a year.",
    ),

    EvalQuery(
        id="ADV-005",
        category="adversarial",
        query="Was Apple's R&D higher than Microsoft's in FY2023?",
        reference_answer=(
            "Yes, Apple's absolute R&D spend ($29,915M) exceeded Microsoft's ($27,195M) in FY2023. "
            "However, Microsoft's R&D intensity (12.8% of revenue) was higher than Apple's (7.8%)."
        ),
        reference_number=None,
        relevant_chunk_ids=[
            "AAPL-FY2023-10K-item8-table-32000",    # Apple R&D table: 29915
            "AAPL-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-operations-text-0023",  # Apple MD&A: R&D
            "MSFT-FY2023-10K-item7-managements-discussion-and-analysis-of-financial-condition-and-results-of-text-0101",  # Microsoft MD&A: R&D discussion
        ],
        companies=["apple", "microsoft"],
        years=[2023],
        expected_route="multi_hop",
        sub_query_specs=[
            SubQuerySpec("Apple R&D research development expense FY2023", "apple", 2023),
            SubQuerySpec("Microsoft R&D research development expense FY2023", "microsoft", 2023),
        ],
        notes="Comparison query — answer requires retrieving both companies' R&D and reasoning.",
    ),
]


# ── Loader helpers ─────────────────────────────────────────────────────────────

def load_eval_dataset(categories: list[str] | None = None) -> list[EvalQuery]:
    """Returns all or category-filtered EvalQuery objects.

    Args:
        categories: Optional list of category strings to include.
                    Valid values: "direct_lookup", "single_hop", "multi_hop", "adversarial".
                    Pass None (default) to return all 30 queries.

    Returns:
        Filtered list of EvalQuery objects.
    """
    if categories is None:
        return EVAL_DATASET
    return [q for q in EVAL_DATASET if q.category in categories]


def load_direct_lookup()  -> list[EvalQuery]: return load_eval_dataset(["direct_lookup"])
def load_single_hop()     -> list[EvalQuery]: return load_eval_dataset(["single_hop"])
def load_multi_hop()      -> list[EvalQuery]: return load_eval_dataset(["multi_hop"])
def load_adversarial()    -> list[EvalQuery]: return load_eval_dataset(["adversarial"])
