# FinSightAgent — Financial Research Agent

> An agentic RAG system that answers complex questions about SEC 10-K filings using hybrid keyword-semantic search, multimodal understanding of tables and charts, and a Python sandbox for on-the-fly financial ratio computation. Built with LangGraph orchestration, Pinecone vector search, and dual-layer caching for production-grade retrieval performance.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Tech Stack](#tech-stack)
5. [Project Structure](#project-structure)
6. [Getting Started](#getting-started)
7. [Usage](#usage)
   - [A. Document Ingestion](#a-document-ingestion)
   - [B. Query Pipeline](#b-query-pipeline)
   - [C. Evaluation](#c-evaluation)
8. [Capabilities in Depth](#capabilities-in-depth)
   - [Agentic Query Routing](#agentic-query-routing)
   - [Hybrid Retrieval](#hybrid-retrieval)
   - [Multimodal Retrieval](#multimodal-retrieval)
   - [Cross-Encoder Reranking](#cross-encoder-reranking)
   - [Sufficiency Check & Retry Loop](#sufficiency-check--retry-loop)
   - [Python Sandbox for Financial Computation](#python-sandbox-for-financial-computation)
   - [Answer Verification](#answer-verification)
   - [Dual-Layer Caching](#dual-layer-caching)
9. [Evaluation Framework](#evaluation-framework)
10. [Data Sources](#data-sources)

---

## Overview

FinSightAgent is an end-to-end **agentic Retrieval-Augmented Generation (RAG)** system designed specifically for financial research over SEC 10-K annual reports.

Given a natural language question, the system:
1. Classifies the query complexity (single fact vs. multi-year comparison vs. cross-company analysis)
2. Decomposes it into atomic sub-queries when needed
3. Retrieves evidence via **hybrid keyword + semantic search** over multimodal chunks (text, tables, charts)
4. Runs a **sufficiency check** — if retrieved chunks are incomplete, it reformulates and retries
5. Executes sandboxed Python for **financial computations** (growth rates, ratios, percentages)
6. Generates a cited, structured answer and **verifies every claim** against the retrieved evidence
7. Caches the result (exact + semantic) so repeated or similar questions are served instantly

The system currently covers **3 companies × 3 fiscal years = 9 10-K filings**:
- **Apple (AAPL)** — FY2022, FY2023, FY2024
- **Microsoft (MSFT)** — FY2022, FY2023, FY2024
- **NVIDIA (NVDA)** — FY2022, FY2023, FY2024

---

## Key Features

- **Agentic Query Routing** — Gemini LLM classifies every query as `direct_lookup`, `single_hop`, or `multi_hop` and extracts company/year entities, skipping unnecessary pipeline stages to save quota
- **Hybrid Retrieval (BM25 + Semantic)** — keyword search for exact financial figures fused with dense vector search for semantic matches using Reciprocal Rank Fusion (RRF)
- **Multimodal Chunks** — tables and charts are extracted as images, embedded with `voyage-multimodal-3`, and retrieved alongside text chunks in a unified vector space
- **Cross-Encoder Reranking** — Pinecone-hosted `bge-reranker-v2-m3` re-scores all retrieved chunks jointly against the query for a final relevance ranking
- **Sufficiency Check & Retry Loop** — a Gemini judge evaluates all sub-query chunks in one call; if evidence is insufficient it reformulates queries using a financial synonym map and retries (up to 2×)
- **Python Sandbox** — financial computations (YoY growth, percentage of revenue, etc.) run in an isolated subprocess with a 5-second timeout — computed values are always traceable to exact retrieved chunks, never LLM-estimated
- **Answer Verification** — a separate Gemini judge fact-checks every claim in the draft answer against retrieved chunks and removes or flags unsupported claims
- **Dual-Layer Redis Cache** — exact-match cache for repeated queries and semantic similarity cache (cosine ≥ 0.95) for paraphrases, both backed by Upstash Redis with a 30-day TTL
- **Comprehensive Evaluation Suite** — Precision@5, Recall@5 (retrieval), and Faithfulness / Answer Relevance (generation via LLM-as-Judge) over 30 labelled ground-truth queries

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LangGraph StateGraph                        │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────┐    cache hit    ┌─────┐                            │
│  │ Cache Check │───────────────► │ END │                            │
│  │ (Redis)     │                 └─────┘                            │
│  └──────┬──────┘                                                    │
│         │ cache miss                                                │
│         ▼                                                           │
│  ┌─────────────┐  multi_hop  ┌────────────┐                        │
│  │   Router    │────────────►│ Decomposer │                        │
│  │  (Gemini)   │             └─────┬──────┘                        │
│  └──────┬──────┘                   │                               │
│         │ direct/single_hop        │                               │
│         └──────────────────────────┘                               │
│                          │                                          │
│                          ▼                                          │
│              ┌───────────────────────┐                             │
│              │ Retrieval + Sufficiency│  ◄── BM25 (keyword)        │
│              │      Loop             │  ◄── Pinecone (semantic)    │
│              │  (up to 2 retries)    │  ◄── RRF fusion             │
│              └──────────┬────────────┘  ◄── Cross-encoder rerank   │
│                         │                                          │
│                         ▼                                          │
│              ┌────────────────────┐                                │
│              │   Tool-Use Agent   │  ◄── Sandboxed Python          │
│              │  (computation)     │      financial calculators     │
│              └──────────┬─────────┘                                │
│                         │                                          │
│                         ▼                                          │
│              ┌────────────────────┐                                │
│              │  Generator Agent   │  ◄── Gemini (cited answer)    │
│              └──────────┬─────────┘                                │
│                         │                                          │
│                         ▼                                          │
│              ┌────────────────────┐                                │
│              │  Verifier Agent    │  ◄── Gemini (fact-check)      │
│              └──────────┬─────────┘                                │
│                         │                                          │
│                         ▼                                          │
│              ┌────────────────────┐                                │
│              │   Cache Write      │  ──► Redis (exact + semantic)  │
│              └──────────┬─────────┘                                │
│                         ▼                                          │
│                    Final Answer                                    │
│              (answer + citations + verdict)                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Category | Technology | Purpose |
|---|---|---|
| **LLM** | Google Gemini 3.1 Flash Lite | Routing, decomposition, generation, verification, sufficiency judging |
| **Embeddings** | Voyage AI `voyage-multimodal-3` | Unified text + table/chart image embeddings (1024-dim) |
| **Vector Database** | Pinecone (serverless, AWS us-east-1) | ANN semantic search with namespace-level company isolation |
| **Keyword Search** | BM-25 (`rank-bm25`) | Exact-term retrieval for financial figures and codes, stored in-process |
| **Reranker** | Pinecone Inference `bge-reranker-v2-m3` | Cross-encoder reranking of fused BM25 + semantic candidates |
| **Graph Orchestration** | LangGraph `StateGraph` | Conditional agentic pipeline with typed state |
| **Cache** | Upstash Redis (REST) | Exact-match + semantic similarity cache with 30-day TTL |
| **PDF Parsing** | PyMuPDF (`fitz`), pdfplumber | Layout-aware text extraction, table structure detection |
| **Computation Sandbox** | Python subprocess | Isolated financial calculator execution (5s timeout) |
| **Evaluation LLM** | Google Gemini (same model) | LLM-as-Judge for Faithfulness and Answer Relevance |

---

## Project Structure

```
Financial_Research_Agent/
│
├── Annual Reports/             # Raw 10-K PDF files (source documents)
│   ├── Apple/
│   ├── Microsoft/
│   └── NVIDIA/
│
├── ingestion/                  # Part A — Offline document processing
│   ├── config.py               #   All ingestion constants (chunk size, models, paths)
│   ├── pipeline.py             #   Top-level orchestrator: parse → chunk → embed → upsert
│   ├── parsers/                #   PDF text extraction (PyMuPDF + pdfplumber)
│   ├── preprocessing/          #   Section detection, figure extraction
│   ├── chunking/               #   Text chunker, table chunker
│   └── indexing/               #   Voyage embedder, Pinecone upserter, BM25 indexer
│
├── query/                      # Part B — Online query pipeline
│   ├── config.py               #   All query constants (top-k, RRF_K, cache TTL, etc.)
│   ├── pipeline.py             #   init_pipeline_resources() + run_query() entry points
│   ├── graph/
│   │   └── state_graph.py      #   LangGraph graph definition (7 nodes + conditional edges)
│   ├── agents/                 #   One file per LangGraph node
│   │   ├── router_agent.py     #     Classifies query → direct_lookup/single_hop/multi_hop
│   │   ├── decomposer_agent.py #     Breaks multi-hop queries into atomic sub-queries
│   │   ├── sufficiency_agent.py#     Judges evidence completeness; triggers retry
│   │   ├── tool_use_agent.py   #     Identifies and runs financial computations
│   │   ├── generator_agent.py  #     Drafts cited answer from chunks + computed metrics
│   │   └── verifier_agent.py   #     Fact-checks draft against retrieved evidence
│   ├── retrieval/              #   Hybrid retrieval stack
│   │   ├── hybrid_retriever.py #     Orchestrates BM25 → Semantic → RRF → Filter → Rerank
│   │   ├── bm25_retriever.py   #     BM25Okapi keyword search
│   │   ├── semantic_retriever.py#    Pinecone ANN search + chunk hydration
│   │   ├── fusion.py           #     Reciprocal Rank Fusion (RRF)
│   │   ├── metadata_filter.py  #     Company/year namespace filtering
│   │   └── reranker.py         #     Cross-encoder reranking via Pinecone Inference API
│   ├── cache/                  #   Redis caching layer
│   │   ├── exact_cache.py      #     Normalised exact-match cache
│   │   └── semantic_cache.py   #     Cosine similarity semantic cache
│   └── tools/                  #   Sandboxed computation
│       ├── financial_calculators.py  # Growth, ratio, percentage extractors
│       └── sandbox_executor.py       # Subprocess-based secure execution
│
├── eval/                       # Part C — Evaluation framework
│   ├── dataset.py              #   30-query ground truth dataset (EvalQuery dataclass)
│   ├── retrieval_eval.py       #   Precision@5 and Recall@5 metrics
│   ├── generation_eval.py      #   Faithfulness and Answer Relevance (LLM-as-Judge)
│   └── run_eval.py             #   CLI runner for the full evaluation pipeline
│
├── run_ingestion.py            # CLI entry point — ingestion pipeline
├── run_query.py                # CLI entry point — query pipeline
├── requirements.txt            # Python dependencies
└── .env.example                # Environment variable template
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- API keys for: **Gemini**, **Voyage AI**, **Pinecone**, **Upstash Redis**

### 1. Clone and install

```bash
git clone https://github.com/your-username/Financial_Research_Agent.git
cd Financial_Research_Agent
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
VOYAGE_API_KEY=...          # https://dash.voyageai.com/api-keys
PINECONE_API_KEY=...        # https://app.pinecone.io/
UPSTASH_REDIS_REST_URL=...  # https://console.upstash.com/
UPSTASH_REDIS_REST_TOKEN=...
GEMINI_API_KEY=...          # https://aistudio.google.com/app/apikey
```

### 3. Place annual report PDFs

Put the 10-K PDF files in `Annual Reports/` following this structure:

```
Annual Reports/
├── Apple/
│   ├── aapl-2022-10K.pdf
│   ├── aapl-2023-10K.pdf
│   └── ...
├── Microsoft/
│   └── ...
└── NVIDIA/
    └── ...
```

PDF filenames must contain the company ticker (`AAPL`, `MSFT`, `NVDA`) and fiscal year (e.g. `2023`).

---

## Usage

### A. Document Ingestion

Run once (or after adding new PDFs) to parse, chunk, embed, and index all documents:

```bash
# Full ingest (all companies, all years)
python run_ingestion.py

# Parse and chunk only — no API calls (useful for debugging)
python run_ingestion.py --dry-run

# Ingest only one company
python run_ingestion.py --company Apple

# Ingest only one company + year
python run_ingestion.py --company Apple --year 2023

# Delete and re-ingest one company's Pinecone namespace
python run_ingestion.py --reset-namespace Apple
```

**What ingestion does:**
1. Parses each 10-K PDF — extracts text by section (Item 1–15), tables, and chart images
2. Chunks narrative text (512 tokens, 64-token overlap) and tables (≤50 rows per chunk)
3. Embeds all chunk types using `voyage-multimodal-3` (text as text, tables/charts as images)
4. Upserts vectors to Pinecone with company/year/section metadata
5. Builds a BM25 in-process keyword index over all chunks and serialises it to `data/bm25_index/`

> **Expected output:** ~1,699 chunks across 9 filings, stored in both Pinecone and the local BM25 index.

---

### B. Query Pipeline

```bash
# Basic query (formatted answer)
python run_query.py "What was Apple's total revenue in FY2023?"

# Output full JSON payload
python run_query.py "What was NVIDIA's net income in FY2024?" --json

# Bypass cache read (useful for testing changes)
python run_query.py "Compare R&D spending: Apple vs Microsoft FY2023" --no-cache
```

**Example output:**

```
========================================================================
ANSWER
========================================================================
Apple's total net sales for FY2023 were $383,285 million, a decrease of 2.8% 
compared to $394,328 million in FY2022.

Sources: AAPL-FY2023-10K-item8-table-1000, AAPL-FY2022-10K-item8-table-1000

Verdict: VERIFIED
========================================================================
```

**Response fields (JSON mode):**

```json
{
  "final_answer": "...",
  "citations": ["AAPL-FY2023-10K-item8-table-1000", "..."],
  "verdict": "verified",
  "flagged_claims": [],
  "cached_at": "2026-07-29T14:00:00+00:00"
}
```

---

### C. Evaluation

```bash
# Retrieval metrics only (Precision@5 + Recall@5) — no Gemini calls
python eval/run_eval.py --retrieval-only

# Evaluate a single category
python eval/run_eval.py --category direct_lookup --retrieval-only

# Evaluate multiple categories together
python eval/run_eval.py --category single_hop --category multi_hop --retrieval-only

# Full pipeline evaluation (includes generation + LLM-as-Judge scoring)
python eval/run_eval.py --category direct_lookup

# Save JSON report
python eval/run_eval.py --retrieval-only --output eval/results/run1.json

# Pipe JSON to jq
python eval/run_eval.py --json --retrieval-only | jq '.aggregate'
```

---

## Capabilities in Depth

### Agentic Query Routing

The **Router Agent** (`query/agents/router_agent.py`) is the first Gemini call in every pipeline run. It classifies the query into one of three routes and extracts all named entities:

| Route | Description | Example |
|---|---|---|
| `direct_lookup` | Single fact from one document | "What was Apple's FY2023 revenue?" |
| `single_hop` | Same metric across 2 years (YoY) | "How did Microsoft's net income change from FY2022 to FY2023?" |
| `multi_hop` | Cross-company or derived metric | "Compare R&D as % of revenue: Apple vs Microsoft FY2023" |

**Why routing matters:** `direct_lookup` skips the Decomposer agent entirely and bypasses the sufficiency retry loop, saving up to 4+ Gemini calls per query. `single_hop` also skips the Decomposer. Only `multi_hop` routes through the Decomposer.

The router also sets a `needs_computation` flag that determines whether the Tool-Use Agent runs.

---

### Hybrid Retrieval

Each sub-query goes through a 7-step retrieval pipeline (`query/retrieval/hybrid_retriever.py`):

1. **BM25 keyword search** — `BM-25` scores all 1,699 chunks in-process against tokenised query terms; top-20 candidates
2. **Semantic ANN search** — Query is embedded with `voyage-multimodal-3` and searched against Pinecone with a native metadata filter (company + year namespace); top-20 candidates
3. **Reciprocal Rank Fusion (RRF)** — Merges both ranked lists: score = Σ 1/(rank_i + 60), preserving rank information from both signals
4. **Post-fusion metadata filter** — Normalises BM25 results to the same company/year constraints as the semantic filter
5. **Chunk hydration** — Chart and table chunks lacking `image_bytes`/`table_data` are re-hydrated from the BM25 pickle (Pinecone stores only embeddings + metadata, not raw content)
6. **Cross-encoder reranking** — Top-N candidates are jointly scored against the query by `bge-reranker-v2-m3`; long documents are truncated to 1024 tokens at the END
7. **Final top-5** — `RERANK_TOP_N = 5` chunks per sub-query

**Key parameters** (`query/config.py`):
```python
BM25_TOP_K    = 20    # candidates from BM25
SEMANTIC_TOP_K = 20   # candidates from Pinecone ANN
RRF_K          = 60   # RRF damping constant
RERANK_TOP_N   = 5    # final chunks per sub-query
```

---

### Multimodal Retrieval

Financial 10-Ks are dense with tables and charts that purely text-based RAG misses. FinSightAgent handles three content types:

| Type | Extraction | Embedding |
|---|---|---|
| **Narrative text** | PyMuPDF section-aware extraction | Text input to `voyage-multimodal-3` |
| **Tables** | pdfplumber structure detection, rendered as Markdown | Image of table rendered as PIL image, embedded multimodally |
| **Charts** | PyMuPDF `xref`-based image extraction | Image embedded with `voyage-multimodal-3` |

All three types share a **single unified Pinecone index and vector space**, so a semantic query can retrieve the most relevant evidence regardless of whether it lives in prose, a financial statement table, or a chart.

---

### Cross-Encoder Reranking

The bi-encoder stage (BM25 + semantic) retrieves broad candidates based on independent query and document embeddings. The **cross-encoder** (`bge-reranker-v2-m3` via Pinecone Inference API) then jointly encodes each (query, document) pair — this is fundamentally more accurate but computationally expensive, which is why it runs only on the top-N fused candidates.

For multi-hop queries with 2+ sub-queries, a **second cross-query reranking pass** runs over the union of all sub-query results using the *original* full query as the reranker query, collapsing ~10 candidates back to 5.

---

### Sufficiency Check & Retry Loop

After retrieval, the **Sufficiency Agent** (`query/agents/sufficiency_agent.py`) asks Gemini: *"Do these chunks contain enough information to answer each sub-query?"*

Critically: **all sub-queries are evaluated in a single Gemini call** regardless of count (not N calls).

If any sub-query is marked insufficient:
- **Attempt 2**: Rule-based synonym substitution — e.g. `"revenue"` → `"net sales"`, `"R&D"` → `"research and development"` — no extra Gemini call
- **Attempt 3**: Same substitution with broader phrasing
- After `MAX_SUFFICIENCY_RETRIES = 2` failed attempts, the pipeline proceeds with the best available evidence

---

### Python Sandbox for Financial Computation

When `needs_computation = True` (set by the router for queries involving ratios, percentages, or growth rates), the **Tool-Use Agent** (`query/agents/tool_use_agent.py`):

1. Identifies required computations via regex patterns (no Gemini call) — e.g. `r"growth|change|increase|yoy"` signals a YoY growth computation
2. Extracts source numeric values directly from the top retrieved table chunks
3. Generates Python code that performs the computation
4. Executes it in an isolated subprocess with a **5-second timeout** (`SANDBOX_TIMEOUT_SECONDS = 5`)
5. Returns `ComputedMetric` objects with the result *and* the source chunk IDs

This ensures computed values (e.g. "NVIDIA's revenue grew 125.9%") are **always derived from retrieved data**, never hallucinated by the LLM.

---

### Answer Verification

The **Verifier Agent** (`query/agents/verifier_agent.py`) does a final Gemini call to fact-check the draft answer:
- Checks every numerical claim against the retrieved chunks
- Sets `verdict = "verified"` if all claims are supported
- Sets `verdict = "partial"` and populates `flagged_claims` if any claim is unsupported or contradicted
- Returns a cleaned `final_answer` with flagged claims removed

---

### Dual-Layer Caching

All caching is backed by **Upstash Redis** (serverless REST API) with a **30-day TTL**:

| Layer | Trigger | Mechanism |
|---|---|---|
| **Exact cache** | Query string matches (after normalisation: lowercase, strip whitespace) | Redis key = `hash(normalized_query + corpus_version)` |
| **Semantic cache** | No exact match, but cosine similarity ≥ 0.95 to a cached query embedding | Query embedded, scanned against stored embeddings in Redis list |

Cache keys include `CORPUS_VERSION = "v1"`, so re-ingesting the corpus (bumping the version) automatically orphans all stale entries without manual invalidation. The `--no-cache` flag on `run_query.py` bypasses cache reads while still writing results.

---

## Evaluation Framework

The evaluation suite in `eval/` measures four metrics over a **30-query ground truth dataset** across four categories:

| Category | Queries | Description |
|---|---|---|
| `direct_lookup` | 10 | Single-fact retrieval from one filing |
| `single_hop` | 8 | Year-over-year comparisons |
| `multi_hop` | 7 | Cross-company or derived metrics |
| `adversarial` | 5 | Out-of-scope, ambiguous, or missing-year queries |

### Metrics

| Metric | What it measures |
|---|---|
| **Precision@5** | Are the top-5 chunks relevant? |
| **Recall@5** | Are all relevant chunks found? |
| **Faithfulness** | Are all answer claims grounded in retrieved chunks? |
| **Answer Relevance** | Does the answer address the actual question? |

### Running the evaluation

```bash
# Retrieval only (all 30 queries, no Gemini calls)
python eval/run_eval.py --retrieval-only

# Full evaluation (retrieval + generation + LLM judges)
python eval/run_eval.py

# Category-specific
python eval/run_eval.py --category multi_hop --retrieval-only --verbose
```

---

## Data Sources

| Company | Ticker | Filing |
|---|---|---|
| Apple Inc. | AAPL | 10-K |
| Microsoft Corp. | MSFT | 10-K |
| NVIDIA Corp. | NVDA | 10-K |

All filings are publicly available from the SEC EDGAR database. They are **not included in this repository** due to size; download them from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar) and place them under `Annual Reports/`.

---