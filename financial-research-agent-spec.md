# Financial Research & Investment Analyst Agent

## 1. Project Overview

**One-line pitch**: An agentic RAG system that answers multi-hop financial research questions by reasoning over 10-K filings, annual reports, and earnings materials — retrieving text, tables, and charts, doing its own arithmetic, and verifying its own claims before answering.

**Purpose**: Most RAG portfolio projects retrieve-then-generate on a single hop. This project demonstrates the harder, more realistic version of the problem: real financial analysis questions require decomposing a question into sub-questions, retrieving different pieces of evidence for each, computing derived metrics correctly (not hallucinating them), and checking your own work before presenting a conclusion. The project exists to prove you can build a system that reasons *about* its own retrieval process, not just execute it once.

**Who it's "for"**: Framed as a tool for an equity research analyst or informed retail investor who wants quick, cited answers to comparative/quantitative questions across companies' public filings — without reading 200-page PDFs themselves.

---

## 2. Core Capabilities

| Capability | What it means in practice |
|---|---|
| **Multimodal document understanding** | Parses 10-Ks/annual reports and extracts narrative text (MD&A, risk factors), structured tables (balance sheet, income statement), and charts/figures — treating each as first-class retrievable content, not just flattened text |
| **Hybrid retrieval** | Combines BM25 (good for exact terms — ticker symbols, line-item names like "Total Stockholders' Equity") with semantic search (good for conceptual questions — "risk related to supply chain") |
| **Metadata-aware filtering** | Every chunk is tagged with company, fiscal year, filing type, section (e.g. "Item 7 - MD&A"), and content type (text/table/chart) — so retrieval can be scoped precisely ("only FY2023 balance sheets for Company A") |
| **Query decomposition** | Breaks multi-company, multi-year, or multi-metric questions into independently retrievable sub-questions |
| **Tool-augmented computation** | Pulls raw numbers from retrieved tables and computes derived metrics (YoY growth, margins, ratios) with an actual calculator/code tool instead of letting the LLM "eyeball" arithmetic |
| **Self-correcting retrieval** | Checks whether retrieved context is actually sufficient before generating an answer; if not, reformulates the query or widens the search and retries (bounded loop) |
| **Answer verification** | A second pass checks the draft answer's claims against the retrieved evidence and flags/strips anything unsupported |
| **Full evaluation suite** | Retrieval quality (recall, precision), generation quality (answer relevancy, faithfulness), and consistency/fairness (does phrasing change the answer?) |
| **Response caching** | Repeated or near-duplicate queries are served from cache rather than re-running the full pipeline |

---

## 3. System Architecture

```
                          ┌─────────────────┐
                          │   User Query     │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │  Cache Check      │──► (hit) return cached answer
                          └────────┬─────────┘
                                   │ (miss)
                          ┌────────▼─────────┐
                          │  Router Agent     │  decides: simple lookup vs.
                          └────────┬─────────┘  decompose vs. needs computation
                                   │
                    ┌──────────────┴───────────────┐
                    │ (multi-hop)                   │ (single-hop)
           ┌────────▼─────────┐                     │
           │ Decomposer Agent  │                     │
           │ → N sub-queries   │                     │
           └────────┬─────────┘                     │
                     │                               │
           ┌─────────▼───────────────────────────────▼────────┐
           │         Hybrid Retriever (per sub-query)          │
           │   BM25 + Semantic (Voyage multimodal) + Metadata   │
           │              filter → Reranker                    │
           └─────────────────────┬──────────────────────────────┘
                                  │
                        ┌─────────▼─────────┐
                        │ Sufficiency Check  │── insufficient ──► reformulate,
                        │      Agent         │                    retry (max 2)
                        └─────────┬─────────┘
                                  │ sufficient
                        ┌─────────▼─────────┐
                        │  Tool-Use Agent    │  (optional: calculator/code
                        │  (compute metrics) │   exec on extracted table data)
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Generator (Gemini)│  drafts answer + citations
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Verifier Agent    │  checks claims vs. context,
                        │                    │  flags unsupported statements
                        └─────────┬─────────┘
                                  │
                        ┌─────────▼─────────┐
                        │  Cache write +     │
                        │  Final Answer      │
                        └────────────────────┘
```

---

## 4. Feature Breakdown by Pipeline Stage

### 4.1 Data Ingestion & Preprocessing
- Download 10-K/10-Q/annual report PDFs from SEC EDGAR (free, bulk-accessible) for a curated set of ~10–20 companies across a couple of sectors (for good comparison questions)
- Custom preprocessing pipeline:
  - **Layout-aware PDF parsing** to separate narrative text, tables, and embedded charts/images (rather than one flat text dump)
  - **Table extraction** preserving row/column structure (so a "Total Revenue" row stays associated with its year columns)
  - **Chart/figure extraction** as images, passed to a multimodal embedder
  - **Text cleaning**: de-hyphenation, header/footer stripping, boilerplate legal-disclaimer removal
  - **Metadata tagging**: company name/ticker, fiscal year, filing type, section heading, content type

### 4.2 Chunking Strategy
- **Section-aware chunking** for narrative text (respect Item 1/1A/7/7A boundaries rather than fixed token windows crossing sections)
- **Table-as-unit chunking**: each table (or logical sub-table) is kept whole as one chunk with a generated text summary prefix (e.g. "Balance sheet excerpt, Company A, FY2023") so it's retrievable by both BM25 and semantic search
- **Sliding window with overlap** for long narrative sections (e.g. Risk Factors) to preserve context across chunk boundaries
- Chunk size tuned per content type — smaller for dense tables, larger for narrative prose

### 4.3 Hybrid Search + Metadata
- **BM25** index (e.g. via `rank_bm25` or Elasticsearch-lite) for exact term/line-item matching
- **Semantic search** via Voyage AI multimodal embeddings (text + table + chart embeddings in the same vector space) stored in Pinecone
- **Score fusion**: reciprocal rank fusion or weighted linear combination of BM25 + semantic scores
- **Metadata pre-filtering**: company/year/section filters applied before or alongside vector search to narrow the candidate pool
- **Reranker**: cross-encoder reranking pass on the fused top-K candidates before they reach the agent (e.g. a free/open reranker model, or Voyage's reranker if within free tier)

### 4.4 Agentic Layer
- **Router Agent**: classifies incoming query as direct-lookup, single-hop-semantic, or multi-hop-decompose
- **Decomposer Agent**: for multi-hop questions, generates an explicit list of sub-questions with dependency awareness (e.g., "get Company A's R&D spend FY2023" and "get Company A's revenue FY2023" before "compute R&D as % of revenue")
- **Sufficiency Check Agent**: structured-output check (`{"sufficient": bool, "missing": [...]}`) on retrieved context per sub-query; triggers a bounded retry loop (max 2–3 iterations) with reformulated queries if insufficient
- **Tool-Use Agent**: executes calculations (growth rates, ratios, CAGR) via a sandboxed calculator/code tool using numbers pulled from retrieved tables — never lets the LLM freehand arithmetic
- **Verifier Agent**: cross-checks the draft answer's factual claims against the retrieved evidence set; strips or flags claims it can't support

### 4.5 Generation
- Gemini (free tier) synthesizes the final answer from verified sub-answers, with inline citations back to source chunk (company, filing, page/section)

### 4.6 Caching
- Semantic cache (embedding-similarity match against prior queries) to catch near-duplicate questions, not just exact-string matches
- Cache invalidation tied to the underlying corpus version (so re-ingesting updated filings invalidates stale cached answers)

### 4.7 Evaluation Pipeline
**Retrieval-level**:
- Recall@K, Precision@K against a hand-labeled set of (query → relevant chunk IDs) pairs
- Impact of reranking measured as a before/after ablation

**Generation-level**:
- **Answer relevancy**: LLM-as-judge scoring of whether the answer addresses the actual question
- **Faithfulness/groundedness**: does every claim trace back to a retrieved chunk (leverages your Verifier Agent's output directly as a metric)
- **Answer correctness**: for numeric questions, exact-match or tolerance-based comparison against computed ground truth

**Fairness/consistency**:
- Paraphrase-invariance test: same underlying question asked in 3–5 different phrasings — measure answer consistency
- Cross-company coverage: check whether smaller/less-covered companies in your corpus get systematically worse retrieval quality than well-covered large-caps (a real, documented bias in financial NLP)

**Agentic-specific metrics**:
- Self-correction rate: % of sub-queries that triggered the sufficiency-check retry loop, and whether the retry actually improved the final retrieved set
- Decomposition accuracy: for a labeled subset of multi-hop questions, did the decomposer produce the correct sub-questions

---

## 5. Example Queries the System Should Handle

- *Direct lookup*: "What was Company A's total revenue in FY2023?"
- *Single computation*: "What was Company A's YoY revenue growth from FY2022 to FY2023?"
- *Multi-hop comparison*: "How does Company A's R&D spend as a percentage of revenue compare to Company B and Company C over the last 2 years?"
- *Qualitative + quantitative*: "What risk factors did Company A cite for FY2023, and did their debt levels increase that year?"
- *Trend/chart-based*: "Summarize the revenue trend shown in Company A's investor presentation chart for the last 3 years."

---

## 6. Tech Stack

| Layer | Tool |
|---|---|
| PDF parsing / preprocessing | `unstructured`, `pdfplumber`, or `PyMuPDF` for layout-aware extraction; custom table/figure separation logic |
| Embeddings | Voyage AI multimodal embeddings (free tier) |
| Vector DB | Pinecone (free serverless tier) |
| Keyword search | `rank_bm25` or lightweight Elasticsearch/OpenSearch |
| Reranker | Cross-encoder reranker (open-source or Voyage's reranker within free tier) |
| Agent orchestration | LangGraph (explicit state machine, supports the retry/loop pattern cleanly) |
| Generation LLM | Gemini Flash (free tier) |
| Caching | Redis (local) or simple SQLite-backed semantic cache for a personal project |
| Evaluation | Custom eval harness + LLM-as-judge (Gemini) for relevancy/faithfulness scoring; `ragas` library as an optional accelerant for standard retrieval/generation metrics |

---

## 7. Suggested Scope for a Personal Project

To keep this achievable while still showcasing everything:
- **Corpus**: 8–12 companies, 2–3 fiscal years each (roughly 20–35 filings) — comfortably within Pinecone's free-tier vector limits
- **Eval set**: hand-label 40–60 queries spanning direct lookup, computation, and multi-hop comparison, with ground-truth answers and relevant-chunk annotations
- **Build order**: ingestion/chunking → hybrid retrieval → baseline single-hop RAG (get this evaluated and working first) → add router/decomposer → add sufficiency loop → add tool-use → add verifier → add caching → full eval pass with ablations (with/without reranker, with/without self-correction, with/without verifier) to show measurable impact of each component

---

## 8. What Makes This a Strong Portfolio Project

- Every agentic component is *load-bearing* — decomposition, tool-use, and verification are necessary for the domain, not decorative
- Ablation studies (component on/off) give you concrete, defensible numbers to talk about rather than vague claims
- The fairness/consistency angle is something most RAG projects skip entirely
- Multimodal table/chart handling is a genuine technical challenge, not just "chunk the text"
