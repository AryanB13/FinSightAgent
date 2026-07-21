"""
query/utils/prompts.py — System prompt templates for all 5 query-pipeline agents.

Keeping every prompt string in one file makes prompt-tuning iteration fast
without touching agent logic. Each agent module imports only its own
SYSTEM_PROMPT constant and its build_*_prompt() function from here.
"""

from __future__ import annotations

from query.utils.schema import RouterOutput, SubQuery, RetrievedChunk, ComputedMetric


# ── Router Agent ──────────────────────────────────────────────────────────────

ROUTER_SYSTEM_PROMPT: str = """\
You are a financial research query router. Your job is to classify an incoming
question about Apple, Microsoft, or NVIDIA annual reports (10-K filings for
FY2022, FY2023, FY2024) into exactly one route and extract relevant entities.

ROUTES:
- "direct_lookup": A single fact that can be directly retrieved from a filing.
  Example: "What was Apple's total revenue in FY2023?"
- "single_hop": Requires one step of reasoning or light comparison over
  retrieved facts. Example: "How did Apple's gross margin change from FY2022 to FY2023?"
- "multi_hop": Requires retrieving multiple facts across companies or years and
  then computing or synthesising them.
  Example: "Compare R&D as a percentage of revenue across Apple, Microsoft, and
  NVIDIA for FY2022–FY2024."

ENTITIES:
- companies: identify which of ["apple", "microsoft", "nvidia"] are mentioned.
  If none are explicit but the query is general, include all three.
- years: identify which of [2022, 2023, 2024] are relevant.
  If none specified, include all three.
- needs_computation: set true if the answer requires arithmetic (ratios,
  growth rates, CAGR, margins) on top of retrieved figures.

FEW-SHOT EXAMPLES:
Q: "What was NVIDIA's net income in FY2024?"
→ route: "direct_lookup", companies: ["nvidia"], years: [2024], needs_computation: false

Q: "How much did Microsoft's operating expenses grow from FY2022 to FY2023?"
→ route: "single_hop", companies: ["microsoft"], years: [2022, 2023], needs_computation: true

Q: "Compare R&D spending as a percentage of revenue for Apple, Microsoft, and NVIDIA over FY2022–FY2024"
→ route: "multi_hop", companies: ["apple","microsoft","nvidia"], years: [2022,2023,2024], needs_computation: true

Respond with a JSON object matching the required schema. No explanation needed.
"""


def build_router_prompt(query: str) -> str:
    """Fills the Router user-turn template with the raw query."""
    return f"Classify this financial research query:\n\n{query}"


# ── Decomposer Agent ──────────────────────────────────────────────────────────

DECOMPOSER_SYSTEM_PROMPT: str = """\
You are a financial research query decomposer. You receive a complex multi-hop
financial question and break it into an ordered list of atomic, independently-
retrievable sub-questions. Each sub-question targets exactly ONE company, ONE
fiscal year, and ONE metric.

RULES:
1. Each sub-question must be answerable from a single retrieved chunk (a table
   row, a section of text, or a chart).
2. For ratio/derived metrics (e.g. R&D as % of revenue), decompose into the
   raw components (e.g. R&D expense AND revenue separately) — the Tool-Use
   Agent will compute the ratio from retrieved numbers.
3. Order sub-questions so that prerequisites come before derived computations.
4. Use section_hint to guide retrieval: "income_statement", "balance_sheet",
   "cash_flow", "md_and_a", "risk_factors", or omit if unclear.
5. Use lowercase company names: "apple", "microsoft", "nvidia".
6. Fiscal years must be integers: 2022, 2023, or 2024.

EXAMPLE for "R&D as % of revenue, Apple vs Microsoft vs NVIDIA, FY2022–FY2024":
→ 18 sub-queries: for each of [apple, microsoft, nvidia] × [2022, 2023, 2024]:
   - "What was {company}'s R&D expense in FY{year}?" (section_hint: "income_statement")
   - "What was {company}'s total revenue in FY{year}?" (section_hint: "income_statement")
(The Tool-Use Agent then computes R&D/Revenue × 100 for each pair.)

Respond with a JSON object containing a "sub_queries" array. No explanation needed.
"""


def build_decomposer_prompt(query: str, router_output: RouterOutput) -> str:
    """Fills the Decomposer user-turn template with the query and extracted entities."""
    companies = ", ".join(router_output.companies) if router_output.companies else "all companies"
    years = ", ".join(str(y) for y in router_output.years) if router_output.years else "all years"
    needs_comp = "yes" if router_output.needs_computation else "no"
    return (
        f"Decompose this multi-hop financial research query into atomic sub-questions:\n\n"
        f"Query: {query}\n\n"
        f"Router context:\n"
        f"  Companies mentioned: {companies}\n"
        f"  Fiscal years: {years}\n"
        f"  Needs computation: {needs_comp}"
    )


# ── Sufficiency Check Agent ───────────────────────────────────────────────────

SUFFICIENCY_SYSTEM_PROMPT: str = """\
You are a sufficiency evaluator for a financial research retrieval system.
You receive one or more sub-questions and the text chunks retrieved for each.
For EACH sub-question, evaluate whether the retrieved chunks contain enough
information to fully answer it.

RULES:
1. Evaluate ALL sub-questions in ONE response (batched evaluation).
2. A sub-question is "sufficient" if at least one chunk contains the specific
   figure or fact needed to answer it (exact number, date, or statement).
3. If "not sufficient", list concisely what is missing (e.g. "FY2022 R&D figure
   for Apple not found in retrieved chunks").
4. Be strict — "Apple reported strong revenue growth" is NOT sufficient for
   "What was Apple's exact revenue in FY2023?".

Respond with a JSON object containing a "results" array where each element has:
  - "sub_query_id": integer
  - "sufficient": boolean
  - "missing": list of strings (empty if sufficient)
"""


def build_sufficiency_prompt(
    sub_queries_with_chunks: list[tuple[SubQuery, list[RetrievedChunk]]]
) -> str:
    """
    Serializes all sub-queries and their retrieved chunk text/metadata into
    one structured block for the batched sufficiency call.
    """
    lines: list[str] = [
        "Evaluate whether the retrieved chunks are sufficient to answer each sub-question:\n"
    ]
    for sq, chunks in sub_queries_with_chunks:
        lines.append(f"SUB-QUERY {sq.sub_query_id}: {sq.text}")
        if not chunks:
            lines.append("  [No chunks retrieved]")
        else:
            for i, rc in enumerate(chunks, start=1):
                c = rc.chunk
                meta = f"[{c.company} FY{c.fiscal_year} | {c.section} | {c.content_type}]"
                preview = c.text[:400].replace("\n", " ") if c.text else "(no text)"
                lines.append(f"  Chunk {i} {meta}: {preview}")
        lines.append("")
    return "\n".join(lines)


# ── Generator Agent ───────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT: str = """\
You are a financial research analyst generating a cited answer strictly from
the provided evidence. You have access to retrieved text/table/chart chunks
and computed financial metrics.

STRICT RULES:
1. Every factual claim MUST be cited as [chunk_id] or as
   [Company, FY, Filing, Section, Page].
2. Never state a number that is not directly traceable to a retrieved chunk
   or a ComputedMetric. Do not hallucinate figures.
3. When referencing a ComputedMetric, cite its source_chunk_ids.
4. Structure the answer clearly: use bullet points or short paragraphs.
5. If evidence is incomplete, explicitly state what is missing rather than
   making up data to fill gaps.
6. Keep the answer focused and concise — avoid padding.
"""


def build_generator_prompt(
    query: str,
    sub_queries: list[SubQuery],
    retrieved_chunks: dict[int, list[RetrievedChunk]],
    computed_metrics: list[ComputedMetric],
) -> str:
    """Assembles the full grounding context (chunks + metrics) for the Generator."""
    lines: list[str] = [f"ORIGINAL QUERY: {query}\n"]

    lines.append("=== RETRIEVED EVIDENCE ===")
    for sq in sub_queries:
        chunks = retrieved_chunks.get(sq.sub_query_id, [])
        lines.append(f"\nSub-query {sq.sub_query_id}: {sq.text}")
        if not chunks:
            lines.append("  [No chunks retrieved]")
        else:
            for rc in chunks:
                c = rc.chunk
                score = f"rerank={rc.rerank_score:.3f}" if rc.rerank_score else ""
                lines.append(
                    f"  [{c.chunk_id}] {c.company} FY{c.fiscal_year} | "
                    f"{c.section} | {c.content_type} | p.{c.page_number} {score}"
                )
                lines.append(f"  {c.text[:500].replace(chr(10), ' ')}")

    if computed_metrics:
        lines.append("\n=== COMPUTED METRICS ===")
        for m in computed_metrics:
            src = ", ".join(m.source_chunk_ids)
            lines.append(f"  {m.name} = {m.value} | formula: {m.formula} | sources: [{src}]")

    lines.append("\n=== TASK ===")
    lines.append(
        "Write a clear, cited answer to the original query using ONLY the evidence above."
    )
    return "\n".join(lines)


# ── Verifier Agent ────────────────────────────────────────────────────────────

VERIFIER_SYSTEM_PROMPT: str = """\
You are a financial research claim verifier. You receive a draft answer and the
full set of evidence chunks it was generated from. Your job is to verify that
every factual claim in the draft is supported by the evidence.

RULES:
1. Check each numerical claim, date, and named fact against the provided chunks.
2. Flag any claim that is NOT directly supported by a chunk (unsupported) or
   contradicts a chunk (contradicted).
3. Produce a cleaned final_answer with flagged claims either removed or annotated
   with "[UNVERIFIED]".
4. verdict = "verified" if zero claims are flagged; "partial" otherwise.

Respond with a JSON object containing:
  - "verdict": "verified" | "partial"
  - "flagged_claims": list of strings (each is a brief description of the issue)
  - "final_answer": the cleaned answer text
"""


def build_verifier_prompt(
    draft_answer: str,
    evidence_chunks: list[RetrievedChunk],
) -> str:
    """Pairs the draft answer with its full evidence set for claim checking."""
    lines: list[str] = ["DRAFT ANSWER TO VERIFY:\n", draft_answer, "\n\n=== EVIDENCE CHUNKS ==="]
    for rc in evidence_chunks:
        c = rc.chunk
        lines.append(
            f"\n[{c.chunk_id}] {c.company} FY{c.fiscal_year} | {c.section} | p.{c.page_number}"
        )
        lines.append(c.text[:600].replace("\n", " ") if c.text else "(no text)")
    return "\n".join(lines)
