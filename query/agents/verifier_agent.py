"""
query/agents/verifier_agent.py — Verifier Agent (Phase 7).

Independently re-derives the final answer by checking every factual claim in
the Generator's draft against the retrieved evidence. Flags or removes any
claim not traceable to a retrieved chunk.

ONE call — the last Gemini call in the pipeline per query (per §5 budget math).
"""

from __future__ import annotations

import logging

from query.utils.gemini_client import GeminiCallCounter, call_structured
from query.utils.prompts import VERIFIER_SYSTEM_PROMPT, build_verifier_prompt
from query.utils.schema import RetrievedChunk, VerifierResult

logger = logging.getLogger(__name__)

# ── Gemini response schema ────────────────────────────────────────────────────

VERIFIER_RESPONSE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["verified", "partial"],
        },
        "flagged_claims": {
            "type": "array",
            "items": {"type": "string"},
        },
        "final_answer": {"type": "string"},
    },
    "required": ["verdict", "flagged_claims", "final_answer"],
}


# ── Public API ────────────────────────────────────────────────────────────────

def verify_answer(
    draft_answer: str,
    retrieved_chunks: dict[int, list[RetrievedChunk]],
    gemini_client,
    call_counter: GeminiCallCounter,
) -> VerifierResult:
    """
    Verifies the Generator's draft against all retrieved evidence.

    Steps:
    1. Flatten ``retrieved_chunks`` values into one ``list[RetrievedChunk]``.
    2. Build the verifier prompt via
       :func:`~query.utils.prompts.build_verifier_prompt`.
    3. Call Gemini with ``VERIFIER_RESPONSE_SCHEMA`` structured output.
    4. Return :class:`~query.utils.schema.VerifierResult`.

    Gemini is instructed to independently re-derive ``final_answer`` with any
    unsupported claims either removed or annotated ``[UNVERIFIED]`` — the
    pipeline trusts this ``final_answer`` over the raw draft.

    ONE call — the last Gemini call in the pipeline per query.

    Args:
        draft_answer:     Draft text from :func:`~query.agents.generator_agent.generate_answer`.
        retrieved_chunks: ``dict[sub_query_id, list[RetrievedChunk]]``.
        gemini_client:    Initialised ``google.genai.Client``.
        call_counter:     Shared :class:`~query.utils.gemini_client.GeminiCallCounter`.

    Returns:
        :class:`~query.utils.schema.VerifierResult` with ``verdict``,
        ``flagged_claims``, and ``final_answer``.
    """
    all_evidence: list[RetrievedChunk] = [
        rc for chunks in retrieved_chunks.values() for rc in chunks
    ]
    prompt = build_verifier_prompt(draft_answer, all_evidence)
    result = call_structured(
        gemini_client,
        VERIFIER_SYSTEM_PROMPT,
        prompt,
        VERIFIER_RESPONSE_SCHEMA,
        call_counter,
    )
    verdict = result.get("verdict", "partial")
    flagged_claims = list(result.get("flagged_claims", []))
    final_answer = result.get("final_answer", draft_answer)

    logger.info(
        "verify_answer: verdict=%s, flagged_claims=%d, final_answer_len=%d",
        verdict, len(flagged_claims), len(final_answer),
    )
    return VerifierResult(
        verdict=verdict,
        flagged_claims=flagged_claims,
        final_answer=final_answer,
    )


def strip_flagged_claims(draft_answer: str, flagged_claims: list[str]) -> str:
    """
    Fallback safety net: removes each flagged claim substring from the draft
    if ``verifier_result.final_answer`` is empty or malformed.

    This is NOT the primary path (``VerifierResult.final_answer`` from Gemini
    is primary). This exists so the pipeline never crashes or ships an
    unverified claim if the Verifier's structured output is incomplete.

    Performs a simple substring removal for each item in ``flagged_claims``.
    Strips any resulting double-spaces/newlines.

    Args:
        draft_answer:   Raw draft from the Generator.
        flagged_claims: Flagged claim strings from :func:`verify_answer`.

    Returns:
        Draft with flagged claim substrings removed.
    """
    result = draft_answer
    for claim in flagged_claims:
        if claim and claim in result:
            result = result.replace(claim, "").strip()
    # Normalise double spaces/newlines left by removals
    import re
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"  +", " ", result)
    return result.strip()
