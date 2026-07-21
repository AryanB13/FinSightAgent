"""
query/utils/gemini_client.py — Centralised Gemini API wrapper.

All Gemini calls in the query pipeline go through this module — the
query-pipeline equivalent of ``ingestion.indexing.voyage_embedder``.
Retry logic, call counting, and daily quota warnings stay here so agent
modules never touch the raw SDK directly.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date
from typing import Optional

from google import genai
from google.genai import types as genai_types

from query.config import (
    GEMINI_MODEL,
    GEMINI_MAX_RETRIES,
    GEMINI_RETRY_BACKOFF_BASE,
    GEMINI_DAILY_QUOTA_WARNING,
)

logger = logging.getLogger(__name__)


# ── Call counter ───────────────────────────────────────────────────────────────

class GeminiCallCounter:
    """
    Tracks Gemini calls made during the current process lifetime and
    optionally persists a daily count to Redis so the counter survives
    across process restarts within the same day.

    If no ``redis_client`` is provided, falls back to an in-memory counter
    that resets when the process exits.
    """

    def __init__(self, redis_client=None) -> None:
        self._redis = redis_client
        self._today: str = str(date.today())
        self._in_memory_count: int = 0

    def _redis_key(self) -> str:
        return f"gemini_calls:{self._today}"

    def increment(self) -> None:
        """Increment the call counter by 1."""
        self._in_memory_count += 1
        if self._redis is not None:
            try:
                self._redis.incr(self._redis_key())
            except Exception:
                pass  # Redis failure must never crash a query

    def count_today(self) -> int:
        """
        Returns today's call count. Resets if the date has changed.
        Reads from Redis if available, otherwise from in-memory counter.
        """
        self.reset_if_new_day()
        if self._redis is not None:
            try:
                raw = self._redis.get(self._redis_key())
                return int(raw) if raw is not None else self._in_memory_count
            except Exception:
                pass
        return self._in_memory_count

    def reset_if_new_day(self) -> None:
        """Resets the in-memory counter when the calendar date has changed."""
        today = str(date.today())
        if today != self._today:
            self._today = today
            self._in_memory_count = 0


# ── Client init ────────────────────────────────────────────────────────────────

def init_gemini_client(api_key: str) -> genai.Client:
    """
    Instantiates and returns a ``google.genai.Client`` configured with
    ``api_key`` and bound to ``GEMINI_MODEL``.

    Instantiated once in ``pipeline.py`` and passed to every agent function.

    Args:
        api_key: Gemini API key from ``GEMINI_API_KEY`` env var.

    Raises:
        EnvironmentError: If ``api_key`` is empty or None.

    Returns:
        Configured ``genai.Client`` instance.
    """
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Export it before running the query pipeline."
        )
    client = genai.Client(api_key=api_key)
    logger.info("init_gemini_client: using model '%s'.", GEMINI_MODEL)
    return client


# ── Internal retry helper ──────────────────────────────────────────────────────

def _is_rate_limit_error(exc: Exception) -> bool:
    """Returns True if the exception looks like a Gemini 429 / quota error."""
    exc_str = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    return any(kw in exc_str or kw in exc_name for kw in (
        "ratelimit", "rate_limit", "resourceexhausted", "429", "quota"
    ))


def _check_quota_warning(call_counter: GeminiCallCounter) -> None:
    today = call_counter.count_today()
    if today > GEMINI_DAILY_QUOTA_WARNING:
        logger.warning(
            "gemini_client: daily call count is %d — approaching free-tier limit of ~250.",
            today,
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def call_structured(
    model: genai.Client,
    system_prompt: str,
    user_content: str,
    response_schema: dict,
    call_counter: GeminiCallCounter,
) -> dict:
    """
    Makes one Gemini call with structured JSON output enforced via
    ``generation_config`` ``response_schema`` + ``response_mime_type``.

    Steps:
    1. Increment ``call_counter``; warn if daily quota threshold crossed.
    2. Call ``client.models.generate_content`` with the structured config.
    3. ``json.loads(response.text)`` and return the dict.
    4. Retry up to ``GEMINI_MAX_RETRIES`` on rate-limit errors with
       exponential backoff (``GEMINI_RETRY_BACKOFF_BASE × 2^attempt``).

    Args:
        model:           Initialised ``genai.Client``.
        system_prompt:   Agent system instructions.
        user_content:    User-turn content (query / context block).
        response_schema: JSON schema dict for ``response_schema`` param.
        call_counter:    Shared ``GeminiCallCounter`` instance.

    Returns:
        Parsed JSON dict from Gemini's response.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=response_schema,
    )

    last_exc: Optional[Exception] = None
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            call_counter.increment()
            _check_quota_warning(call_counter)

            response = model.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_content,
                config=config,
            )
            return json.loads(response.text)

        except Exception as exc:
            if _is_rate_limit_error(exc):
                sleep_s = GEMINI_RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "call_structured: rate-limited (attempt %d/%d), sleeping %.1fs.",
                    attempt + 1, GEMINI_MAX_RETRIES, sleep_s,
                )
                time.sleep(sleep_s)
                last_exc = exc
            else:
                raise

    raise RuntimeError(
        f"call_structured: failed after {GEMINI_MAX_RETRIES} retries."
    ) from last_exc


def call_freeform(
    model: genai.Client,
    system_prompt: str,
    user_content: str,
    call_counter: GeminiCallCounter,
) -> str:
    """
    Same as ``call_structured`` but without a ``response_schema`` — used by
    the Generator Agent which produces free-text prose with inline citations
    rather than strict JSON.

    Same retry/backoff/counting behaviour as ``call_structured``.

    Args:
        model:         Initialised ``genai.Client``.
        system_prompt: Agent system instructions.
        user_content:  User-turn content.
        call_counter:  Shared ``GeminiCallCounter`` instance.

    Returns:
        Raw response text string.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt,
    )

    last_exc: Optional[Exception] = None
    for attempt in range(GEMINI_MAX_RETRIES):
        try:
            call_counter.increment()
            _check_quota_warning(call_counter)

            response = model.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_content,
                config=config,
            )
            return response.text

        except Exception as exc:
            if _is_rate_limit_error(exc):
                sleep_s = GEMINI_RETRY_BACKOFF_BASE * (2 ** attempt)
                logger.warning(
                    "call_freeform: rate-limited (attempt %d/%d), sleeping %.1fs.",
                    attempt + 1, GEMINI_MAX_RETRIES, sleep_s,
                )
                time.sleep(sleep_s)
                last_exc = exc
            else:
                raise

    raise RuntimeError(
        f"call_freeform: failed after {GEMINI_MAX_RETRIES} retries."
    ) from last_exc
