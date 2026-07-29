"""
query/tools/sandbox_executor.py — Sandboxed subprocess execution for financial calculators.

Executes a single named function from ``financial_calculators.py`` in a
subprocess that has:
  - No network access (proxy/network env vars stripped)
  - No ambient file I/O (subprocess gets a clean /tmp working directory)
  - A hard timeout (default ``SANDBOX_TIMEOUT_SECONDS`` = 5s)

Inputs and outputs are exchanged as JSON over stdin/stdout using the
``_sandbox_worker.py`` entry point — never ``eval`` or ``exec``.

Only functions in ``ALLOWED_FUNCTIONS`` can be invoked; any other name raises
``ValueError`` before the subprocess is even spawned.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Callable

from query.config import SANDBOX_TIMEOUT_SECONDS
from query.tools.financial_calculators import (
    yoy_growth, cagr, percent_of, operating_margin, debt_to_equity,
)

logger = logging.getLogger(__name__)

# ── Explicit allow-list ───────────────────────────────────────────────────────

ALLOWED_FUNCTIONS: dict[str, Callable] = {
    "yoy_growth":       yoy_growth,
    "cagr":             cagr,
    "percent_of":       percent_of,
    "operating_margin": operating_margin,
    "debt_to_equity":   debt_to_equity,
}

# Network/proxy env vars to strip from the subprocess environment
_STRIP_ENV_VARS: frozenset[str] = frozenset({
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "no_proxy", "NO_PROXY", "ALL_PROXY", "all_proxy",
    "CURL_CA_BUNDLE", "REQUESTS_CA_BUNDLE",
})


def run_in_sandbox(
    function_name: str,
    args: dict,
    timeout_seconds: int | None = None,
) -> float:
    """
    Executes a named financial calculator function in an isolated subprocess.

    Steps:
    1. Validate ``function_name`` against ``ALLOWED_FUNCTIONS`` — raises
       ``ValueError`` immediately if not found (no subprocess spawned).
    2. Build a clean subprocess environment (strip network/proxy vars).
    3. Spawn ``python query/tools/_sandbox_worker.py`` with a ``/tmp``
       working directory and a hard timeout.
    4. Pass ``{"function_name": ..., "args": ...}`` as JSON over stdin.
    5. Parse the JSON result from stdout and return it as ``float``.

    Raises:
        ValueError:   If ``function_name`` is not in ``ALLOWED_FUNCTIONS``.
        TimeoutError: If the subprocess exceeds ``timeout_seconds``.
        RuntimeError: If the worker returns an error payload.

    Args:
        function_name:   One of the keys in ``ALLOWED_FUNCTIONS``.
        args:            Keyword arguments for the function as a dict.
        timeout_seconds: Hard timeout; defaults to ``SANDBOX_TIMEOUT_SECONDS``.

    Returns:
        ``float`` result of the computation.
    """
    if function_name not in ALLOWED_FUNCTIONS:
        raise ValueError(
            f"run_in_sandbox: '{function_name}' is not in the allow-list. "
            f"Permitted functions: {sorted(ALLOWED_FUNCTIONS)}"
        )

    _timeout = timeout_seconds if timeout_seconds is not None else SANDBOX_TIMEOUT_SECONDS

    # Build restricted environment (strip network/proxy vars but keep PYTHONPATH
    # so the subprocess can locate the `query` package from the project root)
    clean_env = {k: v for k, v in os.environ.items() if k not in _STRIP_ENV_VARS}
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    existing_pypath = clean_env.get("PYTHONPATH", "")
    clean_env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{existing_pypath}" if existing_pypath else project_root
    )

    payload = json.dumps({"function_name": function_name, "args": args})
    worker_path = os.path.join(os.path.dirname(__file__), "_sandbox_worker.py")

    try:
        proc = subprocess.run(
            [sys.executable, worker_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=_timeout,
            cwd=tempfile.gettempdir(),
            env=clean_env,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(
            f"run_in_sandbox: '{function_name}' exceeded timeout ({_timeout}s)"
        )

    stdout = proc.stdout.strip()
    if not stdout:
        stderr = proc.stderr.strip()
        raise RuntimeError(
            f"run_in_sandbox: worker produced no output "
            f"(exit={proc.returncode}). stderr: {stderr}"
        )

    try:
        result_dict = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"run_in_sandbox: could not parse worker stdout as JSON: {stdout!r}"
        ) from exc

    if "error" in result_dict:
        raise RuntimeError(
            f"run_in_sandbox: worker error for '{function_name}': {result_dict['error']}"
        )

    logger.debug(
        "run_in_sandbox: %s(%s) = %s", function_name, args, result_dict["result"]
    )
    return float(result_dict["result"])
