"""
query/tools/_sandbox_worker.py — Subprocess worker for the sandboxed calculator.

NEVER import this module directly. It is only invoked as a subprocess by
``sandbox_executor.run_in_sandbox``. It reads a JSON payload from stdin,
executes the named function from financial_calculators, and writes the
JSON result (or error) to stdout.

Stdin payload:  {"function_name": str, "args": dict}
Stdout result:  {"result": float}   or   {"error": str}
"""

import json
import sys


def main() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        function_name: str = payload["function_name"]
        args: dict = payload["args"]
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": f"bad stdin payload: {exc}"}))
        sys.exit(1)

    # Import here (inside subprocess) so the allow-list is enforced at runtime
    from query.tools.financial_calculators import (
        yoy_growth, cagr, percent_of, operating_margin, debt_to_equity,
    )

    ALLOWED: dict = {
        "yoy_growth":       yoy_growth,
        "cagr":             cagr,
        "percent_of":       percent_of,
        "operating_margin": operating_margin,
        "debt_to_equity":   debt_to_equity,
    }

    fn = ALLOWED.get(function_name)
    if fn is None:
        sys.stdout.write(json.dumps({"error": f"function '{function_name}' not in allow-list"}))
        sys.exit(1)

    try:
        result = fn(**args)
        sys.stdout.write(json.dumps({"result": result}))
    except Exception as exc:
        sys.stdout.write(json.dumps({"error": str(exc)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
