"""Compute a CI verdict over the GitHub check-runs API and pull failed-run logs.

``verdict <sha>`` fetches ``gh api repos/<repo>/commits/<sha>/check-runs`` and
reduces it to one verdict:

- **pending** — any run with ``status != "completed"``.
- **green** — every run completed with a passing conclusion (``success``, or
  the non-blocking ``skipped``/``neutral`` that conditional jobs produce).
- **failure** — any completed run with another (or unrecognized) conclusion;
  failed workflow-run IDs are parsed from each ``details_url`` and a stable
  failure signature is printed so a repeated, unchanged failure is a string
  comparison rather than a judgment call.
- **non-verdict** — empty check set, malformed JSON, a truncated payload
  (``total_count`` disagreeing with the returned list — a red check beyond the
  page limit must never read as green), a failed ``gh`` call, or any crash in
  the command itself. A non-verdict is never green.

Exit codes: 0 green, 1 pending, 2 failure, 3 non-verdict.

``logs <run-id>`` wraps ``gh run view <run-id> --log-failed``.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys

REPO = "Vzlentin/calibre"

EXIT_GREEN = 0
EXIT_PENDING = 1
EXIT_FAILURE = 2
EXIT_NON_VERDICT = 3

# Conditional jobs conclude as skipped/neutral on runs where they don't apply;
# GitHub treats both as non-blocking, so they must not read as red.
PASSING_CONCLUSIONS = frozenset({"success", "skipped", "neutral"})

_RUN_ID_RE = re.compile(r"/runs/(\d+)")


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing text output without raising on failure."""
    return subprocess.run(args, capture_output=True, text=True, check=False)


def parse_run_id(details_url: str) -> str | None:
    """Extract the workflow-run ID from a check run's ``details_url``."""
    match = _RUN_ID_RE.search(details_url)
    return match.group(1) if match else None


def failure_signature(failed_runs: list[dict]) -> str:
    """Compute a stable signature for a set of failed check runs.

    The signature is a short hash over the sorted failed check names and the
    first line of each check's output title/summary, so an identical failure
    recurring across fix iterations produces an identical string.
    """
    parts = []
    for run in sorted(failed_runs, key=lambda r: str(r.get("name", ""))):
        output = run.get("output") or {}
        lines = str(output.get("title") or output.get("summary") or "").splitlines()
        first_line = lines[0] if lines else ""
        parts.append(f"{run.get('name', '')}|{run.get('conclusion', '')}|{first_line}")
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:12]
    return digest


def evaluate(payload_text: str) -> tuple[int, dict]:
    """Reduce a raw check-runs API payload to a verdict.

    Args:
        payload_text: The JSON text returned by the check-runs API.

    Returns:
        A ``(exit_code, report)`` pair; ``report`` carries ``verdict`` plus,
        on failure, ``failed`` entries (name, conclusion, run_id) and a
        ``signature``.
    """
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        return EXIT_NON_VERDICT, {"verdict": "non-verdict", "reason": f"malformed payload: {exc}"}
    check_runs = payload.get("check_runs") if isinstance(payload, dict) else None
    if not isinstance(check_runs, list):
        return EXIT_NON_VERDICT, {
            "verdict": "non-verdict",
            "reason": "payload carries no check_runs list",
        }

    if not check_runs:
        return EXIT_NON_VERDICT, {"verdict": "non-verdict", "reason": "empty check set"}

    if any(not isinstance(run, dict) for run in check_runs):
        return EXIT_NON_VERDICT, {
            "verdict": "non-verdict",
            "reason": "malformed check_runs entry (not an object)",
        }

    total_count = payload.get("total_count")
    if isinstance(total_count, int) and total_count != len(check_runs):
        return EXIT_NON_VERDICT, {
            "verdict": "non-verdict",
            "reason": (
                f"truncated check-runs payload: total_count={total_count}, "
                f"received {len(check_runs)}"
            ),
        }

    if any(run.get("status") != "completed" for run in check_runs):
        return EXIT_PENDING, {"verdict": "pending"}

    failed = [run for run in check_runs if run.get("conclusion") not in PASSING_CONCLUSIONS]
    if not failed:
        return EXIT_GREEN, {"verdict": "green"}

    return EXIT_FAILURE, {
        "verdict": "failure",
        "failed": [
            {
                "name": run.get("name", ""),
                "conclusion": run.get("conclusion", ""),
                "run_id": parse_run_id(str(run.get("details_url", ""))),
            }
            for run in failed
        ],
        "signature": failure_signature(failed),
    }


def cmd_verdict(sha: str) -> int:
    """Fetch check-runs for a commit and print the verdict report as JSON."""
    proc = _run(["gh", "api", f"repos/{REPO}/commits/{sha}/check-runs?per_page=100"])
    if proc.returncode != 0:
        report = {"verdict": "non-verdict", "reason": f"gh api failed: {proc.stderr.strip()}"}
        print(json.dumps(report, indent=2))
        return EXIT_NON_VERDICT
    code, report = evaluate(proc.stdout)
    print(json.dumps(report, indent=2))
    return code


def cmd_logs(run_id: str) -> int:
    """Stream the failed-step logs of one workflow run."""
    return subprocess.run(["gh", "run", "view", run_id, "--log-failed"], check=False).returncode


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a CI command."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_verdict = sub.add_parser("verdict", help="print the CI verdict for a commit SHA")
    p_verdict.add_argument("sha")

    p_logs = sub.add_parser("logs", help="pull failed logs for a workflow run")
    p_logs.add_argument("run_id")

    args = parser.parse_args(argv)
    # A crash must not alias a verdict: unhandled exceptions would exit 1
    # (= pending, which the CI loop would poll forever). Fail to non-verdict.
    try:
        if args.command == "verdict":
            return cmd_verdict(args.sha)
        return cmd_logs(args.run_id)
    except Exception as exc:  # noqa: BLE001 — exit-code contract over traceback
        print(
            json.dumps({"verdict": "non-verdict", "reason": f"crash: {exc!r}"}, indent=2),
            file=sys.stderr,
        )
        return EXIT_NON_VERDICT


if __name__ == "__main__":
    sys.exit(main())
