"""Emit the single-SHA Gate A technical report (KTD-A13, GA6).

Runs in the gate-a aggregate job at the C1 checkout after every lane job
succeeded. Records the candidate SHA, the digests of every committed
evidence surface, the C0→C1 merge-discipline check (the diff from the
promoted record's subject SHA to C1 must touch only allowlisted evidence
paths), and the budget check against the Gate issue's immutable activation
record. The owner decision is a separate timestamped record binding this
report's digest; this script never writes to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from stage3_clock import find_activation_record, record_is_schema_complete

EVIDENCE_ALLOWLIST = ("stage3/evidence/",)
TRACKING_SERIES = Path("stage3/evidence/tracking/series.jsonl")
INPUT_INVENTORY = Path("stage3/evidence/vn2-input-digests.json")
CAPTURES_DIR = Path("stage3/evidence/captures")


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Run a git command and return stripped stdout."""
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> int:
    """Assemble and write the Gate report; nonzero on a failed discipline check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate = os.environ["CANDIDATE_SHA"]
    problems: list[str] = []

    # Tracking record 1 (Gate precondition) and the C0→C1 merge discipline.
    c0 = None
    record_digest = None
    if TRACKING_SERIES.exists():
        lines = [ln for ln in TRACKING_SERIES.read_text(encoding="utf-8").splitlines() if ln]
        if lines:
            record_digest = hashlib.sha256((lines[-1] + "\n").encode("utf-8")).hexdigest()
            try:
                c0 = json.loads(lines[-1]).get("subject_sha")
            except json.JSONDecodeError:
                problems.append("tracking series last line is not valid JSON")
    else:
        problems.append("gate precondition unmet: no promoted tracking record (U9b)")

    diff_check = None
    if c0:
        changed = git("diff", "--name-only", f"{c0}..HEAD").splitlines()
        offending = [f for f in changed if not f.startswith(EVIDENCE_ALLOWLIST)]
        diff_check = {"c0": c0, "changed": changed, "offending": offending}
        if offending:
            problems.append(
                "C0→C1 diff leaves the evidence-path allowlist; "
                "candidate void — re-mint at a new C0"
            )

    # Budget check against the immutable activation record.
    now = datetime.now(UTC)
    budget = None
    record = find_activation_record()
    if record_is_schema_complete(record):
        deadline = datetime.fromisoformat(str(record["deadline"]).replace("Z", "+00:00"))
        budget = {
            "deadline": record["deadline"],
            "evaluated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inside_budget": now <= deadline,
        }
        if now > deadline:
            problems.append("evidence completed after the immutable deadline")
    else:
        problems.append("no schema-complete activation record on the Gate issue")

    report = {
        "candidate_sha": candidate,
        "head_sha": git("rev-parse", "HEAD"),
        "workflow_sha": os.environ.get("GITHUB_WORKFLOW_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_url": (
            f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}"
            f"/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
        ),
        "emitted_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "arch": platform.machine(),
            "os_release": platform.freedesktop_os_release().get("PRETTY_NAME", "unknown"),
            "python": platform.python_version(),
            "runner_image": (
                f"{os.environ.get('ImageOS', '?')}/{os.environ.get('ImageVersion', '?')}"  # noqa: SIM112
            ),
        },
        "digests": {
            "input_inventory": (sha256_file(INPUT_INVENTORY) if INPUT_INVENTORY.exists() else None),
            "capture_manifests": {
                p.relative_to(CAPTURES_DIR).as_posix(): sha256_file(p)
                for p in sorted(CAPTURES_DIR.rglob("manifest.json"))
            }
            if CAPTURES_DIR.exists()
            else {},
            "tracking_record": record_digest,
        },
        "c0_c1_discipline": diff_check,
        "budget": budget,
        "problems": problems,
        "lanes_note": (
            "this aggregate job runs only after lint/unit-isolated/consistency/"
            "oracle/vn2-verify all succeeded at this same candidate SHA"
        ),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_digest = sha256_file(args.out)
    print(f"gate report digest: {report_digest}")
    print(json.dumps({"problems": problems}, indent=2))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
