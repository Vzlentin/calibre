"""Record oracle-process provenance and validate a minted VN2 capture bundle."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from newcalibre.oracle.capture import (  # noqa: E402
    GITHUB_REPOSITORY,
    THREAD_VARIABLES,
    validate_capture_bundle,
    validate_promoted_capture,
)


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("model name") and ":" in line:
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
    return platform.processor() or platform.machine()


def _environment() -> dict[str, object]:
    import numpy as np

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    numpy_config = output.getvalue().strip()
    if not numpy_config:
        raise SystemExit("NumPy/BLAS configuration is unavailable")
    thread_policy = {name: os.environ.get(name, "") for name in THREAD_VARIABLES}
    if set(thread_policy.values()) != {"1"}:
        raise SystemExit("capture workflow must explicitly pin every thread variable to 1")
    os_release = platform.freedesktop_os_release()
    return {
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "numpy": np.__version__,
        "numpy_config": numpy_config,
        "os": {
            "id": os_release.get("ID", "unknown"),
            "pretty_name": os_release.get("PRETTY_NAME", "unknown"),
            "version_id": os_release.get("VERSION_ID", "unknown"),
        },
        "python": platform.python_version(),
        "runner_image": (
            f"{os.environ.get('ImageOS', '?')}/{os.environ.get('ImageVersion', '?')}"  # noqa: SIM112
        ),
        "thread_policy": thread_policy,
    }


def _record_environment(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_environment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"recorded oracle environment -> {path}")  # noqa: T201
    return 0


def _github_json(endpoint: str) -> object:
    completed = subprocess.run(
        ("gh", "api", endpoint),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"GitHub metadata lookup failed: {completed.stderr.strip()}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise SystemExit("GitHub metadata was not valid JSON") from error


def _github_promotion_metadata(receipt_path: Path, *, repository: str) -> tuple[object, object]:
    if repository != GITHUB_REPOSITORY:
        raise SystemExit(f"promotion repository must equal {GITHUB_REPOSITORY}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        artifact_id = receipt["artifact_id"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit("capture receipt does not expose an artifact_id") from error
    if not isinstance(artifact_id, str) or re.fullmatch(r"[1-9][0-9]*", artifact_id) is None:
        raise SystemExit("capture receipt artifact_id must be a positive decimal string")
    artifact = _github_json(f"repos/{repository}/actions/artifacts/{artifact_id}")
    if not isinstance(artifact, dict):
        raise SystemExit("GitHub artifact metadata must be an object")
    workflow_run = artifact.get("workflow_run")
    if not isinstance(workflow_run, dict):
        raise SystemExit("GitHub artifact metadata does not bind a workflow run")
    run_id = workflow_run.get("id")
    if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id < 1:
        raise SystemExit("GitHub artifact workflow run ID must be a positive integer")
    run = _github_json(f"repos/{repository}/actions/runs/{run_id}")
    return artifact, run


def main() -> int:
    """Dispatch provenance recording or strict post-mint validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    environment = commands.add_parser("environment")
    environment.add_argument("--out", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--bundle", type=Path, required=True)
    validate.add_argument("--candidate-sha", required=True)
    validate.add_argument("--workflow-sha", required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--input-inventory", type=Path, required=True)
    promote = commands.add_parser("promote")
    promote.add_argument("--bundle", type=Path, required=True)
    promote.add_argument("--receipt", type=Path, required=True)
    promote.add_argument("--config", type=Path, required=True)
    promote.add_argument("--input-inventory", type=Path, required=True)
    promote.add_argument("--repository", required=True)
    args = parser.parse_args()
    if args.command == "environment":
        return _record_environment(args.out)
    if args.command == "validate":
        bundle = validate_capture_bundle(
            args.bundle,
            expected_candidate_sha=args.candidate_sha,
            expected_workflow_sha=args.workflow_sha,
            expected_run_id=args.run_id,
            expected_config_path=args.config,
            expected_input_inventory_path=args.input_inventory,
        )
    else:
        artifact_metadata, run_metadata = _github_promotion_metadata(
            args.receipt,
            repository=args.repository,
        )
        bundle, _receipt = validate_promoted_capture(
            args.bundle,
            args.receipt,
            artifact_metadata=artifact_metadata,
            run_metadata=run_metadata,
            expected_config_path=args.config,
            expected_input_inventory_path=args.input_inventory,
        )
    print(f"validated capture inner digest: {bundle.manifest.inner_bundle_digest}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
