"""Record oracle-process provenance and validate a minted VN2 capture bundle."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from newcalibre.oracle.capture import THREAD_VARIABLES, validate_capture_bundle  # noqa: E402


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
    args = parser.parse_args()
    if args.command == "environment":
        return _record_environment(args.out)
    bundle = validate_capture_bundle(
        args.bundle,
        expected_candidate_sha=args.candidate_sha,
        expected_workflow_sha=args.workflow_sha,
        expected_run_id=args.run_id,
        expected_config_path=args.config,
        expected_input_inventory_path=args.input_inventory,
    )
    print(f"validated capture inner digest: {bundle.manifest.inner_bundle_digest}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
