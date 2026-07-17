"""Build compact VN2 tracking proposals or validate an exact history append."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newcalibre.oracle import OracleEvidenceError, load_capture  # noqa: E402
from newcalibre.protocols.vn2 import (  # noqa: E402
    VN2ResultError,
    build_tracking_record,
    load_result_bundle,
)
from newcalibre.protocols.vn2.tracking import (  # noqa: E402
    TrackingError,
    validate_tracking_append,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the compact tracking CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--result-root", type=Path, required=True)
    build.add_argument("--candidate-sha", required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--input-inventory", type=Path, required=True)
    build.add_argument("--lockfile", type=Path, required=True)
    build.add_argument("--capture", type=Path, required=True)
    build.add_argument("--oracle-config", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    append = commands.add_parser("validate-append")
    append.add_argument("--base", type=Path, required=True)
    append.add_argument("--head", type=Path, required=True)
    return parser


def main() -> int:
    """Dispatch proposal construction or trusted-base append validation."""
    args = build_parser().parse_args()
    try:
        if args.command == "validate-append":
            appended = validate_tracking_append(args.base, args.head)
            print(f"validated exact VN2 tracking append: {len(appended)} record(s)")  # noqa: T201
            return 0

        capture = load_capture(
            args.capture,
            config_path=args.oracle_config,
            input_inventory_path=args.input_inventory,
        )
        bundle = load_result_bundle(
            args.result_root,
            expected_candidate_sha=args.candidate_sha,
            config_path=args.config,
            input_inventory_path=args.input_inventory,
            lock_path=args.lockfile,
            expected_capture_digest=capture.capture_digest,
        )
        record = build_tracking_record(bundle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(record.to_bytes())
        print(f"wrote compact VN2 tracking proposal: {args.output}")  # noqa: T201
        return 0
    except (OracleEvidenceError, TrackingError, VN2ResultError) as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    sys.exit(main())
