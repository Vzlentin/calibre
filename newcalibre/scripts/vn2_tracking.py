"""Build and revalidate one proposal-only VN2 tracking record."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newcalibre.protocols.vn2 import (  # noqa: E402
    TrackingError,
    build_tracking_record,
    parse_tracking_record,
    write_proposal_record,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("propose", "validate"):
        command = commands.add_parser(name)
        command.add_argument("--result-root", type=Path, required=True)
        command.add_argument("--capture-root", type=Path, required=True)
        command.add_argument("--candidate-sha", required=True)
        command.add_argument("--workflow-ref", dest="definition_ref", required=True)
        command.add_argument("--workflow-sha", dest="definition_sha", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--run-url", required=True)
        command.add_argument("--result-artifact-id", dest="artifact_id", required=True)
        command.add_argument("--result-artifact-name", dest="artifact_name", required=True)
        command.add_argument("--result-artifact-digest", dest="artifact_digest", required=True)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--input-inventory", type=Path, required=True)
        command.add_argument("--lockfile", type=Path, required=True)
        if name == "propose":
            command.add_argument("--output", type=Path, required=True)
        else:
            command.add_argument("--proposal", type=Path, required=True)
    return parser


def _build(args: argparse.Namespace):
    return build_tracking_record(
        args.result_root,
        args.capture_root,
        candidate_sha=args.candidate_sha,
        definition_ref=args.definition_ref,
        definition_sha=args.definition_sha,
        run_id=args.run_id,
        run_url=args.run_url,
        result_artifact_id=args.artifact_id,
        result_artifact_name=args.artifact_name,
        result_artifact_digest=args.artifact_digest,
        config_path=args.config,
        input_inventory_path=args.input_inventory,
        lockfile_path=args.lockfile,
    )


def main(argv: list[str] | None = None) -> int:
    """Dispatch proposal construction or exact revalidation."""
    args = _parser().parse_args(argv)
    try:
        expected = _build(args)
        if args.command == "propose":
            write_proposal_record(expected, args.output)
        else:
            actual = parse_tracking_record(args.proposal)
            if actual.to_bytes() != expected.to_bytes():
                raise TrackingError(
                    "proposal bytes do not match freshly derived validated evidence"
                )
    except TrackingError as error:
        raise SystemExit(str(error)) from error
    proposal_path = args.output if args.command == "propose" else args.proposal
    print(f"validated {args.command} tracking proposal: {proposal_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
