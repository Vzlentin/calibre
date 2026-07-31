"""Verify VN2 inputs against the successor-approved inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newcalibre.protocols.vn2 import VN2InputError, verify_vn2_inputs  # noqa: E402

DEFAULT_INVENTORY = PROJECT_ROOT / "benchmarks" / "vn2" / "vn2-input-digests.json"


def build_parser() -> argparse.ArgumentParser:
    """Build the verification-only CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--target", type=Path, required=True)
    verify.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    return parser


def main() -> int:
    """Dispatch unconditional verification."""
    args = build_parser().parse_args()
    try:
        inventory = verify_vn2_inputs(args.target, args.inventory)
    except VN2InputError as error:
        raise SystemExit(str(error)) from error
    print(f"verified {len(inventory.files)} VN2 inputs")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
