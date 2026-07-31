"""Verify consumed M5 inputs against the approved inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newcalibre.protocols.m5 import verify_m5_inputs  # noqa: E402
from newcalibre.protocols.m5.inventory import M5InputError  # noqa: E402

DEFAULT_INVENTORY = PROJECT_ROOT / "benchmarks" / "m5" / "m5-inputs.json"


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
        inventory = verify_m5_inputs(args.target, args.inventory)
    except M5InputError as error:
        raise SystemExit(str(error)) from error
    print(f"verified {len(inventory.files)} consumed M5 inputs")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
