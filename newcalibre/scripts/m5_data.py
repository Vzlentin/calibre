"""Download or verify consumed M5 inputs against the approved inventory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newcalibre.protocols.m5 import verify_m5_inputs  # noqa: E402
from newcalibre.protocols.m5.inventory import (  # noqa: E402
    M5InputError,
    download_m5_inputs,
    load_unique_json,
)

DEFAULT_INVENTORY = PROJECT_ROOT / "benchmarks" / "m5" / "m5-inputs.json"


def _sources(path: Path) -> dict[str, str]:
    raw = load_unique_json(path, subject=f"source mapping {path}")
    if not isinstance(raw, dict) or set(raw) != {"files"}:
        raise M5InputError("source mapping must contain exactly one 'files' list")
    files = cast(dict[str, object], raw)["files"]
    if not isinstance(files, list):
        raise M5InputError("source mapping files must be a list")
    sources: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"name", "url"}:
            raise M5InputError("each source entry must contain exact name/url keys")
        payload = cast(dict[str, object], entry)
        name = payload["name"]
        url = payload["url"]
        if not isinstance(name, str) or not isinstance(url, str):
            raise M5InputError("each source name and URL must be a string")
        if name in sources:
            raise M5InputError(f"source mapping contains duplicate name {name!r}")
        sources[name] = url
    return sources


def build_parser() -> argparse.ArgumentParser:
    """Build the acquisition CLI with only download and verify operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--target", type=Path, required=True)
    download.add_argument("--sources", type=Path, required=True)
    download.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    verify = commands.add_parser("verify")
    verify.add_argument("--target", type=Path, required=True)
    verify.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    return parser


def main() -> int:
    """Dispatch script-only acquisition or unconditional verification."""
    args = build_parser().parse_args()
    try:
        if args.command == "download":
            inventory = download_m5_inputs(
                args.target,
                _sources(args.sources),
                args.inventory,
            )
        else:
            inventory = verify_m5_inputs(args.target, args.inventory)
    except M5InputError as error:
        raise SystemExit(str(error)) from error
    print(f"verified {len(inventory.files)} consumed M5 inputs")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
