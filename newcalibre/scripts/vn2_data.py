"""Download or verify VN2 inputs against the successor-approved inventory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from newcalibre.protocols.vn2 import (  # noqa: E402
    VN2InputError,
    download_vn2_inputs,
    verify_vn2_inputs,
)

DEFAULT_INVENTORY = PROJECT_ROOT / "benchmarks" / "vn2" / "vn2-input-digests.json"


def _sources(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VN2InputError(f"source mapping is not readable JSON: {path}") from error
    if not isinstance(raw, dict) or set(raw) != {"files"} or not isinstance(raw["files"], list):
        raise VN2InputError("source mapping must contain exactly one 'files' list")
    sources: dict[str, str] = {}
    for entry in raw["files"]:
        if not isinstance(entry, dict) or set(entry) != {"name", "url"}:
            raise VN2InputError("each source entry must contain exact name/url keys")
        payload = cast(dict[str, object], entry)
        name = payload["name"]
        url = payload["url"]
        if not isinstance(name, str) or not isinstance(url, str):
            raise VN2InputError("each source name and URL must be a string")
        if name in sources:
            raise VN2InputError(f"source mapping contains duplicate name {name!r}")
        sources[name] = url
    return sources


def build_parser() -> argparse.ArgumentParser:
    """Build the acquisition CLI without a digest-mint command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--target", type=Path, required=True)
    download.add_argument("--sources", type=Path, required=True)
    download.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    download.add_argument("--if-missing", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--target", type=Path, required=True)
    verify.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    return parser


def main() -> int:
    """Dispatch successor-owned download or unconditional verification."""
    args = build_parser().parse_args()
    try:
        if args.command == "download":
            inventory = download_vn2_inputs(
                args.target,
                _sources(args.sources),
                args.inventory,
                if_missing=args.if_missing,
            )
        else:
            inventory = verify_vn2_inputs(args.target, args.inventory)
    except VN2InputError as error:
        raise SystemExit(str(error)) from error
    print(f"verified {len(inventory.files)} VN2 inputs")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
