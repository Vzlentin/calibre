"""Bootstrap VN2 data acquisition and digest verification (KTD-A4).

Bootstrap-only root tooling: it reads the frozen repo's link manifest as
*data* (``benchmarks/vn2/vn2_file_links.json``) and never imports frozen
``benchmarks/`` or ``calibre/`` code. Three commands:

    download  Fetch the challenge distribution (exactly the twelve
              source-manifest names) from datasource.ai, falling back to the
              private dataset mirror when ``DATASETS_MIRROR_REPO`` is set.
    mint      Compute the sha256 inventory of a downloaded set — the one-time
              pre-clock digest mint, run on CI and promoted through a PR.
    verify    Revalidate a directory against the committed inventory: the
              exact file set (no missing, no extra CSV) and every digest,
              immediately before any consumption. A mismatch fails loudly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

LINKS_MANIFEST = Path("benchmarks/vn2/vn2_file_links.json")
EXPECTED_FILE_COUNT = 12
MIRROR_RELEASE = "vn2-v1"


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_links() -> list[dict]:
    """Read the frozen link manifest (data only) and check its shape."""
    entries = json.loads(LINKS_MANIFEST.read_text(encoding="utf-8"))["files"]
    names = [entry["file_name"] for entry in entries]
    if len(names) != EXPECTED_FILE_COUNT or len(set(names)) != EXPECTED_FILE_COUNT:
        raise SystemExit(
            f"link manifest must carry exactly {EXPECTED_FILE_COUNT} distinct files, "
            f"found {len(names)}"
        )
    return entries


def fetch_from_mirror(name: str, target: Path) -> bool:
    """Fetch one file from the private dataset mirror, if configured."""
    mirror = os.environ.get("DATASETS_MIRROR_REPO")
    if not mirror:
        return False
    result = subprocess.run(
        [
            "gh",
            "release",
            "download",
            MIRROR_RELEASE,
            "--repo",
            mirror,
            "--pattern",
            name,
            "--dir",
            str(target),
            "--clobber",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"mirror fetch failed for {name}: {result.stderr.strip()}")
        return False
    return True


def cmd_download(target: Path, if_missing: bool) -> int:
    """Download the twelve challenge files into the target directory."""
    target.mkdir(parents=True, exist_ok=True)
    for entry in load_links():
        name, url = entry["file_name"], entry["url"]
        destination = target / name
        if if_missing and destination.exists():
            print(f"kept   {name} (exists; verification is a separate, unconditional step)")
            continue
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                destination.write_bytes(response.read())
            print(f"fetched {name} from upstream")
        except OSError as error:
            print(f"upstream fetch failed for {name}: {error}")
            if not fetch_from_mirror(name, target):
                raise SystemExit(
                    f"could not obtain {name} from upstream or mirror; VN2 inputs are unavailable"
                ) from error
            print(f"fetched {name} from mirror")
    return 0


def cmd_mint(target: Path, out: Path) -> int:
    """Mint the digest inventory from a downloaded set (pre-clock, on CI)."""
    names = [entry["file_name"] for entry in load_links()]
    files = []
    for name in names:
        path = target / name
        if not path.exists():
            raise SystemExit(f"cannot mint: {name} missing from {target}")
        files.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    inventory = {
        "schema": 1,
        "dataset": "vn2",
        "source_manifest": LINKS_MANIFEST.as_posix(),
        "source_manifest_sha256": sha256_file(LINKS_MANIFEST),
        "minted_run_id": os.environ.get("GITHUB_RUN_ID"),
        "minted_sha": os.environ.get("GITHUB_SHA"),
        "files": files,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"minted {len(files)} digests -> {out}")
    return 0


def cmd_verify(target: Path, inventory_path: Path) -> int:
    """Revalidate the exact file set and every digest; fail loudly."""
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected = {entry["name"]: entry for entry in inventory["files"]}
    if len(expected) != EXPECTED_FILE_COUNT:
        raise SystemExit(f"inventory carries {len(expected)} files, expected {EXPECTED_FILE_COUNT}")
    present = {p.name for p in target.glob("*.csv")}
    missing = sorted(set(expected) - present)
    extra = sorted(present - set(expected))
    if missing or extra:
        raise SystemExit(f"file-set mismatch: missing={missing} extra={extra}")
    for name, entry in sorted(expected.items()):
        path = target / name
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise SystemExit(f"{name}: size {size} != inventory {entry['bytes']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise SystemExit(f"{name}: sha256 {digest} != inventory {entry['sha256']}")
    print(f"verified {len(expected)} files against {inventory_path}")
    return 0


def main() -> int:
    """Dispatch the requested data command."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("download")
    download.add_argument("--target", type=Path, required=True)
    download.add_argument("--if-missing", action="store_true")
    mint = sub.add_parser("mint")
    mint.add_argument("--target", type=Path, required=True)
    mint.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--target", type=Path, required=True)
    verify.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "download":
        return cmd_download(args.target, args.if_missing)
    if args.command == "mint":
        return cmd_mint(args.target, args.out)
    return cmd_verify(args.target, args.inventory)


if __name__ == "__main__":
    sys.exit(main())
