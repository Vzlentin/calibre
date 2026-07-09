"""Assemble the manifest-complete oracle capture bundle (KTD-A6).

Bootstrap-only root tooling; runs in the oracle-capture workflow after the
extraction helper has written the order files. Computes per-file sha256s in
deterministic order, derives the inner bundle digest from them, and writes a
``manifest.json`` binding candidate/workflow/oracle identities, run
provenance, config/input digests, and environment facts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Write files.sha256 and manifest.json into the bundle directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-inventory", type=Path, required=True)
    args = parser.parse_args()

    payload_files = sorted(
        p
        for p in args.bundle.rglob("*")
        if p.is_file() and p.name not in ("manifest.json", "files.sha256")
    )
    if not payload_files:
        print("bundle contains no payload files; refusing to mint an empty capture")
        return 1
    lines = [f"{sha256_file(p)}  {p.relative_to(args.bundle).as_posix()}" for p in payload_files]
    files_listing = "\n".join(lines) + "\n"
    (args.bundle / "files.sha256").write_text(files_listing, encoding="utf-8")
    inner_digest = hashlib.sha256(files_listing.encode("utf-8")).hexdigest()

    manifest = {
        "candidate_sha": os.environ["CANDIDATE_SHA"],
        "workflow_sha": os.environ["GITHUB_WORKFLOW_SHA"],
        "oracle_tag": os.environ["ORACLE_TAG"],
        "oracle_commit": os.environ["ORACLE_COMMIT"],
        "oracle_lock_sha256": os.environ["ORACLE_LOCK_SHA256"],
        "run_id": os.environ["GITHUB_RUN_ID"],
        "run_url": (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        ),
        "config_digest": sha256_file(args.config),
        "input_inventory": args.input_inventory.as_posix(),
        "input_inventory_digest": sha256_file(args.input_inventory),
        "environment": {
            "arch": platform.machine(),
            "os_release": platform.freedesktop_os_release().get("PRETTY_NAME", "unknown"),
            "python": platform.python_version(),
            "runner_image": (
                f"{os.environ.get('ImageOS', '?')}/{os.environ.get('ImageVersion', '?')}"  # noqa: SIM112
            ),
        },
        "files": lines,
        "inner_bundle_digest": inner_digest,
    }
    manifest_path = args.bundle / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"inner bundle digest: {inner_digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
