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
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_digest(
    *,
    environment: dict[str, object],
    config_digest: str,
    input_digest: str,
    capture_digest: str,
    actuals_semantics: str,
    lockfile_sha256: str,
) -> str:
    """Hash only GA1's canonical comparability key, never per-run provenance."""
    os_release = environment["os"]
    if not isinstance(os_release, dict):
        raise TypeError("environment.os must be an object")
    comparability_key = {
        "actuals_semantics": actuals_semantics,
        "architecture": environment["arch"],
        "capture_digest": capture_digest,
        "config_digest": config_digest,
        "input_digest": input_digest,
        "lockfile_sha256": lockfile_sha256,
        "os_release": {
            "id": os_release["id"],
            "version_id": os_release["version_id"],
        },
    }
    canonical = json.dumps(
        comparability_key,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def main() -> int:
    """Write files.sha256 and manifest.json into the bundle directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-inventory", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--actuals-semantics", required=True)
    args = parser.parse_args()

    payload_files = sorted(
        (
            p
            for p in args.bundle.rglob("*")
            if p.is_file() and p.name not in ("manifest.json", "files.sha256")
        ),
        key=lambda path: path.relative_to(args.bundle).as_posix().encode("utf-8"),
    )
    if not payload_files:
        print("bundle contains no payload files; refusing to mint an empty capture")
        return 1
    files = [
        {
            "path": path.relative_to(args.bundle).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in payload_files
    ]
    lines = [f"{entry['sha256']}  {entry['path']}" for entry in files]
    files_listing = "\n".join(lines) + "\n"
    (args.bundle / "files.sha256").write_text(files_listing, encoding="utf-8")
    inner_digest = hashlib.sha256(files_listing.encode("utf-8")).hexdigest()
    capture_listing = "".join(
        f"{entry['sha256']}  {entry['path']}\n"
        for entry in files
        if str(entry["path"]).startswith("orders/")
    )
    if not capture_listing:
        print("bundle contains no oracle order payloads; refusing to mint")
        return 1
    capture_digest = hashlib.sha256(capture_listing.encode("utf-8")).hexdigest()

    config_digest = sha256_file(args.config)
    input_inventory_digest = sha256_file(args.input_inventory)
    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    manifest = {
        "schema": 1,
        "artifact_kind": "vn2-oracle-orders",
        "artifact_name": f"oracle-capture-{os.environ['CANDIDATE_SHA']}",
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
        "config_digest": config_digest,
        "input_inventory": args.input_inventory.as_posix(),
        "input_inventory_digest": input_inventory_digest,
        "actuals_semantics": args.actuals_semantics,
        "capture_digest": capture_digest,
        "environment": environment,
        "environment_digest": environment_digest(
            environment=environment,
            config_digest=config_digest,
            input_digest=input_inventory_digest,
            capture_digest=capture_digest,
            actuals_semantics=args.actuals_semantics,
            lockfile_sha256=os.environ["ORACLE_LOCK_SHA256"],
        ),
        "files": files,
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
