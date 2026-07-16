"""Validate and safely extract live Stage 3 tracking artifacts.

This script is executed from the immutable trusted base checkout. It binds the
live workflow run and both downloaded artifacts to the receipt fields before it
stages either archive. ZIP contents are validated as a complete namespace,
extracted into private temporary directories, and published only after both
archives have extracted successfully.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

_SUCCESSOR_SRC = Path(__file__).resolve().parents[2] / "newcalibre" / "src"
sys.path.insert(0, str(_SUCCESSOR_SRC))

from newcalibre.protocols.vn2.tracking import (  # noqa: E402
    TRACKING_SERIES_PATH,
    parse_promotion_receipt,
    promotion_receipt_path,
    validate_promotion_paths,
)

TRUSTED_REPOSITORY = "Vzlentin/calibre"
TRUSTED_WORKFLOW_PATH = ".github/workflows/newcalibre.yml"
TRUSTED_BRANCH = "main"
MAX_TOTAL_EXPANDED_BYTES = 1024 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")


class AdmissionError(ValueError):
    """Report an artifact that cannot cross the tracking admission boundary."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate key: {key}")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    """Read a strict JSON object, rejecting invalid text and duplicate keys."""
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise AdmissionError(f"{label} metadata is malformed") from error
    if not isinstance(value, dict):
        raise AdmissionError(f"{label} metadata must be a JSON object")
    return value


def _is_integer(value: object) -> bool:
    return type(value) is int


def validate_run_metadata(
    metadata: dict[str, Any],
    *,
    run_id: int,
    candidate_sha: str,
    run_url: str,
) -> int:
    """Validate the live workflow run and return its repository ID."""
    repository = metadata.get("repository")
    if (
        not _is_integer(run_id)
        or metadata.get("id") != run_id
        or not _is_integer(metadata.get("id"))
        or metadata.get("event") != "workflow_dispatch"
        or metadata.get("path") != TRUSTED_WORKFLOW_PATH
        or metadata.get("head_branch") != TRUSTED_BRANCH
        or metadata.get("head_sha") != candidate_sha
        or metadata.get("status") != "completed"
        or metadata.get("conclusion") != "success"
        or metadata.get("run_attempt") != 1
        or not _is_integer(metadata.get("run_attempt"))
        or metadata.get("html_url") != run_url
        or not isinstance(repository, dict)
        or repository.get("full_name") != TRUSTED_REPOSITORY
        or not _is_integer(repository.get("id"))
    ):
        raise AdmissionError("workflow-run metadata is not the trusted successful C0 mint")
    return repository["id"]


def validate_artifact_metadata(
    metadata: dict[str, Any],
    *,
    label: str,
    artifact_id: int,
    artifact_name: str,
    receipt_digest: str,
    run_id: int,
    candidate_sha: str,
    repository_id: int,
) -> None:
    """Bind one artifact metadata response to the trusted live workflow run."""
    workflow_run = metadata.get("workflow_run")
    if (
        _DIGEST_PATTERN.fullmatch(receipt_digest) is None
        or not _is_integer(artifact_id)
        or metadata.get("id") != artifact_id
        or not _is_integer(metadata.get("id"))
        or metadata.get("name") != artifact_name
        or metadata.get("digest") != f"sha256:{receipt_digest}"
        or metadata.get("expired") is not False
        or not isinstance(workflow_run, dict)
        or workflow_run.get("id") != run_id
        or not _is_integer(workflow_run.get("id"))
        or workflow_run.get("head_branch") != TRUSTED_BRANCH
        or workflow_run.get("head_sha") != candidate_sha
        or workflow_run.get("repository_id") != repository_id
        or not _is_integer(workflow_run.get("repository_id"))
        or workflow_run.get("head_repository_id") != repository_id
        or not _is_integer(workflow_run.get("head_repository_id"))
    ):
        raise AdmissionError(f"{label} artifact metadata is not from the trusted C0 mint")


def stream_sha256(path: Path) -> str:
    """Hash a file without loading the archive into memory."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(_COPY_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError as error:
        raise AdmissionError(f"cannot read artifact archive {path}") from error
    return digest.hexdigest()


def validate_archive_digest(
    archive: Path,
    *,
    expected_digest: str,
    label: str,
) -> None:
    """Require the downloaded bytes to match the receipt and GitHub metadata."""
    if _DIGEST_PATTERN.fullmatch(expected_digest) is None:
        raise AdmissionError(f"{label} archive digest binding is malformed")
    actual_digest = stream_sha256(archive)
    if actual_digest != expected_digest:
        raise AdmissionError(f"downloaded {label} archive digest does not match GitHub metadata")


def _entry_path(entry: zipfile.ZipInfo) -> tuple[str, ...]:
    raw_path = entry.filename
    directory = entry.is_dir()
    path = raw_path[:-1] if directory and raw_path.endswith("/") else raw_path
    if (
        not path
        or "\\" in raw_path
        or "\x00" in raw_path
        or PurePosixPath(raw_path).is_absolute()
        or PureWindowsPath(raw_path).is_absolute()
    ):
        raise AdmissionError(f"artifact archive has an unsafe path: {raw_path!r}")
    parts = tuple(path.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise AdmissionError(f"artifact archive has an unsafe path: {raw_path!r}")
    return parts


def _validated_entries(
    bundle: zipfile.ZipFile,
    *,
    max_total_size: int,
) -> list[tuple[zipfile.ZipInfo, tuple[str, ...]]]:
    if max_total_size < 0:
        raise ValueError("max_total_size must be non-negative")

    validated: list[tuple[zipfile.ZipInfo, tuple[str, ...]]] = []
    kinds: dict[tuple[str, ...], str] = {}
    total_size = 0
    for entry in bundle.infolist():
        parts = _entry_path(entry)
        kind = "directory" if entry.is_dir() else "file"
        if parts in kinds:
            raise AdmissionError(f"artifact archive has a duplicate path: {entry.filename!r}")
        for depth in range(1, len(parts)):
            if kinds.get(parts[:depth]) == "file":
                raise AdmissionError("artifact archive has a file-directory prefix collision")
        if kind == "file" and any(
            len(existing) > len(parts) and existing[: len(parts)] == parts for existing in kinds
        ):
            raise AdmissionError("artifact archive has a file-directory prefix collision")

        mode = entry.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        expected_types = {0, stat.S_IFDIR} if entry.is_dir() else {0, stat.S_IFREG}
        if file_type not in expected_types or entry.flag_bits & 0x1:
            raise AdmissionError(
                f"artifact archive entry has a special or mismatched mode: {entry.filename!r}"
            )
        if not entry.is_dir() and mode & 0o111:
            raise AdmissionError(
                f"artifact archive entry has an executable mode: {entry.filename!r}"
            )

        total_size += entry.file_size
        if total_size > max_total_size:
            raise AdmissionError("artifact archive expands beyond the limit")
        kinds[parts] = kind
        validated.append((entry, parts))
    return validated


def _extract_entries(
    bundle: zipfile.ZipFile,
    entries: list[tuple[zipfile.ZipInfo, tuple[str, ...]]],
    stage: Path,
    *,
    max_total_size: int,
) -> None:
    total_written = 0
    for entry, parts in entries:
        target = stage.joinpath(*parts)
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        entry_written = 0
        with bundle.open(entry, "r") as source, target.open("xb") as output:
            while chunk := source.read(_COPY_CHUNK_BYTES):
                entry_written += len(chunk)
                total_written += len(chunk)
                if total_written > max_total_size or entry_written > entry.file_size:
                    raise AdmissionError("artifact archive expands beyond the limit")
                output.write(chunk)
        if entry_written != entry.file_size:
            raise AdmissionError("artifact archive has a short or corrupt entry")


def stage_safe_zip(
    archive: Path,
    destination: Path,
    *,
    max_total_size: int = MAX_TOTAL_EXPANDED_BYTES,
) -> Path:
    """Validate and extract an archive into a private sibling staging directory."""
    if destination.exists() or destination.is_symlink():
        raise AdmissionError(f"artifact destination already exists: {destination}")
    if not destination.parent.is_dir():
        raise AdmissionError(f"artifact destination parent does not exist: {destination.parent}")

    stage: Path | None = None
    completed = False
    try:
        try:
            with zipfile.ZipFile(archive, "r") as bundle:
                entries = _validated_entries(bundle, max_total_size=max_total_size)
                stage = Path(
                    tempfile.mkdtemp(
                        prefix=f".stage3-admission-{destination.name}-",
                        dir=destination.parent,
                    )
                )
                _extract_entries(
                    bundle,
                    entries,
                    stage,
                    max_total_size=max_total_size,
                )
        except AdmissionError:
            raise
        except (zipfile.BadZipFile, EOFError, RuntimeError, OSError) as error:
            raise AdmissionError("artifact archive is short or corrupt") from error
        completed = True
        return stage
    finally:
        if not completed and stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def publish_staged_directory(stage: Path, destination: Path) -> None:
    """Atomically rename a completed private stage to its accepted destination."""
    if destination.exists() or destination.is_symlink():
        raise AdmissionError(f"artifact destination already exists: {destination}")
    try:
        os.replace(stage, destination)
    except OSError as error:
        raise AdmissionError(f"cannot publish artifact destination: {destination}") from error


def extract_safe_zip(
    archive: Path,
    destination: Path,
    *,
    max_total_size: int = MAX_TOTAL_EXPANDED_BYTES,
) -> None:
    """Safely extract and atomically publish one ZIP archive."""
    stage = stage_safe_zip(
        archive,
        destination,
        max_total_size=max_total_size,
    )
    try:
        publish_staged_directory(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def admit_artifact_archive(
    archive: Path,
    destination: Path,
    *,
    expected_digest: str,
    label: str,
    max_total_size: int = MAX_TOTAL_EXPANDED_BYTES,
) -> None:
    """Validate an archive digest, safely extract it, and atomically publish it."""
    validate_archive_digest(
        archive,
        expected_digest=expected_digest,
        label=label,
    )
    extract_safe_zip(
        archive,
        destination,
        max_total_size=max_total_size,
    )


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _sha256_digest(value: str) -> str:
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase SHA-256 digest")
    return value


def _commit_sha(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise argparse.ArgumentTypeError("must be a lowercase 40-hex commit SHA")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect",
        help="inspect a promotion directly from the base and head Git trees",
    )
    inspect.add_argument("--base-sha", required=True, type=_commit_sha)
    inspect.add_argument("--head-sha", required=True, type=_commit_sha)
    inspect.add_argument("--github-output", required=True, type=Path)
    inspect.add_argument("--repository-root", default=Path.cwd(), type=Path)

    admit = commands.add_parser(
        "admit",
        help="validate and extract the live artifact archives",
    )
    admit.add_argument("--candidate-sha", required=True, type=_commit_sha)
    admit.add_argument("--run-id", required=True, type=_positive_integer)
    admit.add_argument("--run-url", required=True)
    admit.add_argument("--run-metadata", required=True, type=Path)
    for label in ("result", "proposal"):
        admit.add_argument(f"--{label}-artifact-id", required=True, type=_positive_integer)
        admit.add_argument(f"--{label}-artifact-name", required=True)
        admit.add_argument(f"--{label}-artifact-digest", required=True, type=_sha256_digest)
        admit.add_argument(f"--{label}-artifact-metadata", required=True, type=Path)
        admit.add_argument(f"--{label}-archive", required=True, type=Path)
        admit.add_argument(f"--{label}-destination", required=True, type=Path)
    return parser


def _git(repository: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise AdmissionError(
            f"cannot inspect the requested Git trees: git {' '.join(args)}"
        ) from error


def _decode_nul_paths(value: bytes, *, label: str) -> list[str]:
    if not value:
        return []
    if not value.endswith(b"\0"):
        raise AdmissionError(f"{label} did not return a NUL-terminated path list")
    encoded = value[:-1].split(b"\0")
    if any(not path for path in encoded):
        raise AdmissionError(f"{label} returned an empty path")
    try:
        return [path.decode("utf-8", errors="strict") for path in encoded]
    except UnicodeDecodeError as error:
        raise AdmissionError(f"{label} paths must be valid UTF-8") from error


def _changed_paths(repository: Path, *, base_sha: str, head_sha: str) -> list[str]:
    for sha in (base_sha, head_sha):
        _git(repository, "cat-file", "-e", f"{sha}^{{commit}}")
    return _decode_nul_paths(
        _git(
            repository,
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            base_sha,
            head_sha,
            "--",
        ),
        label="changed Git",
    )


def _head_blob(
    repository: Path,
    *,
    head_sha: str,
    path: str,
) -> bytes:
    entries = _git(repository, "ls-tree", "-z", head_sha, "--", path)
    if not entries.endswith(b"\0") or entries.count(b"\0") != 1:
        raise AdmissionError("tracking receipt must be one exact entry in the head Git tree")
    record = entries[:-1]
    try:
        metadata, encoded_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.split(b" ", 2)
        decoded_path = encoded_path.decode("utf-8", errors="strict")
    except (ValueError, UnicodeDecodeError) as error:
        raise AdmissionError("tracking receipt Git entry is malformed") from error
    if decoded_path != path:
        raise AdmissionError("tracking receipt Git entry path is not exact")
    if mode != b"100644" or object_type != b"blob":
        raise AdmissionError("tracking receipt must be a regular 100644 blob")
    try:
        oid = object_id.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise AdmissionError("tracking receipt Git object ID is malformed") from error
    return _git(repository, "cat-file", "blob", oid)


def _write_github_outputs(path: Path, fields: dict[str, str]) -> None:
    for name, value in fields.items():
        if not value or "\n" in value or "\r" in value:
            raise AdmissionError(f"GitHub output {name} is invalid")
    try:
        with path.open("a", encoding="utf-8", newline="\n") as output:
            output.writelines(f"{name}={value}\n" for name, value in fields.items())
    except OSError as error:
        raise AdmissionError("cannot write the explicit GitHub output file") from error


def inspect_promotion(args: argparse.Namespace) -> None:
    """Inspect the real base/head Git trees and publish trusted receipt outputs."""
    repository = args.repository_root.resolve()
    paths = _changed_paths(repository, base_sha=args.base_sha, head_sha=args.head_sha)
    tracking_root = TRACKING_SERIES_PATH.rsplit("/", 1)[0]
    if not any(path == tracking_root or path.startswith(f"{tracking_root}/") for path in paths):
        _write_github_outputs(args.github_output, {"present": "false"})
        return

    validate_promotion_paths(paths, candidate_sha=args.base_sha)
    receipt_path = promotion_receipt_path(args.base_sha)
    receipt = parse_promotion_receipt(
        _head_blob(repository, head_sha=args.head_sha, path=receipt_path)
    )
    if receipt.candidate_sha != args.base_sha:
        raise AdmissionError("tracking receipt candidate does not equal the PR base")
    if receipt.definition_sha != receipt.candidate_sha:
        raise AdmissionError("tracking workflow definition SHA does not bind the candidate")
    if receipt.result_artifact.id == receipt.proposal_artifact.id:
        raise AdmissionError("tracking result and proposal artifacts must be distinct")

    _write_github_outputs(
        args.github_output,
        {
            "present": "true",
            "receipt": receipt_path,
            "candidate": receipt.candidate_sha,
            "workflow_ref": receipt.definition_ref,
            "workflow_sha": receipt.definition_sha,
            "run_id": receipt.run_id,
            "run_url": receipt.run_url,
            "result_id": receipt.result_artifact.id,
            "result_name": receipt.result_artifact.name,
            "result_digest": receipt.result_artifact.digest,
            "proposal_id": receipt.proposal_artifact.id,
            "proposal_name": receipt.proposal_artifact.name,
            "proposal_digest": receipt.proposal_artifact.digest,
        },
    )


def admit_artifacts(args: argparse.Namespace) -> None:
    """Validate both live artifacts, then stage and publish them as one operation."""
    run = read_json_object(args.run_metadata, label="workflow-run")
    repository_id = validate_run_metadata(
        run,
        run_id=args.run_id,
        candidate_sha=args.candidate_sha,
        run_url=args.run_url,
    )
    if args.result_artifact_id == args.proposal_artifact_id:
        raise AdmissionError("result and proposal artifact IDs must be distinct")

    artifacts = []
    for label in ("result", "proposal"):
        artifact_id = getattr(args, f"{label}_artifact_id")
        artifact_name = getattr(args, f"{label}_artifact_name")
        receipt_digest = getattr(args, f"{label}_artifact_digest")
        metadata_path = getattr(args, f"{label}_artifact_metadata")
        archive = getattr(args, f"{label}_archive")
        destination = getattr(args, f"{label}_destination")
        metadata = read_json_object(metadata_path, label=f"{label} artifact")
        validate_artifact_metadata(
            metadata,
            label=label,
            artifact_id=artifact_id,
            artifact_name=artifact_name,
            receipt_digest=receipt_digest,
            run_id=args.run_id,
            candidate_sha=args.candidate_sha,
            repository_id=repository_id,
        )
        validate_archive_digest(
            archive,
            expected_digest=receipt_digest,
            label=label,
        )
        artifacts.append((archive, destination))

    destinations = [destination for _, destination in artifacts]
    if len(set(destinations)) != len(destinations):
        raise AdmissionError("artifact destinations must be distinct")

    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for archive, destination in artifacts:
            staged.append((stage_safe_zip(archive, destination), destination))
        for stage, destination in staged:
            publish_staged_directory(stage, destination)
            published.append(destination)
    except Exception:
        for destination in reversed(published):
            shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        for stage, _ in staged:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    """Validate the requested Stage 3 tracking admission operation."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            inspect_promotion(args)
        else:
            admit_artifacts(args)
    except (AdmissionError, ValueError) as error:
        parser.exit(1, f"stage3 tracking admission failed: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
