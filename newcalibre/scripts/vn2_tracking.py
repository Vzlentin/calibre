"""Build, receipt, and validate exact VN2 tracking promotions."""

from __future__ import annotations

import argparse
import re
import subprocess
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
from newcalibre.protocols.vn2._tracking_promotion import (  # noqa: E402
    TRACKING_SERIES_PATH,
    build_promotion_receipt,
    load_promotion_metadata,
    promotion_receipt_path,
    validate_promotion_paths,
    validate_tracking_promotion,
    write_promotion_receipt,
)

_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


def _add_record_arguments(command: argparse.ArgumentParser) -> None:
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


def _add_live_promotion_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--proposal", type=Path, required=True)
    command.add_argument("--result-artifact-metadata", type=Path, required=True)
    command.add_argument("--proposal-artifact-metadata", type=Path, required=True)
    command.add_argument("--run-metadata", type=Path, required=True)
    command.add_argument("--result-archive", type=Path, required=True)
    command.add_argument("--proposal-archive", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("propose", "validate", "receipt", "promote"):
        command = commands.add_parser(name)
        _add_record_arguments(command)
        if name == "propose":
            command.add_argument("--output", type=Path, required=True)
        elif name == "validate":
            command.add_argument("--proposal", type=Path, required=True)
        else:
            _add_live_promotion_arguments(command)
            if name == "receipt":
                command.add_argument("--output", type=Path, required=True)
            else:
                command.add_argument("--repository-root", type=Path, required=True)
                command.add_argument("--base-sha", required=True)
                command.add_argument("--head-sha", required=True)
                command.add_argument("--default-branch-sha", required=True)
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


def _live_inputs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "result_artifact_metadata": load_promotion_metadata(
            args.result_artifact_metadata,
            name="result artifact metadata",
        ),
        "proposal_artifact_metadata": load_promotion_metadata(
            args.proposal_artifact_metadata,
            name="proposal artifact metadata",
        ),
        "run_metadata": load_promotion_metadata(
            args.run_metadata,
            name="workflow-run metadata",
        ),
        "result_archive": args.result_archive,
        "proposal_archive": args.proposal_archive,
    }


def _require_commit(value: str, *, name: str) -> str:
    if _COMMIT_SHA.fullmatch(value) is None:
        raise TrackingError(f"{name} must be a full lowercase 40-hex commit SHA")
    return value


def _git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
    )
    if completed.returncode == 0:
        return completed.stdout
    detail = completed.stderr.decode("utf-8", errors="replace").strip()
    raise TrackingError(f"Git object inspection failed: {detail}")


def _git_blob(
    root: Path,
    revision: str,
    path: str,
    *,
    allow_missing: bool = False,
) -> bytes | None:
    spec = f"{revision}:{path}"
    classified = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-t", spec],
        check=False,
        capture_output=True,
    )
    if classified.returncode != 0:
        if allow_missing:
            return None
        detail = classified.stderr.decode("utf-8", errors="replace").strip()
        raise TrackingError(f"required Git blob is missing: {detail}")
    if classified.stdout.strip() != b"blob":
        raise TrackingError("tracking evidence Git object must be a regular blob")
    return _git(root, "cat-file", "blob", spec)


def _promotion_git_blobs(
    repository_root: Path,
    *,
    base_sha: str,
    head_sha: str,
    candidate_sha: str,
) -> tuple[bytes | None, bytes, bytes]:
    root = Path(repository_root).absolute()
    if (
        root.is_symlink()
        or (root / ".git").is_symlink()
        or not (root / ".git").exists()
        or not (root / "newcalibre" / "pyproject.toml").is_file()
    ):
        raise TrackingError("repository root must identify the checked-out Calibre worktree")
    base = _require_commit(base_sha, name="promotion base SHA")
    head = _require_commit(head_sha, name="promotion head SHA")
    if base == head:
        raise TrackingError("promotion head SHA must differ from its base")
    for name, revision in (("base", base), ("head", head)):
        if _git(root, "cat-file", "-t", revision).strip() != b"commit":
            raise TrackingError(f"promotion {name} SHA must identify a commit")
    changed_raw = _git(root, "diff", "--name-only", "-z", base, head, "--")
    try:
        changed = [item.decode("utf-8") for item in changed_raw.split(b"\0") if item]
    except UnicodeError as error:
        raise TrackingError("promotion changed paths must be valid UTF-8") from error
    validate_promotion_paths(changed, candidate_sha=candidate_sha)
    receipt_path = promotion_receipt_path(candidate_sha)
    promoted = _git_blob(root, head, TRACKING_SERIES_PATH)
    receipt = _git_blob(root, head, receipt_path)
    prior = _git_blob(root, base, TRACKING_SERIES_PATH, allow_missing=True)
    assert promoted is not None
    assert receipt is not None
    return prior, promoted, receipt


def main(argv: list[str] | None = None) -> int:
    """Dispatch proposal construction, receipt construction, or promotion validation."""
    args = _parser().parse_args(argv)
    try:
        expected = _build(args)
        if args.command == "propose":
            write_proposal_record(expected, args.output)
            validated_subject = str(args.output)
        elif args.command == "validate":
            actual = parse_tracking_record(args.proposal)
            if actual.to_bytes() != expected.to_bytes():
                raise TrackingError(
                    "proposal bytes do not match freshly derived validated evidence"
                )
            validated_subject = str(args.proposal)
        elif args.command == "receipt":
            receipt = build_promotion_receipt(expected, args.proposal, **_live_inputs(args))
            write_promotion_receipt(receipt, args.output)
            validated_subject = str(args.output)
        else:
            prior, promoted, receipt_bytes = _promotion_git_blobs(
                args.repository_root,
                base_sha=args.base_sha,
                head_sha=args.head_sha,
                candidate_sha=args.candidate_sha,
            )
            validate_tracking_promotion(
                expected,
                args.proposal,
                receipt_bytes,
                promoted,
                prior_history=prior,
                base_sha=args.base_sha,
                default_branch_sha=args.default_branch_sha,
                **_live_inputs(args),
            )
            validated_subject = args.head_sha
    except TrackingError as error:
        raise SystemExit(str(error)) from error
    print(f"validated {args.command} tracking evidence: {validated_subject}")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
