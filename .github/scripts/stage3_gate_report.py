"""Emit the single-SHA Gate A technical report (KTD-A13, GA6).

Runs in the gate-a aggregate job at the C1 checkout after every lane reaches
a terminal result. Records each lane's result and common workflow provenance,
the candidate SHA, the digests of every committed evidence surface, the C0→C1
merge-discipline check, and the immutable budget check. The owner decision is
a separate timestamped record binding this report's digest; this script never
writes to the repository.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from stage3_clock import find_activation_record, parse_utc_timestamp, record_is_schema_complete

EVIDENCE_ALLOWLIST = ("stage3/evidence/",)
TRACKING_SERIES = Path("stage3/evidence/tracking/series.jsonl")
INPUT_INVENTORY = Path("stage3/evidence/vn2-input-digests.json")
CAPTURES_DIR = Path("stage3/evidence/captures")
LANE_NAMES = ("lint", "unit-isolated", "consistency", "oracle", "vn2-verify")
LANE_RESULTS = frozenset({"success", "failure", "cancelled", "skipped"})
TIER3_INVENTORY_PATH = (
    Path(__file__).resolve().parents[2] / "newcalibre" / "tests" / "tier3" / "oracle_inventory.json"
)


def _load_required_oracle_outcomes(
    path: Path,
    *,
    problems: list[str],
) -> dict[str, dict[str, str]]:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        problems.append(f"missing required Tier 3 oracle inventory: {path.as_posix()}")
        return {}
    except OSError as error:
        problems.append(f"cannot inspect required Tier 3 oracle inventory: {error}")
        return {}
    if stat.S_ISLNK(mode):
        problems.append("required Tier 3 oracle inventory must not be a symbolic link")
        return {}
    if not stat.S_ISREG(mode):
        problems.append("required Tier 3 oracle inventory must be a regular file")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f"invalid JSON in required Tier 3 oracle inventory: {error}")
        return {}
    except (OSError, UnicodeError) as error:
        problems.append(f"cannot read required Tier 3 oracle inventory: {error}")
        return {}
    if not isinstance(value, dict) or set(value) != {"named", "schema", "tier"}:
        problems.append("required Tier 3 oracle inventory has an invalid root schema")
        return {}
    if value["schema"] != 1 or value["tier"] != "tier3":
        problems.append("required Tier 3 oracle inventory has an invalid identity")
        return {}
    named = value["named"]
    if not isinstance(named, dict) or set(named) != {"gate", "witness"}:
        problems.append("required Tier 3 oracle inventory must name one gate and witness")
        return {}

    outcomes: dict[str, dict[str, str]] = {}
    for role in ("gate", "witness"):
        item = named[role]
        if not isinstance(item, dict) or set(item) != {"id", "node"}:
            problems.append(f"required Tier 3 oracle inventory {role} has an invalid schema")
            continue
        identifier = item["id"]
        node = item["node"]
        if identifier != "vn2-conditional-replay":
            problems.append(
                f"required Tier 3 oracle inventory {role} must use vn2-conditional-replay"
            )
            continue
        if not isinstance(node, str) or not node.startswith("tests.tier3.") or "::" not in node:
            problems.append(
                f"required Tier 3 oracle inventory {role} must use a stable Tier 3 node ID"
            )
            continue
        outcomes[role] = {"id": identifier, "node": node}
    if (
        set(outcomes) == {"gate", "witness"}
        and outcomes["gate"]["node"] == outcomes["witness"]["node"]
    ):
        problems.append("required Tier 3 oracle gate and witness must use distinct nodes")
        return {}
    return outcomes


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    """Run a git command and return stripped stdout."""
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def _open_regular_file(path: Path, *, name: str) -> int:
    absolute = Path(os.path.abspath(path))
    parts = absolute.parts
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    leaf_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    opened: list[int] = []
    try:
        parent_fd = os.open(absolute.anchor, directory_flags)
        opened.append(parent_fd)
        for part in parts[1:-1]:
            child_fd = os.open(part, directory_flags, dir_fd=parent_fd)
            opened.append(child_fd)
            parent_fd = child_fd
        fd = os.open(parts[-1], leaf_flags, dir_fd=parent_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError(f"{name} must be a regular non-symlink file")
        except BaseException:
            with contextlib.suppress(OSError):
                os.close(fd)
            raise
    except ValueError:
        raise
    except OSError as error:
        raise ValueError(f"{name} and every ancestor must be readable non-symlinks") from error
    finally:
        for directory_fd in reversed(opened):
            with contextlib.suppress(OSError):
                os.close(directory_fd)
    return fd


def _read_regular_bytes(path: Path, *, name: str, maximum: int = 4 * 1024 * 1024) -> bytes:
    fd = _open_regular_file(path, name=name)
    chunks: list[bytes] = []
    size = 0
    try:
        while True:
            chunk = os.read(fd, min(1024 * 1024, maximum + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum:
                raise ValueError(f"{name} exceeds the maximum size")
    except OSError as error:
        raise ValueError(f"{name} is unreadable") from error
    finally:
        os.close(fd)
    return b"".join(chunks)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _canonical_json_line(line: bytes, *, name: str) -> dict[str, object]:
    if not line or b"\n" in line or b"\r" in line:
        raise ValueError(f"{name} must contain exactly one LF-free JSON object")
    try:
        value = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object)
        canonical = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (UnicodeError, ValueError, TypeError, OverflowError, RecursionError) as error:
        raise ValueError(f"{name} must be finite canonical UTF-8 JSON") from error
    if not isinstance(value, dict) or canonical != line:
        raise ValueError(f"{name} bytes are not one canonical JSON object")
    return value


def _tracking_candidate_sha(record: dict[str, object]) -> str:
    subject = record.get("subject")
    if not isinstance(subject, dict) or set(subject) != {"candidate_sha", "repository"}:
        raise ValueError("tracking record subject has an invalid schema")
    candidate = subject.get("candidate_sha")
    if (
        not isinstance(candidate, str)
        or re.fullmatch(r"[0-9a-f]{40}", candidate) is None
        or subject.get("repository") != "Vzlentin/calibre"
    ):
        raise ValueError("tracking record subject does not bind the canonical repository and C0")
    return candidate


def _receipt_binds_record(
    receipt: dict[str, object],
    record: dict[str, object],
    *,
    candidate: str,
    record_digest: str,
) -> bool:
    expected_receipt_keys = {
        "candidate_sha",
        "proposal_artifact",
        "receipt_kind",
        "record_sha256",
        "repository",
        "result_artifact",
        "schema",
        "workflow",
    }
    if (
        set(receipt) != expected_receipt_keys
        or receipt.get("schema") != 1
        or receipt.get("receipt_kind") != "vn2-tracking-promotion-receipt"
        or receipt.get("repository") != "Vzlentin/calibre"
        or receipt.get("candidate_sha") != candidate
        or receipt.get("record_sha256") != record_digest
    ):
        return False
    workflow = receipt.get("workflow")
    result = receipt.get("result_artifact")
    proposal = receipt.get("proposal_artifact")
    if (
        not isinstance(workflow, dict)
        or set(workflow) != {"definition_ref", "definition_sha", "run_id", "run_url"}
        or workflow != record.get("workflow")
        or workflow.get("definition_ref")
        != "Vzlentin/calibre/.github/workflows/newcalibre.yml@refs/heads/main"
        or workflow.get("definition_sha") != candidate
        or not isinstance(workflow.get("run_id"), str)
        or re.fullmatch(r"[1-9][0-9]*", workflow["run_id"]) is None
        or workflow.get("run_url")
        != f"https://github.com/Vzlentin/calibre/actions/runs/{workflow['run_id']}"
        or not _valid_receipt_artifact(
            result,
            expected_name=f"vn2-acceptance-{candidate}",
        )
        or not _valid_receipt_artifact(
            proposal,
            expected_name=f"vn2-tracking-proposal-{candidate}",
        )
        or result != record.get("result_artifact")
    ):
        return False
    return result["id"] != proposal["id"]


def _valid_receipt_artifact(value: object, *, expected_name: str) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"digest", "id", "name"}
        and isinstance(value.get("id"), str)
        and re.fullmatch(r"[1-9][0-9]*", value["id"]) is not None
        and value.get("name") == expected_name
        and isinstance(value.get("digest"), str)
        and re.fullmatch(r"[0-9a-f]{64}", value["digest"]) is not None
    )


def _tracking_evidence(
    series_path: Path,
    *,
    problems: list[str],
) -> tuple[str | None, str | None, str | None]:
    try:
        payload = _read_regular_bytes(series_path, name="tracking series")
    except ValueError as error:
        if not os.path.lexists(series_path):
            problems.append("gate precondition unmet: no promoted tracking record (U9b)")
        else:
            problems.append(str(error))
        return None, None, None
    if not payload or not payload.endswith(b"\n") or payload.endswith(b"\r\n"):
        problems.append("tracking series must be non-empty canonical LF-terminated JSONL")
        return None, None, None
    rows = payload[:-1].split(b"\n")
    if not rows or any(not row for row in rows):
        problems.append("tracking series must not contain blank records")
        return None, None, None
    identities: list[object] = []
    latest: dict[str, object] | None = None
    try:
        for index, row in enumerate(rows):
            latest = _canonical_json_line(row, name=f"tracking series row {index + 1}")
            identities.append(latest.get("identity"))
    except ValueError as error:
        problems.append(str(error))
        return None, None, None
    if any(
        not isinstance(identity, str) or re.fullmatch(r"[0-9a-f]{64}", identity) is None
        for identity in identities
    ) or len(set(identities)) != len(identities):
        problems.append("tracking series identities must be unique lowercase SHA-256 values")
        return None, None, None
    assert latest is not None
    expected_record_keys = {
        "environment",
        "evidence",
        "identity",
        "objective",
        "record_kind",
        "result_artifact",
        "result_bundle",
        "schema",
        "subject",
        "workflow",
    }
    if (
        set(latest) != expected_record_keys
        or latest.get("schema") != 1
        or latest.get("record_kind") != "vn2-gate-a-tracking-record"
    ):
        problems.append("tracking series latest record has an invalid schema")
        return None, None, None
    try:
        c0 = _tracking_candidate_sha(latest)
    except ValueError as error:
        problems.append(str(error))
        return None, None, None
    record_digest = hashlib.sha256(rows[-1] + b"\n").hexdigest()
    receipt_path = series_path.parent / f"{c0}-receipt.json"
    try:
        receipt_bytes = _read_regular_bytes(receipt_path, name="tracking promotion receipt")
        if not receipt_bytes.endswith(b"\n") or receipt_bytes.endswith(b"\r\n"):
            raise ValueError("tracking promotion receipt must end with exactly one LF")
        receipt = _canonical_json_line(
            receipt_bytes[:-1],
            name="tracking promotion receipt",
        )
    except ValueError as error:
        problems.append(str(error))
        return None, record_digest, None
    if not _receipt_binds_record(
        receipt,
        latest,
        candidate=c0,
        record_digest=record_digest,
    ):
        problems.append("tracking promotion receipt does not bind the latest canonical record")
        return None, record_digest, None
    return c0, record_digest, hashlib.sha256(receipt_bytes).hexdigest()


def _capture_manifest_digests(root: Path, *, problems: list[str]) -> dict[str, str]:
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError:
        problems.append(f"promoted capture root is missing: {root.as_posix()}")
        return {}
    except OSError as error:
        problems.append(f"cannot inspect promoted capture root {root.as_posix()}: {error}")
        return {}
    if stat.S_ISLNK(root_mode):
        problems.append(f"promoted capture root must not be a symbolic link: {root.as_posix()}")
        return {}
    if not stat.S_ISDIR(root_mode):
        problems.append(f"promoted capture root must be a directory: {root.as_posix()}")
        return {}

    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        problems.append(f"cannot inspect promoted capture root {root.as_posix()}: {error}")
        return {}

    digests: dict[str, str] = {}
    for path in entries:
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            problems.append(f"cannot classify promoted capture entry {path.as_posix()}: {error}")
            continue
        if stat.S_ISLNK(mode):
            problems.append(
                f"promoted capture entry must not be a symbolic link: {path.as_posix()}"
            )
            continue
        if stat.S_ISREG(mode):
            continue
        if not stat.S_ISDIR(mode):
            problems.append(f"promoted capture entry has an invalid file type: {path.as_posix()}")
            continue

        manifest = path / "manifest.json"
        try:
            manifest_mode = manifest.lstat().st_mode
        except FileNotFoundError:
            problems.append(f"capture manifest is missing: {manifest.as_posix()}")
            continue
        except OSError as error:
            problems.append(f"cannot inspect capture manifest {manifest.as_posix()}: {error}")
            continue
        if stat.S_ISLNK(manifest_mode):
            problems.append(f"capture manifest must not be a symbolic link: {manifest.as_posix()}")
            continue
        if not stat.S_ISREG(manifest_mode):
            problems.append(f"capture manifest must be a regular file: {manifest.as_posix()}")
            continue
        try:
            digest = sha256_file(manifest)
        except OSError as error:
            problems.append(f"cannot digest capture manifest {manifest.as_posix()}: {error}")
            continue
        digests[manifest.relative_to(root).as_posix()] = digest
    return digests


def _oracle_binding(
    evidence_path: Path,
    junit_path: Path,
    *,
    candidate: str,
    required_outcomes: dict[str, dict[str, str]],
) -> tuple[dict[str, object] | None, list[str]]:
    problems: list[str] = []
    if not evidence_path.is_file():
        return None, ["aggregate is missing the oracle evidence JSON"]
    if not junit_path.is_file():
        return None, ["aggregate is missing the oracle pytest JUnit XML"]
    try:
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, ["aggregate oracle evidence JSON is invalid"]
    if not isinstance(evidence, dict):
        return None, ["aggregate oracle evidence root is not an object"]

    identity = evidence.get("identity")
    tests = evidence.get("tests")
    if not isinstance(identity, dict) or not isinstance(tests, dict):
        return None, ["aggregate oracle evidence lacks identity or test outcomes"]
    named = tests.get("named")
    if not isinstance(named, dict):
        named = {}

    if evidence.get("candidate_sha") != candidate:
        problems.append("oracle evidence candidate does not match Gate candidate")
    if identity.get("candidate_sha") != candidate or identity.get("head_sha") != candidate:
        problems.append("oracle evidence does not bind the checked-out Gate candidate")
    if evidence.get("status") != "passed" or evidence.get("problems") != []:
        problems.append("oracle evidence is not a clean passing result")
    if evidence.get("actuals_semantics") != "censored_sales_surrogate":
        problems.append("oracle evidence actuals_semantics is not the ratified surrogate label")
    for role in ("gate", "witness"):
        required = required_outcomes.get(role)
        if required is None:
            problems.append(f"oracle evidence cannot bind {role} without a valid Tier 3 inventory")
            continue
        outcome = named.get(role)
        if (
            not isinstance(outcome, dict)
            or set(outcome) != {"id", "node", "status"}
            or outcome.get("id") != required["id"]
            or outcome.get("node") != required["node"]
            or outcome.get("status") != "passed"
        ):
            problems.append(f"oracle evidence lacks a passing named {role} outcome")
    junit_digest = sha256_file(junit_path)
    if tests.get("junit_sha256") != junit_digest:
        problems.append("oracle evidence does not bind the downloaded JUnit bytes")
    if tests.get("junit_file") != junit_path.name:
        problems.append("oracle evidence names a different JUnit artifact")

    expected_workflow_sha = os.environ.get("GITHUB_WORKFLOW_SHA")
    if expected_workflow_sha and identity.get("workflow_sha") != expected_workflow_sha:
        problems.append("oracle evidence workflow SHA differs from the aggregate workflow SHA")
    expected_run_id = os.environ.get("GITHUB_RUN_ID")
    if expected_run_id and identity.get("run_id") != expected_run_id:
        problems.append("oracle evidence run ID differs from the aggregate run ID")

    return (
        {
            "candidate_sha": evidence.get("candidate_sha"),
            "evidence_sha256": sha256_file(evidence_path),
            "junit_sha256": junit_digest,
            "outcomes": named,
            "ref": identity.get("ref"),
            "run_id": identity.get("run_id"),
            "run_url": identity.get("run_url"),
            "workflow_ref": identity.get("workflow_ref"),
            "workflow_sha": identity.get("workflow_sha"),
        },
        problems,
    )


def _lane_provenance(
    lane_specs: list[str],
    *,
    candidate: str,
    problems: list[str],
) -> dict[str, dict[str, object]] | None:
    if not lane_specs:
        problems.append("Gate report requires every required lane result")
        return None
    results: dict[str, str] = {}
    for spec in lane_specs:
        name, separator, result = spec.partition("=")
        if not separator or not name or not result:
            problems.append(f"invalid lane result specification: {spec!r}")
            continue
        if name in results:
            problems.append(f"duplicate lane result: {name}")
            continue
        results[name] = result
    if set(results) != set(LANE_NAMES):
        problems.append("Gate report must record every required lane exactly once")

    common = {
        "candidate_sha": candidate,
        "event_name": os.environ.get("GITHUB_EVENT_NAME"),
        "ref": os.environ.get("GITHUB_REF"),
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_url": (
            f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}"
            f"/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
        ),
        "workflow_ref": os.environ.get("GITHUB_WORKFLOW_REF"),
        "workflow_sha": os.environ.get("GITHUB_WORKFLOW_SHA"),
    }
    for key in (
        "event_name",
        "ref",
        "repository",
        "run_attempt",
        "run_id",
        "workflow_ref",
        "workflow_sha",
    ):
        if not isinstance(common[key], str) or not common[key]:
            problems.append(f"lane provenance {key} is missing")

    lanes: dict[str, dict[str, object]] = {}
    for name in LANE_NAMES:
        result = results.get(name)
        lanes[name] = {"job": name, "result": result, **common}
        if result not in LANE_RESULTS:
            problems.append(f"lane {name} has invalid result {result!r}")
        elif result != "success":
            problems.append(f"lane {name} result={result}")
    return lanes


def _check_report(path: Path) -> int:
    if not path.is_file():
        print("Gate report was not emitted")
        return 1
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        print("Gate report is not valid JSON")
        return 1
    if not isinstance(report, dict) or not isinstance(report.get("problems"), list):
        print("Gate report does not expose a problem list")
        return 1
    lanes = report.get("lanes")
    lane_failure = not isinstance(lanes, dict) or set(lanes) != set(LANE_NAMES)
    if not lane_failure:
        lane_failure = any(
            not isinstance(value, dict) or value.get("result") != "success"
            for value in lanes.values()
        )
    problems = report["problems"]
    print(json.dumps({"problems": problems}, indent=2, sort_keys=True))
    return 1 if problems or lane_failure else 0


def main() -> int:
    """Assemble and write the Gate report; nonzero on a failed discipline check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-report", type=Path)
    parser.add_argument("--lane", action="append", default=[])
    parser.add_argument("--no-fail", action="store_true")
    parser.add_argument("--oracle-evidence", type=Path)
    parser.add_argument("--oracle-junit", type=Path)
    args = parser.parse_args()

    if args.check_report is not None:
        if args.out is not None or args.lane or args.oracle_evidence or args.oracle_junit:
            parser.error("--check-report cannot be combined with report-emission arguments")
        return _check_report(args.check_report)
    if args.out is None:
        parser.error("--out is required when emitting a Gate report")

    candidate = os.environ["CANDIDATE_SHA"]
    problems: list[str] = []
    required_outcomes = _load_required_oracle_outcomes(
        TIER3_INVENTORY_PATH,
        problems=problems,
    )
    lanes = _lane_provenance(args.lane, candidate=candidate, problems=problems)

    oracle_evidence = None
    if (args.oracle_evidence is None) != (args.oracle_junit is None):
        problems.append("Gate report requires oracle evidence JSON and JUnit XML together")
    elif args.oracle_evidence is not None and args.oracle_junit is not None:
        oracle_evidence, oracle_problems = _oracle_binding(
            args.oracle_evidence,
            args.oracle_junit,
            candidate=candidate,
            required_outcomes=required_outcomes,
        )
        problems.extend(oracle_problems)

    # Tracking record 1 (Gate precondition) and the C0→candidate merge discipline.
    c0, record_digest, receipt_digest = _tracking_evidence(
        TRACKING_SERIES,
        problems=problems,
    )
    diff_check = None
    if c0:
        try:
            changed = git("diff", "--name-only", f"{c0}..HEAD").splitlines()
        except subprocess.CalledProcessError as error:
            problems.append(f"C0→candidate diff inspection failed: {error}")
        else:
            offending = [path for path in changed if not path.startswith(EVIDENCE_ALLOWLIST)]
            diff_check = {"c0": c0, "changed": changed, "offending": offending}
            if offending:
                problems.append(
                    "C0→C1 diff leaves the evidence-path allowlist; "
                    "candidate void — re-mint at a new C0"
                )
    if diff_check is None:
        problems.append("C0→candidate evidence-only diff could not be established")

    # Budget check against the immutable activation record.
    now = datetime.now(UTC)
    budget = None
    record = find_activation_record()
    deadline = None
    deadline_text = None
    if record is not None and record_is_schema_complete(record):
        deadline_text = record["deadline"]
        deadline = parse_utc_timestamp(deadline_text)
    if deadline is not None:
        budget = {
            "deadline": deadline_text,
            "evaluated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inside_budget": now <= deadline,
        }
        if now > deadline:
            problems.append("evidence completed after the immutable deadline")
    else:
        problems.append("no schema-complete activation record on the Gate issue")

    head_sha = git("rev-parse", "HEAD")
    if head_sha != candidate:
        problems.append("aggregate checkout HEAD does not equal the Gate candidate")
    capture_manifest_digests = _capture_manifest_digests(CAPTURES_DIR, problems=problems)

    report = {
        "candidate_sha": candidate,
        "head_sha": head_sha,
        "workflow_sha": os.environ.get("GITHUB_WORKFLOW_SHA"),
        "run_id": os.environ.get("GITHUB_RUN_ID"),
        "run_url": (
            f"{os.environ.get('GITHUB_SERVER_URL')}/{os.environ.get('GITHUB_REPOSITORY')}"
            f"/actions/runs/{os.environ.get('GITHUB_RUN_ID')}"
        ),
        "emitted_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "environment": {
            "arch": platform.machine(),
            "os_release": platform.freedesktop_os_release().get("PRETTY_NAME", "unknown"),
            "python": platform.python_version(),
            "runner_image": (
                f"{os.environ.get('ImageOS', '?')}/{os.environ.get('ImageVersion', '?')}"  # noqa: SIM112
            ),
        },
        "digests": {
            "input_inventory": (sha256_file(INPUT_INVENTORY) if INPUT_INVENTORY.exists() else None),
            "capture_manifests": capture_manifest_digests,
            "tracking_record": record_digest,
            "tracking_receipt": receipt_digest,
        },
        "oracle_evidence": oracle_evidence,
        "lanes": lanes,
        "c0_c1_discipline": diff_check,
        "budget": budget,
        "problems": problems,
        "lanes_note": "every required needs.<lane>.result is recorded with run provenance",
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_digest = sha256_file(args.out)
    print(f"gate report digest: {report_digest}")
    print(json.dumps({"problems": problems}, indent=2))
    return 0 if args.no_fail else (1 if problems else 0)


if __name__ == "__main__":
    sys.exit(main())
