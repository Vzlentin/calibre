"""Emit and validate candidate-qualified Stage 3 oracle evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import stat
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, cast

ARTIFACT_KIND = "stage3-oracle-evidence"
ACTUALS_SEMANTICS = "censored_sales_surrogate"
THREAD_VARIABLES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TIER3_INVENTORY_PATH = REPOSITORY_ROOT / "newcalibre" / "tests" / "tier3" / "oracle_inventory.json"


def _load_required_tier3_outcomes(
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
    """Return the sha256 of a file's exact bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, *, problems: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        problems.append(f"missing required file: {path.as_posix()}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        problems.append(f"invalid JSON at {path.as_posix()}: {error}")
        return None
    if not isinstance(value, dict):
        problems.append(f"JSON root must be an object: {path.as_posix()}")
        return None
    return value


def _command_version(command: str) -> str | None:
    try:
        completed = subprocess.run(
            (command, "--version"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = completed.stdout.strip() or completed.stderr.strip()
    return value if completed.returncode == 0 and value else None


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and HEX40.fullmatch(value) else None


def _read_actuals_semantics(path: Path, *, problems: list[str]) -> str | None:
    if not path.is_file():
        problems.append(f"missing required file: {path.as_posix()}")
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"actuals_semantics:\s*([^\s#]+)\s*(?:#.*)?", line)
        if match:
            return match.group(1).strip("\"'")
    problems.append(f"actuals_semantics is absent from {path.as_posix()}")
    return None


def _capture_paths(
    root: Path,
    *,
    problems: list[str],
) -> tuple[Path | None, Path | None, bool]:
    try:
        root_mode = root.lstat().st_mode
    except FileNotFoundError:
        return None, None, True
    except OSError as error:
        problems.append(f"invalid promoted capture root {root.as_posix()}: {error}")
        return None, None, False
    if stat.S_ISLNK(root_mode):
        problems.append(f"promoted capture root must not be a symbolic link: {root.as_posix()}")
        return None, None, False
    if not stat.S_ISDIR(root_mode):
        problems.append(f"promoted capture root must be a directory: {root.as_posix()}")
        return None, None, False

    try:
        entries = sorted(root.iterdir(), key=lambda path: path.name)
    except OSError as error:
        problems.append(f"cannot inspect promoted capture root {root.as_posix()}: {error}")
        return None, None, False

    bundle_roots: list[Path] = []
    receipt_paths: list[Path] = []
    for path in entries:
        try:
            mode = path.lstat().st_mode
        except OSError as error:
            problems.append(f"cannot classify promoted capture entry {path.as_posix()}: {error}")
            continue
        if stat.S_ISLNK(mode):
            if HEX40.fullmatch(path.name):
                name = "promoted capture bundle root"
            elif path.name.endswith("-receipt.json"):
                name = "promoted capture receipt"
            else:
                name = "promoted capture entry"
            problems.append(f"{name} must not be a symbolic link: {path.as_posix()}")
        elif stat.S_ISDIR(mode):
            bundle_roots.append(path)
        elif stat.S_ISREG(mode):
            receipt_paths.append(path)
        else:
            problems.append(f"promoted capture entry has an invalid file type: {path.as_posix()}")
    if len(bundle_roots) != 1 or HEX40.fullmatch(bundle_roots[0].name) is None:
        problems.append("promoted capture root must contain exactly one 40-SHA bundle directory")
        return None, None, False
    expected_receipt = root / f"{bundle_roots[0].name}-receipt.json"
    if receipt_paths != [expected_receipt]:
        problems.append("promoted capture root must contain only the bundle's matching receipt")
        return bundle_roots[0], None, False
    return bundle_roots[0], expected_receipt, False


def _capture_manifest_path(bundle_root: Path, *, problems: list[str]) -> Path | None:
    path = bundle_root / "manifest.json"
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        problems.append(f"missing required file: {path.as_posix()}")
        return None
    except OSError as error:
        problems.append(f"cannot inspect capture manifest {path.as_posix()}: {error}")
        return None
    if stat.S_ISLNK(mode):
        problems.append(f"capture manifest must not be a symbolic link: {path.as_posix()}")
        return None
    if not stat.S_ISREG(mode):
        problems.append(f"capture manifest must be a regular file: {path.as_posix()}")
        return None
    return path


def _placeholder_junit(
    path: Path,
    *,
    skipped: bool,
    required_outcomes: dict[str, dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    test_count = len(required_outcomes)
    suite = ET.Element(
        "testsuite",
        {
            "name": "stage3-oracle",
            "tests": str(test_count),
            "failures": "0",
            "errors": "0" if skipped else str(test_count),
            "skipped": str(test_count) if skipped else "0",
        },
    )
    for role in ("gate", "witness"):
        required = required_outcomes.get(role)
        if required is None:
            continue
        node = required["node"]
        classname, name = node.rsplit("::", 1)
        case = ET.SubElement(suite, "testcase", {"classname": classname, "name": name})
        if skipped:
            ET.SubElement(case, "skipped", {"message": "oracle test step was skipped"})
        else:
            ET.SubElement(case, "error", {"message": "pytest did not emit JUnit output"})
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


def _case_status(case: ET.Element) -> str:
    if case.find("error") is not None:
        return "error"
    if case.find("failure") is not None:
        return "failed"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


def _pytest_outcomes(
    junit_path: Path,
    *,
    required_outcomes: dict[str, dict[str, str]],
    problems: list[str],
) -> dict[str, object]:
    try:
        root = ET.parse(junit_path).getroot()
    except (OSError, ET.ParseError) as error:
        problems.append(f"invalid pytest JUnit XML: {error}")
        return {
            "junit_file": junit_path.name,
            "junit_sha256": None,
            "summary": {"total": 0, "passed": 0, "failed": 0, "error": 0, "skipped": 0},
            "named": {},
        }

    cases = list(root.iter("testcase"))
    by_node: dict[str, list[ET.Element]] = {}
    summary = {"total": len(cases), "passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for case in cases:
        status = _case_status(case)
        summary[status] += 1
        node = f"{case.get('classname', '')}::{case.get('name', '')}"
        by_node.setdefault(node, []).append(case)

    named: dict[str, object] = {}
    for role in ("gate", "witness"):
        required = required_outcomes.get(role)
        if required is None:
            continue
        matches = by_node.get(required["node"], [])
        if len(matches) != 1:
            problems.append(f"pytest JUnit must contain exactly one named {role} outcome")
            continue
        case = matches[0]
        named[role] = {
            **required,
            "status": _case_status(case),
        }

    return {
        "junit_file": junit_path.name,
        "junit_sha256": sha256_file(junit_path),
        "summary": summary,
        "named": named,
    }


def _evidence_status(
    test_outcome: str,
    named: object,
    *,
    capture_root_absent: bool,
    problems: list[str],
) -> str:
    if problems:
        return "failed"
    skipped = test_outcome == "skipped"
    expected_named_status = "skipped" if skipped else "passed"
    if not isinstance(named, dict) or set(named) != {"gate", "witness"}:
        return "failed"
    named_values = cast(dict[str, object], named).values()
    if not all(
        isinstance(value, dict)
        and cast(dict[str, object], value).get("status") == expected_named_status
        for value in named_values
    ):
        return "failed"
    if skipped:
        return "skipped" if capture_root_absent else "failed"
    return "passed" if test_outcome == "success" else "failed"


def _environment(
    path: Path,
    lock_path: Path,
    *,
    allow_missing: bool,
    problems: list[str],
) -> dict[str, object]:
    try:
        path.lstat()
    except FileNotFoundError:
        recorded = {} if allow_missing else (_json_object(path, problems=problems) or {})
    except OSError as error:
        problems.append(f"cannot inspect oracle environment file {path.as_posix()}: {error}")
        recorded = {}
    else:
        recorded = _json_object(path, problems=problems) or {}
    os_release = recorded.get("os")
    numerical = {
        "numpy": recorded.get("numpy"),
        "numpy_config": recorded.get("numpy_config"),
    }
    return {
        "arch": recorded.get("arch") or platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_model": recorded.get("cpu_model") or platform.processor() or platform.machine(),
        "lock_sha256": sha256_file(lock_path) if lock_path.is_file() else None,
        "numerical_stack": numerical,
        "os": os_release
        if isinstance(os_release, dict)
        else {"id": None, "pretty_name": None, "version_id": None},
        "python": recorded.get("python") or platform.python_version(),
        "runner_image": recorded.get("runner_image"),
        "thread_policy": recorded.get("thread_policy")
        if isinstance(recorded.get("thread_policy"), dict)
        else {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "uv": _command_version("uv"),
    }


def emit_evidence(
    *,
    candidate_sha: str,
    environment_path: Path,
    junit_path: Path,
    output_path: Path,
    test_outcome: str,
) -> dict[str, object]:
    """Write one canonical evidence object and return its in-memory value."""
    problems: list[str] = []
    required_outcomes = _load_required_tier3_outcomes(
        TIER3_INVENTORY_PATH,
        problems=problems,
    )
    if HEX40.fullmatch(candidate_sha) is None:
        problems.append("candidate_sha must be a full lowercase 40-hex SHA")

    skipped = test_outcome == "skipped"
    if not junit_path.is_file():
        _placeholder_junit(
            junit_path,
            skipped=skipped,
            required_outcomes=required_outcomes,
        )
        if not skipped:
            problems.append("pytest did not emit its required JUnit outcome file")

    config_path = Path("newcalibre/benchmarks/vn2/protocol.yaml")
    oracle_config_path = Path("benchmarks/vn2/config/vn2-winning-loop.yaml")
    input_path = Path("stage3/evidence/vn2-input-digests.json")
    successor_input_path = Path("newcalibre/benchmarks/vn2/vn2-input-digests.json")
    lock_path = Path("newcalibre/uv.lock")
    captures_root = Path("stage3/evidence/captures")
    bundle_root, receipt_path, capture_root_absent = _capture_paths(
        captures_root,
        problems=problems,
    )
    if capture_root_absent and not skipped:
        problems.append(f"missing promoted capture root: {captures_root.as_posix()}")
    elif skipped and not capture_root_absent:
        problems.append("skipped oracle outcome requires the promoted capture root to be absent")
    elif skipped and os.environ.get("GITHUB_EVENT_NAME") != "schedule":
        problems.append("only scheduled runs may skip an absent promoted capture root")
    manifest_path = (
        _capture_manifest_path(bundle_root, problems=problems) if bundle_root is not None else None
    )
    manifest = _json_object(manifest_path, problems=problems) if manifest_path is not None else None
    receipt = _json_object(receipt_path, problems=problems) if receipt_path is not None else None
    semantics = _read_actuals_semantics(config_path, problems=problems)
    if manifest is not None and manifest.get("actuals_semantics") != semantics:
        problems.append("capture and successor actuals_semantics do not match")

    def digest(path: Path) -> str | None:
        if not path.is_file():
            problems.append(f"missing required file: {path.as_posix()}")
            return None
        return sha256_file(path)

    head_sha = _git_head()
    identity = {
        "candidate_sha": candidate_sha,
        "event_name": os.environ.get("GITHUB_EVENT_NAME"),
        "event_sha": os.environ.get("GITHUB_SHA"),
        "head_sha": head_sha,
        "job": os.environ.get("GITHUB_JOB"),
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
    if head_sha != candidate_sha:
        problems.append("checked-out HEAD does not equal candidate_sha")

    outcomes = _pytest_outcomes(
        junit_path,
        required_outcomes=required_outcomes,
        problems=problems,
    )

    digests = {
        "capture_inner_bundle": manifest.get("inner_bundle_digest") if manifest else None,
        "capture_manifest": digest(manifest_path) if manifest_path is not None else None,
        "capture_payload": manifest.get("capture_digest") if manifest else None,
        "input_inventory": digest(input_path),
        "oracle_config": digest(oracle_config_path),
        "receipt": digest(receipt_path) if receipt_path is not None else None,
        "receipt_artifact": receipt.get("artifact_digest") if receipt else None,
        "successor_config": digest(config_path),
        "successor_input_inventory": digest(successor_input_path),
    }
    environment = _environment(
        environment_path,
        lock_path,
        allow_missing=skipped and capture_root_absent,
        problems=problems,
    )
    status = _evidence_status(
        test_outcome,
        outcomes["named"],
        capture_root_absent=capture_root_absent,
        problems=problems,
    )

    report: dict[str, object] = {
        "actuals_semantics": semantics,
        "artifact_kind": ARTIFACT_KIND,
        "candidate_sha": candidate_sha,
        "digests": digests,
        "environment": environment,
        "identity": identity,
        "problems": sorted(set(problems)),
        "schema": 1,
        "status": status,
        "tests": outcomes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"oracle evidence -> {output_path}")
    return report


def _require_mapping(value: object, *, name: str, problems: list[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        problems.append(f"{name} must be an object")
        return {}
    return cast(dict[str, Any], value)


def _require_exact_keys(
    value: dict[str, Any], expected: set[str], *, name: str, problems: list[str]
) -> None:
    if set(value) != expected:
        problems.append(f"{name} must contain exactly {sorted(expected)!r}")


def validate_evidence(
    evidence_path: Path,
    junit_path: Path,
    *,
    allow_skipped: bool,
) -> list[str]:
    """Return every evidence-contract violation without mutating the artifact."""
    problems: list[str] = []
    required_outcomes = _load_required_tier3_outcomes(
        TIER3_INVENTORY_PATH,
        problems=problems,
    )
    report = _json_object(evidence_path, problems=problems)
    if report is None:
        return problems
    _require_exact_keys(
        report,
        {
            "actuals_semantics",
            "artifact_kind",
            "candidate_sha",
            "digests",
            "environment",
            "identity",
            "problems",
            "schema",
            "status",
            "tests",
        },
        name="oracle evidence",
        problems=problems,
    )
    candidate = report.get("candidate_sha")
    if not isinstance(candidate, str) or HEX40.fullmatch(candidate) is None:
        problems.append("candidate_sha must be a full lowercase 40-hex SHA")
        candidate = ""
    if evidence_path.name != f"oracle-evidence-{candidate}.json":
        problems.append("oracle evidence filename must be candidate-qualified")
    if report.get("artifact_kind") != ARTIFACT_KIND or report.get("schema") != 1:
        problems.append("oracle evidence schema identity is invalid")
    if report.get("actuals_semantics") != ACTUALS_SEMANTICS:
        problems.append("actuals_semantics must equal censored_sales_surrogate")

    status = report.get("status")
    if status == "skipped":
        if not allow_skipped:
            problems.append("skipped oracle evidence is not allowed for this run")
    elif status != "passed":
        problems.append("oracle evidence status must be passed")
    recorded_problems = report.get("problems")
    if recorded_problems != []:
        problems.append("passing or skipped oracle evidence must contain no recorded problems")

    identity = _require_mapping(report.get("identity"), name="identity", problems=problems)
    _require_exact_keys(
        identity,
        {
            "candidate_sha",
            "event_name",
            "event_sha",
            "head_sha",
            "job",
            "ref",
            "repository",
            "run_attempt",
            "run_id",
            "run_url",
            "workflow_ref",
            "workflow_sha",
        },
        name="identity",
        problems=problems,
    )
    if identity.get("candidate_sha") != candidate or identity.get("head_sha") != candidate:
        problems.append("identity must bind candidate_sha to checked-out HEAD")
    for name in ("ref", "repository", "run_id", "run_url", "workflow_ref", "workflow_sha"):
        if not isinstance(identity.get(name), str) or not identity[name]:
            problems.append(f"identity.{name} must be non-empty")
    if (
        not isinstance(identity.get("workflow_sha"), str)
        or HEX40.fullmatch(identity.get("workflow_sha", "")) is None
    ):
        problems.append("identity.workflow_sha must be a full lowercase 40-hex SHA")

    environment = _require_mapping(report.get("environment"), name="environment", problems=problems)
    _require_exact_keys(
        environment,
        {
            "arch",
            "cpu_count",
            "cpu_model",
            "lock_sha256",
            "numerical_stack",
            "os",
            "python",
            "runner_image",
            "thread_policy",
            "uv",
        },
        name="environment",
        problems=problems,
    )
    if status != "skipped":
        if environment.get("arch") != "x86_64":
            problems.append("environment.arch must equal x86_64")
        os_release = _require_mapping(
            environment.get("os"),
            name="environment.os",
            problems=problems,
        )
        if os_release.get("id") != "ubuntu" or os_release.get("version_id") != "24.04":
            problems.append("environment.os must bind Ubuntu 24.04")
        if not str(environment.get("python", "")).startswith("3.12."):
            problems.append("environment.python must bind a Python 3.12 patch release")
        if not str(environment.get("runner_image", "")).startswith("ubuntu24/"):
            problems.append("environment.runner_image must bind a versioned ubuntu24 image")
        if not isinstance(environment.get("cpu_count"), int) or environment["cpu_count"] < 1:
            problems.append("environment.cpu_count must be a positive integer")
        for name in ("cpu_model", "uv"):
            if not isinstance(environment.get(name), str) or not environment[name]:
                problems.append(f"environment.{name} must be non-empty")
        if (
            not isinstance(environment.get("lock_sha256"), str)
            or HEX64.fullmatch(environment.get("lock_sha256", "")) is None
        ):
            problems.append("environment.lock_sha256 must be a sha256")
        numerical = _require_mapping(
            environment.get("numerical_stack"),
            name="environment.numerical_stack",
            problems=problems,
        )
        numerical_fields_present = all(
            isinstance(numerical.get(key), str) and numerical[key]
            for key in ("numpy", "numpy_config")
        )
        if not numerical_fields_present:
            problems.append("environment.numerical_stack must bind NumPy and BLAS provenance")
        thread_policy = _require_mapping(
            environment.get("thread_policy"),
            name="environment.thread_policy",
            problems=problems,
        )
        if set(thread_policy) != set(THREAD_VARIABLES) or set(thread_policy.values()) != {"1"}:
            problems.append(
                "environment.thread_policy must pin every numerical thread variable to 1"
            )

    digests = _require_mapping(report.get("digests"), name="digests", problems=problems)
    expected_digest_keys = {
        "capture_inner_bundle",
        "capture_manifest",
        "capture_payload",
        "input_inventory",
        "oracle_config",
        "receipt",
        "receipt_artifact",
        "successor_config",
        "successor_input_inventory",
    }
    _require_exact_keys(digests, expected_digest_keys, name="digests", problems=problems)
    if status != "skipped":
        for name in expected_digest_keys:
            if not isinstance(digests.get(name), str) or HEX64.fullmatch(digests[name]) is None:
                problems.append(f"digests.{name} must be a sha256")

    tests = _require_mapping(report.get("tests"), name="tests", problems=problems)
    _require_exact_keys(
        tests,
        {"junit_file", "junit_sha256", "named", "summary"},
        name="tests",
        problems=problems,
    )
    if tests.get("junit_file") != junit_path.name:
        problems.append("tests.junit_file does not name the uploaded JUnit artifact")
    if not junit_path.is_file() or tests.get("junit_sha256") != sha256_file(junit_path):
        problems.append("tests.junit_sha256 does not bind the uploaded JUnit bytes")
    named = _require_mapping(tests.get("named"), name="tests.named", problems=problems)
    if set(named) != {"gate", "witness"}:
        problems.append("tests.named must contain separate gate and witness outcomes")
    expected_named_status = "skipped" if status == "skipped" else "passed"
    for role in ("gate", "witness"):
        required = required_outcomes.get(role)
        if required is None:
            continue
        outcome = _require_mapping(named.get(role), name=f"tests.named.{role}", problems=problems)
        _require_exact_keys(
            outcome,
            {"id", "node", "status"},
            name=f"tests.named.{role}",
            problems=problems,
        )
        if outcome.get("id") != required["id"]:
            problems.append(f"tests.named.{role}.id must equal {required['id']}")
        if outcome.get("node") != required["node"]:
            problems.append(f"tests.named.{role}.node must equal {required['node']}")
        if outcome.get("status") != expected_named_status:
            problems.append(f"tests.named.{role}.status must equal {expected_named_status}")
    return sorted(set(problems))


def main() -> int:
    """Dispatch evidence emission or validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    emit = commands.add_parser("emit")
    emit.add_argument("--candidate-sha", required=True)
    emit.add_argument("--environment", type=Path, required=True)
    emit.add_argument("--junit", type=Path, required=True)
    emit.add_argument("--out", type=Path, required=True)
    emit.add_argument("--test-outcome", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--evidence", type=Path, required=True)
    validate.add_argument("--junit", type=Path, required=True)
    validate.add_argument("--allow-skipped", choices=("true", "false"), default="false")
    args = parser.parse_args()
    if args.command == "emit":
        emit_evidence(
            candidate_sha=args.candidate_sha,
            environment_path=args.environment,
            junit_path=args.junit,
            output_path=args.out,
            test_outcome=args.test_outcome,
        )
        return 0
    problems = validate_evidence(
        args.evidence,
        args.junit,
        allow_skipped=args.allow_skipped == "true",
    )
    print(json.dumps({"problems": problems}, indent=2, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
