"""Own strict, content-bound oracle capture evidence validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast
from urllib.parse import urlparse

ORACLE_TAG = "oracle-freeze-2026-07-06"
ORACLE_COMMIT = "686a1b284a4f4879123b4095d306f07b88d2ddc3"
ORACLE_LOCK_SHA256 = "5cc585d347195861d81760e16a675bd2a05b51777cf90c13d9af0ab05bb743f3"
CAPTURE_KIND = "vn2-oracle-orders"
ACTUALS_SEMANTICS = "censored_sales_surrogate"
GITHUB_REPOSITORY = "Vzlentin/calibre"
VN2_CONFIG_PATH = "benchmarks/vn2/config/vn2-winning-loop.yaml"
VN2_INPUT_INVENTORY_PATH = "stage3/evidence/vn2-input-digests.json"
EXPECTED_ROUNDS = 6
EXPECTED_SERIES = 599
THREAD_VARIABLES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_RUN_ID = re.compile(r"[1-9][0-9]*")
_ARTIFACT_NAME = re.compile(r"oracle-capture-([0-9a-f]{40})")
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "artifact_kind",
        "artifact_name",
        "candidate_sha",
        "workflow_sha",
        "oracle_tag",
        "oracle_commit",
        "oracle_lock_sha256",
        "run_id",
        "run_url",
        "config_digest",
        "input_inventory",
        "input_inventory_digest",
        "actuals_semantics",
        "capture_digest",
        "environment",
        "environment_digest",
        "files",
        "inner_bundle_digest",
    }
)
_ENVIRONMENT_KEYS = frozenset(
    {
        "arch",
        "cpu_model",
        "os",
        "python",
        "numpy",
        "numpy_config",
        "runner_image",
        "thread_policy",
    }
)
_OS_KEYS = frozenset({"id", "version_id", "pretty_name"})
_FILE_KEYS = frozenset({"path", "bytes", "sha256"})
_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "artifact_id",
        "artifact_digest",
        "artifact_name",
        "producer_sha",
        "workflow_sha",
        "workflow_run_id",
        "run_url",
        "manifest_sha256",
        "inner_bundle_digest",
        "environment_digest",
    }
)


class OracleEvidenceError(ValueError):
    """Report incomplete, malformed, or content-mismatched oracle evidence."""


@dataclass(frozen=True, slots=True)
class CaptureFile:
    """Bind one canonical payload path to its exact bytes and digest."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class CaptureEnvironment:
    """Retain the complete Gate-A provenance facts for the oracle process."""

    arch: str
    cpu_model: str
    os_id: str
    os_version_id: str
    os_pretty_name: str
    python: str
    numpy: str
    numpy_config: str
    runner_image: str
    thread_policy: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CaptureManifest:
    """Expose a fully validated, immutable capture manifest."""

    artifact_name: str
    candidate_sha: str
    workflow_sha: str
    run_id: str
    run_url: str
    config_digest: str
    input_inventory: str
    input_inventory_digest: str
    actuals_semantics: str
    capture_digest: str
    environment: CaptureEnvironment
    environment_digest: str
    files: tuple[CaptureFile, ...]
    inner_bundle_digest: str


@dataclass(frozen=True, slots=True)
class CaptureBundle:
    """Return one content-verified bundle plus the exact manifest digest."""

    root: Path
    manifest: CaptureManifest
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class CaptureReceipt:
    """Bind promoted bytes to the GitHub artifact that supplied them."""

    artifact_id: str
    artifact_digest: str
    artifact_name: str
    producer_sha: str
    workflow_sha: str
    workflow_run_id: str
    run_url: str
    manifest_sha256: str
    inner_bundle_digest: str
    environment_digest: str


def validate_promoted_captures_root(root: Path) -> Path:
    """Require a real directory at the promoted-captures trust boundary."""
    return _require_nonsymlink_directory(Path(root), name="promoted captures root")


def validate_capture_bundle(
    root: Path,
    *,
    expected_candidate_sha: str,
    expected_workflow_sha: str,
    expected_run_id: str,
    expected_config_path: Path,
    expected_input_inventory_path: Path,
) -> CaptureBundle:
    """Validate identity, manifest completeness, payload shape, and every byte."""
    bundle_root = _require_nonsymlink_directory(Path(root), name="capture bundle")
    manifest_path = bundle_root / "manifest.json"
    listing_path = bundle_root / "files.sha256"
    manifest_value, manifest_bytes = _load_json_object(manifest_path, name="capture manifest")
    _require_exact_keys(manifest_value, _MANIFEST_KEYS, name="capture manifest")
    _require_exact_int(manifest_value["schema"], expected=1, name="capture manifest schema")
    if manifest_value["artifact_kind"] != CAPTURE_KIND:
        raise OracleEvidenceError(f"capture artifact_kind must equal {CAPTURE_KIND!r}")

    candidate_sha = _require_commit_sha(manifest_value["candidate_sha"], name="candidate_sha")
    workflow_sha = _require_commit_sha(manifest_value["workflow_sha"], name="workflow_sha")
    run_id = _require_run_id(manifest_value["run_id"], name="run_id")
    artifact_name = _require_text(manifest_value["artifact_name"], name="artifact_name")
    if artifact_name != f"oracle-capture-{candidate_sha}":
        raise OracleEvidenceError("capture artifact_name must bind the candidate SHA")
    _require_expected(candidate_sha, expected_candidate_sha, name="candidate_sha")
    _require_expected(workflow_sha, expected_workflow_sha, name="workflow_sha")
    _require_expected(run_id, expected_run_id, name="run_id")
    if manifest_value["oracle_tag"] != ORACLE_TAG:
        raise OracleEvidenceError(f"capture oracle_tag must equal {ORACLE_TAG!r}")
    if manifest_value["oracle_commit"] != ORACLE_COMMIT:
        raise OracleEvidenceError("capture oracle_commit does not match the pinned tag")
    if manifest_value["oracle_lock_sha256"] != ORACLE_LOCK_SHA256:
        raise OracleEvidenceError("capture oracle lock digest does not match the pin")
    if manifest_value["actuals_semantics"] != ACTUALS_SEMANTICS:
        raise OracleEvidenceError(
            "VN2 oracle capture actuals_semantics must be censored_sales_surrogate"
        )

    run_url = _validate_run_url(manifest_value["run_url"], run_id=run_id)
    config_digest = _require_sha256(manifest_value["config_digest"], name="config_digest")
    _require_trusted_digest(
        config_digest,
        expected_config_path,
        name="config_digest",
    )
    input_inventory = _require_payload_path(
        manifest_value["input_inventory"],
        name="input_inventory",
    )
    if input_inventory != VN2_INPUT_INVENTORY_PATH:
        raise OracleEvidenceError(
            f"capture input_inventory must equal {VN2_INPUT_INVENTORY_PATH!r}"
        )
    input_inventory_digest = _require_sha256(
        manifest_value["input_inventory_digest"],
        name="input_inventory_digest",
    )
    _require_trusted_digest(
        input_inventory_digest,
        expected_input_inventory_path,
        name="input_inventory_digest",
    )
    environment = _validate_environment(manifest_value["environment"])
    files = _validate_file_entries(manifest_value["files"])
    capture_digest = _require_sha256(
        manifest_value["capture_digest"],
        name="capture_digest",
    )
    expected_capture_digest = _capture_digest(files)
    if capture_digest != expected_capture_digest:
        raise OracleEvidenceError(
            "capture_digest does not match the canonical oracle payload listing"
        )
    inner_digest = _require_sha256(
        manifest_value["inner_bundle_digest"],
        name="inner_bundle_digest",
    )
    environment_digest = _require_sha256(
        manifest_value["environment_digest"],
        name="environment_digest",
    )
    expected_environment_digest = _environment_digest(
        environment=environment,
        config_digest=config_digest,
        input_digest=input_inventory_digest,
        capture_digest=capture_digest,
        actuals_semantics=ACTUALS_SEMANTICS,
    )
    if environment_digest != expected_environment_digest:
        raise OracleEvidenceError(
            "capture environment_digest does not match the canonical comparability key"
        )

    expected_listing = "".join(f"{entry.sha256}  {entry.path}\n" for entry in files).encode()
    try:
        actual_listing = listing_path.read_bytes()
    except OSError as error:
        raise OracleEvidenceError("capture bundle is missing files.sha256") from error
    if actual_listing != expected_listing:
        raise OracleEvidenceError("files.sha256 does not exactly match manifest payload entries")
    if _sha256_bytes(actual_listing) != inner_digest:
        raise OracleEvidenceError("capture inner bundle digest does not match files.sha256")

    expected_paths = {"manifest.json", "files.sha256", *(entry.path for entry in files)}
    actual_paths: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise OracleEvidenceError("capture bundle paths must not be symbolic links")
        if not path.is_file():
            continue
        actual_paths.add(path.relative_to(bundle_root).as_posix())
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths, key=str.encode)
        extra = sorted(actual_paths - expected_paths, key=str.encode)
        raise OracleEvidenceError(
            f"capture file set mismatch: missing={missing!r}, extra={extra!r}"
        )
    for entry in files:
        payload = bundle_root / Path(*PurePosixPath(entry.path).parts)
        if payload.stat().st_size != entry.bytes:
            raise OracleEvidenceError(f"capture payload size mismatch: {entry.path}")
        if _sha256_file(payload) != entry.sha256:
            raise OracleEvidenceError(f"capture payload digest mismatch: {entry.path}")

    manifest = CaptureManifest(
        artifact_name=artifact_name,
        candidate_sha=candidate_sha,
        workflow_sha=workflow_sha,
        run_id=run_id,
        run_url=run_url,
        config_digest=config_digest,
        input_inventory=input_inventory,
        input_inventory_digest=input_inventory_digest,
        actuals_semantics=ACTUALS_SEMANTICS,
        capture_digest=capture_digest,
        environment=environment,
        environment_digest=environment_digest,
        files=files,
        inner_bundle_digest=inner_digest,
    )
    _validate_vn2_payloads(bundle_root, manifest)
    return CaptureBundle(
        root=bundle_root.resolve(),
        manifest=manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
    )


def validate_capture_receipt(
    path: Path,
    *,
    bundle: CaptureBundle,
    expected_artifact_id: str,
    expected_artifact_digest: str,
    expected_artifact_name: str,
    expected_producer_sha: str,
    expected_workflow_sha: str,
    expected_workflow_run_id: str,
) -> CaptureReceipt:
    """Validate a promotion receipt against already verified bundle bytes."""
    receipt_path = _require_nonsymlink_file(Path(path), name="capture receipt")
    value, _receipt_bytes = _load_json_object(receipt_path, name="capture receipt")
    _require_exact_keys(value, _RECEIPT_KEYS, name="capture receipt")
    _require_exact_int(value["schema"], expected=1, name="capture receipt schema")
    artifact_id = _require_run_id(value["artifact_id"], name="artifact_id")
    _require_expected(artifact_id, expected_artifact_id, name="artifact_id")
    artifact_digest = _require_sha256(value["artifact_digest"], name="artifact_digest")
    _require_expected(
        artifact_digest,
        expected_artifact_digest,
        name="artifact_digest",
    )
    artifact_name = _require_text(value["artifact_name"], name="artifact_name")
    producer_sha = _require_commit_sha(value["producer_sha"], name="producer_sha")
    workflow_sha = _require_commit_sha(value["workflow_sha"], name="workflow_sha")
    workflow_run_id = _require_run_id(value["workflow_run_id"], name="workflow_run_id")
    run_url = _validate_run_url(value["run_url"], run_id=workflow_run_id)
    manifest_sha256 = _require_sha256(value["manifest_sha256"], name="manifest_sha256")
    inner_bundle_digest = _require_sha256(
        value["inner_bundle_digest"],
        name="inner_bundle_digest",
    )
    environment_digest = _require_sha256(
        value["environment_digest"],
        name="environment_digest",
    )
    manifest = bundle.manifest
    comparisons = {
        "artifact_name (requested)": (artifact_name, expected_artifact_name),
        "producer_sha (requested)": (producer_sha, expected_producer_sha),
        "workflow_sha (requested)": (workflow_sha, expected_workflow_sha),
        "workflow_run_id (requested)": (workflow_run_id, expected_workflow_run_id),
        "artifact_name": (artifact_name, manifest.artifact_name),
        "producer_sha": (producer_sha, manifest.candidate_sha),
        "workflow_sha": (workflow_sha, manifest.workflow_sha),
        "workflow_run_id": (workflow_run_id, manifest.run_id),
        "run_url": (run_url, manifest.run_url),
        "manifest_sha256": (manifest_sha256, bundle.manifest_sha256),
        "inner_bundle_digest": (inner_bundle_digest, manifest.inner_bundle_digest),
        "environment_digest": (environment_digest, manifest.environment_digest),
    }
    for name, (actual, expected) in comparisons.items():
        if actual != expected:
            raise OracleEvidenceError(f"capture receipt {name} does not match the bundle")
    return CaptureReceipt(
        artifact_id=artifact_id,
        artifact_digest=artifact_digest,
        artifact_name=artifact_name,
        producer_sha=producer_sha,
        workflow_sha=workflow_sha,
        workflow_run_id=workflow_run_id,
        run_url=run_url,
        manifest_sha256=manifest_sha256,
        inner_bundle_digest=inner_bundle_digest,
        environment_digest=environment_digest,
    )


def validate_promoted_capture(
    bundle_root: Path,
    receipt_path: Path,
    *,
    artifact_metadata: object,
    run_metadata: object,
    expected_config_path: Path,
    expected_input_inventory_path: Path,
) -> tuple[CaptureBundle, CaptureReceipt]:
    """Validate promoted bytes against live GitHub artifact metadata."""
    metadata = _require_object(artifact_metadata, name="GitHub artifact metadata")
    artifact_id = str(_require_positive_int(metadata.get("id"), name="artifact metadata id"))
    artifact_name = _require_text(metadata.get("name"), name="artifact metadata name")
    name_match = _ARTIFACT_NAME.fullmatch(artifact_name)
    if name_match is None:
        raise OracleEvidenceError("artifact metadata name must bind a full candidate SHA")
    candidate_sha = name_match.group(1)
    digest_value = _require_text(metadata.get("digest"), name="artifact metadata digest")
    if not digest_value.startswith("sha256:"):
        raise OracleEvidenceError("artifact metadata digest must use the sha256 algorithm")
    artifact_digest = _require_sha256(
        digest_value.removeprefix("sha256:"),
        name="artifact metadata digest",
    )
    if metadata.get("expired") is not False:
        raise OracleEvidenceError("GitHub artifact must exist and be unexpired")
    workflow_run = _require_object(
        metadata.get("workflow_run"),
        name="artifact metadata workflow_run",
    )
    workflow_run_id = str(
        _require_positive_int(
            workflow_run.get("id"),
            name="artifact metadata workflow run id",
        )
    )
    artifact_workflow_sha = _require_commit_sha(
        workflow_run.get("head_sha"),
        name="artifact metadata workflow head_sha",
    )
    if workflow_run.get("head_branch") != "main":
        raise OracleEvidenceError("artifact metadata workflow must have run from main")

    run = _require_object(run_metadata, name="GitHub workflow-run metadata")
    run_id = str(_require_positive_int(run.get("id"), name="workflow-run metadata id"))
    if run_id != workflow_run_id:
        raise OracleEvidenceError("artifact and workflow-run metadata IDs do not match")
    workflow_sha = _require_commit_sha(
        run.get("head_sha"),
        name="workflow-run metadata head_sha",
    )
    if workflow_sha != artifact_workflow_sha:
        raise OracleEvidenceError("artifact and workflow-run metadata SHAs do not match")
    run_expectations = {
        "event": "workflow_dispatch",
        "path": ".github/workflows/oracle-capture.yml",
        "head_branch": "main",
        "status": "completed",
        "conclusion": "success",
    }
    for name, expected in run_expectations.items():
        if run.get(name) != expected:
            raise OracleEvidenceError(f"workflow-run metadata {name} must equal {expected!r}")
    repository = _require_object(
        run.get("repository"),
        name="workflow-run metadata repository",
    )
    if repository.get("full_name") != GITHUB_REPOSITORY:
        raise OracleEvidenceError(
            f"workflow-run metadata repository must equal {GITHUB_REPOSITORY!r}"
        )
    run_url = _validate_run_url(run.get("html_url"), run_id=run_id)

    bundle = validate_capture_bundle(
        bundle_root,
        expected_candidate_sha=candidate_sha,
        expected_workflow_sha=workflow_sha,
        expected_run_id=workflow_run_id,
        expected_config_path=expected_config_path,
        expected_input_inventory_path=expected_input_inventory_path,
    )
    receipt = validate_capture_receipt(
        receipt_path,
        bundle=bundle,
        expected_artifact_id=artifact_id,
        expected_artifact_digest=artifact_digest,
        expected_artifact_name=artifact_name,
        expected_producer_sha=candidate_sha,
        expected_workflow_sha=workflow_sha,
        expected_workflow_run_id=workflow_run_id,
    )
    if bundle.manifest.run_url != run_url:
        raise OracleEvidenceError("promoted capture run_url does not match GitHub metadata")
    return bundle, receipt


def _validate_environment(value: object) -> CaptureEnvironment:
    environment = _require_object(value, name="capture environment")
    _require_exact_keys(environment, _ENVIRONMENT_KEYS, name="capture environment")
    arch = _require_text(environment["arch"], name="environment.arch")
    if arch != "x86_64":
        raise OracleEvidenceError("capture environment arch must equal 'x86_64'")
    os_value = _require_object(environment["os"], name="environment.os")
    _require_exact_keys(os_value, _OS_KEYS, name="environment.os")
    os_id = _require_text(os_value["id"], name="environment.os.id")
    os_version_id = _require_text(
        os_value["version_id"],
        name="environment.os.version_id",
    )
    if os_id != "ubuntu" or os_version_id != "24.04":
        raise OracleEvidenceError("capture environment OS must equal Ubuntu 24.04")
    python = _require_text(environment["python"], name="environment.python")
    if re.fullmatch(r"3\.12\.\d+", python) is None:
        raise OracleEvidenceError("capture environment Python must be a 3.12 patch release")
    runner_image = _require_text(
        environment["runner_image"],
        name="environment.runner_image",
    )
    if re.fullmatch(r"ubuntu24/[A-Za-z0-9._-]+", runner_image) is None:
        raise OracleEvidenceError(
            "capture runner_image must identify a versioned ubuntu24 GitHub runner"
        )
    thread_value = _require_object(
        environment["thread_policy"],
        name="environment.thread_policy",
    )
    _require_exact_keys(
        thread_value,
        frozenset(THREAD_VARIABLES),
        name="environment.thread_policy",
    )
    thread_policy = {
        name: _require_text(thread_value[name], name=f"thread_policy.{name}")
        for name in THREAD_VARIABLES
    }
    if set(thread_policy.values()) != {"1"}:
        raise OracleEvidenceError(
            "capture thread policy must explicitly pin every thread count to 1"
        )
    return CaptureEnvironment(
        arch=arch,
        cpu_model=_require_text(environment["cpu_model"], name="environment.cpu_model"),
        os_id=os_id,
        os_version_id=os_version_id,
        os_pretty_name=_require_text(
            os_value["pretty_name"],
            name="environment.os.pretty_name",
        ),
        python=python,
        numpy=_require_text(environment["numpy"], name="environment.numpy"),
        numpy_config=_require_text(
            environment["numpy_config"],
            name="environment.numpy_config",
        ),
        runner_image=runner_image,
        thread_policy=MappingProxyType(thread_policy),
    )


def _validate_file_entries(value: object) -> tuple[CaptureFile, ...]:
    if not isinstance(value, list) or not value:
        raise OracleEvidenceError("capture files must be a non-empty list")
    entries: list[CaptureFile] = []
    for index, raw in enumerate(value):
        item = _require_object(raw, name=f"capture files[{index}]")
        _require_exact_keys(item, _FILE_KEYS, name=f"capture files[{index}]")
        path = _require_payload_path(item["path"], name=f"capture files[{index}].path")
        size = item["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise OracleEvidenceError("capture payload byte counts must be positive integers")
        entries.append(
            CaptureFile(
                path=path,
                bytes=size,
                sha256=_require_sha256(
                    item["sha256"],
                    name=f"capture files[{index}].sha256",
                ),
            )
        )
    paths = [entry.path for entry in entries]
    if len(set(paths)) != len(paths):
        raise OracleEvidenceError("capture payload paths must be unique")
    if paths != sorted(paths, key=str.encode):
        raise OracleEvidenceError("capture payload entries must use canonical UTF-8 path order")
    return tuple(entries)


def _validate_vn2_payloads(root: Path, manifest: CaptureManifest) -> None:
    round_paths = tuple(f"orders/round-{round_number}.json" for round_number in range(1, 7))
    expected = {"environment.json", "orders/extraction-report.json", *round_paths}
    actual = {entry.path for entry in manifest.files}
    if actual != expected:
        raise OracleEvidenceError("VN2 capture must contain environment, report, and six rounds")
    environment_value, _bytes = _load_json_object(root / "environment.json", name="environment")
    if environment_value != _environment_value(manifest.environment):
        raise OracleEvidenceError("environment.json does not match the manifest environment")
    report, _bytes = _load_json_object(
        root / "orders" / "extraction-report.json",
        name="extraction report",
    )
    _require_exact_keys(
        report,
        frozenset({"config", "rounds", "series_per_round", "files"}),
        name="extraction report",
    )
    _require_exact_int(
        report["rounds"],
        expected=EXPECTED_ROUNDS,
        name="extraction report rounds",
    )
    _require_exact_int(
        report["series_per_round"],
        expected=EXPECTED_SERIES,
        name="extraction report series_per_round",
    )
    if report["config"] != VN2_CONFIG_PATH:
        raise OracleEvidenceError(f"extraction report config must equal {VN2_CONFIG_PATH!r}")
    report_files = _require_object(report["files"], name="extraction report files")
    if set(report_files) != {PurePosixPath(path).name for path in round_paths}:
        raise OracleEvidenceError("extraction report must bind exactly the six round files")
    manifest_by_path = {entry.path: entry for entry in manifest.files}
    origins: list[str] = []
    stable_series_keys: tuple[str, ...] | None = None
    for round_number, relative_path in enumerate(round_paths, start=1):
        name = PurePosixPath(relative_path).name
        digest = _require_sha256(report_files[name], name=f"extraction report {name}")
        if digest != manifest_by_path[relative_path].sha256:
            raise OracleEvidenceError(f"extraction report digest mismatch: {name}")
        payload, _bytes = _load_json_object(
            root / Path(*PurePosixPath(relative_path).parts), name=name
        )
        _require_exact_keys(
            payload,
            frozenset({"round_num", "origin", "orders"}),
            name=name,
        )
        _require_exact_int(
            payload["round_num"],
            expected=round_number,
            name=f"{name} round_num",
        )
        origins.append(_require_text(payload["origin"], name=f"{name} origin"))
        orders = _require_object(payload["orders"], name=f"{name} orders")
        if len(orders) != EXPECTED_SERIES:
            raise OracleEvidenceError(f"{name} must contain exactly 599 series orders")
        series_keys = tuple(orders)
        if series_keys != tuple(sorted(series_keys, key=str.encode)):
            raise OracleEvidenceError(f"{name} series keys must use canonical UTF-8 order")
        if stable_series_keys is None:
            stable_series_keys = series_keys
        elif series_keys != stable_series_keys:
            raise OracleEvidenceError("VN2 capture series keys must be identical in every round")
        for series_key, value in orders.items():
            _require_text(series_key, name=f"{name} series key")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OracleEvidenceError(f"{name} order values must be real numbers")
            quantity = float(value)
            if not math.isfinite(quantity) or quantity < 0.0:
                raise OracleEvidenceError(f"{name} order values must be finite and nonnegative")
    if len(set(origins)) != EXPECTED_ROUNDS:
        raise OracleEvidenceError("VN2 capture origins must be distinct")
    if origins != sorted(origins):
        raise OracleEvidenceError("VN2 capture origins must be strictly increasing")


def _environment_value(environment: CaptureEnvironment) -> dict[str, object]:
    return {
        "arch": environment.arch,
        "cpu_model": environment.cpu_model,
        "numpy": environment.numpy,
        "numpy_config": environment.numpy_config,
        "os": {
            "id": environment.os_id,
            "pretty_name": environment.os_pretty_name,
            "version_id": environment.os_version_id,
        },
        "python": environment.python,
        "runner_image": environment.runner_image,
        "thread_policy": dict(environment.thread_policy),
    }


def _require_nonsymlink_directory(path: Path, *, name: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise OracleEvidenceError(f"{name} must be an existing directory") from error
    if stat.S_ISLNK(mode):
        raise OracleEvidenceError(f"{name} must not be a symbolic link")
    if not stat.S_ISDIR(mode):
        raise OracleEvidenceError(f"{name} must be an existing directory")
    return path


def _require_nonsymlink_file(path: Path, *, name: str) -> Path:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise OracleEvidenceError(f"{name} must be an existing regular file") from error
    if stat.S_ISLNK(mode):
        raise OracleEvidenceError(f"{name} must not be a symbolic link")
    if not stat.S_ISREG(mode):
        raise OracleEvidenceError(f"{name} must be an existing regular file")
    return path


def _load_json_object(path: Path, *, name: str) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_unique_object)
    except OracleEvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OracleEvidenceError(f"{name} must be readable UTF-8 JSON") from error
    return _require_object(value, name=name), payload


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise OracleEvidenceError(f"oracle evidence contains duplicate JSON key {key!r}")
        value[key] = item
    return value


def _require_object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise OracleEvidenceError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _require_exact_keys(
    value: Mapping[str, object], expected: frozenset[str], *, name: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual, key=str.encode)
        extra = sorted(actual - expected, key=str.encode)
        raise OracleEvidenceError(f"{name} fields mismatch: missing={missing!r}, extra={extra!r}")


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise OracleEvidenceError(f"{name} must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise OracleEvidenceError(f"{name} must be valid UTF-8") from error
    return value


def _require_exact_int(value: object, *, expected: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise OracleEvidenceError(f"{name} must be the integer {expected}")
    return value


def _require_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise OracleEvidenceError(f"{name} must be a positive integer")
    return value


def _require_commit_sha(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if _COMMIT_SHA.fullmatch(text) is None:
        raise OracleEvidenceError(f"{name} must be a full lowercase 40-hex SHA")
    return text


def _require_sha256(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if _SHA256.fullmatch(text) is None:
        raise OracleEvidenceError(f"{name} must be a lowercase sha256")
    return text


def _require_run_id(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if _RUN_ID.fullmatch(text) is None:
        raise OracleEvidenceError(f"{name} must be a positive decimal identifier")
    return text


def _require_payload_path(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    path = PurePosixPath(text)
    if (
        text == "."
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or path.as_posix() != text
    ):
        raise OracleEvidenceError(f"{name} must be a canonical relative POSIX path")
    if "\\" in text or text in {"manifest.json", "files.sha256"}:
        raise OracleEvidenceError(f"{name} is not an admissible payload path")
    return text


def _validate_run_url(value: object, *, run_id: str) -> str:
    text = _require_text(value, name="run_url")
    parsed = urlparse(text)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "github.com"
        or parsed.query
        or parsed.fragment
        or parsed.path != f"/{GITHUB_REPOSITORY}/actions/runs/{run_id}"
    ):
        raise OracleEvidenceError(
            f"run_url must equal the {GITHUB_REPOSITORY} Actions URL for run_id"
        )
    return text


def _require_expected(actual: str, expected: str, *, name: str) -> None:
    if actual != expected:
        raise OracleEvidenceError(f"capture {name} does not match the requested value")


def _require_trusted_digest(actual: str, path: Path, *, name: str) -> None:
    trusted_path = Path(path)
    try:
        expected = _sha256_file(trusted_path)
    except OSError as error:
        raise OracleEvidenceError(f"trusted {name} input must be a readable file") from error
    if actual != expected:
        raise OracleEvidenceError(f"capture {name} does not match the trusted input bytes")


def _environment_digest(
    *,
    environment: CaptureEnvironment,
    config_digest: str,
    input_digest: str,
    capture_digest: str,
    actuals_semantics: str,
) -> str:
    comparability_key = {
        "actuals_semantics": actuals_semantics,
        "architecture": environment.arch,
        "capture_digest": capture_digest,
        "config_digest": config_digest,
        "input_digest": input_digest,
        "lockfile_sha256": ORACLE_LOCK_SHA256,
        "os_release": {
            "id": environment.os_id,
            "version_id": environment.os_version_id,
        },
    }
    canonical = json.dumps(
        comparability_key,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _capture_digest(files: tuple[CaptureFile, ...]) -> str:
    listing = "".join(
        f"{entry.sha256}  {entry.path}\n" for entry in files if entry.path.startswith("orders/")
    ).encode("utf-8")
    if not listing:
        raise OracleEvidenceError("capture payload listing must contain order evidence")
    return _sha256_bytes(listing)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


__all__ = [
    "ORACLE_COMMIT",
    "ORACLE_LOCK_SHA256",
    "ORACLE_TAG",
    "CaptureBundle",
    "CaptureEnvironment",
    "CaptureFile",
    "CaptureManifest",
    "CaptureReceipt",
    "OracleEvidenceError",
    "validate_capture_bundle",
    "validate_capture_receipt",
    "validate_promoted_capture",
    "validate_promoted_captures_root",
]
