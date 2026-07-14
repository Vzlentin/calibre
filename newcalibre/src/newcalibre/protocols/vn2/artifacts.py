"""Project generic VN2 engine facts into strict Gate-A R1-R4 evidence."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import cast
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from newcalibre.domain import (
    ActualsSemantics,
    GuaranteeClaim,
    GuaranteeDescriptor,
    InventoryPosition,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.ledger import BookedCost, OrderRow, SettlementRecord, StockoutTransition
from newcalibre.ordering import CostValue, SettlementObjective, settle_path_cost
from newcalibre.protocols.vn2.adapter import VN2RunResult
from newcalibre.protocols.vn2.config import VN2ProtocolConfig, load_vn2_config

RESULT_KIND = "vn2-gate-a-results"
GITHUB_REPOSITORY = "Vzlentin/calibre"
CONFIG_PATH = "benchmarks/vn2/protocol.yaml"
INPUT_INVENTORY_PATH = "benchmarks/vn2/vn2-input-digests.json"
LOCK_PATH = "uv.lock"
THREAD_VARIABLES = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

_PAYLOAD_PATHS = frozenset(
    {
        "environment.json",
        "r1-orders.jsonl",
        "r2-cost-ledger.jsonl",
        "r3-final-triple.json",
        "r4-cost-trajectory.json",
    }
)
_ALL_PATHS = frozenset({"manifest.json", "files.sha256", *_PAYLOAD_PATHS})
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "artifact_kind",
        "artifact_name",
        "candidate_sha",
        "workflow_sha",
        "run_id",
        "run_url",
        "config_path",
        "config_digest",
        "input_inventory_path",
        "input_inventory_digest",
        "lock_path",
        "lock_digest",
        "actuals_semantics",
        "session_id",
        "series_count",
        "round_count",
        "realized_period_count",
        "series_identity_digest",
        "provenance_digest",
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
_R1_KEYS = frozenset(
    {
        "schema",
        "actuals_semantics",
        "provenance_digest",
        "session_id",
        "series_key",
        "store",
        "product",
        "round",
        "origin",
        "model_name",
        "quantity",
        "arrival_period",
        "raw_target",
        "target",
        "reorder_point",
        "source_columns",
        "source_descriptor",
        "effective_descriptor",
        "bindings",
        "consumed_claim",
    }
)
_DESCRIPTOR_KEYS = frozenset({"type", "level", "scored_series", "window", "scope"})
_GUARANTEE_TYPE_KEYS = frozenset({"claim", "currency", "declared_slack"})
_SCOPE_KEYS = frozenset({"kind", "class_system_name"})
_BINDING_KEYS = frozenset({"name", "value", "bound"})
_R2_KEYS = frozenset(
    {
        "schema",
        "actuals_semantics",
        "provenance_digest",
        "session_id",
        "series_key",
        "store",
        "product",
        "period_index",
        "period",
        "currency",
        "stockout_rule",
        "start_inventory",
        "arrivals",
        "demand",
        "sales",
        "missed_sales",
        "end_inventory",
        "closing_backorders",
        "on_order",
        "holding_rate",
        "holding_basis",
        "holding_cost",
        "shortage_rate",
        "shortage_basis",
        "shortage_cost",
    }
)
_R3_KEYS = frozenset(
    {
        "schema",
        "actuals_semantics",
        "provenance_digest",
        "holding_total",
        "shortage_total",
        "total_cost",
    }
)
_R4_KEYS = frozenset(
    {
        "schema",
        "actuals_semantics",
        "provenance_digest",
        "decision_rounds",
        "drain_remainder",
    }
)
_ROUND_KEYS = frozenset(
    {
        "round",
        "origin",
        "cumulative_cost",
        "actuals_semantics",
        "provenance_digest",
    }
)
_DRAIN_KEYS = frozenset({"periods", "cost", "actuals_semantics", "provenance_digest"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
_RUN_ID = re.compile(r"[1-9][0-9]*")
_IMAGE_OS_ENV = "ImageOS"
_IMAGE_VERSION_ENV = "ImageVersion"


class VN2ResultError(ValueError):
    """Report incomplete facts or a malformed VN2 R1-R4 result bundle."""


@dataclass(frozen=True, slots=True)
class VN2EvidenceEnvironment:
    """Retain the complete ratified environment facts for one VN2 run."""

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

    def __post_init__(self) -> None:
        arch = _require_text(self.arch, name="environment.arch")
        if arch != "x86_64":
            raise VN2ResultError("VN2 evidence environment arch must equal 'x86_64'")
        os_id = _require_text(self.os_id, name="environment.os.id")
        os_version_id = _require_text(
            self.os_version_id,
            name="environment.os.version_id",
        )
        if os_id != "ubuntu" or os_version_id != "24.04":
            raise VN2ResultError("VN2 evidence environment OS must equal Ubuntu 24.04")
        python = _require_text(self.python, name="environment.python")
        if re.fullmatch(r"3\.12\.\d+", python) is None:
            raise VN2ResultError("VN2 evidence environment Python must be a 3.12 patch release")
        numpy_config = _require_text(
            self.numpy_config,
            name="environment.numpy_config",
        )
        if "blas" not in numpy_config.casefold():
            raise VN2ResultError("VN2 evidence environment must retain BLAS provenance")
        runner_image = _require_text(
            self.runner_image,
            name="environment.runner_image",
        )
        if re.fullmatch(r"ubuntu24/[A-Za-z0-9._-]+", runner_image) is None:
            raise VN2ResultError(
                "VN2 runner_image must identify a versioned ubuntu24 GitHub runner"
            )
        if not isinstance(self.thread_policy, Mapping):
            raise VN2ResultError("environment.thread_policy must be a mapping")
        thread_policy = dict(self.thread_policy)
        _require_exact_keys(
            thread_policy,
            frozenset(THREAD_VARIABLES),
            name="environment.thread_policy",
        )
        normalized_policy = {
            name: _require_text(thread_policy[name], name=f"thread_policy.{name}")
            for name in THREAD_VARIABLES
        }
        if set(normalized_policy.values()) != {"1"}:
            raise VN2ResultError("every VN2 evidence thread count must be pinned to 1")
        object.__setattr__(self, "arch", arch)
        object.__setattr__(self, "cpu_model", _require_text(self.cpu_model, name="cpu_model"))
        object.__setattr__(self, "os_id", os_id)
        object.__setattr__(self, "os_version_id", os_version_id)
        object.__setattr__(
            self,
            "os_pretty_name",
            _require_text(self.os_pretty_name, name="environment.os.pretty_name"),
        )
        object.__setattr__(self, "python", python)
        object.__setattr__(self, "numpy", _require_text(self.numpy, name="numpy"))
        object.__setattr__(self, "numpy_config", numpy_config)
        object.__setattr__(self, "runner_image", runner_image)
        object.__setattr__(self, "thread_policy", MappingProxyType(normalized_policy))


@dataclass(frozen=True, slots=True)
class VN2ResultFile:
    """Bind one canonical payload path to its exact bytes and digest."""

    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class VN2ResultManifest:
    """Expose one fully validated immutable VN2 result manifest."""

    artifact_name: str
    candidate_sha: str
    workflow_sha: str
    run_id: str
    run_url: str
    config_path: str
    config_digest: str
    input_inventory_path: str
    input_inventory_digest: str
    lock_path: str
    lock_digest: str
    actuals_semantics: str
    session_id: str
    series_count: int
    round_count: int
    realized_period_count: int
    series_identity_digest: str
    provenance_digest: str
    environment: VN2EvidenceEnvironment
    environment_digest: str
    files: tuple[VN2ResultFile, ...]
    inner_bundle_digest: str


@dataclass(frozen=True, slots=True)
class VN2ResultBundle:
    """Return one content-verified bundle plus the manifest digest."""

    root: Path
    manifest: VN2ResultManifest
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _ValidatedFacts:
    identities: Mapping[str, tuple[int, int]]
    orders: tuple[OrderRow, ...]
    settlements: tuple[SettlementRecord, ...]
    series_identity_digest: str


def capture_vn2_evidence_environment() -> VN2EvidenceEnvironment:
    """Capture the ratified Ubuntu runner, Python, NumPy, BLAS, and thread facts."""
    release = _os_release()
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        np.show_config()
    return VN2EvidenceEnvironment(
        arch=platform.machine(),
        cpu_model=_cpu_model(),
        os_id=release.get("ID", ""),
        os_version_id=release.get("VERSION_ID", ""),
        os_pretty_name=release.get("PRETTY_NAME", ""),
        python=platform.python_version(),
        numpy=np.__version__,
        numpy_config=output.getvalue().strip(),
        runner_image=(
            f"{os.environ.get(_IMAGE_OS_ENV, '')}/{os.environ.get(_IMAGE_VERSION_ENV, '')}"
        ),
        thread_policy={name: os.environ.get(name, "") for name in THREAD_VARIABLES},
    )


def emit_vn2_result_bundle(
    root: Path,
    *,
    result: VN2RunResult,
    config: VN2ProtocolConfig,
    candidate_sha: str,
    workflow_sha: str,
    run_id: str,
    run_url: str,
    config_path: Path,
    input_inventory_path: Path,
    lock_path: Path,
    environment: VN2EvidenceEnvironment,
) -> VN2ResultBundle:
    """Validate all engine facts, then atomically emit deterministic R1-R4 bytes."""
    bundle_root = Path(root)
    if bundle_root.exists() or bundle_root.is_symlink():
        raise VN2ResultError("VN2 result bundle destination must not already exist")
    if not isinstance(result, VN2RunResult):
        raise VN2ResultError("VN2 result projection requires a VN2RunResult")
    if not isinstance(config, VN2ProtocolConfig):
        raise VN2ResultError("VN2 result projection requires a VN2ProtocolConfig")
    if not isinstance(environment, VN2EvidenceEnvironment):
        raise VN2ResultError("VN2 result projection requires a VN2EvidenceEnvironment")

    identity = _validated_identity(candidate_sha, workflow_sha, run_id, run_url)
    trusted = _trusted_inputs(config, config_path, input_inventory_path, lock_path)
    facts = _validate_engine_facts(result, config=config)
    environment_value = _environment_value(environment)
    environment_digest = _digest_json(environment_value, name="VN2 environment")
    artifact_name = f"vn2-acceptance-{identity['candidate_sha']}"
    provenance_value = {
        "actuals_semantics": config.actuals_semantics.value,
        "artifact_kind": RESULT_KIND,
        "artifact_name": artifact_name,
        "candidate_sha": identity["candidate_sha"],
        "config_digest": trusted["config_digest"],
        "environment_digest": environment_digest,
        "input_inventory_digest": trusted["input_inventory_digest"],
        "lock_digest": trusted["lock_digest"],
        "realized_periods": [period.isoformat() for period in config.realized_periods],
        "run_id": identity["run_id"],
        "run_url": identity["run_url"],
        "series_identity_digest": facts.series_identity_digest,
        "session_id": result.session.value,
        "workflow_sha": identity["workflow_sha"],
    }
    provenance_digest = _digest_json(provenance_value, name="VN2 provenance")
    payloads = _project_payloads(
        result,
        config=config,
        ordered_orders=facts.orders,
        ordered_settlements=facts.settlements,
        identities=facts.identities,
        provenance_digest=provenance_digest,
        environment_value=environment_value,
    )
    files = tuple(
        VN2ResultFile(path=path, bytes=len(payloads[path]), sha256=_sha256(payloads[path]))
        for path in sorted(payloads, key=str.encode)
    )
    listing = "".join(f"{entry.sha256}  {entry.path}\n" for entry in files).encode()
    manifest_value = {
        "actuals_semantics": config.actuals_semantics.value,
        "artifact_kind": RESULT_KIND,
        "artifact_name": artifact_name,
        "candidate_sha": identity["candidate_sha"],
        "config_digest": trusted["config_digest"],
        "config_path": CONFIG_PATH,
        "environment": environment_value,
        "environment_digest": environment_digest,
        "files": [_file_value(entry) for entry in files],
        "inner_bundle_digest": _sha256(listing),
        "input_inventory_digest": trusted["input_inventory_digest"],
        "input_inventory_path": INPUT_INVENTORY_PATH,
        "lock_digest": trusted["lock_digest"],
        "lock_path": LOCK_PATH,
        "provenance_digest": provenance_digest,
        "realized_period_count": len(config.realized_periods),
        "round_count": config.round_count,
        "run_id": identity["run_id"],
        "run_url": identity["run_url"],
        "schema": 1,
        "series_count": config.series_count,
        "series_identity_digest": facts.series_identity_digest,
        "session_id": result.session.value,
        "workflow_sha": identity["workflow_sha"],
    }
    manifest_bytes = _json_bytes(manifest_value, name="VN2 result manifest")

    parent = bundle_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{bundle_root.name}-", dir=parent))
    try:
        for path, payload in payloads.items():
            destination = temporary / Path(*PurePosixPath(path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        (temporary / "files.sha256").write_bytes(listing)
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        validate_vn2_result_bundle(
            temporary,
            expected_candidate_sha=identity["candidate_sha"],
            expected_workflow_sha=identity["workflow_sha"],
            expected_run_id=identity["run_id"],
            expected_config_path=Path(config_path),
            expected_input_inventory_path=Path(input_inventory_path),
            expected_lock_path=Path(lock_path),
        )
        temporary.replace(bundle_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_vn2_result_bundle(
        bundle_root,
        expected_candidate_sha=identity["candidate_sha"],
        expected_workflow_sha=identity["workflow_sha"],
        expected_run_id=identity["run_id"],
        expected_config_path=Path(config_path),
        expected_input_inventory_path=Path(input_inventory_path),
        expected_lock_path=Path(lock_path),
    )


def validate_vn2_result_bundle(
    root: Path,
    *,
    expected_candidate_sha: str,
    expected_workflow_sha: str,
    expected_run_id: str,
    expected_config_path: Path,
    expected_input_inventory_path: Path,
    expected_lock_path: Path,
) -> VN2ResultBundle:
    """Validate identity, exact files, every digest, and the R1-R4 semantics."""
    bundle_root = Path(root)
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise VN2ResultError("VN2 result bundle must be a real existing directory")
    _validate_bundle_paths(bundle_root)
    manifest, manifest_bytes = _load_json_object(
        bundle_root / "manifest.json",
        name="VN2 result manifest",
    )
    _require_exact_keys(manifest, _MANIFEST_KEYS, name="VN2 result manifest")
    _require_int(manifest["schema"], expected=1, name="manifest.schema")
    if manifest["artifact_kind"] != RESULT_KIND:
        raise VN2ResultError(f"artifact_kind must equal {RESULT_KIND!r}")

    candidate_sha = _require_commit_sha(manifest["candidate_sha"], name="candidate_sha")
    workflow_sha = _require_commit_sha(manifest["workflow_sha"], name="workflow_sha")
    run_id = _require_run_id(manifest["run_id"], name="run_id")
    _require_expected(candidate_sha, expected_candidate_sha, name="candidate_sha")
    _require_expected(workflow_sha, expected_workflow_sha, name="workflow_sha")
    _require_expected(run_id, expected_run_id, name="run_id")
    run_url = _validate_run_url(manifest["run_url"], run_id=run_id)
    artifact_name = _require_text(manifest["artifact_name"], name="artifact_name")
    if artifact_name != f"vn2-acceptance-{candidate_sha}":
        raise VN2ResultError("artifact_name must bind the candidate SHA")
    path_expectations = {
        "config_path": CONFIG_PATH,
        "input_inventory_path": INPUT_INVENTORY_PATH,
        "lock_path": LOCK_PATH,
    }
    for name, expected in path_expectations.items():
        if manifest[name] != expected:
            raise VN2ResultError(f"{name} must equal {expected!r}")

    config_digest = _require_sha256(manifest["config_digest"], name="config_digest")
    input_digest = _require_sha256(
        manifest["input_inventory_digest"],
        name="input_inventory_digest",
    )
    lock_digest = _require_sha256(manifest["lock_digest"], name="lock_digest")
    _require_trusted_digest(
        config_digest,
        Path(expected_config_path),
        name="config_digest",
    )
    _require_trusted_digest(
        input_digest,
        Path(expected_input_inventory_path),
        name="input_inventory_digest",
    )
    _require_trusted_digest(lock_digest, Path(expected_lock_path), name="lock_digest")
    config = load_vn2_config(Path(expected_config_path))
    if manifest["actuals_semantics"] != config.actuals_semantics.value:
        raise VN2ResultError("manifest actuals_semantics does not match VN2 configuration")

    environment = _parse_environment(manifest["environment"])
    environment_value = _environment_value(environment)
    environment_digest = _require_sha256(
        manifest["environment_digest"],
        name="environment_digest",
    )
    if environment_digest != _digest_json(environment_value, name="VN2 environment"):
        raise VN2ResultError("environment_digest does not match environment facts")
    files = _parse_files(manifest["files"])
    if {entry.path for entry in files} != _PAYLOAD_PATHS:
        raise VN2ResultError(
            "VN2 result payload file set must contain exactly R1-R4 and environment"
        )
    inner_digest = _require_sha256(
        manifest["inner_bundle_digest"],
        name="inner_bundle_digest",
    )
    expected_listing = "".join(f"{entry.sha256}  {entry.path}\n" for entry in files).encode()
    try:
        actual_listing = (bundle_root / "files.sha256").read_bytes()
    except OSError as error:
        raise VN2ResultError("VN2 result bundle is missing files.sha256") from error
    if actual_listing != expected_listing:
        raise VN2ResultError("files.sha256 does not exactly match manifest payload entries")
    if _sha256(actual_listing) != inner_digest:
        raise VN2ResultError("inner bundle digest does not match files.sha256")

    for entry in files:
        payload_path = bundle_root / Path(*PurePosixPath(entry.path).parts)
        try:
            payload = payload_path.read_bytes()
        except OSError as error:
            raise VN2ResultError(f"VN2 result payload is unreadable: {entry.path}") from error
        if len(payload) != entry.bytes:
            raise VN2ResultError(f"VN2 result payload size mismatch: {entry.path}")
        if _sha256(payload) != entry.sha256:
            raise VN2ResultError(f"VN2 result payload digest mismatch: {entry.path}")

    session_id = _require_sha256(manifest["session_id"], name="session_id")
    series_count = _require_int(
        manifest["series_count"],
        expected=config.series_count,
        name="series_count",
    )
    round_count = _require_int(
        manifest["round_count"],
        expected=config.round_count,
        name="round_count",
    )
    realized_count = _require_int(
        manifest["realized_period_count"],
        expected=len(config.realized_periods),
        name="realized_period_count",
    )
    series_identity_digest = _require_sha256(
        manifest["series_identity_digest"],
        name="series_identity_digest",
    )
    provenance_digest = _require_sha256(
        manifest["provenance_digest"],
        name="provenance_digest",
    )
    provenance_value = {
        "actuals_semantics": config.actuals_semantics.value,
        "artifact_kind": RESULT_KIND,
        "artifact_name": artifact_name,
        "candidate_sha": candidate_sha,
        "config_digest": config_digest,
        "environment_digest": environment_digest,
        "input_inventory_digest": input_digest,
        "lock_digest": lock_digest,
        "realized_periods": [period.isoformat() for period in config.realized_periods],
        "run_id": run_id,
        "run_url": run_url,
        "series_identity_digest": series_identity_digest,
        "session_id": session_id,
        "workflow_sha": workflow_sha,
    }
    if provenance_digest != _digest_json(provenance_value, name="VN2 provenance"):
        raise VN2ResultError("provenance_digest does not match manifest provenance facts")

    environment_payload, _ = _load_json_object(
        bundle_root / "environment.json",
        name="VN2 environment payload",
    )
    if environment_payload != environment_value:
        raise VN2ResultError("environment.json does not match manifest environment facts")
    try:
        identities, reconstructed_session, order_arrivals = _validate_r1_payload(
            bundle_root / "r1-orders.jsonl",
            config=config,
            session_id=session_id,
            provenance_digest=provenance_digest,
        )
        if len(identities) != series_count:
            raise VN2ResultError("R1 series mapping does not match series_count")
        if reconstructed_session.value != session_id:
            raise VN2ResultError("R1 series mapping does not derive the manifest session_id")
        if _series_digest(identities) != series_identity_digest:
            raise VN2ResultError("R1 series mapping does not match series_identity_digest")
        records = _validate_r2_payload(
            bundle_root / "r2-cost-ledger.jsonl",
            config=config,
            session=reconstructed_session,
            identities=identities,
            provenance_digest=provenance_digest,
        )
        settlement_by_key = {(record.series_key, record.period): record for record in records}
        for key, quantity in order_arrivals.items():
            settlement = settlement_by_key.get(key)
            if settlement is None or settlement.arrivals != quantity:
                raise VN2ResultError("R1 order quantity does not match its R2 arrival fact")
        objective = settle_path_cost(
            records,
            actuals_semantics=config.actuals_semantics,
        )
        _validate_r3_payload(
            bundle_root / "r3-final-triple.json",
            objective=objective,
            semantics=config.actuals_semantics,
            provenance_digest=provenance_digest,
        )
        _validate_r4_payload(
            bundle_root / "r4-cost-trajectory.json",
            records=records,
            objective=objective,
            config=config,
            provenance_digest=provenance_digest,
        )
    except VN2ResultError:
        raise
    except (TypeError, ValueError) as error:
        raise VN2ResultError("VN2 result payloads do not reconstruct valid engine facts") from error

    manifest_object = VN2ResultManifest(
        artifact_name=artifact_name,
        candidate_sha=candidate_sha,
        workflow_sha=workflow_sha,
        run_id=run_id,
        run_url=run_url,
        config_path=CONFIG_PATH,
        config_digest=config_digest,
        input_inventory_path=INPUT_INVENTORY_PATH,
        input_inventory_digest=input_digest,
        lock_path=LOCK_PATH,
        lock_digest=lock_digest,
        actuals_semantics=config.actuals_semantics.value,
        session_id=session_id,
        series_count=series_count,
        round_count=round_count,
        realized_period_count=realized_count,
        series_identity_digest=series_identity_digest,
        provenance_digest=provenance_digest,
        environment=environment,
        environment_digest=environment_digest,
        files=files,
        inner_bundle_digest=inner_digest,
    )
    return VN2ResultBundle(
        root=bundle_root.resolve(),
        manifest=manifest_object,
        manifest_sha256=_sha256(manifest_bytes),
    )


def _validate_bundle_paths(root: Path) -> None:
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise VN2ResultError("VN2 result bundle paths must not be symbolic links")
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            raise VN2ResultError(f"VN2 result bundle contains unexpected directory: {relative}")
        if path.is_file():
            actual_paths.add(relative)
    if actual_paths != _ALL_PATHS:
        missing = sorted(_ALL_PATHS - actual_paths, key=str.encode)
        extra = sorted(actual_paths - _ALL_PATHS, key=str.encode)
        raise VN2ResultError(f"VN2 result file set mismatch: missing={missing!r}, extra={extra!r}")


def _trusted_inputs(
    config: VN2ProtocolConfig,
    config_path: Path,
    input_inventory_path: Path,
    lock_path: Path,
) -> dict[str, str]:
    paths = {
        "config_digest": Path(config_path),
        "input_inventory_digest": Path(input_inventory_path),
        "lock_digest": Path(lock_path),
    }
    digests = {name: _sha256_file(path, name=name) for name, path in paths.items()}
    try:
        trusted_config = load_vn2_config(Path(config_path))
    except (OSError, ValueError) as error:
        raise VN2ResultError("trusted VN2 configuration is invalid") from error
    if trusted_config != config:
        raise VN2ResultError("VN2 configuration object does not match trusted config bytes")
    return digests


def _validate_engine_facts(
    result: VN2RunResult,
    *,
    config: VN2ProtocolConfig,
) -> _ValidatedFacts:
    identities = dict(result.series_identities)
    if len(identities) != config.series_count:
        raise VN2ResultError("VN2 series identity mapping must match configured series_count")
    if len(set(identities.values())) != len(identities):
        raise VN2ResultError("VN2 series identity mapping values must be unique")
    expected_session = _derive_session(config, identities)
    if result.session != expected_session:
        raise VN2ResultError("VN2 result session does not match its series mapping and config")
    series_keys = frozenset(identities)
    origins = frozenset(config.decision_origins)
    periods = frozenset(config.realized_periods)

    order_by_key: dict[tuple[str, pd.Timestamp], OrderRow] = {}
    for order in result.orders:
        if order.session != result.session:
            raise VN2ResultError("every VN2 order must share the result session")
        if order.series_key not in series_keys or order.origin not in origins:
            raise VN2ResultError("VN2 order has a foreign series or decision origin")
        key = (order.series_key, order.origin)
        if key in order_by_key:
            raise VN2ResultError("VN2 orders must have one row per series and round")
        if order.arrival_period != config.calendar.advance(
            order.origin,
            config.timing.lead_time,
        ):
            raise VN2ResultError("VN2 order arrival does not match configured lead time")
        if order.evidence is None:
            raise VN2ResultError("every VN2 order must retain complete decision evidence")
        if (
            order.evidence.source_descriptor.type.claim is not GuaranteeClaim.NONE
            or order.evidence.effective_descriptor.type.claim is not GuaranteeClaim.NONE
        ):
            raise VN2ResultError("Gate-A VN2 order evidence must consume claim none")
        order_by_key[key] = order
    expected_order_keys = {(series, origin) for series in series_keys for origin in origins}
    if set(order_by_key) != expected_order_keys:
        raise VN2ResultError("VN2 R1 order spine is incomplete")

    settlement_by_key: dict[tuple[str, pd.Timestamp], SettlementRecord] = {}
    for record in result.settlements:
        if record.session != result.session:
            raise VN2ResultError("every VN2 settlement must share the result session")
        if record.series_key not in series_keys or record.period not in periods:
            raise VN2ResultError("VN2 settlement has a foreign series or realized period")
        if record.actuals_semantics is not config.actuals_semantics:
            raise VN2ResultError("every VN2 settlement must preserve configured semantics")
        if record.transition.rule is not config.stockout_rule:
            raise VN2ResultError("every VN2 settlement must use the configured stockout rule")
        key = (record.series_key, record.period)
        if key in settlement_by_key:
            raise VN2ResultError("VN2 settlements must have one row per series and period")
        settlement_by_key[key] = record
    expected_settlement_keys = {(series, period) for series in series_keys for period in periods}
    if set(settlement_by_key) != expected_settlement_keys:
        raise VN2ResultError("VN2 R2 settlement spine is incomplete")
    ordered_orders = tuple(
        order_by_key[(series, origin)]
        for origin in config.decision_origins
        for series in sorted(series_keys, key=str.encode)
    )
    ordered_settlements = tuple(
        settlement_by_key[(series, period)]
        for period in config.realized_periods
        for series in sorted(series_keys, key=str.encode)
    )
    return _ValidatedFacts(
        identities=MappingProxyType(identities),
        orders=ordered_orders,
        settlements=ordered_settlements,
        series_identity_digest=_series_digest(identities),
    )


def _project_payloads(
    result: VN2RunResult,
    *,
    config: VN2ProtocolConfig,
    ordered_orders: tuple[OrderRow, ...],
    ordered_settlements: tuple[SettlementRecord, ...],
    identities: Mapping[str, tuple[int, int]],
    provenance_digest: str,
    environment_value: dict[str, object],
) -> dict[str, bytes]:
    round_by_origin = {
        origin: index for index, origin in enumerate(config.decision_origins, start=1)
    }
    period_index = {period: index for index, period in enumerate(config.realized_periods, start=1)}
    r1: list[dict[str, object]] = []
    for order in ordered_orders:
        evidence = order.evidence
        assert evidence is not None
        store, product = identities[order.series_key]
        r1.append(
            {
                "actuals_semantics": config.actuals_semantics.value,
                "arrival_period": order.arrival_period.isoformat(),
                "bindings": [
                    {"bound": item.bound, "name": item.name, "value": item.value}
                    for item in evidence.bindings
                ],
                "consumed_claim": evidence.effective_descriptor.type.claim.value,
                "effective_descriptor": _descriptor_value(evidence.effective_descriptor),
                "model_name": order.model_name,
                "origin": order.origin.isoformat(),
                "product": product,
                "provenance_digest": provenance_digest,
                "quantity": order.quantity,
                "raw_target": evidence.raw_target,
                "reorder_point": evidence.reorder_point,
                "round": round_by_origin[order.origin],
                "schema": 1,
                "series_key": order.series_key,
                "session_id": result.session.value,
                "source_columns": list(evidence.source_columns),
                "source_descriptor": _descriptor_value(evidence.source_descriptor),
                "store": store,
                "target": evidence.target,
            }
        )
    r2: list[dict[str, object]] = []
    for record in ordered_settlements:
        store, product = identities[record.series_key]
        r2.append(
            {
                "actuals_semantics": record.actuals_semantics.value,
                "arrivals": record.arrivals,
                "closing_backorders": record.transition.closing_backorders,
                "currency": config.currency,
                "demand": record.transition.demand,
                "end_inventory": record.transition.closing_on_hand,
                "holding_basis": record.holding.basis,
                "holding_cost": record.holding.amount,
                "holding_rate": record.holding.rate,
                "missed_sales": record.transition.unmet_demand,
                "on_order": record.inventory_position.on_order,
                "period": record.period.isoformat(),
                "period_index": period_index[record.period],
                "product": product,
                "provenance_digest": provenance_digest,
                "sales": record.transition.fulfilled_demand,
                "schema": 1,
                "series_key": record.series_key,
                "session_id": result.session.value,
                "shortage_basis": record.shortage.basis,
                "shortage_cost": record.shortage.amount,
                "shortage_rate": record.shortage.rate,
                "start_inventory": record.transition.available_inventory,
                "stockout_rule": record.transition.rule.value,
                "store": store,
            }
        )
    objective = settle_path_cost(
        ordered_settlements,
        actuals_semantics=config.actuals_semantics,
    )
    _require_objective_spine(objective.by_origin, config=config)
    _require_exact_objective_total(objective)
    r3 = _r3_value(
        objective,
        semantics=config.actuals_semantics,
        provenance_digest=provenance_digest,
    )
    r4 = _r4_value(
        ordered_settlements,
        objective=objective,
        config=config,
        provenance_digest=provenance_digest,
    )
    return {
        "environment.json": _json_bytes(environment_value, name="VN2 environment"),
        "r1-orders.jsonl": b"".join(_json_bytes(row, name="R1 row") for row in r1),
        "r2-cost-ledger.jsonl": b"".join(_json_bytes(row, name="R2 row") for row in r2),
        "r3-final-triple.json": _json_bytes(r3, name="R3 final triple"),
        "r4-cost-trajectory.json": _json_bytes(r4, name="R4 cost trajectory"),
    }


def _validate_r1_payload(
    path: Path,
    *,
    config: VN2ProtocolConfig,
    session_id: str,
    provenance_digest: str,
) -> tuple[
    dict[str, tuple[int, int]],
    SessionIdentity,
    dict[tuple[str, pd.Timestamp], float],
]:
    rows = _load_jsonl(path, name="R1 orders")
    expected_count = config.series_count * config.round_count
    if len(rows) != expected_count:
        raise VN2ResultError(f"R1 orders must contain exactly {expected_count} rows")
    origin_by_round = {
        index: origin for index, origin in enumerate(config.decision_origins, start=1)
    }
    identities: dict[str, tuple[int, int]] = {}
    order_arrivals: dict[tuple[str, pd.Timestamp], float] = {}
    spine: list[tuple[int, str]] = []
    model_name = config.model_config.get("model_name")
    for index, row in enumerate(rows):
        name = f"R1 orders[{index}]"
        _require_exact_keys(row, _R1_KEYS, name=name)
        _require_int(row["schema"], expected=1, name=f"{name}.schema")
        _require_semantics(row, config=config, provenance_digest=provenance_digest, name=name)
        if row["session_id"] != session_id:
            raise VN2ResultError(f"{name}.session_id does not match manifest")
        series_key = _require_text(row["series_key"], name=f"{name}.series_key")
        store = _integer(row["store"], name=f"{name}.store")
        product = _integer(row["product"], name=f"{name}.product")
        identity = (store, product)
        prior = identities.setdefault(series_key, identity)
        if prior != identity:
            raise VN2ResultError("R1 series mapping must be stable across rounds")
        round_number = _positive_integer(row["round"], name=f"{name}.round")
        if round_number not in origin_by_round:
            raise VN2ResultError(f"{name}.round is outside the configured spine")
        origin = _timestamp(row["origin"], name=f"{name}.origin")
        if origin != origin_by_round[round_number]:
            raise VN2ResultError(f"{name}.origin does not match its round")
        arrival = _timestamp(row["arrival_period"], name=f"{name}.arrival_period")
        if arrival != config.calendar.advance(origin, config.timing.lead_time):
            raise VN2ResultError(f"{name}.arrival_period does not match lead time")
        if row["model_name"] != model_name:
            raise VN2ResultError(f"{name}.model_name does not match configuration")
        quantity = _finite_nonnegative(row["quantity"], name=f"{name}.quantity")
        if not quantity.is_integer():
            raise VN2ResultError("R1 order quantities must be whole units")
        order_arrivals[(series_key, arrival)] = quantity
        _finite_number(row["raw_target"], name=f"{name}.raw_target")
        _finite_number(row["target"], name=f"{name}.target")
        if row["reorder_point"] is not None:
            _finite_number(row["reorder_point"], name=f"{name}.reorder_point")
        columns = row["source_columns"]
        if (
            not isinstance(columns, list)
            or not columns
            or any(not isinstance(item, str) or not item for item in columns)
            or len(set(columns)) != len(columns)
        ):
            raise VN2ResultError("R1 source_columns must be unique non-empty strings")
        for descriptor_name in ("source_descriptor", "effective_descriptor"):
            claim = _validate_descriptor(row[descriptor_name], name=f"{name}.{descriptor_name}")
            if claim != GuaranteeClaim.NONE.value:
                raise VN2ResultError("R1 descriptors must declare claim none")
        if row["consumed_claim"] != GuaranteeClaim.NONE.value:
            raise VN2ResultError("R1 consumed_claim must equal none")
        bindings = row["bindings"]
        if not isinstance(bindings, list):
            raise VN2ResultError(f"{name}.bindings must be a list")
        for binding_index, binding_value in enumerate(bindings):
            binding = _object(binding_value, name=f"{name}.bindings[{binding_index}]")
            _require_exact_keys(binding, _BINDING_KEYS, name="R1 binding")
            _require_text(binding["name"], name="R1 binding name")
            _finite_number(binding["value"], name="R1 binding value")
            if not isinstance(binding["bound"], bool):
                raise VN2ResultError("R1 binding bound must be boolean")
        spine.append((round_number, series_key))
    expected_spine = [
        (round_number, series_key)
        for round_number in range(1, config.round_count + 1)
        for series_key in sorted(identities, key=str.encode)
    ]
    if spine != expected_spine or len(identities) != config.series_count:
        raise VN2ResultError("R1 orders do not use the exact canonical Cartesian spine")
    if len(set(identities.values())) != len(identities):
        raise VN2ResultError("R1 series identities must be unique")
    return identities, _derive_session(config, identities), order_arrivals


def _validate_r2_payload(
    path: Path,
    *,
    config: VN2ProtocolConfig,
    session: SessionIdentity,
    identities: Mapping[str, tuple[int, int]],
    provenance_digest: str,
) -> tuple[SettlementRecord, ...]:
    rows = _load_jsonl(path, name="R2 cost ledger")
    expected_count = config.series_count * len(config.realized_periods)
    if len(rows) != expected_count:
        raise VN2ResultError(f"R2 cost ledger must contain exactly {expected_count} rows")
    period_by_index = {
        index: period for index, period in enumerate(config.realized_periods, start=1)
    }
    records: list[SettlementRecord] = []
    spine: list[tuple[int, str]] = []
    for index, row in enumerate(rows):
        name = f"R2 cost ledger[{index}]"
        _require_exact_keys(row, _R2_KEYS, name=name)
        _require_int(row["schema"], expected=1, name=f"{name}.schema")
        _require_semantics(row, config=config, provenance_digest=provenance_digest, name=name)
        if row["session_id"] != session.value:
            raise VN2ResultError(f"{name}.session_id does not match manifest")
        series_key = _require_text(row["series_key"], name=f"{name}.series_key")
        if series_key not in identities:
            raise VN2ResultError(f"{name}.series_key is foreign to R1")
        identity = (
            _integer(row["store"], name=f"{name}.store"),
            _integer(row["product"], name=f"{name}.product"),
        )
        if identity != identities[series_key]:
            raise VN2ResultError("R2 series mapping does not match R1")
        period_index = _positive_integer(
            row["period_index"],
            name=f"{name}.period_index",
        )
        if period_index not in period_by_index:
            raise VN2ResultError(f"{name}.period_index is outside the configured spine")
        period = _timestamp(row["period"], name=f"{name}.period")
        if period != period_by_index[period_index]:
            raise VN2ResultError(f"{name}.period does not match period_index")
        if row["currency"] != config.currency:
            raise VN2ResultError(f"{name}.currency does not match configuration")
        if row["stockout_rule"] != StockoutRule.LOST_SALES.value:
            raise VN2ResultError(f"{name}.stockout_rule must equal lost-sales")
        transition = StockoutTransition(
            rule=StockoutRule.LOST_SALES,
            demand=_finite_nonnegative(row["demand"], name=f"{name}.demand"),
            fulfilled_demand=_finite_nonnegative(row["sales"], name=f"{name}.sales"),
            unmet_demand=_finite_nonnegative(
                row["missed_sales"],
                name=f"{name}.missed_sales",
            ),
            closing_on_hand=_finite_nonnegative(
                row["end_inventory"],
                name=f"{name}.end_inventory",
            ),
            closing_backorders=_finite_nonnegative(
                row["closing_backorders"],
                name=f"{name}.closing_backorders",
            ),
        )
        if transition.available_inventory != _finite_nonnegative(
            row["start_inventory"],
            name=f"{name}.start_inventory",
        ):
            raise VN2ResultError("R2 start_inventory does not match booked transition facts")
        holding = BookedCost(
            rate=_finite_nonnegative(row["holding_rate"], name=f"{name}.holding_rate"),
            basis=_finite_nonnegative(
                row["holding_basis"],
                name=f"{name}.holding_basis",
            ),
            amount=_finite_nonnegative(
                row["holding_cost"],
                name=f"{name}.holding_cost",
            ),
        )
        shortage = BookedCost(
            rate=_finite_nonnegative(
                row["shortage_rate"],
                name=f"{name}.shortage_rate",
            ),
            basis=_finite_nonnegative(
                row["shortage_basis"],
                name=f"{name}.shortage_basis",
            ),
            amount=_finite_nonnegative(
                row["shortage_cost"],
                name=f"{name}.shortage_cost",
            ),
        )
        records.append(
            SettlementRecord(
                session=session,
                series_key=series_key,
                period=period,
                arrivals=_finite_nonnegative(row["arrivals"], name=f"{name}.arrivals"),
                actuals_semantics=config.actuals_semantics,
                transition=transition,
                inventory_position=InventoryPosition(
                    on_hand=transition.closing_on_hand,
                    on_order=_finite_nonnegative(row["on_order"], name=f"{name}.on_order"),
                    backorders=transition.closing_backorders,
                ),
                holding=holding,
                shortage=shortage,
            )
        )
        spine.append((period_index, series_key))
    expected_spine = [
        (period_index, series_key)
        for period_index in range(1, len(config.realized_periods) + 1)
        for series_key in sorted(identities, key=str.encode)
    ]
    if spine != expected_spine:
        raise VN2ResultError("R2 ledger does not use the exact canonical Cartesian spine")
    return tuple(records)


def _validate_r3_payload(
    path: Path,
    *,
    objective: SettlementObjective,
    semantics: ActualsSemantics,
    provenance_digest: str,
) -> None:
    value, _ = _load_json_object(path, name="R3 final triple")
    _require_exact_keys(value, _R3_KEYS, name="R3 final triple")
    expected = _r3_value(
        objective,
        semantics=semantics,
        provenance_digest=provenance_digest,
    )
    if value != expected:
        raise VN2ResultError("R3 final triple does not match the generic settlement reducer")


def _validate_r4_payload(
    path: Path,
    *,
    records: tuple[SettlementRecord, ...],
    objective: SettlementObjective,
    config: VN2ProtocolConfig,
    provenance_digest: str,
) -> None:
    value, _ = _load_json_object(path, name="R4 cost trajectory")
    _require_exact_keys(value, _R4_KEYS, name="R4 cost trajectory")
    expected = _r4_value(
        records,
        objective=objective,
        config=config,
        provenance_digest=provenance_digest,
    )
    if value != expected:
        raise VN2ResultError("R4 trajectory does not match the generic settlement reducer")


def _r3_value(
    objective: SettlementObjective,
    *,
    semantics: ActualsSemantics,
    provenance_digest: str,
) -> dict[str, object]:
    _require_exact_objective_total(objective)
    return {
        "actuals_semantics": semantics.value,
        "holding_total": objective.holding.value,
        "provenance_digest": provenance_digest,
        "schema": 1,
        "shortage_total": objective.shortage.value,
        "total_cost": objective.total.value,
    }


def _r4_value(
    records: tuple[SettlementRecord, ...],
    *,
    objective: SettlementObjective,
    config: VN2ProtocolConfig,
    provenance_digest: str,
) -> dict[str, object]:
    _require_objective_spine(objective.by_origin, config=config)
    partials = {partial.origin: partial.cost.value for partial in objective.partials}
    decision_rounds = [
        {
            "actuals_semantics": config.actuals_semantics.value,
            "cumulative_cost": partials[origin],
            "origin": origin.isoformat(),
            "provenance_digest": provenance_digest,
            "round": index,
        }
        for index, origin in enumerate(config.decision_origins, start=1)
    ]
    drain_periods = config.realized_periods[-config.drain_periods :]
    drain_period_set = frozenset(drain_periods)
    drain_records = tuple(record for record in records if record.period in drain_period_set)
    drain = settle_path_cost(
        drain_records,
        actuals_semantics=config.actuals_semantics,
    )
    if tuple(drain.by_origin) != drain_periods:
        raise VN2ResultError("drain settlement reducer does not cover exact drain periods")
    return {
        "actuals_semantics": config.actuals_semantics.value,
        "decision_rounds": decision_rounds,
        "drain_remainder": {
            "actuals_semantics": config.actuals_semantics.value,
            "cost": drain.total.value,
            "periods": [period.isoformat() for period in drain_periods],
            "provenance_digest": provenance_digest,
        },
        "provenance_digest": provenance_digest,
        "schema": 1,
    }


def _require_objective_spine(
    value: Mapping[pd.Timestamp, CostValue],
    *,
    config: VN2ProtocolConfig,
) -> None:
    if tuple(value) != config.realized_periods:
        raise VN2ResultError("generic settlement reducer does not cover exact realized periods")


def _require_exact_objective_total(objective: SettlementObjective) -> None:
    if objective.total.value != objective.holding.value + objective.shortage.value:
        raise VN2ResultError("generic settlement total must equal its exact R3 components")


def _derive_session(
    config: VN2ProtocolConfig,
    identities: Mapping[str, tuple[int, int]],
) -> SessionIdentity:
    series_keys = tuple(sorted(identities, key=str.encode))
    return SessionIdentity.derive(
        tenant=config.dataset,
        series_keys=series_keys,
        calendar=config.calendar,
        horizon=config.task_horizon,
        model_config=config.model_config,
        conformal_config=config.conformal_config,
        ordering_policy=config.ordering_policy,
        decision_series_keys=series_keys,
        cost_structure=config.cost_structure,
        decision_timing=config.timing,
        stockout_rule=config.stockout_rule,
    )


def _series_digest(identities: Mapping[str, tuple[int, int]]) -> str:
    value = [
        {"product": product, "series_key": series, "store": store}
        for series, (store, product) in sorted(
            identities.items(),
            key=lambda item: item[0].encode(),
        )
    ]
    return _digest_json(value, name="VN2 series identities")


def _descriptor_value(descriptor: GuaranteeDescriptor) -> dict[str, object]:
    return {
        "level": descriptor.level,
        "scope": {
            "class_system_name": descriptor.scope.class_system_name,
            "kind": descriptor.scope.kind.value,
        },
        "scored_series": descriptor.scored_series.value,
        "type": {
            "claim": descriptor.type.claim.value,
            "currency": (
                None if descriptor.type.currency is None else descriptor.type.currency.value
            ),
            "declared_slack": descriptor.type.declared_slack,
        },
        "window": descriptor.window.value,
    }


def _validate_descriptor(value: object, *, name: str) -> str:
    descriptor = _object(value, name=name)
    _require_exact_keys(descriptor, _DESCRIPTOR_KEYS, name=name)
    guarantee_type = _object(descriptor["type"], name=f"{name}.type")
    _require_exact_keys(guarantee_type, _GUARANTEE_TYPE_KEYS, name=f"{name}.type")
    claim = _require_text(guarantee_type["claim"], name=f"{name}.type.claim")
    if claim == GuaranteeClaim.NONE.value:
        if guarantee_type["currency"] is not None or guarantee_type["declared_slack"] is not None:
            raise VN2ResultError(f"{name} claim none cannot carry currency or slack")
    else:
        _require_text(guarantee_type["currency"], name=f"{name}.type.currency")
        if guarantee_type["declared_slack"] is not None:
            _finite_nonnegative(
                guarantee_type["declared_slack"],
                name=f"{name}.type.declared_slack",
            )
    level = _finite_number(descriptor["level"], name=f"{name}.level")
    if not 0.0 <= level <= 1.0:
        raise VN2ResultError(f"{name}.level must lie in [0, 1]")
    _require_text(descriptor["scored_series"], name=f"{name}.scored_series")
    _require_text(descriptor["window"], name=f"{name}.window")
    scope = _object(descriptor["scope"], name=f"{name}.scope")
    _require_exact_keys(scope, _SCOPE_KEYS, name=f"{name}.scope")
    _require_text(scope["kind"], name=f"{name}.scope.kind")
    if scope["class_system_name"] is not None:
        _require_text(scope["class_system_name"], name=f"{name}.scope.class_system_name")
    return claim


def _validated_identity(
    candidate_sha: object,
    workflow_sha: object,
    run_id: object,
    run_url: object,
) -> dict[str, str]:
    candidate = _require_commit_sha(candidate_sha, name="candidate_sha")
    workflow = _require_commit_sha(workflow_sha, name="workflow_sha")
    run = _require_run_id(run_id, name="run_id")
    return {
        "candidate_sha": candidate,
        "workflow_sha": workflow,
        "run_id": run,
        "run_url": _validate_run_url(run_url, run_id=run),
    }


def _environment_value(environment: VN2EvidenceEnvironment) -> dict[str, object]:
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


def _parse_environment(value: object) -> VN2EvidenceEnvironment:
    raw = _object(value, name="VN2 result environment")
    _require_exact_keys(raw, _ENVIRONMENT_KEYS, name="VN2 result environment")
    os_value = _object(raw["os"], name="VN2 result environment.os")
    _require_exact_keys(os_value, _OS_KEYS, name="VN2 result environment.os")
    thread_policy = _object(
        raw["thread_policy"],
        name="VN2 result environment.thread_policy",
    )
    return VN2EvidenceEnvironment(
        arch=cast(str, raw["arch"]),
        cpu_model=cast(str, raw["cpu_model"]),
        os_id=cast(str, os_value["id"]),
        os_version_id=cast(str, os_value["version_id"]),
        os_pretty_name=cast(str, os_value["pretty_name"]),
        python=cast(str, raw["python"]),
        numpy=cast(str, raw["numpy"]),
        numpy_config=cast(str, raw["numpy_config"]),
        runner_image=cast(str, raw["runner_image"]),
        thread_policy=cast(Mapping[str, str], thread_policy),
    )


def _parse_files(value: object) -> tuple[VN2ResultFile, ...]:
    if not isinstance(value, list) or not value:
        raise VN2ResultError("manifest files must be a non-empty list")
    files: list[VN2ResultFile] = []
    for index, raw in enumerate(value):
        item = _object(raw, name=f"manifest files[{index}]")
        _require_exact_keys(item, _FILE_KEYS, name=f"manifest files[{index}]")
        path = _payload_path(item["path"], name=f"manifest files[{index}].path")
        size = _positive_integer(item["bytes"], name=f"manifest files[{index}].bytes")
        files.append(
            VN2ResultFile(
                path=path,
                bytes=size,
                sha256=_require_sha256(
                    item["sha256"],
                    name=f"manifest files[{index}].sha256",
                ),
            )
        )
    paths = [entry.path for entry in files]
    if paths != sorted(paths, key=str.encode) or len(set(paths)) != len(paths):
        raise VN2ResultError("manifest payload entries must have unique canonical path order")
    return tuple(files)


def _file_value(entry: VN2ResultFile) -> dict[str, object]:
    return {"bytes": entry.bytes, "path": entry.path, "sha256": entry.sha256}


def _load_json_object(path: Path, *, name: str) -> tuple[dict[str, object], bytes]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except VN2ResultError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise VN2ResultError(f"{name} must be readable strict UTF-8 JSON") from error
    result = _object(value, name=name)
    if payload != _json_bytes(result, name=name):
        raise VN2ResultError(f"{name} must use canonical newline-terminated JSON bytes")
    return result, payload


def _load_jsonl(path: Path, *, name: str) -> list[dict[str, object]]:
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise VN2ResultError(f"{name} must be readable UTF-8 JSONL") from error
    if not payload or not payload.endswith(b"\n"):
        raise VN2ResultError(f"{name} must be non-empty and newline-terminated")
    rows: list[dict[str, object]] = []
    for index, line in enumerate(text.splitlines()):
        try:
            value = json.loads(line, object_pairs_hook=_unique_object)
        except VN2ResultError:
            raise
        except json.JSONDecodeError as error:
            raise VN2ResultError(f"{name}[{index}] must be strict JSON") from error
        row = _object(value, name=f"{name}[{index}]")
        if (line + "\n").encode() != _json_bytes(row, name=f"{name}[{index}]"):
            raise VN2ResultError(f"{name}[{index}] must use canonical JSON bytes")
        rows.append(row)
    return rows


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise VN2ResultError(f"VN2 result contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_bytes(value: object, *, name: str) -> bytes:
    try:
        return canonical_json_bytes(value, path=name) + b"\n"
    except CanonicalJsonError as error:
        raise VN2ResultError(str(error)) from error


def _digest_json(value: object, *, name: str) -> str:
    try:
        return _sha256(canonical_json_bytes(value, path=name))
    except CanonicalJsonError as error:
        raise VN2ResultError(str(error)) from error


def _object(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise VN2ResultError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual, key=str.encode)
        extra = sorted(actual - expected, key=str.encode)
        raise VN2ResultError(f"{name} fields mismatch: missing={missing!r}, extra={extra!r}")


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise VN2ResultError(f"{name} must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise VN2ResultError(f"{name} must be valid UTF-8") from error
    return value


def _require_commit_sha(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if _COMMIT_SHA.fullmatch(text) is None:
        raise VN2ResultError(f"{name} must be a full lowercase 40-hex SHA")
    return text


def _require_sha256(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if _SHA256.fullmatch(text) is None:
        raise VN2ResultError(f"{name} must be a lowercase sha256")
    return text


def _require_run_id(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    if _RUN_ID.fullmatch(text) is None:
        raise VN2ResultError(f"{name} must be a positive decimal identifier")
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
        raise VN2ResultError(f"run_url must equal the {GITHUB_REPOSITORY} Actions URL for run_id")
    return text


def _payload_path(value: object, *, name: str) -> str:
    text = _require_text(value, name=name)
    path = PurePosixPath(text)
    if (
        text == "."
        or path.is_absolute()
        or "." in path.parts
        or ".." in path.parts
        or path.as_posix() != text
        or "\\" in text
        or text in {"manifest.json", "files.sha256"}
    ):
        raise VN2ResultError(f"{name} must be a canonical relative POSIX payload path")
    return text


def _require_expected(actual: str, expected: object, *, name: str) -> None:
    expected_text = _require_text(expected, name=f"expected {name}")
    if actual != expected_text:
        raise VN2ResultError(f"VN2 result {name} does not match the requested value")


def _require_trusted_digest(actual: str, path: Path, *, name: str) -> None:
    if actual != _sha256_file(path, name=name):
        raise VN2ResultError(f"VN2 result {name} does not match trusted input bytes")


def _sha256_file(path: Path, *, name: str) -> str:
    try:
        return _sha256(Path(path).read_bytes())
    except OSError as error:
        raise VN2ResultError(f"trusted {name} input must be a readable file") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_int(value: object, *, expected: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise VN2ResultError(f"{name} must be the integer {expected}")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VN2ResultError(f"{name} must be an integer")
    return value


def _positive_integer(value: object, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 1:
        raise VN2ResultError(f"{name} must be positive")
    return integer


def _finite_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VN2ResultError(f"{name} must be a finite real number")
    number = float(value)
    if not math.isfinite(number):
        raise VN2ResultError(f"{name} must be a finite real number")
    return 0.0 if number == 0.0 else number


def _finite_nonnegative(value: object, *, name: str) -> float:
    number = _finite_number(value, name=name)
    if number < 0.0:
        raise VN2ResultError(f"{name} must be non-negative")
    return number


def _timestamp(value: object, *, name: str) -> pd.Timestamp:
    text = _require_text(value, name=name)
    try:
        timestamp = pd.Timestamp(text)
    except (TypeError, ValueError) as error:
        raise VN2ResultError(f"{name} must be an ISO timestamp") from error
    if timestamp.tz is not None or timestamp.isoformat() != text:
        raise VN2ResultError(f"{name} must be a canonical timezone-naive ISO timestamp")
    return timestamp


def _require_semantics(
    row: Mapping[str, object],
    *,
    config: VN2ProtocolConfig,
    provenance_digest: str,
    name: str,
) -> None:
    if row["actuals_semantics"] != config.actuals_semantics.value:
        raise VN2ResultError(f"{name}.actuals_semantics does not match configuration")
    if row["provenance_digest"] != provenance_digest:
        raise VN2ResultError(f"{name}.provenance_digest does not match manifest provenance")


def _os_release() -> dict[str, str]:
    try:
        lines = Path("/etc/os-release").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise VN2ResultError("VN2 evidence requires readable /etc/os-release") from error
    result: dict[str, str] = {}
    for line in lines:
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        result[key] = raw_value.strip().strip('"')
    return result


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.casefold().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


__all__ = [
    "THREAD_VARIABLES",
    "VN2EvidenceEnvironment",
    "VN2ResultBundle",
    "VN2ResultError",
    "VN2ResultFile",
    "VN2ResultManifest",
    "capture_vn2_evidence_environment",
    "emit_vn2_result_bundle",
    "validate_vn2_result_bundle",
]
