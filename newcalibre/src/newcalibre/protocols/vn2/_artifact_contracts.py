"""Define the strict VN2 result wire contract and canonical value builders."""

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

import pandas as pd

from newcalibre.domain import ActualsSemantics, SessionIdentity
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.ledger import SettlementRecord
from newcalibre.ordering import CostComponents, CostValue, SettlementObjective, settle_path_cost
from newcalibre.protocols.vn2.config import VN2ProtocolConfig

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
    """Return one content-verified bundle plus the manifest digest and R3 costs."""

    root: Path
    manifest: VN2ResultManifest
    manifest_sha256: str
    cost: CostComponents


@dataclass(frozen=True, slots=True)
class _RunIdentity:
    candidate_sha: str
    workflow_sha: str
    run_id: str
    run_url: str


@dataclass(frozen=True, slots=True)
class _TrustedInputs:
    config_digest: str
    input_inventory_digest: str
    lock_digest: str


def _provenance_value(
    *,
    config: VN2ProtocolConfig,
    artifact_name: str,
    identity: _RunIdentity,
    trusted: _TrustedInputs,
    environment_digest: str,
    series_identity_digest: str,
    session_id: str,
) -> dict[str, object]:
    return {
        "actuals_semantics": config.actuals_semantics.value,
        "artifact_kind": RESULT_KIND,
        "artifact_name": artifact_name,
        "candidate_sha": identity.candidate_sha,
        "config_digest": trusted.config_digest,
        "environment_digest": environment_digest,
        "input_inventory_digest": trusted.input_inventory_digest,
        "lock_digest": trusted.lock_digest,
        "realized_periods": [period.isoformat() for period in config.realized_periods],
        "run_id": identity.run_id,
        "run_url": identity.run_url,
        "series_identity_digest": series_identity_digest,
        "session_id": session_id,
        "workflow_sha": identity.workflow_sha,
    }


def _r3_value(
    objective: SettlementObjective,
    *,
    semantics: ActualsSemantics,
    provenance_digest: str,
) -> dict[str, object]:
    return {
        "actuals_semantics": semantics.value,
        "holding_total": objective.holding.value,
        "provenance_digest": provenance_digest,
        "schema": 1,
        "shortage_total": objective.shortage.value,
        "total_cost": objective.holding.value + objective.shortage.value,
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


def _derive_session(
    config: VN2ProtocolConfig,
    series_keys: tuple[str, ...],
) -> SessionIdentity:
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


def _series_digest(
    identities: Mapping[str, tuple[int, int]],
    *,
    series_keys: tuple[str, ...] | None = None,
) -> str:
    ordered_series_keys = (
        tuple(sorted(identities, key=str.encode)) if series_keys is None else series_keys
    )
    value = [
        {
            "product": identities[series][1],
            "series_key": series,
            "store": identities[series][0],
        }
        for series in ordered_series_keys
    ]
    return _digest_json(value, name="VN2 series identities")


def _validated_identity(
    candidate_sha: object,
    workflow_sha: object,
    run_id: object,
    run_url: object,
) -> _RunIdentity:
    candidate = _require_commit_sha(candidate_sha, name="candidate_sha")
    workflow = _require_commit_sha(workflow_sha, name="workflow_sha")
    run = _require_run_id(run_id, name="run_id")
    return _RunIdentity(
        candidate_sha=candidate,
        workflow_sha=workflow,
        run_id=run,
        run_url=_validate_run_url(run_url, run_id=run),
    )


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
    try:
        file_stat = path.lstat()
    except OSError as error:
        raise VN2ResultError(f"trusted {name} input must be a readable file") from error
    if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
        raise VN2ResultError(f"trusted {name} input must be a real non-symlink regular file")
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
