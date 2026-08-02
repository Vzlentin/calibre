"""Publish and validate one attributable five-file M5 Gate C result."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from newcalibre.benchmarking.environment import validate_environment
from newcalibre.benchmarking.profile import ProfileError, validate_profile
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.protocols.m5 import (
    M5ArtifactSet,
    load_m5_config,
    validate_m5_artifact_files,
)

RESULT_FILE_NAMES = frozenset(
    {
        "coverage-summary.json",
        "coverage-by-node.parquet",
        "report.md",
        "profile.json",
        "environment.json",
    }
)
_LEVEL_NODE_COUNTS = {
    "bottom": 30_490,
    "item": 3_049,
    "department": 7,
    "category": 3,
    "store": 10,
    "state": 3,
    "total": 1,
}
_NODE_COUNT = 33_563
_ORIGIN_COUNT = 64
_HORIZON = 28
_ROWS_PER_NODE = _ORIGIN_COUNT * _HORIZON
_ELIGIBLE_PER_NODE = sum(range(27, 55))


class M5GateCResultError(ValueError):
    """Report malformed, incomplete, or incorrectly bound Gate C evidence."""


@dataclass(frozen=True, slots=True)
class M5GateCResult:
    """Carry one validated result and its recomputed binding disposition."""

    root: Path
    artifacts: M5ArtifactSet
    profile: Mapping[str, object]
    environment: Mapping[str, object]
    disposition: str
    failures: tuple[str, ...]
    profile_digest: str
    environment_digest: str


def publish_m5_gate_c_result(
    destination: Path,
    *,
    diagnostics: Path,
    profile: object,
    environment: object,
) -> None:
    """Validate and atomically publish exactly the five ordinary result files."""
    target = Path(destination)
    if target.is_symlink() or target.exists():
        raise M5GateCResultError("Gate C result destination already exists")
    artifacts = validate_m5_artifact_files(
        Path(diagnostics) / "coverage-summary.json",
        Path(diagnostics) / "coverage-by-node.parquet",
        Path(diagnostics) / "report.md",
        root=Path(diagnostics),
    )
    environment_payload = validate_environment(environment)
    profile_payload = validate_profile(profile, environment=environment_payload)
    if profile_payload["valid"] is not True:
        raise M5GateCResultError("invalid measurement evidence cannot be published")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for name in ("coverage-summary.json", "coverage-by-node.parquet", "report.md"):
            shutil.copyfile(artifacts.root / name, temporary / name)
        (temporary / "profile.json").write_bytes(_json_bytes(profile_payload))
        (temporary / "environment.json").write_bytes(_json_bytes(environment_payload))
        if {path.name for path in temporary.iterdir()} != RESULT_FILE_NAMES:
            raise M5GateCResultError("Gate C staging produced an unexpected file set")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_m5_gate_c_result(
    root: Path,
    *,
    config_path: Path,
    inventory_path: Path,
    lock_path: Path,
    expected_candidate_sha: str | None = None,
) -> M5GateCResult:
    """Validate five committed files and recompute the Gate C disposition."""
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise M5GateCResultError("Gate C result root must be a regular directory")
    paths = {path.name: path for path in directory.iterdir()}
    if set(paths) != RESULT_FILE_NAMES or any(path.is_symlink() for path in paths.values()):
        raise M5GateCResultError("Gate C result must contain exactly five regular files")
    try:
        environment = _canonical_json_file(paths["environment.json"], name="environment")
        environment_payload = validate_environment(environment)
        profile = _canonical_json_file(paths["profile.json"], name="profile")
        profile_payload = validate_profile(profile, environment=environment_payload)
        artifacts = validate_m5_artifact_files(
            paths["coverage-summary.json"],
            paths["coverage-by-node.parquet"],
            paths["report.md"],
            root=directory,
        )
        _validate_bindings(
            environment_payload,
            config_path=config_path,
            inventory_path=inventory_path,
            lock_path=lock_path,
            expected_candidate_sha=expected_candidate_sha,
        )
    except (OSError, ValueError, CanonicalJsonError) as error:
        if isinstance(error, M5GateCResultError):
            raise
        raise M5GateCResultError(str(error)) from error
    failures = recompute_gate_c_failures(artifacts=artifacts, profile=profile_payload)
    return M5GateCResult(
        root=directory.resolve(),
        artifacts=artifacts,
        profile=profile_payload,
        environment=environment_payload,
        disposition="GO" if not failures else "NO-GO",
        failures=failures,
        profile_digest=_file_digest(paths["profile.json"]),
        environment_digest=_file_digest(paths["environment.json"]),
    )


def recompute_gate_c_failures(
    *,
    artifacts: M5ArtifactSet,
    profile: Mapping[str, object],
) -> tuple[str, ...]:
    """Recompute binding failures without consulting sales-coverage values."""
    failures: list[str] = []
    summary = artifacts.summary
    context = cast(Mapping[str, object], summary["context"])
    population = cast(Mapping[str, object], summary["population"])
    population_counts = cast(Mapping[str, object], population["counts"])
    mask = cast(Mapping[str, object], summary["mask"])
    if (
        context["dataset"] != "m5"
        or context["phase"] != "evaluation"
        or context["origin_count"] != _ORIGIN_COUNT
        or context["horizon"] != _HORIZON
        or context["node_count"] != _NODE_COUNT
        or context["expected_row_count"] != _NODE_COUNT * _ROWS_PER_NODE
        or context["model_name"] != "seasonal-naive"
        or context["reconciler"] != "wls_struct"
        or context["conformal_method"] != "split-per-step"
        or context["conformal_partition"] != "series-horizon"
    ):
        failures.append("full M5 configured identity is not exact")
    expected_counts = {
        "total": _NODE_COUNT * _ROWS_PER_NODE,
        "resolved": _NODE_COUNT * _ROWS_PER_NODE,
        "eligible": _NODE_COUNT * _ELIGIBLE_PER_NODE,
        "scored": _NODE_COUNT * _ELIGIBLE_PER_NODE,
    }
    if any(population_counts[key] != value for key, value in expected_counts.items()):
        failures.append("population resolution and scored-mask completeness failed")
    if (
        summary["status"] != "VALID"
        or population["mask_equal"] is not True
        or mask["equal"] is not True
        or mask["structural_issue_count"] != 0
        or mask["missing_eligible_count"] != 0
        or mask["early_scored_count"] != 0
    ):
        failures.append("diagnostic structure or exact mask is invalid")
    levels = cast(list[Mapping[str, object]], summary["levels"])
    for level in levels:
        name = cast(str, level["level"])
        node_count = _LEVEL_NODE_COUNTS.get(name)
        counts = cast(Mapping[str, object], level["counts"])
        if (
            node_count is None
            or level["node_count"] != node_count
            or counts["total"] != node_count * _ROWS_PER_NODE
            or counts["resolved"] != node_count * _ROWS_PER_NODE
            or counts["eligible"] != node_count * _ELIGIBLE_PER_NODE
            or counts["scored"] != node_count * _ELIGIBLE_PER_NODE
            or level["mask_equal"] is not True
        ):
            failures.append(f"{name} level completeness failed")
    node_rows = artifacts.nodes.to_pylist()
    if any(
        row["total"] != _ROWS_PER_NODE
        or row["resolved"] != _ROWS_PER_NODE
        or row["eligible"] != _ELIGIBLE_PER_NODE
        or row["scored"] != _ELIGIBLE_PER_NODE
        or row["mask_equal"] is not True
        for row in node_rows
    ):
        failures.append("node-level completeness failed")
    if profile["valid"] is not True:
        failures.append("profile evidence is invalid or incomplete")
    budgets = cast(Mapping[str, Mapping[str, object]], profile["budgets"])
    for name in (
        "wall_seconds",
        "pre_origin_seconds",
        "peak_job_memory_bytes",
        "reconciliation_percent",
    ):
        if budgets[name]["passed"] is not True:
            failures.append(f"{name} budget failed")
    return tuple(failures)


def _validate_bindings(
    environment: Mapping[str, object],
    *,
    config_path: Path,
    inventory_path: Path,
    lock_path: Path,
    expected_candidate_sha: str | None,
) -> None:
    config = load_m5_config(config_path)
    if (
        config.population.kind != "full"
        or config.horizon != _HORIZON
        or config.origin_count != _ORIGIN_COUNT
        or config.reconciliation_strategy != "wls_struct"
        or config.conformal_partition != "series-horizon"
        or config.execution.logical_shards != 16
        or config.execution.workers != 16
        or config.execution.numeric_threads_per_worker != 1
        or config.execution.retries != 0
    ):
        raise M5GateCResultError("committed Gate C configuration is not the selected strict intent")
    inputs = cast(Mapping[str, object], environment["input"])
    provenance = cast(Mapping[str, object], environment["provenance"])
    if inputs["config_sha256"] != _file_digest(config_path):
        raise M5GateCResultError("environment configuration digest differs from the committed file")
    if inputs["inventory_sha256"] != _file_digest(inventory_path):
        raise M5GateCResultError("environment inventory digest differs from the committed file")
    if provenance["lock_sha256"] != _file_digest(lock_path):
        raise M5GateCResultError("environment lock digest differs from the committed file")
    if (
        expected_candidate_sha is not None
        and environment["candidate_sha"] != expected_candidate_sha
    ):
        raise M5GateCResultError("environment candidate SHA differs from the measured candidate")


def _canonical_json_file(path: Path, *, name: str) -> dict[str, object]:
    payload = path.read_bytes()
    raw = json.loads(payload)
    if not isinstance(raw, dict):
        raise M5GateCResultError(f"Gate C {name} must be a JSON object")
    value = cast(dict[str, object], raw)
    if payload != canonical_json_bytes(value, path=f"Gate C {name}") + b"\n":
        raise M5GateCResultError(f"Gate C {name} is not canonical JSON")
    return value


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_bytes(value: object) -> bytes:
    try:
        return canonical_json_bytes(value, path="Gate C result artifact") + b"\n"
    except CanonicalJsonError as error:
        raise ProfileError("Gate C result artifact is not canonical JSON") from error


__all__ = [
    "M5GateCResult",
    "M5GateCResultError",
    "RESULT_FILE_NAMES",
    "load_m5_gate_c_result",
    "publish_m5_gate_c_result",
    "recompute_gate_c_failures",
]
