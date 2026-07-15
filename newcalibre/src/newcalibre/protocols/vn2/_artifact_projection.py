"""Capture the VN2 evidence environment and project engine facts into bundle bytes."""

from __future__ import annotations

import contextlib
import io
import os
import platform
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np
import pandas as pd

from newcalibre.domain import GuaranteeClaim, GuaranteeDescriptor
from newcalibre.ledger import OrderRow, SettlementRecord
from newcalibre.ordering import settle_path_cost
from newcalibre.protocols.vn2._artifact_contracts import (
    CONFIG_PATH,
    INPUT_INVENTORY_PATH,
    LOCK_PATH,
    RESULT_KIND,
    THREAD_VARIABLES,
    VN2EvidenceEnvironment,
    VN2ResultBundle,
    VN2ResultError,
    VN2ResultFile,
    _derive_session,
    _digest_json,
    _environment_value,
    _file_value,
    _json_bytes,
    _provenance_value,
    _r3_value,
    _r4_value,
    _require_objective_spine,
    _series_digest,
    _sha256,
    _sha256_file,
    _TrustedInputs,
    _validated_identity,
)
from newcalibre.protocols.vn2._artifact_validation import (
    _validate_lost_sales_sequence,
    validate_vn2_result_bundle,
)
from newcalibre.protocols.vn2.adapter import VN2RunResult
from newcalibre.protocols.vn2.config import VN2ProtocolConfig, load_vn2_config

_IMAGE_OS_ENV = "ImageOS"
_IMAGE_VERSION_ENV = "ImageVersion"


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
    artifact_name = f"vn2-acceptance-{identity.candidate_sha}"
    provenance_value = _provenance_value(
        config=config,
        artifact_name=artifact_name,
        identity=identity,
        trusted=trusted,
        environment_digest=environment_digest,
        series_identity_digest=facts.series_identity_digest,
        session_id=result.session.value,
    )
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
        "candidate_sha": identity.candidate_sha,
        "config_digest": trusted.config_digest,
        "config_path": CONFIG_PATH,
        "environment": environment_value,
        "environment_digest": environment_digest,
        "files": [_file_value(entry) for entry in files],
        "inner_bundle_digest": _sha256(listing),
        "input_inventory_digest": trusted.input_inventory_digest,
        "input_inventory_path": INPUT_INVENTORY_PATH,
        "lock_digest": trusted.lock_digest,
        "lock_path": LOCK_PATH,
        "provenance_digest": provenance_digest,
        "realized_period_count": len(config.realized_periods),
        "round_count": config.round_count,
        "run_id": identity.run_id,
        "run_url": identity.run_url,
        "schema": 1,
        "series_count": config.series_count,
        "series_identity_digest": facts.series_identity_digest,
        "session_id": result.session.value,
        "workflow_sha": identity.workflow_sha,
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
        validated = validate_vn2_result_bundle(
            temporary,
            expected_candidate_sha=identity.candidate_sha,
            expected_workflow_sha=identity.workflow_sha,
            expected_run_id=identity.run_id,
            expected_config_path=Path(config_path),
            expected_input_inventory_path=Path(input_inventory_path),
            expected_lock_path=Path(lock_path),
        )
        published = replace(validated, root=bundle_root.resolve())
        temporary.replace(bundle_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return published


def _trusted_inputs(
    config: VN2ProtocolConfig,
    config_path: Path,
    input_inventory_path: Path,
    lock_path: Path,
) -> _TrustedInputs:
    config_file = Path(config_path)
    input_inventory_file = Path(input_inventory_path)
    lock_file = Path(lock_path)
    config_digest = _sha256_file(config_file, name="config_digest")
    input_inventory_digest = _sha256_file(
        input_inventory_file,
        name="input_inventory_digest",
    )
    lock_digest = _sha256_file(lock_file, name="lock_digest")
    try:
        trusted_config = load_vn2_config(config_file)
    except (OSError, ValueError) as error:
        raise VN2ResultError("trusted VN2 configuration is invalid") from error
    if trusted_config != config:
        raise VN2ResultError("VN2 configuration object does not match trusted config bytes")
    return _TrustedInputs(
        config_digest=config_digest,
        input_inventory_digest=input_inventory_digest,
        lock_digest=lock_digest,
    )


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
    canonical_series_keys = tuple(sorted(identities, key=str.encode))
    expected_session = _derive_session(config, canonical_series_keys)
    if result.session != expected_session:
        raise VN2ResultError("VN2 result session does not match its series mapping and config")
    series_keys = frozenset(canonical_series_keys)
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
    if len(order_by_key) != config.series_count * config.round_count:
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
        if (
            record.holding.rate != config.holding_rate
            or record.shortage.rate != config.shortage_rate
        ):
            raise VN2ResultError(
                "every VN2 settlement cost rate must match the configured cost structure"
            )
        key = (record.series_key, record.period)
        if key in settlement_by_key:
            raise VN2ResultError("VN2 settlements must have one row per series and period")
        settlement_by_key[key] = record
    if len(settlement_by_key) != config.series_count * len(config.realized_periods):
        raise VN2ResultError("VN2 R2 settlement spine is incomplete")
    ordered_orders = tuple(
        order_by_key[(series, origin)]
        for origin in config.decision_origins
        for series in canonical_series_keys
    )
    ordered_settlements = tuple(
        settlement_by_key[(series, period)]
        for period in config.realized_periods
        for series in canonical_series_keys
    )
    _validate_lost_sales_sequence(ordered_settlements, name="VN2 settlement")
    return _ValidatedFacts(
        identities=MappingProxyType(identities),
        orders=ordered_orders,
        settlements=ordered_settlements,
        series_identity_digest=_series_digest(
            identities,
            series_keys=canonical_series_keys,
        ),
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


def _os_release() -> dict[str, str]:
    try:
        return platform.freedesktop_os_release()
    except OSError as error:
        raise VN2ResultError("VN2 evidence requires readable /etc/os-release") from error


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.casefold().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()
