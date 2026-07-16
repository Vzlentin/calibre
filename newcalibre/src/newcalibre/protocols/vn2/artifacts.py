"""Emit and load deterministic VN2 R1-R4 result bundles."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from newcalibre.ledger import OrderRow, SettlementRecord
from newcalibre.protocols.vn2.adapter import VN2RunResult
from newcalibre.protocols.vn2.config import VN2ProtocolConfig, load_vn2_config

PLATFORM = "ubuntu-24.04/x86_64"
_PAYLOAD_NAMES = (
    "r1-orders.jsonl",
    "r2-cost-ledger.jsonl",
    "r3-final-triple.json",
    "r4-cost-trajectory.json",
)
_ALL_NAMES = frozenset({"manifest.json", *_PAYLOAD_NAMES})
_MANIFEST_KEYS = frozenset(
    {
        "actuals_semantics",
        "candidate_sha",
        "capture_digest",
        "config_digest",
        "files",
        "input_inventory_digest",
        "lock_digest",
        "platform",
        "realized_period_count",
        "round_count",
        "schema",
        "series_count",
        "series_identity_digest",
        "session_id",
        "settlement_count",
    }
)
_R1_KEYS = frozenset(
    {
        "arrival_period",
        "model_name",
        "origin",
        "product",
        "quantity",
        "round",
        "schema",
        "series_key",
        "store",
    }
)
_R2_KEYS = frozenset(
    {
        "actuals_semantics",
        "arrivals",
        "closing_backorders",
        "demand",
        "end_inventory",
        "holding_cost",
        "missed_sales",
        "on_order",
        "period",
        "period_index",
        "product",
        "sales",
        "schema",
        "series_key",
        "shortage_cost",
        "start_inventory",
        "stockout_rule",
        "store",
    }
)
_R3_KEYS = frozenset({"holding_cost", "schema", "shortage_cost", "total_cost"})
_R4_KEYS = frozenset({"decision_rounds", "drain", "schema"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT_SHA = re.compile(r"[0-9a-f]{40}")


class VN2ResultError(ValueError):
    """Report incomplete engine facts or a malformed result bundle."""


class _DuplicateKey(ValueError):
    """Retain duplicate JSON keys for strict parsing."""


@dataclass(frozen=True, slots=True)
class VN2ResultManifest:
    """Expose the compact identities and payload hashes for one VN2 run."""

    candidate_sha: str
    config_digest: str
    input_inventory_digest: str
    capture_digest: str
    lock_digest: str
    platform: str
    actuals_semantics: str
    session_id: str
    series_count: int
    round_count: int
    realized_period_count: int
    settlement_count: int
    series_identity_digest: str
    files: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class VN2ResultBundle:
    """Return a validated result bundle and its reduced final costs."""

    root: Path
    manifest: VN2ResultManifest
    manifest_sha256: str
    holding_cost: float
    shortage_cost: float
    total_cost: float


def emit_result_bundle(
    root: Path,
    *,
    result: VN2RunResult,
    config: VN2ProtocolConfig,
    candidate_sha: str,
    config_path: Path,
    input_inventory_path: Path,
    lock_path: Path,
    capture_digest: str,
) -> VN2ResultBundle:
    """Project generic-engine rows and atomically emit deterministic R1-R4 bytes."""
    destination = Path(root)
    if destination.exists() or destination.is_symlink():
        raise VN2ResultError("result bundle destination must not already exist")
    if not isinstance(result, VN2RunResult) or not isinstance(config, VN2ProtocolConfig):
        raise VN2ResultError("result projection requires VN2RunResult and VN2ProtocolConfig")
    candidate = _commit(candidate_sha, name="candidate")
    capture = _digest(capture_digest, name="capture")
    trusted_config = load_vn2_config(Path(config_path))
    if trusted_config != config:
        raise VN2ResultError("configuration object does not match trusted config bytes")
    trusted = {
        "config_digest": _file_digest(Path(config_path), name="config"),
        "input_inventory_digest": _file_digest(Path(input_inventory_path), name="input inventory"),
        "lock_digest": _file_digest(Path(lock_path), name="lock"),
    }
    identities, orders, settlements = _validated_engine_rows(result, config=config)
    r1 = _project_r1(result, config=config, identities=identities, orders=orders)
    r2 = _project_r2(config=config, identities=identities, settlements=settlements)
    r3 = _reduce_r3(r2)
    r4 = _reduce_r4(r2, config=config)
    payloads = {
        "r1-orders.jsonl": _jsonl_bytes(r1),
        "r2-cost-ledger.jsonl": _jsonl_bytes(r2),
        "r3-final-triple.json": _json_bytes(r3),
        "r4-cost-trajectory.json": _json_bytes(r4),
    }
    manifest_value = {
        "actuals_semantics": config.actuals_semantics.value,
        "candidate_sha": candidate,
        "capture_digest": capture,
        **trusted,
        "files": {name: _sha256(payloads[name]) for name in sorted(payloads)},
        "platform": PLATFORM,
        "realized_period_count": len(config.realized_periods),
        "round_count": config.round_count,
        "schema": 1,
        "series_count": config.series_count,
        "series_identity_digest": _identity_digest(identities),
        "session_id": result.session.value,
        "settlement_count": len(settlements),
    }
    manifest_bytes = _json_bytes(manifest_value)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        for name, payload in payloads.items():
            (temporary / name).write_bytes(payload)
        (temporary / "manifest.json").write_bytes(manifest_bytes)
        validated = load_result_bundle(
            temporary,
            expected_candidate_sha=candidate,
            config_path=Path(config_path),
            input_inventory_path=Path(input_inventory_path),
            lock_path=Path(lock_path),
            expected_capture_digest=capture,
        )
        temporary.replace(destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return VN2ResultBundle(
        root=destination.resolve(),
        manifest=validated.manifest,
        manifest_sha256=validated.manifest_sha256,
        holding_cost=validated.holding_cost,
        shortage_cost=validated.shortage_cost,
        total_cost=validated.total_cost,
    )


def load_result_bundle(
    root: Path,
    *,
    expected_candidate_sha: str,
    config_path: Path,
    input_inventory_path: Path,
    lock_path: Path,
    expected_capture_digest: str,
) -> VN2ResultBundle:
    """Load and validate exact files, trusted identities, shapes, and reduced costs."""
    bundle_root = Path(root)
    if bundle_root.is_symlink() or not bundle_root.is_dir():
        raise VN2ResultError("result bundle must be a real directory")
    actual: set[str] = set()
    for path in bundle_root.rglob("*"):
        relative = path.relative_to(bundle_root).as_posix()
        if path.is_symlink():
            raise VN2ResultError(f"result bundle contains a symbolic link: {relative}")
        if not path.is_file():
            raise VN2ResultError(f"result bundle contains an unexpected directory: {relative}")
        actual.add(relative)
    if actual != _ALL_NAMES:
        raise VN2ResultError("result bundle file set does not match manifest plus R1-R4")

    manifest_bytes = _read_regular(bundle_root / "manifest.json", name="result manifest")
    raw = _json_object(manifest_bytes, name="result manifest")
    if set(raw) != _MANIFEST_KEYS or raw["schema"] != 1:
        raise VN2ResultError("result manifest must use the exact schema 1 keys")
    if _json_bytes(raw) != manifest_bytes:
        raise VN2ResultError("result manifest must use canonical JSON bytes")
    expected_candidate = _commit(expected_candidate_sha, name="expected candidate")
    candidate = _commit(raw["candidate_sha"], name="manifest candidate")
    if candidate != expected_candidate:
        raise VN2ResultError("result candidate does not match expected candidate")
    expected_capture = _digest(expected_capture_digest, name="expected capture")
    capture = _digest(raw["capture_digest"], name="manifest capture")
    if capture != expected_capture:
        raise VN2ResultError("result capture does not match expected capture")
    if raw["platform"] != PLATFORM:
        raise VN2ResultError(f"result platform must equal {PLATFORM!r}")

    config = load_vn2_config(Path(config_path))
    expected_digests = {
        "config_digest": _file_digest(Path(config_path), name="config"),
        "input_inventory_digest": _file_digest(Path(input_inventory_path), name="input inventory"),
        "lock_digest": _file_digest(Path(lock_path), name="lock"),
    }
    for name, expected in expected_digests.items():
        if _digest(raw[name], name=name) != expected:
            raise VN2ResultError(f"result {name} does not match trusted bytes")
    if raw["actuals_semantics"] != config.actuals_semantics.value:
        raise VN2ResultError("result actuals semantics do not match config")

    raw_files = raw["files"]
    if not isinstance(raw_files, dict) or set(raw_files) != set(_PAYLOAD_NAMES):
        raise VN2ResultError("result manifest files must name exactly R1-R4")
    files = cast(dict[str, object], raw_files)
    file_digests: dict[str, str] = {}
    for name in _PAYLOAD_NAMES:
        digest = _digest(files[name], name=f"{name} digest")
        if _sha256(_read_regular(bundle_root / name, name=name)) != digest:
            raise VN2ResultError(f"result payload digest mismatch: {name}")
        file_digests[name] = digest

    r1 = _jsonl(bundle_root / "r1-orders.jsonl", name="R1")
    r2 = _jsonl(bundle_root / "r2-cost-ledger.jsonl", name="R2")
    identities = _validate_r1(r1, config=config)
    _validate_r2(r2, config=config, identities=identities)
    if raw["series_identity_digest"] != _identity_digest(identities):
        raise VN2ResultError("result series identity digest does not match R1")
    expected_shape = {
        "series_count": config.series_count,
        "round_count": config.round_count,
        "realized_period_count": len(config.realized_periods),
        "settlement_count": config.series_count * len(config.realized_periods),
    }
    for name, expected in expected_shape.items():
        if raw[name] != expected:
            raise VN2ResultError(f"result {name} does not match configured shape")
    session_id = _digest(raw["session_id"], name="session id")

    r3 = _json_object(_read_regular(bundle_root / "r3-final-triple.json", name="R3"), name="R3")
    expected_r3 = _reduce_r3(r2)
    if set(r3) != _R3_KEYS or r3 != expected_r3:
        raise VN2ResultError("R3 does not exactly reduce R2 costs")
    r4 = _json_object(_read_regular(bundle_root / "r4-cost-trajectory.json", name="R4"), name="R4")
    if set(r4) != _R4_KEYS or r4 != _reduce_r4(r2, config=config):
        raise VN2ResultError("R4 does not exactly reduce the R2 trajectory")
    holding_cost = _nonnegative(r3["holding_cost"], name="R3 holding cost")
    shortage_cost = _nonnegative(r3["shortage_cost"], name="R3 shortage cost")
    total_cost = _nonnegative(r3["total_cost"], name="R3 total cost")
    if total_cost != holding_cost + shortage_cost:
        raise VN2ResultError("R3 total must equal holding plus shortage")

    manifest = VN2ResultManifest(
        candidate_sha=candidate,
        config_digest=expected_digests["config_digest"],
        input_inventory_digest=expected_digests["input_inventory_digest"],
        capture_digest=capture,
        lock_digest=expected_digests["lock_digest"],
        platform=PLATFORM,
        actuals_semantics=config.actuals_semantics.value,
        session_id=session_id,
        series_count=config.series_count,
        round_count=config.round_count,
        realized_period_count=len(config.realized_periods),
        settlement_count=expected_shape["settlement_count"],
        series_identity_digest=str(raw["series_identity_digest"]),
        files=MappingProxyType(file_digests),
    )
    return VN2ResultBundle(
        root=bundle_root.resolve(),
        manifest=manifest,
        manifest_sha256=_sha256(manifest_bytes),
        holding_cost=holding_cost,
        shortage_cost=shortage_cost,
        total_cost=total_cost,
    )


def _validated_engine_rows(
    result: VN2RunResult,
    *,
    config: VN2ProtocolConfig,
) -> tuple[
    dict[str, tuple[int, int]],
    tuple[OrderRow, ...],
    tuple[SettlementRecord, ...],
]:
    identities = dict(result.series_identities)
    if len(identities) != config.series_count or len(set(identities.values())) != len(identities):
        raise VN2ResultError("result series identities do not match configured series_count")
    series = tuple(sorted(identities, key=str.encode))
    order_by_key = {(row.series_key, row.origin): row for row in result.orders}
    settlement_by_key = {(row.series_key, row.period): row for row in result.settlements}
    if len(order_by_key) != len(result.orders):
        raise VN2ResultError("result orders contain duplicate series/origin keys")
    if len(settlement_by_key) != len(result.settlements):
        raise VN2ResultError("result settlements contain duplicate series/period keys")
    expected_orders = {(key, origin) for origin in config.decision_origins for key in series}
    expected_settlements = {(key, period) for period in config.realized_periods for key in series}
    if set(order_by_key) != expected_orders:
        raise VN2ResultError("result order spine is incomplete")
    if set(settlement_by_key) != expected_settlements:
        raise VN2ResultError("result settlement spine is incomplete")
    orders = tuple(
        order_by_key[(key, origin)] for origin in config.decision_origins for key in series
    )
    settlements = tuple(
        settlement_by_key[(key, period)] for period in config.realized_periods for key in series
    )
    if any(row.session != result.session for row in (*orders, *settlements)):
        raise VN2ResultError("result ledger rows must share the result session")
    return identities, orders, settlements


def _project_r1(
    result: VN2RunResult,
    *,
    config: VN2ProtocolConfig,
    identities: Mapping[str, tuple[int, int]],
    orders: tuple[OrderRow, ...],
) -> list[dict[str, object]]:
    round_by_origin = dict(enumerate(config.decision_origins, start=1))
    inverse_round = {origin: number for number, origin in round_by_origin.items()}
    rows: list[dict[str, object]] = []
    for order in orders:
        series_key = order.series_key
        store, product = identities[series_key]
        rows.append(
            {
                "arrival_period": order.arrival_period.isoformat(),
                "model_name": order.model_name,
                "origin": order.origin.isoformat(),
                "product": product,
                "quantity": order.quantity,
                "round": inverse_round[order.origin],
                "schema": 1,
                "series_key": series_key,
                "store": store,
            }
        )
    del result
    return rows


def _project_r2(
    *,
    config: VN2ProtocolConfig,
    identities: Mapping[str, tuple[int, int]],
    settlements: tuple[SettlementRecord, ...],
) -> list[dict[str, object]]:
    index_by_period = dict(enumerate(config.realized_periods, start=1))
    inverse_index = {period: index for index, period in index_by_period.items()}
    rows: list[dict[str, object]] = []
    for record in settlements:
        series_key = record.series_key
        store, product = identities[series_key]
        transition = record.transition
        rows.append(
            {
                "actuals_semantics": record.actuals_semantics.value,
                "arrivals": record.arrivals,
                "closing_backorders": transition.closing_backorders,
                "demand": transition.demand,
                "end_inventory": transition.closing_on_hand,
                "holding_cost": record.holding.amount,
                "missed_sales": transition.unmet_demand,
                "on_order": record.inventory_position.on_order,
                "period": record.period.isoformat(),
                "period_index": inverse_index[record.period],
                "product": product,
                "sales": transition.fulfilled_demand,
                "schema": 1,
                "series_key": series_key,
                "shortage_cost": record.shortage.amount,
                "start_inventory": transition.available_inventory,
                "stockout_rule": transition.rule.value,
                "store": store,
            }
        )
    return rows


def _reduce_r3(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    holding = sum(_nonnegative(row["holding_cost"], name="holding cost") for row in rows)
    shortage = sum(_nonnegative(row["shortage_cost"], name="shortage cost") for row in rows)
    return {
        "holding_cost": holding,
        "schema": 1,
        "shortage_cost": shortage,
        "total_cost": holding + shortage,
    }


def _reduce_r4(
    rows: Sequence[Mapping[str, object]],
    *,
    config: VN2ProtocolConfig,
) -> dict[str, object]:
    by_period: dict[str, float] = {}
    for row in rows:
        period = cast(str, row["period"])
        by_period[period] = (
            by_period.get(period, 0.0)
            + _nonnegative(row["holding_cost"], name="holding cost")
            + _nonnegative(row["shortage_cost"], name="shortage cost")
        )
    cumulative = 0.0
    partials: dict[str, float] = {}
    for period in config.realized_periods:
        key = period.isoformat()
        cumulative += by_period[key]
        partials[key] = cumulative
    decision_rounds = [
        {
            "cumulative_cost": partials[origin.isoformat()],
            "origin": origin.isoformat(),
            "round": number,
        }
        for number, origin in enumerate(config.decision_origins, start=1)
    ]
    drain_periods = config.realized_periods[-config.drain_periods :]
    return {
        "decision_rounds": decision_rounds,
        "drain": {
            "cost": sum(by_period[period.isoformat()] for period in drain_periods),
            "periods": [period.isoformat() for period in drain_periods],
        },
        "schema": 1,
    }


def _validate_r1(
    rows: Sequence[dict[str, object]],
    *,
    config: VN2ProtocolConfig,
) -> dict[str, tuple[int, int]]:
    expected_count = config.series_count * config.round_count
    if len(rows) != expected_count:
        raise VN2ResultError(f"R1 must contain exactly {expected_count} rows")
    identities: dict[str, tuple[int, int]] = {}
    spine: list[tuple[int, str]] = []
    for row in rows:
        if set(row) != _R1_KEYS or row["schema"] != 1:
            raise VN2ResultError("R1 row has invalid keys or schema")
        round_number = _positive_int(row["round"], name="R1 round")
        if round_number > config.round_count:
            raise VN2ResultError("R1 round is outside the configured spine")
        if row["origin"] != config.decision_origins[round_number - 1].isoformat():
            raise VN2ResultError("R1 origin does not match round")
        series_key = _text(row["series_key"], name="R1 series key")
        identity = (
            _integer(row["store"], name="R1 store"),
            _integer(row["product"], name="R1 product"),
        )
        if identities.setdefault(series_key, identity) != identity:
            raise VN2ResultError("R1 series identity changes between rounds")
        _nonnegative(row["quantity"], name="R1 quantity")
        spine.append((round_number, series_key))
    series = tuple(sorted(identities, key=str.encode))
    expected_spine = [(round_number, key) for round_number in range(1, 7) for key in series]
    if spine != expected_spine or len(series) != config.series_count:
        raise VN2ResultError("R1 rows are not the canonical complete spine")
    return identities


def _validate_r2(
    rows: Sequence[dict[str, object]],
    *,
    config: VN2ProtocolConfig,
    identities: Mapping[str, tuple[int, int]],
) -> None:
    expected_count = config.series_count * len(config.realized_periods)
    if len(rows) != expected_count:
        raise VN2ResultError(f"R2 must contain exactly {expected_count} rows")
    spine: list[tuple[int, str]] = []
    for row in rows:
        if set(row) != _R2_KEYS or row["schema"] != 1:
            raise VN2ResultError("R2 row has invalid keys or schema")
        period_index = _positive_int(row["period_index"], name="R2 period index")
        if period_index > len(config.realized_periods):
            raise VN2ResultError("R2 period is outside the configured spine")
        if row["period"] != config.realized_periods[period_index - 1].isoformat():
            raise VN2ResultError("R2 period does not match period index")
        series_key = _text(row["series_key"], name="R2 series key")
        if (row["store"], row["product"]) != identities.get(series_key):
            raise VN2ResultError("R2 series identity does not match R1")
        if row["actuals_semantics"] != config.actuals_semantics.value:
            raise VN2ResultError("R2 actuals semantics do not match config")
        if row["stockout_rule"] != config.stockout_rule.value:
            raise VN2ResultError("R2 stockout rule does not match config")
        for name in (
            "arrivals",
            "closing_backorders",
            "demand",
            "end_inventory",
            "holding_cost",
            "missed_sales",
            "on_order",
            "sales",
            "shortage_cost",
            "start_inventory",
        ):
            _nonnegative(row[name], name=f"R2 {name}")
        spine.append((period_index, series_key))
    series = tuple(sorted(identities, key=str.encode))
    expected_spine = [
        (period_index, key)
        for period_index in range(1, len(config.realized_periods) + 1)
        for key in series
    ]
    if spine != expected_spine:
        raise VN2ResultError("R2 rows are not the canonical complete spine")


def _identity_digest(identities: Mapping[str, tuple[int, int]]) -> str:
    payload = [
        {"product": identities[key][1], "series_key": key, "store": identities[key][0]}
        for key in sorted(identities, key=str.encode)
    ]
    return _sha256(_json_bytes(payload))


def _json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise VN2ResultError("result values must be canonical JSON") from error


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _jsonl(path: Path, *, name: str) -> list[dict[str, object]]:
    payload = _read_regular(path, name=name)
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise VN2ResultError(f"{name} must be non-empty LF-terminated JSONL")
    rows = [_json_object(line + b"\n", name=name) for line in payload.splitlines()]
    if _jsonl_bytes(rows) != payload:
        raise VN2ResultError(f"{name} must use canonical JSONL bytes")
    return rows


def _json_object(payload: bytes, *, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=_unique_object)
    except (_DuplicateKey, UnicodeError, json.JSONDecodeError) as error:
        raise VN2ResultError(f"{name} is not unique-key UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise VN2ResultError(f"{name} must be a JSON object")
    return cast(dict[str, object], value)


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(key)
        result[key] = value
    return result


def _read_regular(path: Path, *, name: str) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise VN2ResultError(f"{name} must be a regular file")
        return path.read_bytes()
    except OSError as error:
        raise VN2ResultError(f"{name} is unreadable") from error


def _file_digest(path: Path, *, name: str) -> str:
    return _sha256(_read_regular(path, name=name))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise VN2ResultError(f"{name} must be a lowercase sha256 digest")
    return value


def _commit(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _COMMIT_SHA.fullmatch(value) is None:
        raise VN2ResultError(f"{name} must be a lowercase full commit SHA")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise VN2ResultError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VN2ResultError(f"{name} must be an integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _integer(value, name=name)
    if result <= 0:
        raise VN2ResultError(f"{name} must be positive")
    return result


def _nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VN2ResultError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise VN2ResultError(f"{name} must be finite and non-negative")
    return result


__all__ = [
    "PLATFORM",
    "VN2ResultBundle",
    "VN2ResultError",
    "VN2ResultManifest",
    "emit_result_bundle",
    "load_result_bundle",
]
