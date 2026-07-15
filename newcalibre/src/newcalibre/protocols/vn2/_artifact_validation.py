"""Validate VN2 bundle integrity and reconstruct its R1-R4 domain facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import pandas as pd

from newcalibre.domain import (
    ActualsSemantics,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeDescriptorError,
    GuaranteeType,
    InventoryPosition,
    ScoredSeries,
    SessionIdentity,
    StockoutRule,
)
from newcalibre.ledger import (
    BookedCost,
    LedgerError,
    SettlementRecord,
    StockoutTransition,
    validate_lost_sales_transition,
)
from newcalibre.ordering import CostComponents, SettlementObjective, settle_path_cost
from newcalibre.protocols.vn2._artifact_contracts import (
    _ALL_PATHS,
    _BINDING_KEYS,
    _DESCRIPTOR_KEYS,
    _GUARANTEE_TYPE_KEYS,
    _MANIFEST_KEYS,
    _PAYLOAD_PATHS,
    _R1_KEYS,
    _R2_KEYS,
    _R3_KEYS,
    _R4_KEYS,
    _SCOPE_KEYS,
    CONFIG_PATH,
    INPUT_INVENTORY_PATH,
    LOCK_PATH,
    RESULT_KIND,
    VN2ResultBundle,
    VN2ResultError,
    VN2ResultManifest,
    _derive_session,
    _digest_json,
    _environment_value,
    _finite_nonnegative,
    _finite_number,
    _integer,
    _load_json_object,
    _load_jsonl,
    _object,
    _parse_environment,
    _parse_files,
    _positive_integer,
    _provenance_value,
    _r3_value,
    _r4_value,
    _require_commit_sha,
    _require_exact_keys,
    _require_expected,
    _require_int,
    _require_run_id,
    _require_semantics,
    _require_sha256,
    _require_text,
    _require_trusted_digest,
    _RunIdentity,
    _series_digest,
    _sha256,
    _timestamp,
    _TrustedInputs,
    _validate_run_url,
)
from newcalibre.protocols.vn2.config import VN2ProtocolConfig, load_vn2_config


def _validate_lost_sales_sequence(
    records: Sequence[SettlementRecord],
    *,
    name: str,
) -> None:
    previous_positions: dict[str, InventoryPosition] = {}
    for index, record in enumerate(records):
        try:
            validate_lost_sales_transition(
                transition=record.transition,
                arrivals=record.arrivals,
                opening=previous_positions.get(record.series_key),
            )
        except LedgerError as error:
            raise VN2ResultError(
                f"{name}[{index}] violates the lost-sales transition contract"
            ) from error
        previous_positions[record.series_key] = record.inventory_position


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
    identity = _RunIdentity(
        candidate_sha=candidate_sha,
        workflow_sha=workflow_sha,
        run_id=run_id,
        run_url=run_url,
    )
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
    trusted = _TrustedInputs(
        config_digest=config_digest,
        input_inventory_digest=input_digest,
        lock_digest=lock_digest,
    )
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
    provenance_value = _provenance_value(
        config=config,
        artifact_name=artifact_name,
        identity=identity,
        trusted=trusted,
        environment_digest=environment_digest,
        series_identity_digest=series_identity_digest,
        session_id=session_id,
    )
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
        cost = _validate_r3_payload(
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
        cost=cost,
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
            descriptor = _validate_descriptor(
                row[descriptor_name],
                name=f"{name}.{descriptor_name}",
            )
            if descriptor.type.claim is not GuaranteeClaim.NONE:
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
    canonical_series_keys = tuple(sorted(identities, key=str.encode))
    expected_spine = [
        (round_number, series_key)
        for round_number in range(1, config.round_count + 1)
        for series_key in canonical_series_keys
    ]
    if spine != expected_spine or len(identities) != config.series_count:
        raise VN2ResultError("R1 orders do not use the exact canonical Cartesian spine")
    if len(set(identities.values())) != len(identities):
        raise VN2ResultError("R1 series identities must be unique")
    return identities, _derive_session(config, canonical_series_keys), order_arrivals


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
        if holding.rate != config.holding_rate or shortage.rate != config.shortage_rate:
            raise VN2ResultError(f"{name} cost rates must match the configured cost structure")
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
    _validate_lost_sales_sequence(records, name="R2 cost ledger")
    return tuple(records)


def _validate_r3_payload(
    path: Path,
    *,
    objective: SettlementObjective,
    semantics: ActualsSemantics,
    provenance_digest: str,
) -> CostComponents:
    value, _ = _load_json_object(path, name="R3 final triple")
    _require_exact_keys(value, _R3_KEYS, name="R3 final triple")
    expected = _r3_value(
        objective,
        semantics=semantics,
        provenance_digest=provenance_digest,
    )
    if value != expected:
        raise VN2ResultError("R3 final triple does not match the generic settlement reducer")
    return CostComponents(objective.holding, objective.shortage)


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


def _validate_descriptor(value: object, *, name: str) -> GuaranteeDescriptor:
    descriptor = _object(value, name=name)
    _require_exact_keys(descriptor, _DESCRIPTOR_KEYS, name=name)
    guarantee_type = _object(descriptor["type"], name=f"{name}.type")
    _require_exact_keys(guarantee_type, _GUARANTEE_TYPE_KEYS, name=f"{name}.type")
    claim_value = _require_text(guarantee_type["claim"], name=f"{name}.type.claim")
    currency_value = guarantee_type["currency"]
    if currency_value is not None:
        currency_value = _require_text(currency_value, name=f"{name}.type.currency")
    declared_slack = guarantee_type["declared_slack"]
    if declared_slack is not None:
        declared_slack = _finite_number(
            declared_slack,
            name=f"{name}.type.declared_slack",
        )
    level = _finite_number(descriptor["level"], name=f"{name}.level")
    scored_series_value = _require_text(
        descriptor["scored_series"],
        name=f"{name}.scored_series",
    )
    window_value = _require_text(descriptor["window"], name=f"{name}.window")
    scope = _object(descriptor["scope"], name=f"{name}.scope")
    _require_exact_keys(scope, _SCOPE_KEYS, name=f"{name}.scope")
    scope_kind_value = _require_text(scope["kind"], name=f"{name}.scope.kind")
    class_system_name = scope["class_system_name"]
    if class_system_name is not None:
        class_system_name = _require_text(
            class_system_name,
            name=f"{name}.scope.class_system_name",
        )

    try:
        claim = GuaranteeClaim(claim_value)
        currency = GuaranteeCurrency(currency_value) if currency_value is not None else None
        scored_series = ScoredSeries(scored_series_value)
        window = EmissionScope(window_value)
        scope_kind = DecisionScopeKind(scope_kind_value)
    except ValueError as error:
        raise VN2ResultError(f"{name} contains unknown descriptor vocabulary") from error

    try:
        return GuaranteeDescriptor(
            type=GuaranteeType(
                claim=claim,
                currency=currency,
                declared_slack=declared_slack,
            ),
            level=level,
            scored_series=scored_series,
            window=window,
            scope=DecisionScope(
                kind=scope_kind,
                class_system_name=class_system_name,
            ),
        )
    except GuaranteeDescriptorError as error:
        raise VN2ResultError(f"{name} violates the guarantee descriptor contract") from error
