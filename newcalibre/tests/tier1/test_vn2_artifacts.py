"""Prove the deterministic VN2 R1-R4 evidence bundle contract."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd
import pytest
from tests.vn2_fixtures import (
    BASE_WEEKS,
    synthetic_config_payload,
    write_config,
    write_dataset,
)

from newcalibre.domain import (
    ActualsSemantics,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeType,
    SessionIdentity,
)
from newcalibre.ordering import settle_path_cost
from newcalibre.protocols.vn2 import (
    THREAD_VARIABLES,
    VN2EvidenceEnvironment,
    VN2ProtocolConfig,
    VN2ResultError,
    VN2RunResult,
    emit_vn2_result_bundle,
    load_vn2_config,
    load_vn2_dataset,
    run_vn2,
    validate_vn2_result_bundle,
)

pytestmark = pytest.mark.tier1

CANDIDATE_SHA = "c" * 40
WORKFLOW_SHA = "d" * 40
RUN_ID = "123456"
RUN_URL = f"https://github.com/Vzlentin/calibre/actions/runs/{RUN_ID}"


def _environment() -> VN2EvidenceEnvironment:
    return VN2EvidenceEnvironment(
        arch="x86_64",
        cpu_model="Synthetic x86_64",
        os_id="ubuntu",
        os_version_id="24.04",
        os_pretty_name="Ubuntu 24.04.2 LTS",
        python="3.12.10",
        numpy="2.3.1",
        numpy_config="OpenBLAS synthetic provenance",
        runner_image="ubuntu24/20250701.1",
        thread_policy={name: "1" for name in THREAD_VARIABLES},
    )


def _run(
    root: Path,
) -> tuple[VN2RunResult, VN2ProtocolConfig, Path, Path, Path]:
    data, inventory_path, config_path = write_dataset(root)
    payload = synthetic_config_payload()
    payload["model_config"]["m"] = len(BASE_WEEKS)  # type: ignore[index]
    write_config(config_path, payload)
    config = load_vn2_config(config_path)
    dataset = load_vn2_dataset(data, inventory_path, config)
    lock_path = root / "uv.lock"
    lock_path.write_bytes(b"synthetic locked environment\n")
    return run_vn2(dataset), config, config_path, inventory_path, lock_path


def _emit(
    root: Path,
    *,
    result: VN2RunResult,
    config: VN2ProtocolConfig,
    config_path: Path,
    inventory_path: Path,
    lock_path: Path,
):
    return emit_vn2_result_bundle(
        root,
        result=result,
        config=config,
        candidate_sha=CANDIDATE_SHA,
        workflow_sha=WORKFLOW_SHA,
        run_id=RUN_ID,
        run_url=RUN_URL,
        config_path=config_path,
        input_inventory_path=inventory_path,
        lock_path=lock_path,
        environment=_environment(),
    )


def _validate(root: Path, *, config_path: Path, inventory_path: Path, lock_path: Path):
    return validate_vn2_result_bundle(
        root,
        expected_candidate_sha=CANDIDATE_SHA,
        expected_workflow_sha=WORKFLOW_SHA,
        expected_run_id=RUN_ID,
        expected_config_path=config_path,
        expected_input_inventory_path=inventory_path,
        expected_lock_path=lock_path,
    )


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _rebind_outer_digests(root: Path) -> None:
    manifest = _json(root / "manifest.json")
    files = manifest["files"]
    assert isinstance(files, list)
    rebound = []
    for raw in files:
        assert isinstance(raw, dict)
        path = root / str(raw["path"])
        payload = path.read_bytes()
        rebound.append(
            {
                "bytes": len(payload),
                "path": raw["path"],
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    manifest["files"] = rebound
    listing = "".join(f"{entry['sha256']}  {entry['path']}\n" for entry in rebound).encode("utf-8")
    (root / "files.sha256").write_bytes(listing)
    manifest["inner_bundle_digest"] = hashlib.sha256(listing).hexdigest()
    (root / "manifest.json").write_bytes(_canonical_json(manifest))


def _foreign_session(
    result: VN2RunResult,
    config: VN2ProtocolConfig,
) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="foreign-vn2",
        series_keys=result.series_identities,
        calendar=config.calendar,
        horizon=config.task_horizon,
        model_config=config.model_config,
        conformal_config=config.conformal_config,
        ordering_policy=config.ordering_policy,
        decision_series_keys=result.series_identities,
        cost_structure=config.cost_structure,
        decision_timing=config.timing,
        stockout_rule=config.stockout_rule,
    )


def test_bundle_projects_exact_r1_r4_spines_semantics_and_math(tmp_path: Path) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")
    root = tmp_path / "bundle"

    bundle = _emit(
        root,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    validated = _validate(
        root,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )

    assert validated == bundle
    assert bundle.manifest.config_path == "benchmarks/vn2/protocol.yaml"
    assert bundle.manifest.input_inventory_path == ("benchmarks/vn2/vn2-input-digests.json")
    assert bundle.manifest.lock_path == "uv.lock"
    r1 = _jsonl(root / "r1-orders.jsonl")
    r2 = _jsonl(root / "r2-cost-ledger.jsonl")
    r3 = _json(root / "r3-final-triple.json")
    r4 = _json(root / "r4-cost-trajectory.json")
    manifest = _json(root / "manifest.json")
    series_keys = tuple(sorted(result.series_identities, key=str.encode))

    assert len(r1) == len(series_keys) * config.round_count
    assert {(row["series_key"], row["round"]) for row in r1} == set(
        product(series_keys, range(1, config.round_count + 1))
    )
    assert [(row["round"], row["series_key"]) for row in r1] == sorted(
        ((row["round"], row["series_key"]) for row in r1),
        key=lambda value: (value[0], value[1].encode()),
    )
    assert all(row["quantity"] >= 0.0 for row in r1)
    assert {row["consumed_claim"] for row in r1} == {GuaranteeClaim.NONE.value}

    periods = tuple(period.isoformat() for period in config.realized_periods)
    assert len(r2) == len(series_keys) * len(periods)
    assert {(row["series_key"], row["period"]) for row in r2} == set(product(series_keys, periods))
    first_record = next(
        record
        for record in result.settlements
        if record.series_key == "0_126" and record.period == config.realized_periods[0]
    )
    first_row = next(
        row for row in r2 if row["series_key"] == "0_126" and row["period"] == periods[0]
    )
    assert first_row["start_inventory"] == first_record.transition.available_inventory
    assert first_row["arrivals"] == first_record.arrivals
    assert first_row["demand"] == first_record.transition.demand
    assert first_row["sales"] == first_record.transition.fulfilled_demand
    assert first_row["missed_sales"] == first_record.transition.unmet_demand
    assert first_row["end_inventory"] == first_record.transition.closing_on_hand
    assert first_row["holding_cost"] == first_record.holding.amount
    assert first_row["shortage_cost"] == first_record.shortage.amount

    objective = settle_path_cost(
        result.settlements,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    assert r3["holding_total"] == objective.holding.value
    assert r3["shortage_total"] == objective.shortage.value
    assert r3["total_cost"] == objective.total.value
    assert r3["total_cost"] == r3["holding_total"] + r3["shortage_total"]

    partial_by_origin = {partial.origin: partial.cost.value for partial in objective.partials}
    decision_rows = r4["decision_rounds"]
    assert isinstance(decision_rows, list)
    assert [row["round"] for row in decision_rows] == list(range(1, config.round_count + 1))
    assert [row["cumulative_cost"] for row in decision_rows] == [
        partial_by_origin[origin] for origin in config.decision_origins
    ]
    drain_periods = config.realized_periods[-config.drain_periods :]
    drain_objective = settle_path_cost(
        (record for record in result.settlements if record.period in drain_periods),
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
    )
    drain = r4["drain_remainder"]
    assert isinstance(drain, dict)
    assert drain["periods"] == [period.isoformat() for period in drain_periods]
    assert drain["cost"] == drain_objective.total.value

    provenance = bundle.manifest.provenance_digest
    assert all(
        row["actuals_semantics"] == "censored_sales_surrogate"
        and row["provenance_digest"] == provenance
        for row in r2
    )
    assert r3["actuals_semantics"] == "censored_sales_surrogate"
    assert r3["provenance_digest"] == provenance
    assert all(
        row["actuals_semantics"] == "censored_sales_surrogate"
        and row["provenance_digest"] == provenance
        for row in decision_rows
    )
    assert drain["actuals_semantics"] == "censored_sales_surrogate"
    assert drain["provenance_digest"] == provenance
    expected_provenance = {
        "actuals_semantics": "censored_sales_surrogate",
        "artifact_kind": "vn2-gate-a-results",
        "artifact_name": f"vn2-acceptance-{CANDIDATE_SHA}",
        "candidate_sha": CANDIDATE_SHA,
        "config_digest": manifest["config_digest"],
        "environment_digest": manifest["environment_digest"],
        "input_inventory_digest": manifest["input_inventory_digest"],
        "lock_digest": manifest["lock_digest"],
        "realized_periods": [period.isoformat() for period in config.realized_periods],
        "run_id": RUN_ID,
        "run_url": RUN_URL,
        "series_identity_digest": manifest["series_identity_digest"],
        "session_id": result.session.value,
        "workflow_sha": WORKFLOW_SHA,
    }
    expected_provenance_bytes = json.dumps(
        expected_provenance,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert provenance == hashlib.sha256(expected_provenance_bytes).hexdigest()


def test_bundle_is_byte_identical_for_shuffled_engine_facts(tmp_path: Path) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")
    shuffled = replace(
        result,
        orders=tuple(reversed(result.orders)),
        settlements=tuple(reversed(result.settlements)),
    )
    left = tmp_path / "left"
    right = tmp_path / "right"

    _emit(
        left,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    _emit(
        right,
        result=shuffled,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )

    assert {
        path.relative_to(left).as_posix(): path.read_bytes()
        for path in left.rglob("*")
        if path.is_file()
    } == {
        path.relative_to(right).as_posix(): path.read_bytes()
        for path in right.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "missing-evidence", "claim"])
def test_bundle_refuses_incomplete_or_unproven_order_spines_before_writing(
    tmp_path: Path,
    mutation: str,
) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")
    orders = result.orders
    if mutation == "missing":
        orders = orders[:-1]
    elif mutation == "duplicate":
        orders = (*orders, orders[0])
    elif mutation == "missing-evidence":
        orders = (replace(orders[0], evidence=None), *orders[1:])
    else:
        evidence = orders[0].evidence
        assert evidence is not None
        wrong_type = GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        )
        orders = (
            replace(
                orders[0],
                evidence=replace(
                    evidence,
                    effective_descriptor=replace(evidence.effective_descriptor, type=wrong_type),
                ),
            ),
            *orders[1:],
        )
    invalid = replace(result, orders=orders)
    root = tmp_path / "bundle"

    with pytest.raises(VN2ResultError):
        _emit(
            root,
            result=invalid,
            config=config,
            config_path=config_path,
            inventory_path=inventory_path,
            lock_path=lock_path,
        )

    assert not root.exists()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "foreign-period", "semantics"])
def test_bundle_refuses_invalid_settlement_spines_before_writing(
    tmp_path: Path,
    mutation: str,
) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")
    settlements = result.settlements
    if mutation == "missing":
        settlements = settlements[:-1]
    elif mutation == "duplicate":
        settlements = (*settlements, settlements[0])
    elif mutation == "foreign-period":
        settlements = (
            replace(settlements[0], period=pd.Timestamp("2030-01-07")),
            *settlements[1:],
        )
    else:
        settlements = (
            replace(settlements[0], actuals_semantics=ActualsSemantics.DEMAND),
            *settlements[1:],
        )
    invalid = replace(result, settlements=settlements)
    root = tmp_path / "bundle"

    with pytest.raises(VN2ResultError):
        _emit(
            root,
            result=invalid,
            config=config,
            config_path=config_path,
            inventory_path=inventory_path,
            lock_path=lock_path,
        )

    assert not root.exists()


def test_bundle_refuses_foreign_session_and_identity_mapping_before_writing(
    tmp_path: Path,
) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")
    foreign = _foreign_session(result, config)
    invalid_session = replace(
        result,
        orders=(replace(result.orders[0], session=foreign), *result.orders[1:]),
    )
    invalid_identity = replace(
        result,
        series_identities={**result.series_identities, "foreign": (9, 9)},
    )

    for index, invalid in enumerate((invalid_session, invalid_identity)):
        root = tmp_path / f"bundle-{index}"
        with pytest.raises(VN2ResultError):
            _emit(
                root,
                result=invalid,
                config=config,
                config_path=config_path,
                inventory_path=inventory_path,
                lock_path=lock_path,
            )
        assert not root.exists()


def test_validator_refuses_tamper_extra_symlink_duplicate_keys_and_wrong_identity(
    tmp_path: Path,
) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")

    tampered = tmp_path / "tampered"
    _emit(
        tampered,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    (tampered / "r1-orders.jsonl").write_bytes(b"{}\n")
    with pytest.raises(VN2ResultError, match="digest|size|listing"):
        _validate(
            tampered,
            config_path=config_path,
            inventory_path=inventory_path,
            lock_path=lock_path,
        )

    extra = tmp_path / "extra"
    _emit(
        extra,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    (extra / "r5-coverage.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(VN2ResultError, match="file set"):
        _validate(
            extra,
            config_path=config_path,
            inventory_path=inventory_path,
            lock_path=lock_path,
        )

    duplicate = tmp_path / "duplicate"
    _emit(
        duplicate,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    manifest = (duplicate / "manifest.json").read_text(encoding="utf-8")
    (duplicate / "manifest.json").write_text(
        manifest.replace(
            '{"actuals_semantics"',
            '{"schema":1,"schema":1,"actuals_semantics"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(VN2ResultError, match="duplicate JSON key"):
        _validate(
            duplicate,
            config_path=config_path,
            inventory_path=inventory_path,
            lock_path=lock_path,
        )

    symlink = tmp_path / "symlink"
    _emit(
        symlink,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    try:
        os.symlink(symlink / "manifest.json", symlink / "linked-manifest.json")
    except OSError:
        pass
    else:
        with pytest.raises(VN2ResultError, match="symbolic link"):
            _validate(
                symlink,
                config_path=config_path,
                inventory_path=inventory_path,
                lock_path=lock_path,
            )

    valid = tmp_path / "identity"
    _emit(
        valid,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    with pytest.raises(VN2ResultError, match="candidate_sha"):
        validate_vn2_result_bundle(
            valid,
            expected_candidate_sha="e" * 40,
            expected_workflow_sha=WORKFLOW_SHA,
            expected_run_id=RUN_ID,
            expected_config_path=config_path,
            expected_input_inventory_path=inventory_path,
            expected_lock_path=lock_path,
        )


def test_validator_refuses_internally_rebound_wrong_provenance(tmp_path: Path) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")
    root = tmp_path / "bundle"
    _emit(
        root,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    rows = _jsonl(root / "r2-cost-ledger.jsonl")
    rows[0]["provenance_digest"] = "f" * 64
    (root / "r2-cost-ledger.jsonl").write_bytes(b"".join(_canonical_json(row) for row in rows))
    _rebind_outer_digests(root)

    with pytest.raises(VN2ResultError, match="provenance"):
        _validate(
            root,
            config_path=config_path,
            inventory_path=inventory_path,
            lock_path=lock_path,
        )


def test_validator_refuses_internally_rebound_r1_r2_mismatch(tmp_path: Path) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")
    root = tmp_path / "bundle"
    _emit(
        root,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )
    rows = _jsonl(root / "r1-orders.jsonl")
    rows[0]["quantity"] = float(rows[0]["quantity"]) + 1.0
    (root / "r1-orders.jsonl").write_bytes(b"".join(_canonical_json(row) for row in rows))
    _rebind_outer_digests(root)

    with pytest.raises(VN2ResultError, match="arrival"):
        _validate(
            root,
            config_path=config_path,
            inventory_path=inventory_path,
            lock_path=lock_path,
        )


def test_bundle_has_no_r5_surface_and_never_mutates_tracking(tmp_path: Path) -> None:
    result, config, config_path, inventory_path, lock_path = _run(tmp_path / "source")
    tracking = tmp_path / "stage3" / "evidence" / "tracking" / "series.jsonl"
    tracking.parent.mkdir(parents=True)
    tracking.write_bytes(b'{"existing":true}\n')
    before = tracking.read_bytes()
    root = tmp_path / "bundle"

    _emit(
        root,
        result=result,
        config=config,
        config_path=config_path,
        inventory_path=inventory_path,
        lock_path=lock_path,
    )

    assert tracking.read_bytes() == before
    assert {path.name for path in root.iterdir()} == {
        "environment.json",
        "files.sha256",
        "manifest.json",
        "r1-orders.jsonl",
        "r2-cost-ledger.jsonl",
        "r3-final-triple.json",
        "r4-cost-trajectory.json",
    }
    assert all("r5" not in path.name.lower() for path in root.rglob("*"))
