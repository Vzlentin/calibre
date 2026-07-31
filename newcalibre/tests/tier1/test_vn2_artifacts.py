"""Test the compact deterministic VN2 R1-R4 result bundle."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.vn2_fixtures import (
    BASE_WEEKS,
    REVEAL_WEEKS,
    SALES_FILES,
    calibrated_config_payload,
    refresh_inventory,
    synthetic_config_payload,
    write_config,
    write_dataset,
)

from newcalibre.domain import interval_columns
from newcalibre.protocols.vn2 import (
    PLATFORM,
    VN2ProtocolConfig,
    VN2ResultError,
    VN2RunResult,
    emit_result_bundle,
    load_result_bundle,
    load_vn2_config,
    load_vn2_dataset,
    render_advisory_result,
    run_vn2,
)

pytestmark = pytest.mark.tier1
CANDIDATE = "c" * 40
CAPTURE_DIGEST = "d" * 64


def _run(
    root: Path,
    *,
    round_count: int = 6,
) -> tuple[VN2RunResult, VN2ProtocolConfig, Path, Path, Path]:
    data, inventory, config_path = write_dataset(root)
    payload = synthetic_config_payload()
    payload["model_config"]["m"] = len(BASE_WEEKS)  # type: ignore[index]
    if round_count != 6:
        decision = payload["decision"]
        files = payload["files"]
        decision["round_count"] = round_count
        decision["origins"] = decision["origins"][:round_count]
        files["sales_reveals"] = files["sales_reveals"][: round_count + 3]
        in_stock_path = data / files["in_stock"]
        in_stock = pd.read_csv(in_stock_path)
        visible_columns = 2 + len(BASE_WEEKS) + round_count + 2
        in_stock.iloc[:, :visible_columns].to_csv(
            in_stock_path,
            index=False,
            lineterminator="\n",
        )
        refresh_inventory(data, inventory)
    write_config(config_path, payload)
    config = load_vn2_config(config_path)
    lock = root / "uv.lock"
    lock.write_bytes(b"synthetic lock\n")
    return run_vn2(load_vn2_dataset(data, inventory, config)), config, config_path, inventory, lock


def _emit(
    root: Path,
    facts: tuple[VN2RunResult, VN2ProtocolConfig, Path, Path, Path],
):
    result, config, config_path, inventory, lock = facts
    return emit_result_bundle(
        root,
        result=result,
        config=config,
        candidate_sha=CANDIDATE,
        config_path=config_path,
        input_inventory_path=inventory,
        lock_path=lock,
        capture_digest=CAPTURE_DIGEST,
    )


def _load(
    root: Path,
    facts: tuple[VN2RunResult, VN2ProtocolConfig, Path, Path, Path],
):
    _, _, config_path, inventory, lock = facts
    return load_result_bundle(
        root,
        expected_candidate_sha=CANDIDATE,
        config_path=config_path,
        input_inventory_path=inventory,
        lock_path=lock,
        expected_capture_digest=CAPTURE_DIGEST,
    )


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_double_emission_is_byte_identical_and_loadable(tmp_path: Path) -> None:
    """Emit deterministic bytes twice and load the same validated result."""
    facts = _run(tmp_path / "fixture")
    first = _emit(tmp_path / "first", facts)
    second = _emit(tmp_path / "second", facts)

    assert first.manifest == second.manifest
    for name in (*first.manifest.files, "manifest.json"):
        assert (first.root / name).read_bytes() == (second.root / name).read_bytes()
    assert _load(first.root, facts) == first


def test_non_six_round_bundle_is_loadable(tmp_path: Path) -> None:
    """Derive the R1 spine from configuration rather than Gate A constants."""
    facts = _run(tmp_path / "fixture", round_count=5)
    bundle = _emit(tmp_path / "bundle", facts)

    assert bundle.manifest.round_count == 5
    assert _load(bundle.root, facts) == bundle


def test_payload_corruption_is_rejected(tmp_path: Path) -> None:
    """Reject a changed payload before interpreting its rows."""
    facts = _run(tmp_path / "fixture")
    bundle = _emit(tmp_path / "bundle", facts)
    path = bundle.root / "r1-orders.jsonl"
    path.write_bytes(path.read_bytes().replace(b'"quantity":', b'"quantity":1', 1))

    with pytest.raises(VN2ResultError, match="digest"):
        _load(bundle.root, facts)


def test_manifest_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    """Refuse a result bound to a different candidate or capture."""
    facts = _run(tmp_path / "fixture")
    bundle = _emit(tmp_path / "bundle", facts)

    with pytest.raises(VN2ResultError, match="candidate"):
        load_result_bundle(
            bundle.root,
            expected_candidate_sha="e" * 40,
            config_path=facts[2],
            input_inventory_path=facts[3],
            lock_path=facts[4],
            expected_capture_digest=CAPTURE_DIGEST,
        )
    with pytest.raises(VN2ResultError, match="capture"):
        load_result_bundle(
            bundle.root,
            expected_candidate_sha=CANDIDATE,
            config_path=facts[2],
            input_inventory_path=facts[3],
            lock_path=facts[4],
            expected_capture_digest="f" * 64,
        )


def test_r1_and_r2_are_direct_ledger_projections(tmp_path: Path) -> None:
    """Project order and settlement rows without a parallel result source."""
    facts = _run(tmp_path / "fixture")
    result, config, *_ = facts
    bundle = _emit(tmp_path / "bundle", facts)
    r1 = _rows(bundle.root / "r1-orders.jsonl")
    r2 = _rows(bundle.root / "r2-cost-ledger.jsonl")

    first_order = result.orders[0]
    first_r1 = next(
        row
        for row in r1
        if row["series_key"] == first_order.series_key
        and row["origin"] == first_order.origin.isoformat()
    )
    assert first_r1["quantity"] == first_order.quantity
    assert first_r1["arrival_period"] == first_order.arrival_period.isoformat()

    first_settlement = result.settlements[0]
    first_r2 = next(
        row
        for row in r2
        if row["series_key"] == first_settlement.series_key
        and row["period"] == first_settlement.period.isoformat()
    )
    assert first_r2["holding_cost"] == first_settlement.holding.amount
    assert first_r2["shortage_cost"] == first_settlement.shortage.amount
    assert len(r1) == config.series_count * config.round_count
    assert len(r2) == config.series_count * len(config.realized_periods)


def test_r3_and_r4_are_reduced_from_r2_costs(tmp_path: Path) -> None:
    """Keep the final triple and trajectory as reduced settlement views."""
    facts = _run(tmp_path / "fixture")
    bundle = _emit(tmp_path / "bundle", facts)
    r2 = _rows(bundle.root / "r2-cost-ledger.jsonl")
    r3 = json.loads((bundle.root / "r3-final-triple.json").read_text(encoding="utf-8"))
    r4 = json.loads((bundle.root / "r4-cost-trajectory.json").read_text(encoding="utf-8"))

    holding = sum(float(row["holding_cost"]) for row in r2)
    shortage = sum(float(row["shortage_cost"]) for row in r2)
    assert r3 == {
        "holding_cost": holding,
        "schema": 1,
        "shortage_cost": shortage,
        "total_cost": holding + shortage,
    }
    assert [row["round"] for row in r4["decision_rounds"]] == list(range(1, 7))
    assert len(r4["drain"]["periods"]) == 2
    assert bundle.holding_cost + bundle.shortage_cost == bundle.total_cost


def test_advisory_projection_recomputes_only_calibrated_ordinary_ledger_facts(
    tmp_path: Path,
) -> None:
    data, inventory, config_path = write_dataset(tmp_path)
    for filename in SALES_FILES:
        path = data / filename
        frame = pd.read_csv(path)
        changed = False
        for week in REVEAL_WEEKS[-3:]:
            if week in frame:
                frame.loc[0, week] = 50.0
                changed = True
        if changed:
            frame.to_csv(path, index=False, lineterminator="\n")
    refresh_inventory(data, inventory)
    payload = calibrated_config_payload()
    payload["model_config"]["m"] = len(BASE_WEEKS)
    write_config(config_path, payload)
    config = load_vn2_config(config_path)
    result = run_vn2(load_vn2_dataset(data, inventory, config))
    lock = tmp_path / "uv.lock"
    lock.write_bytes(b"synthetic lock\n")

    rendered = render_advisory_result(
        result,
        config=config,
        config_path=config_path,
        input_inventory_path=inventory,
        lock_path=lock,
    )
    advisory = json.loads(rendered)

    coverage = config.cost_structure.critical_ratio
    lower, upper = interval_columns(coverage)
    outcomes = [
        outcome for outcome in result.coverage_report.outcomes if outcome.bound_key == (upper,)
    ]
    scored = [outcome for outcome in outcomes if outcome.scored]
    rows = {row.key: row for row in result.forecasts}
    widths = [
        float(rows[outcome.forecast_key].values[upper])
        - float(rows[outcome.forecast_key].values[lower])
        for outcome in scored
    ]
    assert rendered.endswith(b"\n") and b"\r" not in rendered
    assert advisory["status"] == "advisory"
    assert advisory["calibration"]["bound_key"] == [upper]
    assert advisory["calibration"]["counts"] == {
        "covered": sum(outcome.covered is True for outcome in outcomes),
        "pending": sum(not outcome.resolved for outcome in outcomes),
        "resolved": sum(outcome.resolved for outcome in outcomes),
        "scored": len(scored),
        "total": len(outcomes),
        "unscored": sum(outcome.resolved and not outcome.scored for outcome in outcomes),
    }
    direct_covered = []
    for outcome in scored:
        row = rows[outcome.forecast_key]
        window = [
            candidate
            for candidate in result.forecasts
            if candidate.series_key == row.series_key
            and candidate.origin == row.origin
            and candidate.model_name == row.model_name
            and candidate.horizon_step <= config.timing.protection_period
        ]
        actual_sum = math.fsum(float(candidate.actual_value) for candidate in window)
        covered = actual_sum <= float(row.values[upper])
        direct_covered.append(covered)
        assert outcome.covered is covered
    assert advisory["calibration"]["realized_coverage"] == (
        sum(direct_covered) / len(direct_covered)
    )
    assert advisory["calibration"]["widths"] == {
        "method": "linear",
        "quantiles": {
            str(level): float(np.quantile(widths, level, method="linear"))
            for level in (0, 0.25, 0.5, 0.75, 1)
        },
    }
    assert all(math.isfinite(value) and value >= 0.0 for value in widths)
    assert all(float(rows[outcome.forecast_key].values[lower]) == 0.0 for outcome in scored)

    holding = math.fsum(record.holding.amount for record in result.settlements)
    shortage = math.fsum(record.shortage.amount for record in result.settlements)
    assert advisory["costs"] == {
        "currency": config.currency,
        "holding": holding,
        "shortage": shortage,
        "total": math.fsum((holding, shortage)),
    }
    assert advisory["identity"] == {
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "input_inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "platform": PLATFORM,
        "session_id": result.session.value,
    }
    assert advisory["semantics"] == {
        "actuals": "censored_sales_surrogate",
        "censoring_contract": "recorded-sales-under-censored-sales-surrogate",
        "scored_series": "recorded-sales",
    }
    assert advisory["clamps"] == {
        "configured": {"upper_cap": None, "upper_floor": None},
        "issued": [],
    }

    truncated = replace(result, settlements=result.settlements[:-1])
    with pytest.raises(VN2ResultError, match="settlement spine is incomplete"):
        render_advisory_result(
            truncated,
            config=config,
            config_path=config_path,
            input_inventory_path=inventory,
            lock_path=lock,
        )

    mismatched_inventory = tmp_path / "mismatched-inventory.json"
    mismatched_inventory.write_bytes(inventory.read_bytes() + b" ")
    with pytest.raises(VN2ResultError, match="does not match the VN2 run input inventory"):
        render_advisory_result(
            result,
            config=config,
            config_path=config_path,
            input_inventory_path=mismatched_inventory,
            lock_path=lock,
        )


def test_manifest_digest_binds_exact_manifest_bytes(tmp_path: Path) -> None:
    """Expose the digest consumed by compact tracking records."""
    facts = _run(tmp_path / "fixture")
    bundle = _emit(tmp_path / "bundle", facts)

    assert (
        bundle.manifest_sha256
        == hashlib.sha256((bundle.root / "manifest.json").read_bytes()).hexdigest()
    )
