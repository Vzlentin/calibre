"""Test the compact deterministic VN2 R1-R4 result bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from tests.vn2_fixtures import BASE_WEEKS, synthetic_config_payload, write_config, write_dataset

from newcalibre.protocols.vn2 import (
    VN2ProtocolConfig,
    VN2ResultError,
    VN2RunResult,
    emit_result_bundle,
    load_result_bundle,
    load_vn2_config,
    load_vn2_dataset,
    run_vn2,
)

pytestmark = pytest.mark.tier1
CANDIDATE = "c" * 40
CAPTURE_DIGEST = "d" * 64


def _run(root: Path) -> tuple[VN2RunResult, VN2ProtocolConfig, Path, Path, Path]:
    data, inventory, config_path = write_dataset(root)
    payload = synthetic_config_payload()
    payload["model_config"]["m"] = len(BASE_WEEKS)  # type: ignore[index]
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


def test_manifest_digest_binds_exact_manifest_bytes(tmp_path: Path) -> None:
    """Expose the digest consumed by compact tracking records."""
    facts = _run(tmp_path / "fixture")
    bundle = _emit(tmp_path / "bundle", facts)

    assert (
        bundle.manifest_sha256
        == hashlib.sha256((bundle.root / "manifest.json").read_bytes()).hexdigest()
    )
