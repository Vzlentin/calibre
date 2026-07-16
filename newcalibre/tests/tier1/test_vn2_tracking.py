"""Test compact append-only VN2 regression tracking."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.vn2_fixtures import BASE_WEEKS, synthetic_config_payload, write_config, write_dataset

from newcalibre.protocols.vn2 import (
    emit_result_bundle,
    load_vn2_config,
    load_vn2_dataset,
    run_vn2,
)
from newcalibre.protocols.vn2.tracking import (
    TrackingError,
    VN2TrackingRecord,
    build_tracking_record,
    compare_tracking_records,
    load_tracking_history,
    validate_tracking_append,
)

pytestmark = pytest.mark.tier1


def _record(**changes: object) -> VN2TrackingRecord:
    values: dict[str, object] = {
        "candidate_sha": "a" * 40,
        "config_digest": "b" * 64,
        "input_inventory_digest": "c" * 64,
        "capture_digest": "d" * 64,
        "lock_digest": "e" * 64,
        "platform": "ubuntu-24.04/x86_64",
        "actuals_semantics": "censored_sales_surrogate",
        "result_manifest_digest": "f" * 64,
        "holding_cost": 3.0,
        "shortage_cost": 4.0,
        "total_cost": 7.0,
    }
    values.update(changes)
    return VN2TrackingRecord(**values)  # type: ignore[arg-type]


def _bundle(tmp_path: Path):
    data, inventory, config_path = write_dataset(tmp_path / "fixture")
    payload = synthetic_config_payload()
    payload["model_config"]["m"] = len(BASE_WEEKS)  # type: ignore[index]
    write_config(config_path, payload)
    config = load_vn2_config(config_path)
    lock = tmp_path / "fixture" / "uv.lock"
    lock.write_bytes(b"synthetic lock\n")
    result = run_vn2(load_vn2_dataset(data, inventory, config))
    bundle = emit_result_bundle(
        tmp_path / "bundle",
        result=result,
        config=config,
        candidate_sha="1" * 40,
        config_path=config_path,
        input_inventory_path=inventory,
        lock_path=lock,
        capture_digest="2" * 64,
    )
    return bundle


def test_migrated_history_shape_is_readable_and_preserves_gate_a_result() -> None:
    """Read record one after its clean migration to the compact schema."""
    migrated = _record(
        candidate_sha="860ccdbfcc2f6b0b30d4b31e5072e73bd88feeb2",
        config_digest="979b6d39e0d0e1c52a91995c8d2fdf6de7430e0d202a442c354de2678033a370",
        input_inventory_digest="54f8556a811eac81c9597c0fb0d2ef16dca5a6d84936188b7322fcdb8f15ed97",
        capture_digest="9c69bef8922df65ef25e6e54212c6e7c813b5acf1c3abff90f3b0ee746c8f6d4",
        lock_digest="78a0e2d123d110639bfeae5d8e13d0f054b68621627f1a00095ec2efab9a0719",
        result_manifest_digest=("4996915e1c6b3632b018eb8ec4aaca448ed3f60e3635c31000b85c170df7762c"),
        holding_cost=3643.6000000000004,
        shortage_cost=2764.0,
        total_cost=6407.6,
    )
    (record,) = load_tracking_history(migrated.to_bytes())

    assert record == migrated


def test_build_tracking_record_reduces_validated_bundle(tmp_path: Path) -> None:
    """Build only the compact identities and final cost triple."""
    bundle = _bundle(tmp_path)
    record = build_tracking_record(bundle)

    assert record.candidate_sha == bundle.manifest.candidate_sha
    assert record.result_manifest_digest == bundle.manifest_sha256
    assert record.total_cost == record.holding_cost + record.shortage_cost


def test_comparison_refuses_every_comparability_mismatch() -> None:
    """Refuse cost comparison when any protocol identity differs."""
    baseline = _record()
    changes = {
        "config_digest": "0" * 64,
        "input_inventory_digest": "1" * 64,
        "capture_digest": "2" * 64,
        "lock_digest": "3" * 64,
        "platform": "other/x86_64",
        "actuals_semantics": "demand",
    }
    for field, value in changes.items():
        with pytest.raises(TrackingError, match="comparability"):
            compare_tracking_records(baseline, _record(**{field: value}))


def test_matching_comparison_reports_synthetic_cost_jump() -> None:
    """Return signed component deltas for comparable records."""
    comparison = compare_tracking_records(
        _record(),
        _record(
            candidate_sha="9" * 40,
            result_manifest_digest="8" * 64,
            holding_cost=4.5,
            shortage_cost=5.0,
            total_cost=9.5,
        ),
    )

    assert comparison.holding_delta == 1.5
    assert comparison.shortage_delta == 1.0
    assert comparison.total_delta == 2.5


def test_validate_append_requires_exact_prefix_and_order(tmp_path: Path) -> None:
    """Accept an exact append and reject mutation or reordering of history."""
    first = _record()
    second = _record(candidate_sha="9" * 40, result_manifest_digest="8" * 64)
    base = tmp_path / "base.jsonl"
    head = tmp_path / "head.jsonl"
    base.write_bytes(first.to_bytes())
    head.write_bytes(first.to_bytes() + second.to_bytes())

    assert validate_tracking_append(base, head) == (second,)

    head.write_bytes(second.to_bytes() + first.to_bytes())
    with pytest.raises(TrackingError, match="exact append"):
        validate_tracking_append(base, head)


def test_tracking_record_requires_exact_total() -> None:
    """Require total cost to be the exact sum of its components."""
    with pytest.raises(TrackingError, match="holding plus shortage"):
        _record(total_cost=8.0)
