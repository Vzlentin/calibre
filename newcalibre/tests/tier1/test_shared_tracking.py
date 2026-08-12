"""Exercise shared compact tracking across protocol record families."""

from __future__ import annotations

import pytest

from newcalibre.tracking import (
    M5TrackingRecord,
    TrackingError,
    VN2TrackingRecord,
    load_tracking_history,
    validate_tracking_append,
)

pytestmark = pytest.mark.tier1


def _record(**changes: object) -> M5TrackingRecord:
    values: dict[str, object] = {
        "candidate_sha": "a" * 40,
        "config_digest": "b" * 64,
        "input_inventory_digest": "c" * 64,
        "lock_digest": "d" * 64,
        "coverage_summary_digest": "e" * 64,
        "coverage_by_node_digest": "f" * 64,
        "report_digest": "1" * 64,
        "profile_digest": "2" * 64,
        "environment_digest": "3" * 64,
        "disposition": "GO",
    }
    values.update(changes)
    return M5TrackingRecord(**values)  # type: ignore[arg-type]


def test_m5_record_round_trips_as_canonical_discriminated_jsonl() -> None:
    """Round-trip all five file identities and the recomputed disposition."""
    record = _record()

    assert load_tracking_history(record.to_bytes()) == (record,)
    assert b'"record_kind":"m5-performance"' in record.to_bytes()


def test_shared_history_preserves_existing_vn2_bytes_as_exact_prefix() -> None:
    """Append M5 after canonical VN2 bytes without rewriting any prefix byte."""
    vn2 = VN2TrackingRecord(
        candidate_sha="a" * 40,
        config_digest="b" * 64,
        input_inventory_digest="c" * 64,
        capture_digest="d" * 64,
        lock_digest="e" * 64,
        platform="linux/x86_64",
        actuals_semantics="censored_sales_surrogate",
        result_manifest_digest="f" * 64,
        holding_cost=1.5,
        shortage_cost=2.25,
        total_cost=3.75,
    )
    base = vn2.to_bytes()
    record = _record()

    appended = validate_tracking_append(base, base + record.to_bytes())

    assert appended == (record,)


@pytest.mark.parametrize("disposition", ["go", "PASS", "", "NO GO"])
def test_m5_record_refuses_unregistered_dispositions(disposition: str) -> None:
    """Accept only the two Gate C dispositions recomputed by validators."""
    with pytest.raises(TrackingError, match="disposition"):
        _record(disposition=disposition)
