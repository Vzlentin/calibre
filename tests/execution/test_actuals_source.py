"""ActualsSource resolution: frame-backed parity and lazy hierarchy lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
)
from calibre.evaluation.forecast_metrics import resolve_actuals
from calibre.execution.actuals import (
    FrameActualsSource,
    HierarchyActualsSource,
    as_actuals_source,
)
from calibre.execution.task_builder import build_node_history
from calibre.reconciliation.summing import TOTAL_LABEL, build_hierarchy_index


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["item_a_s1", "item_a_s2", "item_b_s1", "item_b_s2"],
            "item_id": ["item_a", "item_a", "item_b", "item_b"],
            "store_id": ["s1", "s2", "s1", "s2"],
        }
    )


def _bottom_history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    rows = []
    for i, uid in enumerate(["item_a_s1", "item_a_s2", "item_b_s1", "item_b_s2"]):
        for j, ds in enumerate(dates):
            rows.append({UNIQUE_ID: uid, DS: ds, Y: float(10 * i + j)})
    frame = pd.DataFrame(rows)
    # item_b_s2 is unobserved on the last date: aggregates that include it
    # are incomplete there and must stay pending.
    return frame[~((frame[UNIQUE_ID] == "item_b_s2") & (frame[DS] == dates[-1]))].reset_index(
        drop=True
    )


def _ledger(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: [uid for uid, _ in rows],
            DS: pd.to_datetime([ds for _, ds in rows]),
            Y_HAT: np.arange(len(rows), dtype="float64"),
            MODEL_NAME: "m",
            FORECAST_ORIGIN: pd.Timestamp("2024-01-01"),
            H: 1,
            Y: np.nan,
        }
    )


def test_frame_source_matches_resolve_actuals() -> None:
    history = _bottom_history()
    ledger = _ledger([("item_a_s1", "2024-01-02"), ("item_b_s1", "2024-01-05")])
    origin = pd.Timestamp("2024-01-03")

    expected_updated, expected_new = resolve_actuals(ledger, history, origin)
    updated, new = FrameActualsSource(history).resolve(ledger, origin)

    pd.testing.assert_frame_equal(updated, expected_updated)
    pd.testing.assert_frame_equal(new, expected_new)


def test_as_actuals_source_wraps_frames_and_passes_sources_through() -> None:
    frame_source = as_actuals_source(_bottom_history())
    assert isinstance(frame_source, FrameActualsSource)

    lazy = HierarchyActualsSource(_bottom_history(), build_hierarchy_index(_hierarchy()))
    assert as_actuals_source(lazy) is lazy


def test_hierarchy_source_parity_with_eager_node_history() -> None:
    history = _bottom_history()
    hierarchy = _hierarchy()
    node_history = build_node_history(history, build_hierarchy_index(hierarchy))
    ledger = _ledger(
        [
            ("item_a_s1", "2024-01-02"),
            ("item_id=item_a", "2024-01-02"),
            ("item_id=item_b", "2024-01-04"),
            ("store_id=s1", "2024-01-03"),
            ("store_id=s2", "2024-01-04"),
            (TOTAL_LABEL, "2024-01-03"),
            (TOTAL_LABEL, "2024-01-04"),
            ("item_b_s2", "2024-01-04"),
        ]
    )
    origin = pd.Timestamp("2024-01-04")

    expected_updated, expected_new = resolve_actuals(ledger, node_history, origin)
    updated, new = HierarchyActualsSource(history, build_hierarchy_index(hierarchy)).resolve(
        ledger, origin
    )

    pd.testing.assert_frame_equal(updated, expected_updated)
    pd.testing.assert_frame_equal(new, expected_new)


def test_sparse_aggregate_resolution_sums_members() -> None:
    source = HierarchyActualsSource(_bottom_history(), build_hierarchy_index(_hierarchy()))
    ledger = _ledger([("item_id=item_a", "2024-01-02"), ("store_id=s2", "2024-01-02")])

    updated, new = source.resolve(ledger, pd.Timestamp("2024-01-02"))

    # item_a members on day 2: y = 1 (s1) + 11 (s2); s2 members: 11 + 31.
    assert updated.loc[0, Y] == pytest.approx(12.0)
    assert updated.loc[1, Y] == pytest.approx(42.0)
    assert len(new) == 2


def test_incomplete_aggregate_dates_stay_pending() -> None:
    history = _bottom_history()
    hierarchy = _hierarchy()
    source = HierarchyActualsSource(history, build_hierarchy_index(hierarchy))
    last_day = "2024-01-04"
    ledger = _ledger(
        [
            ("item_id=item_b", last_day),  # member item_b_s2 missing that day
            ("store_id=s2", last_day),
            (TOTAL_LABEL, last_day),
            ("item_id=item_a", last_day),  # complete: both members observed
        ]
    )

    updated, new = source.resolve(ledger, pd.Timestamp(last_day))

    assert updated.loc[:2, Y].isna().all()
    assert updated.loc[3, Y] == pytest.approx(3.0 + 13.0)
    assert list(new.index) == [3]

    # Eager node history has no rows for the incomplete dates either.
    node_history = build_node_history(history, build_hierarchy_index(hierarchy))
    expected_updated, _ = resolve_actuals(ledger, node_history, pd.Timestamp(last_day))
    pd.testing.assert_frame_equal(updated, expected_updated)


def test_partial_due_window_resolves_only_due_rows() -> None:
    source = HierarchyActualsSource(_bottom_history(), build_hierarchy_index(_hierarchy()))
    ledger = _ledger(
        [
            ("item_id=item_a", "2024-01-02"),
            ("item_id=item_a", "2024-01-03"),
            ("item_id=item_a", "2024-01-04"),
        ]
    )

    updated, new = source.resolve(ledger, pd.Timestamp("2024-01-03"))

    assert updated.loc[0, Y] == pytest.approx(12.0)
    assert updated.loc[1, Y] == pytest.approx(14.0)
    assert pd.isna(updated.loc[2, Y])
    assert list(new.index) == [0, 1]

    # The remaining row resolves once its date comes due.
    updated_later, new_later = source.resolve(updated, pd.Timestamp("2024-01-04"))
    assert updated_later.loc[2, Y] == pytest.approx(16.0)
    assert list(new_later.index) == [2]


def test_unknown_ledger_node_raises() -> None:
    source = HierarchyActualsSource(_bottom_history(), build_hierarchy_index(_hierarchy()))
    ledger = _ledger([("nope", "2024-01-02"), ("item_id=missing", "2024-01-02")])

    with pytest.raises(ValueError, match=r"not present in hierarchy.*item_id=missing.*nope"):
        source.resolve(ledger, pd.Timestamp("2024-01-02"))


def test_unknown_history_unique_id_raises() -> None:
    history = _bottom_history()
    history.loc[0, UNIQUE_ID] = "rogue"

    with pytest.raises(ValueError, match="rogue"):
        HierarchyActualsSource(history, build_hierarchy_index(_hierarchy()))


def test_duplicate_bottom_keys_raise() -> None:
    history = pd.concat([_bottom_history(), _bottom_history().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        HierarchyActualsSource(history, build_hierarchy_index(_hierarchy()))


def test_engine_resolves_identically_through_lazy_hierarchy_source() -> None:
    """BackendEngine delayed feedback is unchanged when actuals come lazily."""
    from calibre.conformal.runtime import SymmetricIntervalConfig
    from calibre.execution.backend import BackendEngine, ConformalOptions, ExecutionOptions
    from calibre.execution.task_builder import build_tasks

    history = _bottom_history()
    hierarchy = _hierarchy()
    node_history = build_node_history(history, build_hierarchy_index(hierarchy))
    model_configs = [
        {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 2, "name": "snaive"}
    ]
    tasks = build_tasks(node_history, model_configs, 1)
    origins = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]

    def _run(actuals) -> pd.DataFrame:
        engine = BackendEngine(
            execution=ExecutionOptions(freq="D", backend="local", seed=42),
            conformal=ConformalOptions(
                config=SymmetricIntervalConfig(
                    method="aci", coverage=0.9, calibration_window=4, gamma=0.05
                )
            ),
        )
        try:
            return engine.execute(tasks, actuals, origins).ledger.to_df()
        finally:
            engine.close()

    eager = _run(node_history)
    lazy = _run(HierarchyActualsSource(history, build_hierarchy_index(hierarchy)))
    pd.testing.assert_frame_equal(lazy, eager)


def test_bottom_only_requests_skip_aggregate_work() -> None:
    source = HierarchyActualsSource(_bottom_history(), build_hierarchy_index(_hierarchy()))
    ledger = _ledger([("item_a_s1", "2024-01-01"), ("item_b_s1", "2024-01-02")])

    updated, new = source.resolve(ledger, pd.Timestamp("2024-01-02"))

    assert updated.loc[0, Y] == pytest.approx(0.0)
    assert updated.loc[1, Y] == pytest.approx(21.0)
    assert len(new) == 2


# ---------------------------------------------------------------------------
# U3: per-run aggregate cache — complete (node, ds) sums computed at most once.
# ---------------------------------------------------------------------------


def _resolve_sequence(source: HierarchyActualsSource, ledger: pd.DataFrame, origins) -> list:
    """Resolve the same ledger across a sequence of origins, carrying forward the
    updated frame (mirrors the engine's per-origin carry-forward)."""
    outputs = []
    current = ledger
    for origin in origins:
        current, new = source.resolve(current, origin)
        outputs.append((current.copy(), new.copy()))
    return outputs


def test_cached_resolution_matches_fresh_instance_per_origin() -> None:
    history = _bottom_history()
    hierarchy = _hierarchy()
    ledger = _ledger(
        [
            ("item_id=item_a", "2024-01-02"),
            ("store_id=s1", "2024-01-02"),
            (TOTAL_LABEL, "2024-01-02"),
            ("item_id=item_a", "2024-01-03"),
            ("item_b_s1", "2024-01-03"),
            ("item_id=item_b", "2024-01-04"),  # incomplete on last day, stays pending
        ]
    )
    origins = [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]

    cached_source = HierarchyActualsSource(history, build_hierarchy_index(hierarchy))
    cached = _resolve_sequence(cached_source, ledger, origins)

    # A fresh instance per origin can never serve a cache hit, so it is the
    # uncached reference. Carry the same frame forward to keep inputs identical.
    fresh = []
    current = ledger
    for origin in origins:
        current, new = HierarchyActualsSource(history, build_hierarchy_index(hierarchy)).resolve(
            current, origin
        )
        fresh.append((current.copy(), new.copy()))

    for (c_updated, c_new), (f_updated, f_new) in zip(cached, fresh, strict=True):
        pd.testing.assert_frame_equal(c_updated, f_updated)
        pd.testing.assert_frame_equal(c_new, f_new)


def test_repeated_origin_lookup_does_no_bottom_history_rebuild(monkeypatch) -> None:
    source = HierarchyActualsSource(_bottom_history(), build_hierarchy_index(_hierarchy()))
    ledger = _ledger([("item_id=item_a", "2024-01-02"), (TOTAL_LABEL, "2024-01-02")])

    calls: list[set] = []
    real_compute = source._compute_lookup

    def _spy(uids: pd.Series, ds_values: pd.Series) -> pd.Series:
        calls.append(set(zip(uids, ds_values, strict=True)))
        return real_compute(uids, ds_values)

    monkeypatch.setattr(source, "_compute_lookup", _spy)

    first, _ = source.resolve(ledger, pd.Timestamp("2024-01-02"))
    assert len(calls) == 1  # first call computes the two complete aggregates
    assert first.loc[0, Y] == pytest.approx(12.0)

    # Second resolve of the same (node, ds) pairs: every pair is cached, so
    # _compute_lookup is never reached again (no window scan / merge / group-by).
    again = _ledger([("item_id=item_a", "2024-01-02"), (TOTAL_LABEL, "2024-01-02")])
    source.resolve(again, pd.Timestamp("2024-01-02"))
    assert len(calls) == 1  # unchanged: no rebuild


# ---------------------------------------------------------------------------
# U1 (#148): mixed-dtype attr-value collisions merge coherently through the
# cached resolve path; category-dtype phantom groups never appear.
# ---------------------------------------------------------------------------


def _collision_hierarchy() -> pd.DataFrame:
    # int 1 and str "1" on different bottom ids collide under str(): the
    # stringified predicate merges them into one "grp=1" aggregate counting
    # both members, matching the node labels and dense summing matrix.
    return pd.DataFrame(
        {
            UNIQUE_ID: ["m_int", "m_str", "m_two"],
            "grp": [1, "1", 2],
        }
    )


def _collision_history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    rows = []
    for i, uid in enumerate(["m_int", "m_str", "m_two"]):
        for j, ds in enumerate(dates):
            rows.append({UNIQUE_ID: uid, DS: ds, Y: float(10 * i + j + 1)})
    frame = pd.DataFrame(rows)
    # m_str is unobserved on the last date: the merged grp=1 aggregate is then
    # incomplete (only m_int present) and must stay pending.
    return frame[~((frame[UNIQUE_ID] == "m_str") & (frame[DS] == dates[-1]))].reset_index(drop=True)


def test_str_collision_resolves_one_merged_aggregate_through_cache() -> None:
    source = HierarchyActualsSource(
        _collision_history(), build_hierarchy_index(_collision_hierarchy())
    )
    # Day 1: both m_int (y=1) and m_str (y=11) observed -> grp=1 = 12.
    ledger = _ledger([("grp=1", "2024-01-01")])

    updated, new = source.resolve(ledger, pd.Timestamp("2024-01-01"))

    assert updated.loc[0, Y] == pytest.approx(1.0 + 11.0)  # both members counted
    assert list(new.index) == [0]


def test_str_collision_aggregate_stays_pending_when_one_member_missing() -> None:
    source = HierarchyActualsSource(
        _collision_history(), build_hierarchy_index(_collision_hierarchy())
    )
    # Day 2: m_str unobserved, so the merged grp=1 aggregate is incomplete.
    ledger = _ledger([("grp=1", "2024-01-02")])

    updated, new = source.resolve(ledger, pd.Timestamp("2024-01-02"))

    assert pd.isna(updated.loc[0, Y])
    assert new.empty


def test_categorical_attr_column_resolves_without_phantom_groups() -> None:
    grp = pd.Categorical(["A", "A", "B"], categories=["A", "B", "C"])
    hierarchy = pd.DataFrame({UNIQUE_ID: ["a", "b", "c"], "grp": grp})
    dates = pd.date_range("2024-01-01", periods=1, freq="D")
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["a", "b", "c"],
            DS: list(dates) * 3,
            Y: [2.0, 3.0, 5.0],
        }
    )
    source = HierarchyActualsSource(history, build_hierarchy_index(hierarchy))
    # grp=A (members a, b) resolves; the unobserved category C is not a node.
    ledger = _ledger([("grp=A", "2024-01-01")])

    updated, new = source.resolve(ledger, pd.Timestamp("2024-01-01"))

    assert updated.loc[0, Y] == pytest.approx(2.0 + 3.0)
    assert list(new.index) == [0]
    with pytest.raises(ValueError, match="not present in hierarchy"):
        source.resolve(_ledger([("grp=C", "2024-01-01")]), pd.Timestamp("2024-01-01"))


def test_cache_recomputes_only_new_ds_values(monkeypatch) -> None:
    source = HierarchyActualsSource(_bottom_history(), build_hierarchy_index(_hierarchy()))

    seen_ds: list[set] = []
    real_compute = source._compute_lookup

    def _spy(uids: pd.Series, ds_values: pd.Series) -> pd.Series:
        seen_ds.append(set(pd.Timestamp(ds) for ds in ds_values))
        return real_compute(uids, ds_values)

    monkeypatch.setattr(source, "_compute_lookup", _spy)

    source.resolve(_ledger([("item_id=item_a", "2024-01-02")]), pd.Timestamp("2024-01-02"))
    # Re-request the cached ds plus a new one: only the new ds reaches compute.
    source.resolve(
        _ledger([("item_id=item_a", "2024-01-02"), ("item_id=item_a", "2024-01-03")]),
        pd.Timestamp("2024-01-03"),
    )
    assert seen_ds[0] == {pd.Timestamp("2024-01-02")}
    assert seen_ds[1] == {pd.Timestamp("2024-01-03")}  # cached ds not recomputed
