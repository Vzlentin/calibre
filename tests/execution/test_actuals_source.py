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
from calibre.reconciliation.summing import TOTAL_LABEL


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

    lazy = HierarchyActualsSource(_bottom_history(), _hierarchy())
    assert as_actuals_source(lazy) is lazy


def test_hierarchy_source_parity_with_eager_node_history() -> None:
    history = _bottom_history()
    hierarchy = _hierarchy()
    node_history = build_node_history(history, hierarchy)
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
    updated, new = HierarchyActualsSource(history, hierarchy).resolve(ledger, origin)

    pd.testing.assert_frame_equal(updated, expected_updated)
    pd.testing.assert_frame_equal(new, expected_new)


def test_sparse_aggregate_resolution_sums_members() -> None:
    source = HierarchyActualsSource(_bottom_history(), _hierarchy())
    ledger = _ledger([("item_id=item_a", "2024-01-02"), ("store_id=s2", "2024-01-02")])

    updated, new = source.resolve(ledger, pd.Timestamp("2024-01-02"))

    # item_a members on day 2: y = 1 (s1) + 11 (s2); s2 members: 11 + 31.
    assert updated.loc[0, Y] == pytest.approx(12.0)
    assert updated.loc[1, Y] == pytest.approx(42.0)
    assert len(new) == 2


def test_incomplete_aggregate_dates_stay_pending() -> None:
    history = _bottom_history()
    hierarchy = _hierarchy()
    source = HierarchyActualsSource(history, hierarchy)
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
    node_history = build_node_history(history, hierarchy)
    expected_updated, _ = resolve_actuals(ledger, node_history, pd.Timestamp(last_day))
    pd.testing.assert_frame_equal(updated, expected_updated)


def test_partial_due_window_resolves_only_due_rows() -> None:
    source = HierarchyActualsSource(_bottom_history(), _hierarchy())
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
    source = HierarchyActualsSource(_bottom_history(), _hierarchy())
    ledger = _ledger([("nope", "2024-01-02"), ("item_id=missing", "2024-01-02")])

    with pytest.raises(ValueError, match=r"not present in hierarchy.*item_id=missing.*nope"):
        source.resolve(ledger, pd.Timestamp("2024-01-02"))


def test_unknown_history_unique_id_raises() -> None:
    history = _bottom_history()
    history.loc[0, UNIQUE_ID] = "rogue"

    with pytest.raises(ValueError, match="rogue"):
        HierarchyActualsSource(history, _hierarchy())


def test_duplicate_bottom_keys_raise() -> None:
    history = pd.concat([_bottom_history(), _bottom_history().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        HierarchyActualsSource(history, _hierarchy())


def test_engine_resolves_identically_through_lazy_hierarchy_source() -> None:
    """BackendEngine delayed feedback is unchanged when actuals come lazily."""
    from calibre.conformal.runtime import SymmetricIntervalConfig
    from calibre.execution.backend import BackendEngine, ConformalOptions, ExecutionOptions
    from calibre.execution.task_builder import build_tasks

    history = _bottom_history()
    hierarchy = _hierarchy()
    node_history = build_node_history(history, hierarchy)
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
    lazy = _run(HierarchyActualsSource(history, hierarchy))
    pd.testing.assert_frame_equal(lazy, eager)


def test_bottom_only_requests_skip_aggregate_work() -> None:
    source = HierarchyActualsSource(_bottom_history(), _hierarchy())
    ledger = _ledger([("item_a_s1", "2024-01-01"), ("item_b_s1", "2024-01-02")])

    updated, new = source.resolve(ledger, pd.Timestamp("2024-01-02"))

    assert updated.loc[0, Y] == pytest.approx(0.0)
    assert updated.loc[1, Y] == pytest.approx(21.0)
    assert len(new) == 2
