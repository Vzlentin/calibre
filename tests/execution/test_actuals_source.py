from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.execution.actuals import FrameActualsSource, HierarchyActualsSource
from calibre.execution.task_builder import build_node_history
from calibre.reconciliation.summing import TOTAL_LABEL


def _ledger(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: [uid for uid, _ds, _y_hat in rows],
            DS: pd.to_datetime([ds for _uid, ds, _y_hat in rows]),
            Y: [np.nan] * len(rows),
            Y_HAT: [y_hat for _uid, _ds, y_hat in rows],
            H: [1] * len(rows),
            FORECAST_ORIGIN: pd.to_datetime(["2024-01-01"] * len(rows)),
            MODEL_NAME: pd.Series(["m"] * len(rows), dtype="object"),
        }
    )


def test_frame_actuals_source_preserves_flat_resolution_behavior() -> None:
    ledger = _ledger(
        [
            ("A", "2024-01-01", 10.0),
            ("A", "2024-01-02", 20.0),
            ("A", "2024-01-03", 30.0),
        ]
    )
    actuals = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A", "A"],
            DS: pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-03"]),
            Y: [1.0, 99.0, 3.0],
        }
    )

    updated, newly_resolved = FrameActualsSource(actuals).resolve(
        ledger,
        pd.Timestamp("2024-01-02"),
    )

    assert updated.loc[0, Y] == pytest.approx(1.0)
    assert pd.isna(updated.loc[1, Y])
    assert pd.isna(updated.loc[2, Y])
    assert newly_resolved.index.tolist() == [0]


def test_hierarchy_actuals_source_resolves_bottom_aggregate_and_total_rows() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A", "B"], "dept_id": ["D", "D"]})
    bottom_actuals = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: pd.to_datetime(["2024-01-02", "2024-01-02"]),
            Y: [2.0, 3.0],
        }
    )
    ledger = _ledger(
        [
            ("A", "2024-01-02", 2.0),
            ("dept_id=D", "2024-01-02", 5.0),
            (TOTAL_LABEL, "2024-01-02", 5.0),
        ]
    )

    updated, newly_resolved = HierarchyActualsSource(bottom_actuals, hierarchy).resolve(
        ledger,
        pd.Timestamp("2024-01-02"),
    )

    resolved = updated.set_index(UNIQUE_ID)[Y]
    assert resolved.loc["A"] == pytest.approx(2.0)
    assert resolved.loc["dept_id=D"] == pytest.approx(5.0)
    assert resolved.loc[TOTAL_LABEL] == pytest.approx(5.0)
    assert newly_resolved.index.tolist() == [0, 1, 2]


def test_hierarchy_actuals_source_leaves_incomplete_aggregate_unresolved() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A", "B"], "dept_id": ["D", "D"]})
    bottom_actuals = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            DS: pd.to_datetime(["2024-01-02"]),
            Y: [2.0],
        }
    )
    ledger = _ledger(
        [
            ("A", "2024-01-02", 2.0),
            ("dept_id=D", "2024-01-02", 5.0),
            (TOTAL_LABEL, "2024-01-02", 5.0),
        ]
    )

    updated, newly_resolved = HierarchyActualsSource(bottom_actuals, hierarchy).resolve(
        ledger,
        pd.Timestamp("2024-01-02"),
    )

    resolved = updated.set_index(UNIQUE_ID)[Y]
    assert resolved.loc["A"] == pytest.approx(2.0)
    assert pd.isna(resolved.loc["dept_id=D"])
    assert pd.isna(resolved.loc[TOTAL_LABEL])
    assert newly_resolved.index.tolist() == [0]


def test_hierarchy_actuals_source_rejects_unknown_requested_nodes() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A"], "dept_id": ["D"]})
    bottom_actuals = pd.DataFrame({UNIQUE_ID: ["A"], DS: pd.to_datetime(["2024-01-02"]), Y: [2.0]})
    ledger = _ledger([("dept_id=BOGUS", "2024-01-02", 5.0)])

    with pytest.raises(ValueError, match="requested hierarchy node labels.*dept_id=BOGUS"):
        HierarchyActualsSource(bottom_actuals, hierarchy).resolve(
            ledger,
            pd.Timestamp("2024-01-02"),
        )


def test_hierarchy_actuals_source_validates_due_nodes_only() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A"], "dept_id": ["D"]})
    bottom_actuals = pd.DataFrame({UNIQUE_ID: ["A"], DS: pd.to_datetime(["2024-01-02"]), Y: [2.0]})
    ledger = _ledger(
        [
            ("dept_id=D", "2024-01-02", 2.0),
            ("dept_id=BOGUS", "2024-01-03", 5.0),
        ]
    )

    updated, newly_resolved = HierarchyActualsSource(bottom_actuals, hierarchy).resolve(
        ledger,
        pd.Timestamp("2024-01-02"),
    )

    assert updated.loc[0, Y] == pytest.approx(2.0)
    assert pd.isna(updated.loc[1, Y])
    assert newly_resolved.index.tolist() == [0]


def test_hierarchy_actuals_source_rejects_duplicate_bottom_actual_keys() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A"], "dept_id": ["D"]})
    bottom_actuals = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "A"],
            DS: pd.to_datetime(["2024-01-02", "2024-01-02"]),
            Y: [2.0, 3.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate.*A"):
        HierarchyActualsSource(bottom_actuals, hierarchy)


def test_hierarchy_actuals_source_matches_eager_node_history_for_requested_rows() -> None:
    hierarchy = pd.DataFrame(
        {
            UNIQUE_ID: ["A_CA", "B_CA", "A_TX"],
            "dept_id": ["A", "B", "A"],
            "state_id": ["CA", "CA", "TX"],
        }
    )
    bottom_actuals = pd.DataFrame(
        {
            UNIQUE_ID: ["A_CA", "B_CA", "A_TX"] * 2,
            DS: pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
            Y: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    requested = _ledger(
        [
            ("A_CA", "2024-01-02", 0.0),
            ("dept_id=A", "2024-01-02", 0.0),
            ("state_id=CA", "2024-01-02", 0.0),
            (TOTAL_LABEL, "2024-01-02", 0.0),
        ]
    )
    eager = FrameActualsSource(build_node_history(bottom_actuals, hierarchy)).resolve(
        requested,
        pd.Timestamp("2024-01-02"),
    )[0]

    lazy = HierarchyActualsSource(bottom_actuals, hierarchy).resolve(
        requested,
        pd.Timestamp("2024-01-02"),
    )[0]

    pd.testing.assert_series_equal(lazy[Y], eager[Y])


def test_hierarchy_actuals_source_uses_canonical_stringified_attribute_labels() -> None:
    hierarchy = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            "launch_date": [pd.Timestamp("2024-01-01")],
        }
    )
    bottom_actuals = pd.DataFrame({UNIQUE_ID: ["A"], DS: pd.to_datetime(["2024-01-02"]), Y: [2.0]})
    ledger = _ledger([("launch_date=2024-01-01", "2024-01-02", 2.0)])

    updated, _newly_resolved = HierarchyActualsSource(bottom_actuals, hierarchy).resolve(
        ledger,
        pd.Timestamp("2024-01-02"),
    )

    assert updated.loc[0, Y] == pytest.approx(2.0)


def test_hierarchy_actuals_source_matches_eager_node_history_for_present_nan_values() -> None:
    hierarchy = pd.DataFrame({UNIQUE_ID: ["A", "B"], "dept_id": ["D", "D"]})
    bottom_actuals = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: pd.to_datetime(["2024-01-02", "2024-01-02"]),
            Y: [np.nan, 3.0],
        }
    )
    requested = _ledger([("dept_id=D", "2024-01-02", 0.0)])
    eager = FrameActualsSource(build_node_history(bottom_actuals, hierarchy)).resolve(
        requested,
        pd.Timestamp("2024-01-02"),
    )[0]

    lazy = HierarchyActualsSource(bottom_actuals, hierarchy).resolve(
        requested,
        pd.Timestamp("2024-01-02"),
    )[0]

    pd.testing.assert_series_equal(lazy[Y], eager[Y])
