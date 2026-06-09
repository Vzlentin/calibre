"""Native bottom_up point reconciler: aggregate synthesis from bottom-only rows."""

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
from calibre.reconciliation import BottomUpReconciler, ReconciliationContext, resolve_reconciler
from calibre.reconciliation.nixtla_adapter import NixtlaReconciler
from calibre.reconciliation.summing import TOTAL_LABEL, build_summing_matrix

_CONTEXT = ReconciliationContext()


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame(
        {
            UNIQUE_ID: ["a_s1", "a_s2", "b_s1"],
            "item_id": ["a", "a", "b"],
            "store_id": ["s1", "s2", "s1"],
        }
    )


def _bottom_frame(h: int = 1, model: str = "m", y_hats: list[float] | None = None) -> pd.DataFrame:
    values = y_hats if y_hats is not None else [1.0, 2.0, 4.0]
    return pd.DataFrame(
        {
            UNIQUE_ID: ["a_s1", "a_s2", "b_s1"],
            DS: pd.Timestamp("2024-01-08"),
            Y_HAT: np.asarray(values, dtype="float64"),
            MODEL_NAME: model,
            FORECAST_ORIGIN: pd.Timestamp("2024-01-07"),
            H: h,
            Y: np.nan,
        }
    )


def test_registry_resolves_native_bottom_up() -> None:
    assert isinstance(resolve_reconciler("bottom_up"), BottomUpReconciler)


def test_nixtla_reconciler_rejects_point_bottom_up() -> None:
    with pytest.raises(ValueError, match="native"):
        NixtlaReconciler("bottom_up")


def test_synthesizes_aggregates_equal_to_summing_matrix_product() -> None:
    frame = _bottom_frame()
    result = BottomUpReconciler()(frame, _hierarchy(), _CONTEXT)

    summing = build_summing_matrix(_hierarchy())
    expected = summing.S @ np.array([1.0, 2.0, 4.0])
    values = result.set_index(UNIQUE_ID)[Y_HAT]
    np.testing.assert_allclose(
        values.reindex(summing.node_labels).to_numpy(dtype=np.float64), expected
    )

    # Bottom rows are unchanged and come first, then aggregates in canonical order.
    assert list(result[UNIQUE_ID]) == [
        "a_s1",
        "a_s2",
        "b_s1",
        "item_id=a",
        "item_id=b",
        "store_id=s1",
        "store_id=s2",
        TOTAL_LABEL,
    ]
    pd.testing.assert_frame_equal(result.iloc[:3].reset_index(drop=True), frame)


def test_synthesized_rows_preserve_frame_columns() -> None:
    frame = _bottom_frame()
    result = BottomUpReconciler()(frame, _hierarchy(), _CONTEXT)

    assert list(result.columns) == list(frame.columns)
    aggregates = result.iloc[3:]
    assert (aggregates[MODEL_NAME] == "m").all()
    assert (aggregates[H] == 1).all()
    assert (aggregates[DS] == pd.Timestamp("2024-01-08")).all()
    assert (aggregates[FORECAST_ORIGIN] == pd.Timestamp("2024-01-07")).all()
    assert aggregates[Y].isna().all()


def test_partial_bottom_subset_synthesizes_only_covered_nodes() -> None:
    frame = _bottom_frame()
    frame = frame[frame[UNIQUE_ID] != "a_s2"].reset_index(drop=True)

    result = BottomUpReconciler()(frame, _hierarchy(), _CONTEXT)

    values = result.set_index(UNIQUE_ID)[Y_HAT]
    assert values["item_id=a"] == pytest.approx(1.0)  # only a_s1 present
    assert values["item_id=b"] == pytest.approx(4.0)
    assert values["store_id=s1"] == pytest.approx(5.0)
    assert values[TOTAL_LABEL] == pytest.approx(5.0)
    assert "store_id=s2" not in set(values.index)


def test_groups_are_expanded_independently() -> None:
    frame = pd.concat(
        [
            _bottom_frame(h=1, y_hats=[1.0, 2.0, 4.0]),
            _bottom_frame(h=2, y_hats=[10.0, 20.0, 40.0]),
            _bottom_frame(h=1, model="other", y_hats=[100.0, 200.0, 400.0]),
        ],
        ignore_index=True,
    )
    result = BottomUpReconciler()(frame, _hierarchy(), _CONTEXT)

    keyed = result.set_index([MODEL_NAME, H, UNIQUE_ID])[Y_HAT]
    assert keyed[("m", 1, TOTAL_LABEL)] == pytest.approx(7.0)
    assert keyed[("m", 2, TOTAL_LABEL)] == pytest.approx(70.0)
    assert keyed[("other", 1, TOTAL_LABEL)] == pytest.approx(700.0)
    assert len(result) == 3 * 8


def test_aggregate_input_rows_are_rejected() -> None:
    frame = _bottom_frame()
    extra = frame.iloc[[0]].assign(**{UNIQUE_ID: "item_id=a"})
    frame = pd.concat([frame, extra], ignore_index=True)

    with pytest.raises(ValueError, match="bottom-level forecast rows only"):
        BottomUpReconciler()(frame, _hierarchy(), _CONTEXT)


def test_unknown_node_rows_are_rejected() -> None:
    frame = _bottom_frame()
    frame.loc[0, UNIQUE_ID] = "rogue"

    with pytest.raises(ValueError, match="rogue"):
        BottomUpReconciler()(frame, _hierarchy(), _CONTEXT)


def test_duplicate_bottom_rows_are_rejected() -> None:
    frame = pd.concat([_bottom_frame(), _bottom_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate"):
        BottomUpReconciler()(frame, _hierarchy(), _CONTEXT)


def test_quantile_columns_are_rejected() -> None:
    frame = _bottom_frame()
    frame["q_0.9"] = 1.0

    with pytest.raises(ValueError, match="quantile columns"):
        BottomUpReconciler()(frame, _hierarchy(), _CONTEXT)


def test_no_hierarchy_or_empty_frame_pass_through() -> None:
    frame = _bottom_frame()
    reconciler = BottomUpReconciler()

    assert reconciler(frame, None, _CONTEXT) is frame
    empty = frame.iloc[0:0]
    assert reconciler(empty, _hierarchy(), _CONTEXT) is empty
