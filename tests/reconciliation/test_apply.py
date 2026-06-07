from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibre.core.forecast_frame import (
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    validate_forecast_frame,
)
from calibre.reconciliation.apply import VectorReconciler
from calibre.reconciliation.summing import SummingMatrix, build_summing_matrix


class _DoubleBottom(VectorReconciler):
    """Stub strategy that doubles the bottom vector (then re-sums aggregates)."""

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        bottom = base[: summing.n_bottom]
        return summing.S @ (2.0 * bottom)


class _BottomSumFill(VectorReconciler):
    """Replace each bottom node with the cross-section's bottom sum.

    Used to prove cross-section isolation: the written-back value depends only on
    the members of the same ``(model, origin, h)`` group.
    """

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        bottom = base[: summing.n_bottom]
        filled = np.full(summing.n_bottom, float(bottom.sum()))
        return summing.S @ filled


class _CaptureBase(VectorReconciler):
    def __init__(self) -> None:
        self.base: np.ndarray | None = None

    def reconcile_vector(self, base: np.ndarray, summing: SummingMatrix) -> np.ndarray:
        self.base = base.copy()
        return summing.S @ base[: summing.n_bottom]


def _hierarchy(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({UNIQUE_ID: ids, "store": [f"S{i}" for i in range(len(ids))]})


def _forecast_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame[UNIQUE_ID] = frame[UNIQUE_ID].astype("object")
    frame["ds"] = pd.to_datetime(frame["ds"]).astype("datetime64[ns]")
    frame["y"] = np.nan
    frame["y"] = frame["y"].astype("float64")
    frame[Y_HAT] = frame[Y_HAT].astype("float64")
    frame[H] = frame[H].astype("int64")
    frame[FORECAST_ORIGIN] = pd.to_datetime(frame[FORECAST_ORIGIN]).astype("datetime64[ns]")
    frame[MODEL_NAME] = frame[MODEL_NAME].astype("object")
    return frame[[UNIQUE_ID, "ds", "y", Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]]


def _single_section(ids: list[str], yhats: list[float]) -> pd.DataFrame:
    return _forecast_frame(
        [
            {
                UNIQUE_ID: uid,
                "ds": "2024-01-14",
                Y_HAT: yhat,
                H: 1,
                FORECAST_ORIGIN: "2024-01-07",
                MODEL_NAME: "m",
            }
            for uid, yhat in zip(ids, yhats, strict=True)
        ]
    )


def _node_section(
    ids: list[str],
    bottom_yhats: list[float],
    hierarchy: pd.DataFrame | None = None,
    **shared,
) -> pd.DataFrame:
    hierarchy = _hierarchy(ids) if hierarchy is None else hierarchy
    summing = build_summing_matrix(hierarchy).subset(ids)
    coherent = summing.S @ np.array(bottom_yhats, dtype=np.float64)
    rows = []
    for uid, yhat in zip(summing.node_labels, coherent, strict=True):
        row = {
            UNIQUE_ID: uid,
            "ds": "2024-01-14",
            Y_HAT: float(yhat),
            H: 1,
            FORECAST_ORIGIN: "2024-01-07",
            MODEL_NAME: "m",
        }
        row.update(shared)
        rows.append(row)
    return _forecast_frame(rows)


def test_hierarchy_none_is_exact_identity() -> None:
    frame = _single_section(["a", "b"], [1.0, 2.0])
    out = _DoubleBottom()(frame, None)
    pd.testing.assert_frame_equal(out, frame)


def test_empty_frame_returns_empty_unchanged() -> None:
    empty = _single_section(["a", "b"], [1.0, 2.0]).iloc[:0]
    out = _DoubleBottom()(empty, _hierarchy(["a", "b"]))
    assert out.empty


def test_each_cross_section_reconciled_independently() -> None:
    frame = pd.concat(
        [
            _node_section(["a", "b"], [1.0, 2.0], **{MODEL_NAME: "A"}),
            _node_section(
                ["a", "b"],
                [3.0, 4.0],
                **{MODEL_NAME: "A", H: 2, "ds": "2024-01-21"},
            ),
            _node_section(["a", "b"], [5.0, 6.0], **{MODEL_NAME: "B"}),
        ],
        ignore_index=True,
    )
    out = _BottomSumFill()(frame, _hierarchy(["a", "b"]))
    # (A, h1): bottom fill 3 -> total 6; (A, h2): fill 7 -> total 14;
    # (B, h1): fill 11 -> total 22.
    expected = [3.0, 3.0, 3.0, 3.0, 6.0, 7.0, 7.0, 7.0, 7.0, 14.0, 11.0, 11.0, 11.0, 11.0, 22.0]
    np.testing.assert_array_equal(out[Y_HAT].to_numpy(), expected)


def test_doubling_stub_writes_back_expected_yhat() -> None:
    frame = _node_section(["a", "b", "c"], [1.0, 2.0, 4.0])
    out = _DoubleBottom()(frame, _hierarchy(["a", "b", "c"]))
    np.testing.assert_array_equal(out[Y_HAT].to_numpy(), [2.0, 4.0, 8.0, 2.0, 4.0, 8.0, 14.0])


def test_write_back_preserves_order_index_dtypes_and_contract() -> None:
    frame = _node_section(["a", "b", "c"], [1.0, 2.0, 4.0]).iloc[[1, 0, 2, 5, 3, 4, 6]]
    original = frame.copy()
    out = _DoubleBottom()(frame, _hierarchy(["a", "b", "c"]))

    # Row order preserved (note input order is deliberately not sorted).
    assert out[UNIQUE_ID].tolist() == [
        "b",
        "a",
        "c",
        "store=S2",
        "store=S0",
        "store=S1",
        "__total__",
    ]
    pd.testing.assert_index_equal(out.index, original.index)
    assert list(out.dtypes) == list(original.dtypes)
    validate_forecast_frame(out)
    assert dict(zip(out[UNIQUE_ID], out[Y_HAT], strict=True)) == {
        "a": 2.0,
        "b": 4.0,
        "c": 8.0,
        "store=S0": 2.0,
        "store=S1": 4.0,
        "store=S2": 8.0,
        "__total__": 14.0,
    }


def test_write_back_preserves_duplicate_index_labels_without_expanding_rows() -> None:
    frame = _node_section(["a", "b"], [1.0, 2.0])
    frame.index = [0, 0, 1, 1, 2]

    out = _DoubleBottom()(frame, _hierarchy(["a", "b"]))

    assert len(out) == len(frame)
    pd.testing.assert_index_equal(out.index, frame.index)
    assert out[UNIQUE_ID].tolist() == frame[UNIQUE_ID].tolist()
    np.testing.assert_array_equal(out[Y_HAT].to_numpy(), [2.0, 4.0, 2.0, 4.0, 6.0])


def test_cross_section_missing_some_bottom_ids_aligns_to_subset() -> None:
    # Hierarchy has a, b, c but this cross-section only forecasts a and c.
    hierarchy = _hierarchy(["a", "b", "c"])
    frame = _node_section(["a", "c"], [3.0, 5.0], hierarchy=hierarchy)
    out = _DoubleBottom()(frame, hierarchy)
    np.testing.assert_array_equal(out[Y_HAT].to_numpy(), [6.0, 10.0, 6.0, 10.0, 16.0])
    validate_forecast_frame(out)


def test_supplied_aggregate_base_is_passed_to_strategy() -> None:
    frame = _node_section(["a", "b"], [1.0, 2.0])
    frame.loc[frame[UNIQUE_ID] == "__total__", Y_HAT] = 99.0
    reconciler = _CaptureBase()

    reconciler(frame, _hierarchy(["a", "b"]))

    assert reconciler.base is not None
    np.testing.assert_array_equal(reconciler.base, [1.0, 2.0, 1.0, 2.0, 99.0])


def test_missing_required_aggregate_row_fails_clearly() -> None:
    frame = _node_section(["a", "b"], [1.0, 2.0])
    frame = frame[frame[UNIQUE_ID] != "__total__"]

    with pytest.raises(ValueError, match="missing required hierarchy node forecast"):
        _DoubleBottom()(frame, _hierarchy(["a", "b"]))
