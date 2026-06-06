from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    validate_forecast_frame,
)
from calibre.reconciliation.apply import NODE_LABEL, VectorReconciler
from calibre.reconciliation.summing import TOTAL_LABEL, SummingMatrix


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


def test_hierarchy_none_is_exact_identity() -> None:
    frame = _single_section(["a", "b"], [1.0, 2.0])
    out = _DoubleBottom()(frame, None)
    pd.testing.assert_frame_equal(out, frame)


def test_empty_frame_returns_empty_unchanged() -> None:
    empty = _single_section(["a", "b"], [1.0, 2.0]).iloc[:0]
    out = _DoubleBottom()(empty, _hierarchy(["a", "b"]))
    assert out.empty


def test_each_cross_section_reconciled_independently() -> None:
    frame = _forecast_frame(
        [
            {
                UNIQUE_ID: "a",
                "ds": "2024-01-14",
                Y_HAT: 1.0,
                H: 1,
                FORECAST_ORIGIN: "2024-01-07",
                MODEL_NAME: "A",
            },
            {
                UNIQUE_ID: "b",
                "ds": "2024-01-14",
                Y_HAT: 2.0,
                H: 1,
                FORECAST_ORIGIN: "2024-01-07",
                MODEL_NAME: "A",
            },
            {
                UNIQUE_ID: "a",
                "ds": "2024-01-21",
                Y_HAT: 3.0,
                H: 2,
                FORECAST_ORIGIN: "2024-01-07",
                MODEL_NAME: "A",
            },
            {
                UNIQUE_ID: "b",
                "ds": "2024-01-21",
                Y_HAT: 4.0,
                H: 2,
                FORECAST_ORIGIN: "2024-01-07",
                MODEL_NAME: "A",
            },
            {
                UNIQUE_ID: "a",
                "ds": "2024-01-14",
                Y_HAT: 5.0,
                H: 1,
                FORECAST_ORIGIN: "2024-01-07",
                MODEL_NAME: "B",
            },
            {
                UNIQUE_ID: "b",
                "ds": "2024-01-14",
                Y_HAT: 6.0,
                H: 1,
                FORECAST_ORIGIN: "2024-01-07",
                MODEL_NAME: "B",
            },
        ]
    )
    out = _BottomSumFill()(frame, _hierarchy(["a", "b"]))
    # (A, h1): sum 1+2=3; (A, h2): sum 3+4=7; (B, h1): sum 5+6=11.
    expected = [3.0, 3.0, 7.0, 7.0, 11.0, 11.0]
    np.testing.assert_array_equal(out[Y_HAT].to_numpy(), expected)


def test_doubling_stub_writes_back_expected_yhat() -> None:
    frame = _single_section(["a", "b", "c"], [1.0, 2.0, 4.0])
    out = _DoubleBottom()(frame, _hierarchy(["a", "b", "c"]))
    np.testing.assert_array_equal(out[Y_HAT].to_numpy(), [2.0, 4.0, 8.0])


def test_write_back_preserves_order_index_dtypes_and_contract() -> None:
    frame = _single_section(["b", "a", "c"], [2.0, 1.0, 4.0])
    original = frame.copy()
    out = _DoubleBottom()(frame, _hierarchy(["a", "b", "c"]))

    # Row order preserved (note input order is b, a, c — not sorted).
    assert out[UNIQUE_ID].tolist() == ["b", "a", "c"]
    pd.testing.assert_index_equal(out.index, original.index)
    assert list(out.dtypes) == list(original.dtypes)
    validate_forecast_frame(out)
    # Each bottom doubled regardless of frame order.
    assert dict(zip(out[UNIQUE_ID], out[Y_HAT], strict=True)) == {"b": 4.0, "a": 2.0, "c": 8.0}


def test_cross_section_missing_some_bottom_ids_aligns_to_subset() -> None:
    # Hierarchy has a, b, c but this cross-section only forecasts a and c.
    frame = _single_section(["a", "c"], [3.0, 5.0])
    out = _DoubleBottom()(frame, _hierarchy(["a", "b", "c"]))
    np.testing.assert_array_equal(out[Y_HAT].to_numpy(), [6.0, 10.0])
    validate_forecast_frame(out)


def test_reconciled_all_levels_exposes_aggregate_nodes() -> None:
    frame = _single_section(["a", "b"], [1.0, 2.0])
    all_levels = _DoubleBottom().reconciled_all_levels(frame, _hierarchy(["a", "b"]))

    by_node = dict(zip(all_levels[NODE_LABEL], all_levels[Y_HAT], strict=True))
    # Doubling stub: bottom doubled, aggregates re-summed from doubled bottom.
    assert by_node["a"] == 2.0
    assert by_node["b"] == 4.0
    assert by_node[TOTAL_LABEL] == 6.0
    # The frame itself keeps only its original bottom node row-set (KTD1).
    assert set(frame[UNIQUE_ID]) == {"a", "b"}
