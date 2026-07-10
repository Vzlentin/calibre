"""Exercise the distinct immutable fitted-values sidecar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
    FITTED_VALUE,
    FITTED_VALUE_KEY_COLUMNS,
    MODEL_NAME,
    REQUIRED_FITTED_VALUE_COLUMNS,
    SERIES_KEY,
    TIMESTAMP,
    FittedValues,
    FittedValuesError,
)

pytestmark = pytest.mark.tier1


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-b", "sku-a", "sku-a"], dtype="string"),
            TIMESTAMP: pd.to_datetime(["2026-01-12", "2026-01-12", "2026-01-05"]),
            ACTUAL_VALUE: pd.Series([2, 3, 1], dtype="int64"),
            FITTED_VALUE: pd.Series([2.5, 3.5, 1.5], dtype="float64"),
            MODEL_NAME: pd.Series(["model", "model", "model"], dtype="string"),
        }
    )


def test_fitted_values_declares_its_exact_distinct_schema_and_key() -> None:
    assert REQUIRED_FITTED_VALUE_COLUMNS == (
        "series_key",
        "timestamp",
        "actual_value",
        "fitted_value",
        "model_name",
    )
    assert FITTED_VALUE_KEY_COLUMNS == ("series_key", "timestamp", "model_name")


def test_fitted_values_normalizes_numeric_columns_and_canonicalizes_key_order() -> None:
    sidecar = FittedValues.from_frame(_frame()[list(reversed(_frame().columns))])

    assert sidecar.frame[SERIES_KEY].tolist() == ["sku-a", "sku-a", "sku-b"]
    assert sidecar.frame[ACTUAL_VALUE].dtype == np.dtype("float64")
    assert sidecar.frame[FITTED_VALUE].dtype == np.dtype("float64")


def test_fitted_values_drops_rows_with_missing_actual_or_fitted_value() -> None:
    frame = _frame()
    frame.loc[0, ACTUAL_VALUE] = np.nan
    frame.loc[1, FITTED_VALUE] = np.nan

    sidecar = FittedValues.from_frame(frame)

    assert len(sidecar.frame) == 1
    assert sidecar.frame[TIMESTAMP].tolist() == [pd.Timestamp("2026-01-05")]


def test_fitted_values_nullable_reals_drop_missing_and_upcast_exactly() -> None:
    frame = _frame()
    frame[ACTUAL_VALUE] = pd.Series([1, pd.NA, 3], dtype="Int64")
    frame[FITTED_VALUE] = pd.Series([1.5, 2.5, pd.NA], dtype="Float32")

    sidecar = FittedValues.from_frame(frame)

    assert len(sidecar.frame) == 1
    assert sidecar.frame[ACTUAL_VALUE].dtype == np.dtype("float64")
    assert sidecar.frame[FITTED_VALUE].dtype == np.dtype("float64")
    assert sidecar.frame[[ACTUAL_VALUE, FITTED_VALUE]].values.tolist() == [[1.0, 1.5]]


def test_fitted_values_sparse_reals_densify_drop_missing_and_upcast() -> None:
    frame = _frame()
    frame[ACTUAL_VALUE] = pd.Series(
        [1.0, np.nan, 3.0], dtype=pd.SparseDtype("float32", fill_value=np.nan)
    )
    frame[FITTED_VALUE] = pd.Series([1, 2, 0], dtype=pd.SparseDtype("int16", fill_value=0))

    sidecar = FittedValues.from_frame(frame)

    assert len(sidecar.frame) == 2
    assert sidecar.frame[ACTUAL_VALUE].dtype == np.dtype("float64")
    assert sidecar.frame[FITTED_VALUE].dtype == np.dtype("float64")


def test_fitted_values_arrow_backed_reals_drop_missing_and_upcast() -> None:
    frame = _frame()
    frame[ACTUAL_VALUE] = pd.Series([1, None, 3], dtype=pd.ArrowDtype(pa.int64()))
    frame[FITTED_VALUE] = pd.Series([1.5, 2.5, None], dtype=pd.ArrowDtype(pa.float32()))

    sidecar = FittedValues.from_frame(frame)

    assert len(sidecar.frame) == 1
    assert sidecar.frame[ACTUAL_VALUE].dtype == np.dtype("float64")
    assert sidecar.frame[FITTED_VALUE].dtype == np.dtype("float64")


def test_fitted_values_defensively_snapshots_input_and_output() -> None:
    frame = _frame()
    sidecar = FittedValues.from_frame(frame)
    frame.loc[:, FITTED_VALUE] = 999
    exposed = sidecar.frame
    exposed.loc[:, FITTED_VALUE] = 888
    assert sidecar.frame[FITTED_VALUE].tolist() == [1.5, 3.5, 2.5]


def test_fitted_values_rejects_duplicate_full_keys() -> None:
    frame = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(FittedValuesError, match="duplicate full key"):
        FittedValues.from_frame(frame)


@pytest.mark.parametrize(
    ("frame", "pattern"),
    [
        (_frame().drop(columns=FITTED_VALUE), "exact schema"),
        (_frame().assign(extra=1.0), "exact schema"),
        (_frame().assign(series_key=_frame()[SERIES_KEY].astype("object")), "string dtype"),
        (_frame().assign(timestamp=pd.Series([1, 2, 3], dtype="int64")), "datetime64"),
        (_frame().assign(actual_value=pd.Series([True, False, True])), "numeric"),
        (_frame().assign(fitted_value=pd.Series(["1", "2", "3"])), "numeric"),
    ],
)
def test_fitted_values_rejects_schema_and_type_drift(frame: pd.DataFrame, pattern: str) -> None:
    with pytest.raises(FittedValuesError, match=pattern):
        FittedValues.from_frame(frame)
