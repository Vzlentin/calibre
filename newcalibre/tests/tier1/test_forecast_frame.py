"""Exercise the final calendar and chapter 02 forecast-frame contract."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from newcalibre.domain import Calendar, CalendarError
from newcalibre.domain.forecast_frame import (
    ACTUAL_VALUE,
    FRAME_KEY_COLUMNS,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    REQUIRED_FRAME_COLUMNS,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    ForecastFrameError,
    interval_columns,
    quantile_column,
    target_timestamp,
    validate_forecast_frame,
)

pytestmark = pytest.mark.tier1
WEEKLY = Calendar("W-MON").bind(pd.Timestamp("2026-01-12"))


def _weekly_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-a", "sku-a"], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(["2026-01-12", "2026-01-19"]),
            ACTUAL_VALUE: pd.Series([np.nan, np.nan], dtype="float64"),
            POINT_FORECAST: pd.Series([7.0, 8.0], dtype="float64"),
            HORIZON_STEP: pd.Series([1, 2], dtype="int64"),
            ORIGIN: pd.to_datetime(["2026-01-12", "2026-01-12"]),
            MODEL_NAME: pd.Series(["seasonal-naive", "seasonal-naive"], dtype="string"),
        }
    )


def test_calendar_normalizes_and_advances_explicit_anchored_frequencies() -> None:
    origin = pd.Timestamp("2026-01-12")
    declaration = Calendar("1W-MON")
    calendar = declaration.bind(origin)

    assert calendar.frequency == "W-MON"
    assert calendar.phase == origin
    assert calendar.contains(origin)
    assert calendar.advance(origin, 0) == origin
    assert calendar.advance(origin, 2) == pd.Timestamp("2026-01-26")
    assert calendar.retreat(origin, 1) == pd.Timestamp("2026-01-05")


@pytest.mark.parametrize("frequency", ["", "W", "2W", "not-a-frequency", 7, "0D", "-1D"])
def test_calendar_rejects_ambiguous_invalid_or_non_advancing_declarations(
    frequency: object,
) -> None:
    with pytest.raises(CalendarError, match="non-empty|explicit anchor|invalid pandas|positive"):
        Calendar(cast(str, frequency))


def test_calendar_rejects_off_grid_and_timezone_aware_timestamps() -> None:
    with pytest.raises(CalendarError, match="does not lie"):
        WEEKLY.advance(pd.Timestamp("2026-01-13"), 0)
    with pytest.raises(CalendarError, match="timezone-naive"):
        WEEKLY.contains(pd.Timestamp("2026-01-12", tz="UTC"))


def test_required_columns_and_full_row_key_are_literal() -> None:
    assert REQUIRED_FRAME_COLUMNS == (
        "series_key",
        "target_timestamp",
        "actual_value",
        "point_forecast",
        "horizon_step",
        "origin",
        "model_name",
    )
    assert FRAME_KEY_COLUMNS == ("series_key", "origin", "horizon_step", "model_name")


def test_frame_validation_accepts_exact_schema_and_derived_timestamps() -> None:
    frame = _weekly_frame()
    validated = validate_forecast_frame(frame, calendar=WEEKLY)

    pd.testing.assert_frame_equal(validated, frame)
    assert target_timestamp(pd.Timestamp("2026-01-12"), 1, calendar=WEEKLY) == pd.Timestamp(
        "2026-01-12"
    )
    assert target_timestamp(pd.Timestamp("2026-01-12"), 2, calendar=WEEKLY) == pd.Timestamp(
        "2026-01-19"
    )


@pytest.mark.parametrize("column", REQUIRED_FRAME_COLUMNS)
def test_frame_validation_rejects_each_missing_required_column(column: str) -> None:
    with pytest.raises(ForecastFrameError, match="missing required columns"):
        validate_forecast_frame(_weekly_frame().drop(columns=column), calendar=WEEKLY)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        (SERIES_KEY, pd.Series(["a", "a"], dtype="object")),
        (TARGET_TIMESTAMP, pd.Series([1, 2], dtype="int64")),
        (ACTUAL_VALUE, pd.Series([1.0, 2.0], dtype="float32")),
        (POINT_FORECAST, pd.Series(["7", "8"], dtype="string")),
        (HORIZON_STEP, pd.Series([1.0, 2.0], dtype="float64")),
        (ORIGIN, pd.Series(["2026-01-12"] * 2, dtype="string")),
        (MODEL_NAME, pd.Series(["m", "m"], dtype="object")),
    ],
)
def test_frame_validation_rejects_mistyped_required_columns(
    column: str, replacement: pd.Series
) -> None:
    frame = _weekly_frame()
    frame[column] = replacement
    with pytest.raises(ForecastFrameError, match=column):
        validate_forecast_frame(frame, calendar=WEEKLY)


def test_frame_validation_preserves_datetime_resolution_and_integer_width() -> None:
    frame = _weekly_frame()
    frame[TARGET_TIMESTAMP] = frame[TARGET_TIMESTAMP].astype("datetime64[ms]")
    frame[ORIGIN] = frame[ORIGIN].astype("datetime64[ms]")
    frame[HORIZON_STEP] = frame[HORIZON_STEP].astype("int32")

    validated = validate_forecast_frame(frame, calendar=WEEKLY)

    assert validated[TARGET_TIMESTAMP].dtype == np.dtype("datetime64[ms]")
    assert validated[ORIGIN].dtype == np.dtype("datetime64[ms]")
    assert validated[HORIZON_STEP].dtype == np.dtype("int32")


def test_frame_integer_values_upcast_atomically_without_mutating_input() -> None:
    lower, upper = interval_columns(0.9)
    frame = _weekly_frame()
    for column, values in {
        ACTUAL_VALUE: [1, 2],
        POINT_FORECAST: [7, 8],
        lower: [5, 6],
        upper: [9, 10],
        quantile_column(0.5): [7, 8],
    }.items():
        frame[column] = pd.Series(values, dtype="int64")
    original = frame.copy(deep=True)

    validated = validate_forecast_frame(frame, calendar=WEEKLY)

    pd.testing.assert_frame_equal(frame, original)
    for column in (ACTUAL_VALUE, POINT_FORECAST, lower, upper, quantile_column(0.5)):
        assert validated[column].dtype == np.dtype("float64")


def test_frame_nullable_integers_upcast_across_all_float_value_columns() -> None:
    lower, upper = interval_columns(0.9)
    quantile = quantile_column(0.5)
    frame = _weekly_frame()
    frame[ACTUAL_VALUE] = pd.Series([pd.NA, 2], dtype="Int64")
    frame[POINT_FORECAST] = pd.Series([7.0, 8.0], dtype="Float32")
    frame[lower] = pd.Series([5, pd.NA], dtype="Int32")
    frame[upper] = pd.Series([9, 10], dtype="UInt32")
    frame[quantile] = pd.Series([7, 8], dtype="Int8")

    validated = validate_forecast_frame(frame, calendar=WEEKLY)

    for column in (ACTUAL_VALUE, POINT_FORECAST, lower, upper, quantile):
        assert validated[column].dtype == np.dtype("float64")
    assert np.isnan(validated.loc[0, ACTUAL_VALUE])
    assert np.isnan(validated.loc[1, lower])


def test_frame_sparse_real_values_densify_and_normalize_to_float64() -> None:
    lower, upper = interval_columns(0.9)
    frame = _weekly_frame()
    frame[ACTUAL_VALUE] = pd.Series([1, 0], dtype=pd.SparseDtype("int16", fill_value=0))
    frame[POINT_FORECAST] = pd.Series(
        [7.0, 8.0], dtype=pd.SparseDtype("float32", fill_value=np.nan)
    )
    frame[lower] = pd.Series([5, 6], dtype=pd.SparseDtype("uint8", fill_value=0))
    frame[upper] = pd.Series([9.0, 10.0], dtype=pd.SparseDtype("float16", fill_value=np.nan))

    validated = validate_forecast_frame(frame, calendar=WEEKLY)

    for column in (ACTUAL_VALUE, POINT_FORECAST, lower, upper):
        assert validated[column].dtype == np.dtype("float64")


def test_frame_arrow_backed_real_values_normalize_to_float64() -> None:
    lower, upper = interval_columns(0.9)
    frame = _weekly_frame()
    frame[ACTUAL_VALUE] = pd.Series([None, 2], dtype=pd.ArrowDtype(pa.int64()))
    frame[POINT_FORECAST] = pd.Series([7, 8], dtype=pd.ArrowDtype(pa.uint64()))
    frame[lower] = pd.Series([5.0, None], dtype=pd.ArrowDtype(pa.float32()))
    frame[upper] = pd.Series([9.0, 10.0], dtype=pd.ArrowDtype(pa.float64()))

    validated = validate_forecast_frame(frame, calendar=WEEKLY)

    for column in (ACTUAL_VALUE, POINT_FORECAST, lower, upper):
        assert validated[column].dtype == np.dtype("float64")
    assert np.isnan(validated.loc[0, ACTUAL_VALUE])
    assert np.isnan(validated.loc[1, lower])


def test_frame_validation_rejects_duplicate_keys_and_wrong_target_derivation() -> None:
    duplicate = pd.concat([_weekly_frame(), _weekly_frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ForecastFrameError, match="duplicate full row key"):
        validate_forecast_frame(duplicate, calendar=WEEKLY)

    wrong_target = _weekly_frame()
    wrong_target.loc[1, TARGET_TIMESTAMP] = pd.Timestamp("2026-01-20")
    with pytest.raises(ForecastFrameError, match="target timestamp"):
        validate_forecast_frame(wrong_target, calendar=WEEKLY)


def test_frame_validation_reports_the_first_offending_target_across_repeated_pairs() -> None:
    """Fan one advance per unique origin/step back to every row that shares it."""
    monthly = Calendar("MS").bind(pd.Timestamp("2026-01-01"))
    frame = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-a", "sku-b", "sku-a", "sku-b"], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(
                ["2026-02-01", "2026-02-01", "2026-01-01", "2026-01-01"]
            ),
            ACTUAL_VALUE: pd.Series([np.nan] * 4, dtype="float64"),
            POINT_FORECAST: pd.Series([1.0, 2.0, 3.0, 4.0], dtype="float64"),
            HORIZON_STEP: pd.Series([2, 2, 1, 1], dtype="int64"),
            ORIGIN: pd.to_datetime(["2026-01-01"] * 4),
            MODEL_NAME: pd.Series(["seasonal-naive"] * 4, dtype="string"),
        }
    )

    pd.testing.assert_frame_equal(validate_forecast_frame(frame, calendar=monthly), frame)

    mismatched = frame.copy(deep=True)
    mismatched.loc[3, TARGET_TIMESTAMP] = pd.Timestamp("2026-03-01")
    with pytest.raises(
        ForecastFrameError,
        match="row 3 has 2026-03-01 00:00:00, expected 2026-01-01 00:00:00",
    ):
        validate_forecast_frame(mismatched, calendar=monthly)


def test_frame_validation_rejects_off_calendar_origin() -> None:
    frame = _weekly_frame().iloc[[0]].copy()
    frame[ORIGIN] = pd.to_datetime(["2026-01-13"])
    frame[TARGET_TIMESTAMP] = pd.to_datetime(["2026-01-13"])
    with pytest.raises(ForecastFrameError, match="does not lie on calendar"):
        validate_forecast_frame(frame, calendar=WEEKLY)


@pytest.mark.parametrize(
    "columns",
    [("lower_0.9",), ("upper_0.9",), ("lower_0.8", "upper_0.9")],
)
def test_interval_columns_are_all_or_nothing(columns: tuple[str, ...]) -> None:
    frame = _weekly_frame()
    for column in columns:
        frame[column] = pd.Series([5.0, 6.0], dtype="float64")
    with pytest.raises(ForecastFrameError, match="complete lower/upper pairs"):
        validate_forecast_frame(frame, calendar=WEEKLY)


@pytest.mark.parametrize(
    "column", ["lower_0.90", "upper_not-a-level", "quantile_1e-1", "quantile_NaN"]
)
def test_optional_columns_require_canonical_finite_decimal_names(column: str) -> None:
    frame = _weekly_frame()
    frame[column] = pd.Series([1.0, 2.0], dtype="float64")
    with pytest.raises(ForecastFrameError, match="canonical|decimal"):
        validate_forecast_frame(frame, calendar=WEEKLY)


def test_frame_preserves_unregistered_extension_columns() -> None:
    frame = _weekly_frame()
    frame["adapter_note"] = pd.Series(["a", "b"], dtype="string")
    validated = validate_forecast_frame(frame, calendar=WEEKLY)
    pd.testing.assert_series_equal(validated["adapter_note"], frame["adapter_note"])


def test_frame_accepts_and_canonicalizes_flat_primitive_extensions() -> None:
    frame = _weekly_frame()
    frame["rank"] = pd.Series([1, 2], dtype="int16")
    frame["eligible"] = pd.Series([True, False], dtype="bool")
    frame["score"] = pd.Series([1.5, np.nan], dtype="float16")
    frame["event_time"] = pd.Series(
        pd.to_datetime(["2026-01-01", "2026-01-02"]).astype("datetime64[ms]")
    )
    frame["note"] = pd.Series(["a", "b"], dtype=pd.StringDtype(storage="python"))
    frame["nullable_rank"] = pd.Series([1, pd.NA], dtype="Int64")
    frame["arrow_score"] = pd.Series([1.5, None], dtype=pd.ArrowDtype(pa.float32()))
    frame["sparse_count"] = pd.Series([1, 0], dtype=pd.SparseDtype("int16", fill_value=0))

    validated = validate_forecast_frame(frame, calendar=WEEKLY)

    assert validated["rank"].dtype == np.dtype("int16")
    assert validated["eligible"].dtype == np.dtype("bool")
    assert validated["score"].dtype == np.dtype("float16")
    assert validated["event_time"].dtype == np.dtype("datetime64[ms]")
    assert validated["note"].dtype.storage == "pyarrow"
    assert str(validated["nullable_rank"].dtype) == "Int64"
    assert validated["arrow_score"].dtype == pd.ArrowDtype(pa.float32())
    assert validated["sparse_count"].dtype == np.dtype("int16")


@pytest.mark.parametrize(
    "extension",
    [
        pd.Series([object(), object()], dtype="object"),
        pd.Series([1 + 2j, 3 + 4j], dtype="complex128"),
    ],
)
def test_frame_rejects_nonprimitive_extension_dtypes(extension: pd.Series) -> None:
    frame = _weekly_frame()
    frame["adapter_extension"] = extension

    with pytest.raises(ForecastFrameError, match="flat transport-safe primitive"):
        validate_forecast_frame(frame, calendar=WEEKLY)


def test_frame_rejects_non_string_and_non_utf8_column_labels() -> None:
    integer_label = _weekly_frame()
    integer_label[7] = pd.Series([1, 2], dtype="int64")
    with pytest.raises(ForecastFrameError, match="column labels must be strings"):
        validate_forecast_frame(integer_label, calendar=WEEKLY)

    surrogate_label = _weekly_frame()
    surrogate_label.columns = pd.Index(
        ["\ud800" if column == MODEL_NAME else column for column in surrogate_label.columns],
        dtype="object",
    )
    with pytest.raises(ForecastFrameError, match="valid UTF-8"):
        validate_forecast_frame(surrogate_label, calendar=WEEKLY)


@pytest.mark.parametrize("level", [True, "NaN", "Infinity", "not-a-level"])
def test_column_helpers_reject_non_finite_or_non_decimal_levels(level: object) -> None:
    with pytest.raises(ForecastFrameError, match="finite decimal|valid decimal"):
        interval_columns(level)
    with pytest.raises(ForecastFrameError, match="finite decimal|valid decimal"):
        quantile_column(level)


def test_column_helpers_return_canonical_names() -> None:
    assert interval_columns("0.90") == ("lower_0.9", "upper_0.9")
    assert quantile_column("0.50") == "quantile_0.5"
