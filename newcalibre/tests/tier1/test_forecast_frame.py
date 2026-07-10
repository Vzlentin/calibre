"""Exercise the chapter 02 forecast-frame contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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


def test_required_columns_and_full_row_key_are_declared_explicitly() -> None:
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

    validated = validate_forecast_frame(frame, calendar_frequency="W-MON")

    pd.testing.assert_frame_equal(validated, frame)
    assert target_timestamp(
        pd.Timestamp("2026-01-12"), 1, calendar_frequency="W-MON"
    ) == pd.Timestamp("2026-01-12")
    assert target_timestamp(
        pd.Timestamp("2026-01-12"), 2, calendar_frequency="W-MON"
    ) == pd.Timestamp("2026-01-19")


def test_target_timestamp_advances_daily_without_shifting_step_one() -> None:
    origin = pd.Timestamp("2026-01-12")

    assert target_timestamp(origin, 1, calendar_frequency="D") == origin
    assert target_timestamp(origin, 4, calendar_frequency="D") == pd.Timestamp("2026-01-15")


def test_target_timestamp_keeps_step_one_on_an_off_grid_anchored_origin() -> None:
    origin = pd.Timestamp("2026-01-13")  # Tuesday, deliberately off W-MON.

    assert target_timestamp(origin, 1, calendar_frequency="W-MON") == origin
    assert target_timestamp(origin, 2, calendar_frequency="W-MON") == pd.Timestamp("2026-01-19")


def test_frame_validation_keeps_step_one_on_an_off_grid_anchored_origin() -> None:
    frame = _weekly_frame().iloc[[0]].copy()
    frame[ORIGIN] = pd.to_datetime(["2026-01-13"])
    frame[TARGET_TIMESTAMP] = pd.to_datetime(["2026-01-13"])

    validated = validate_forecast_frame(frame, calendar_frequency="W-MON")

    assert validated.loc[0, TARGET_TIMESTAMP] == pd.Timestamp("2026-01-13")


@pytest.mark.parametrize("column", REQUIRED_FRAME_COLUMNS)
def test_frame_validation_rejects_every_missing_required_column(column: str) -> None:
    frame = _weekly_frame().drop(columns=column)

    with pytest.raises(ForecastFrameError, match="missing required columns"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        (SERIES_KEY, pd.Series(["sku-a", "sku-a"], dtype="object")),
        (TARGET_TIMESTAMP, pd.Series([1, 2], dtype="int64")),
        (ACTUAL_VALUE, pd.Series([1.0, 2.0], dtype="float32")),
        (POINT_FORECAST, pd.Series(["7", "8"], dtype="string")),
        (HORIZON_STEP, pd.Series([1.0, 2.0], dtype="float64")),
        (ORIGIN, pd.Series(["2026-01-12", "2026-01-12"], dtype="string")),
        (MODEL_NAME, pd.Series(["seasonal-naive", "seasonal-naive"], dtype="object")),
    ],
)
def test_frame_validation_rejects_mistyped_required_columns(
    column: str, replacement: pd.Series[object]
) -> None:
    frame = _weekly_frame()
    frame[column] = replacement

    with pytest.raises(ForecastFrameError, match=column):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


def test_frame_validation_accepts_integer_widths_and_datetime_resolutions() -> None:
    frame = _weekly_frame()
    frame[HORIZON_STEP] = frame[HORIZON_STEP].astype("int32")
    frame[TARGET_TIMESTAMP] = frame[TARGET_TIMESTAMP].astype("datetime64[ns]")
    frame[ORIGIN] = frame[ORIGIN].astype("datetime64[ns]")

    validated = validate_forecast_frame(frame, calendar_frequency="W-MON")

    assert validated[HORIZON_STEP].dtype == np.dtype("int32")
    assert validated[TARGET_TIMESTAMP].dtype == np.dtype("datetime64[ns]")


@pytest.mark.parametrize("column", [TARGET_TIMESTAMP, ORIGIN])
def test_frame_validation_rejects_timezone_aware_timestamps(column: str) -> None:
    frame = _weekly_frame()
    frame[column] = pd.date_range("2026-01-12", periods=2, tz="UTC")

    with pytest.raises(ForecastFrameError, match="timezone-naive"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


def test_frame_validation_upcasts_integer_value_columns_without_mutating_input() -> None:
    lower, upper = interval_columns(0.9)
    quantile = quantile_column(0.5)
    frame = _weekly_frame()
    frame[ACTUAL_VALUE] = pd.Series([1, 2], dtype="int64")
    frame[POINT_FORECAST] = pd.Series([7, 8], dtype="int64")
    frame[lower] = pd.Series([5, 6], dtype="int64")
    frame[upper] = pd.Series([9, 10], dtype="int64")
    frame[quantile] = pd.Series([7, 8], dtype="int64")
    original = frame.copy(deep=True)

    validated = validate_forecast_frame(frame, calendar_frequency="W-MON")

    pd.testing.assert_frame_equal(frame, original)
    for column in (ACTUAL_VALUE, POINT_FORECAST, lower, upper, quantile):
        assert validated[column].dtype == np.dtype("float64")


def test_frame_validation_is_atomic_when_a_later_row_check_fails() -> None:
    lower, upper = interval_columns(0.9)
    frame = _weekly_frame()
    frame[ACTUAL_VALUE] = pd.Series([1, 2], dtype="int64")
    frame[lower] = pd.Series([5, 6], dtype="int64")
    frame[upper] = pd.Series([9, 10], dtype="int64")
    frame.loc[1, TARGET_TIMESTAMP] = pd.Timestamp("2026-01-20")
    original = frame.copy(deep=True)

    with pytest.raises(ForecastFrameError, match="target timestamp"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")

    pd.testing.assert_frame_equal(frame, original)


def test_frame_validation_preserves_unregistered_extension_columns() -> None:
    frame = _weekly_frame()
    frame["adapter_note"] = pd.Series(["a", "b"], dtype="string")
    frame["lowercase_note"] = pd.Series(["c", "d"], dtype="string")

    validated = validate_forecast_frame(frame, calendar_frequency="W-MON")

    pd.testing.assert_series_equal(validated["adapter_note"], frame["adapter_note"])
    pd.testing.assert_series_equal(validated["lowercase_note"], frame["lowercase_note"])


def test_frame_validation_rejects_duplicate_full_row_keys() -> None:
    frame = pd.concat([_weekly_frame(), _weekly_frame().iloc[[0]]], ignore_index=True)

    with pytest.raises(ForecastFrameError, match="full row key"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


def test_frame_validation_allows_same_series_origin_and_step_for_another_model() -> None:
    frame = _weekly_frame().iloc[[0]].copy()
    other_model = frame.copy()
    other_model[MODEL_NAME] = pd.Series(["other-model"], dtype="string")
    frame = pd.concat([frame, other_model], ignore_index=True)

    validated = validate_forecast_frame(frame, calendar_frequency="W-MON")

    assert len(validated) == 2


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (HORIZON_STEP, 0),
        (TARGET_TIMESTAMP, pd.Timestamp("2026-01-20")),
    ],
)
def test_frame_validation_rejects_invalid_row_derivation(column: str, value: object) -> None:
    frame = _weekly_frame()
    frame.loc[1, column] = value

    with pytest.raises(ForecastFrameError, match="horizon|target timestamp"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


def test_frame_validation_requires_explicit_weekly_anchor() -> None:
    with pytest.raises(ForecastFrameError, match="explicit anchor"):
        validate_forecast_frame(_weekly_frame(), calendar_frequency="W")


@pytest.mark.parametrize("frequency", ["", "not-a-frequency", 7])
def test_frame_validation_rejects_invalid_calendar_syntax(frequency: object) -> None:
    with pytest.raises(ForecastFrameError, match="non-empty string|invalid pandas"):
        validate_forecast_frame(
            _weekly_frame(),
            calendar_frequency=frequency,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("frequency", ["0D", "-1D", "0W-MON", "-1W-MON"])
def test_frame_validation_requires_a_forward_moving_calendar(frequency: str) -> None:
    with pytest.raises(ForecastFrameError, match="positive period"):
        validate_forecast_frame(_weekly_frame(), calendar_frequency=frequency)


def test_target_timestamp_rejects_timezone_aware_origin() -> None:
    with pytest.raises(ForecastFrameError, match="timezone-naive"):
        target_timestamp(pd.Timestamp("2026-01-12", tz="UTC"), 1, calendar_frequency="D")


@pytest.mark.parametrize(
    "columns",
    [
        ("lower_0.9",),
        ("upper_0.9",),
        ("lower_0.8", "upper_0.9"),
    ],
)
def test_interval_columns_are_all_or_nothing(columns: tuple[str, ...]) -> None:
    frame = _weekly_frame()
    for column in columns:
        frame[column] = pd.Series([5.0, 6.0], dtype="float64")

    with pytest.raises(ForecastFrameError, match="complete lower/upper pairs"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


@pytest.mark.parametrize(
    "column",
    [
        "lower_0.90",
        "upper_not-a-level",
        "quantile_1e-1",
        "quantile_NaN",
    ],
)
def test_optional_forecast_columns_require_canonical_level_names(column: str) -> None:
    frame = _weekly_frame()
    frame[column] = pd.Series([1.0, 2.0], dtype="float64")

    with pytest.raises(ForecastFrameError, match="canonical|decimal"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


def test_optional_forecast_columns_must_be_float64_or_integer_upcastable() -> None:
    lower, upper = interval_columns(0.9)
    frame = _weekly_frame()
    frame[lower] = pd.Series([5.0, 6.0], dtype="float64")
    frame[upper] = pd.Series([9.0, 10.0], dtype="float32")

    with pytest.raises(ForecastFrameError, match=upper):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


@pytest.mark.parametrize("dtype", ["float32", "bool"])
def test_point_forecast_rejects_non_float64_non_integer_dtypes(dtype: str) -> None:
    frame = _weekly_frame()
    frame[POINT_FORECAST] = pd.Series([True, False], dtype=dtype)

    with pytest.raises(ForecastFrameError, match=POINT_FORECAST):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


@pytest.mark.parametrize("column", ["lower0.9", "upper", "quantile-level"])
def test_optional_forecast_columns_reject_malformed_reserved_names(column: str) -> None:
    frame = _weekly_frame()
    frame[column] = pd.Series([1.0, 2.0], dtype="float64")

    with pytest.raises(ForecastFrameError, match="malformed reserved name"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")


def test_column_name_helpers_canonicalize_syntax_without_owning_level_admissibility() -> None:
    assert quantile_column(0) == "quantile_0"
    assert quantile_column("1.10") == "quantile_1.1"
    assert interval_columns("-0.10") == ("lower_-0.1", "upper_-0.1")


@pytest.mark.parametrize("level", [True, "NaN", "Infinity", "not-a-level"])
def test_column_name_helpers_reject_non_finite_or_non_decimal_levels(level: object) -> None:
    with pytest.raises(ForecastFrameError, match="finite decimal|valid decimal"):
        interval_columns(level)
    with pytest.raises(ForecastFrameError, match="finite decimal|valid decimal"):
        quantile_column(level)


def test_frame_validation_rejects_duplicate_column_labels() -> None:
    frame = _weekly_frame()
    frame.columns = [*frame.columns[:-1], ORIGIN]

    with pytest.raises(ForecastFrameError, match="duplicate column labels"):
        validate_forecast_frame(frame, calendar_frequency="W-MON")
