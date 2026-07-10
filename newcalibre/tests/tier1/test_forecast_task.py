"""Exercise construction-time temporal hygiene for forecast tasks."""

from __future__ import annotations

from typing import cast

import pandas as pd
import pytest

from newcalibre.domain.forecast_task import (
    HISTORY_TIMESTAMP,
    ForecastTask,
    ForecastTaskError,
)

pytestmark = pytest.mark.tier1


def _history(timestamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_key": pd.Series(["sku-a"] * len(timestamps), dtype="string"),
            HISTORY_TIMESTAMP: pd.to_datetime(timestamps),
            "value": pd.Series(range(1, len(timestamps) + 1), dtype="float64"),
        }
    )


def test_task_accepts_history_strictly_before_origin_at_construction() -> None:
    task = ForecastTask(
        history=_history(["2026-01-05", "2026-01-12"]),
        horizon=2,
        origin=pd.Timestamp("2026-01-19"),
        calendar_frequency="W-MON",
        model_config={"backend": "seasonal-naive"},
    )

    assert task.origin == pd.Timestamp("2026-01-19")
    assert task.horizon == 2
    assert task.calendar_frequency == "W-MON"


@pytest.mark.parametrize("timestamp", ["2026-01-19", "2026-01-26"])
def test_task_rejects_history_at_or_after_origin(timestamp: str) -> None:
    with pytest.raises(ForecastTaskError, match="strictly before origin"):
        ForecastTask(
            history=_history(["2026-01-12", timestamp]),
            horizon=2,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W-MON",
            model_config={"backend": "seasonal-naive"},
        )


def test_task_checks_every_history_position_not_only_the_last() -> None:
    with pytest.raises(ForecastTaskError, match="strictly before origin"):
        ForecastTask(
            history=_history(["2026-01-19", "2026-01-12"]),
            horizon=2,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W-MON",
            model_config={"backend": "seasonal-naive"},
        )


def test_task_rejects_missing_history_timestamp_as_not_strictly_pre_origin() -> None:
    history = _history(["2026-01-12"])
    history.loc[0, HISTORY_TIMESTAMP] = pd.NaT

    with pytest.raises(ForecastTaskError, match="strictly before origin"):
        ForecastTask(
            history=history,
            horizon=2,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W-MON",
            model_config={"backend": "seasonal-naive"},
        )


def test_task_rejects_one_future_row_hidden_in_a_multi_series_history() -> None:
    history = pd.DataFrame(
        {
            "series_key": pd.Series(["sku-a", "sku-b", "sku-b"], dtype="string"),
            HISTORY_TIMESTAMP: pd.to_datetime(["2026-01-05", "2026-01-12", "2026-01-26"]),
            "value": pd.Series([1.0, 2.0, 3.0], dtype="float64"),
        }
    )

    with pytest.raises(ForecastTaskError, match="strictly before origin"):
        ForecastTask(
            history=history,
            horizon=2,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W-MON",
            model_config={},
        )


def test_task_requires_pandas_timestamp_origin() -> None:
    with pytest.raises(ForecastTaskError, match="pandas Timestamp"):
        ForecastTask(
            history=_history(["2026-01-12"]),
            horizon=1,
            origin="2026-01-19",  # type: ignore[arg-type]
            calendar_frequency="W-MON",
            model_config={},
        )


def test_task_requires_timezone_naive_datetime_history() -> None:
    history = _history(["2026-01-12"])
    history[HISTORY_TIMESTAMP] = pd.Series(["2026-01-12"], dtype="string")

    with pytest.raises(ForecastTaskError, match="timezone-naive"):
        ForecastTask(
            history=history,
            horizon=1,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W-MON",
            model_config={},
        )


def test_task_accepts_explicit_nanosecond_history() -> None:
    history = _history(["2026-01-12"])
    history[HISTORY_TIMESTAMP] = history[HISTORY_TIMESTAMP].astype("datetime64[ns]")

    task = ForecastTask(
        history=history,
        horizon=1,
        origin=pd.Timestamp("2026-01-19"),
        calendar_frequency="W-MON",
        model_config={},
    )

    assert task.history[HISTORY_TIMESTAMP].dtype == "datetime64[ns]"


def test_task_rejects_timezone_aware_history_and_origin() -> None:
    history = _history(["2026-01-12"])
    history[HISTORY_TIMESTAMP] = pd.date_range("2026-01-12", periods=1, tz="UTC")

    with pytest.raises(ForecastTaskError, match="timezone-naive"):
        ForecastTask(
            history=history,
            horizon=1,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W-MON",
            model_config={},
        )

    with pytest.raises(ForecastTaskError, match="timezone-naive"):
        ForecastTask(
            history=_history(["2026-01-12"]),
            horizon=1,
            origin=pd.Timestamp("2026-01-19", tz="UTC"),
            calendar_frequency="W-MON",
            model_config={},
        )


def test_task_requires_history_timestamp_column() -> None:
    with pytest.raises(ForecastTaskError, match=HISTORY_TIMESTAMP):
        ForecastTask(
            history=_history(["2026-01-12"]).drop(columns=HISTORY_TIMESTAMP),
            horizon=1,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W-MON",
            model_config={},
        )


@pytest.mark.parametrize("horizon", [0, -1, True])
def test_task_requires_a_positive_integer_horizon(horizon: object) -> None:
    with pytest.raises(ForecastTaskError, match="positive integer"):
        ForecastTask(
            history=_history(["2026-01-12"]),
            horizon=horizon,  # type: ignore[arg-type]
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W-MON",
            model_config={},
        )


def test_task_requires_an_explicit_weekly_anchor() -> None:
    with pytest.raises(ForecastTaskError, match="explicit anchor"):
        ForecastTask(
            history=_history(["2026-01-12"]),
            horizon=1,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency="W",
            model_config={},
        )


@pytest.mark.parametrize("frequency", ["", "not-a-frequency", 7])
def test_task_rejects_invalid_calendar_syntax(frequency: object) -> None:
    with pytest.raises(ForecastTaskError, match="non-empty string|invalid pandas"):
        ForecastTask(
            history=_history(["2026-01-12"]),
            horizon=1,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency=frequency,  # type: ignore[arg-type]
            model_config={},
        )


@pytest.mark.parametrize("frequency", ["0D", "-1D", "0W-MON", "-1W-MON"])
def test_task_requires_a_forward_moving_calendar(frequency: str) -> None:
    with pytest.raises(ForecastTaskError, match="positive period"):
        ForecastTask(
            history=_history(["2026-01-12"]),
            horizon=1,
            origin=pd.Timestamp("2026-01-19"),
            calendar_frequency=frequency,
            model_config={},
        )


def test_task_defensively_copies_history_at_and_after_construction() -> None:
    history = _history(["2026-01-12"])
    task = ForecastTask(
        history=history,
        horizon=1,
        origin=pd.Timestamp("2026-01-19"),
        calendar_frequency="W-MON",
        model_config={},
    )

    history.loc[0, HISTORY_TIMESTAMP] = pd.Timestamp("2026-01-19")

    exposed_history = task.history
    exposed_history.loc[0, HISTORY_TIMESTAMP] = pd.Timestamp("2026-01-19")

    assert task.history.loc[0, HISTORY_TIMESTAMP] == pd.Timestamp("2026-01-12")


def test_task_defensively_copies_model_configuration_at_and_after_construction() -> None:
    model_config: dict[str, object] = {"backend": "seasonal-naive", "nested": {"m": 7}}
    task = ForecastTask(
        history=_history(["2026-01-12"]),
        horizon=1,
        origin=pd.Timestamp("2026-01-19"),
        calendar_frequency="W-MON",
        model_config=model_config,
    )

    cast(dict[str, int], model_config["nested"])["m"] = 1
    exposed_config = task.model_config
    cast(dict[str, int], exposed_config["nested"])["m"] = 2

    assert cast(dict[str, int], task.model_config["nested"])["m"] == 7
