"""Exercise the source-closed seasonal-naive first brick.

Structural, schema, and rejection assertions are tolerance class 1. The
``m = 7`` lookup is hand-derived fixture arithmetic (class 2). Deterministic
serialized output is same-engine byte identity (class 4).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
import pytest

from newcalibre.domain import (
    ACTUAL_VALUE,
    HISTORY_TIMESTAMP,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    Calendar,
    ForecastTask,
    Panel,
    Scope,
    validate_forecast_frame,
)
from newcalibre.forecasting import (
    AdapterCapabilityError,
    AdapterConfigurationError,
    AdapterDataError,
    AdapterLifecycleError,
    ForecastAdapter,
    SeasonalNaiveAdapter,
    resolve_adapter,
)

pytestmark = pytest.mark.tier1

ORIGIN_TIMESTAMP = pd.Timestamp("2026-01-15")


def _config(**overrides: object) -> dict[str, object]:
    return {
        "backend": "seasonal-naive",
        "m": 7,
        "model_name": "daily-snaive-7",
        **overrides,
    }


def _history(
    series_values: Mapping[str, list[float]], *, start: str = "2026-01-01"
) -> pd.DataFrame:
    keys: list[str] = []
    timestamps: list[pd.Timestamp] = []
    values: list[float] = []
    for series_key, observations in series_values.items():
        calendar = pd.date_range(start, periods=len(observations), freq="D")
        keys.extend([series_key] * len(observations))
        timestamps.extend(calendar.tolist())
        values.extend(observations)
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(keys, dtype="string"),
            HISTORY_TIMESTAMP: pd.Series(pd.to_datetime(timestamps)),
            "value": pd.Series(values, dtype="float64"),
        }
    )


def _task(
    history: pd.DataFrame,
    *,
    horizon: int = 4,
    config: Mapping[str, object] | None = None,
) -> ForecastTask:
    model_config = dict(config or _config())
    return Panel.from_frame(history, calendar=Calendar("D")).forecast_tasks(
        horizon=horizon,
        origin=ORIGIN_TIMESTAMP,
        scope=Scope.GLOBAL,
        model_config=model_config,
    )[0]


def _adapter(config: Mapping[str, object] | None = None) -> ForecastAdapter:
    return resolve_adapter(dict(config or _config()))


def _sorted_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values([SERIES_KEY, HORIZON_STEP], kind="stable").reset_index(drop=True)


def _forecast_bytes(task: ForecastTask, config: Mapping[str, object]) -> bytes:
    adapter = _adapter(config)
    adapter.fit(task)
    return adapter.predict(task).to_csv(index=False, lineterminator="\n").encode()


def test_m7_walkthrough_emits_the_hand_checkable_validated_frame() -> None:
    task = _task(_history({"sku-a": [float(value) for value in range(1, 15)]}))
    adapter = _adapter()
    adapter.fit(task)

    frame = adapter.predict(task)

    pd.testing.assert_frame_equal(
        frame,
        validate_forecast_frame(frame, calendar=task.calendar),
    )
    assert frame[SERIES_KEY].tolist() == ["sku-a"] * 4
    assert frame[TARGET_TIMESTAMP].tolist() == [
        pd.Timestamp("2026-01-15"),
        pd.Timestamp("2026-01-16"),
        pd.Timestamp("2026-01-17"),
        pd.Timestamp("2026-01-18"),
    ]
    assert frame[POINT_FORECAST].tolist() == [8.0, 9.0, 10.0, 11.0]
    assert frame[HORIZON_STEP].tolist() == [1, 2, 3, 4]
    assert frame[ORIGIN].tolist() == [ORIGIN_TIMESTAMP] * 4
    assert frame[MODEL_NAME].tolist() == ["daily-snaive-7"] * 4
    assert frame[ACTUAL_VALUE].isna().all()
    assert frame[POINT_FORECAST].dtype == np.dtype("float64")


def test_horizons_longer_than_one_season_repeat_the_same_phase_lookup() -> None:
    task = _task(
        _history({"sku-a": [float(value) for value in range(1, 15)]}),
        horizon=9,
    )
    adapter = _adapter()
    adapter.fit(task)

    frame = adapter.predict(task)

    assert frame[POINT_FORECAST].tolist() == [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 8.0, 9.0]


def test_predict_requires_a_successful_fit() -> None:
    task = _task(_history({"sku-a": [float(value) for value in range(1, 15)]}))

    with pytest.raises(AdapterLifecycleError, match="successful fit"):
        _adapter().predict(task)


@pytest.mark.parametrize(
    "overrides",
    [
        {"m": 2},
        {"model_name": "task-selected-name"},
        {"quantile_levels": [0.5]},
        {"censoring_aware": True},
    ],
)
def test_fit_and_predict_reject_a_task_with_different_model_configuration(
    overrides: dict[str, object],
) -> None:
    config = _config()
    history = _history({"sku-a": [float(value) for value in range(1, 15)]})
    mismatched_task = _task(history, config={**config, **overrides})
    unfitted_adapter = _adapter(config)

    with pytest.raises(AdapterConfigurationError, match="must match"):
        unfitted_adapter.fit(mismatched_task)
    with pytest.raises(AdapterLifecycleError, match="successful fit"):
        unfitted_adapter.predict(mismatched_task)

    fitted_task = _task(history, config=config)
    fitted_adapter = _adapter(config)
    fitted_adapter.fit(fitted_task)
    with pytest.raises(AdapterConfigurationError, match="must match"):
        fitted_adapter.predict(mismatched_task)


@pytest.mark.parametrize(
    ("history", "error_pattern"),
    [
        (
            _history({"sku-a": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]}, start="2026-01-09"),
            "complete season.*missing",
        ),
        (
            _history({"sku-a": [float(value) for value in range(1, 15)]}).drop(index=9),
            "2026-01-10",
        ),
    ],
    ids=["short-season", "missing-phase"],
)
def test_predict_fails_loudly_for_short_or_gapped_season(
    history: pd.DataFrame, error_pattern: str
) -> None:
    task = _task(history)
    adapter = _adapter()
    adapter.fit(task)

    with pytest.raises(AdapterDataError, match=error_pattern):
        adapter.predict(task)


def test_predict_requires_the_complete_season_not_only_requested_phases() -> None:
    history = _history({"sku-a": [float(value) for value in range(1, 15)]})
    history = history[history[HISTORY_TIMESTAMP] != pd.Timestamp("2026-01-14")]
    task = _task(history, horizon=1)
    adapter = _adapter()
    adapter.fit(task)

    with pytest.raises(AdapterDataError, match="2026-01-14"):
        adapter.predict(task)


def test_predict_treats_a_missing_phase_value_as_missing_history() -> None:
    history = _history({"sku-a": [float(value) for value in range(1, 15)]})
    history.loc[history[HISTORY_TIMESTAMP] == pd.Timestamp("2026-01-10"), "value"] = np.nan
    task = _task(history)
    adapter = _adapter()
    adapter.fit(task)

    with pytest.raises(AdapterDataError, match="complete season.*2026-01-10"):
        adapter.predict(task)


@pytest.mark.parametrize(
    "predict_history",
    [
        _history({"sku-a": [9.0, 10.0, 11.0, 12.0, 13.0, 14.0]}, start="2026-01-09"),
        _history({"sku-a": [float(value) for value in range(1, 15)]}).assign(
            value=lambda frame: frame["value"].where(
                frame[HISTORY_TIMESTAMP] != pd.Timestamp("2026-01-08"),
                999.0,
            )
        ),
    ],
    ids=["shorter-history", "changed-retained-value"],
)
def test_predict_rejects_history_that_differs_from_the_fitted_season(
    predict_history: pd.DataFrame,
) -> None:
    config = _config()
    fit_task = _task(
        _history({"sku-a": [float(value) for value in range(1, 15)]}),
        config=config,
    )
    predict_task = _task(predict_history, config=config)
    adapter = _adapter(config)
    adapter.fit(fit_task)

    with pytest.raises(AdapterDataError, match="retained season must match"):
        adapter.predict(predict_task)


def test_adapter_is_scope_blind_for_one_or_many_series() -> None:
    config = _config()
    values_by_series = {
        "sku-b": [float(value) for value in range(101, 115)],
        "sku-a": [float(value) for value in range(1, 15)],
    }
    global_task = _task(
        _history(values_by_series),
        config=config,
    )
    global_adapter = _adapter(config)
    global_adapter.fit(global_task)

    global_frame = global_adapter.predict(global_task)
    local_frames: list[pd.DataFrame] = []
    for series_key, values in values_by_series.items():
        local_task = _task(_history({series_key: values}), config=config)
        local_adapter = _adapter(config)
        local_adapter.fit(local_task)
        local_frames.append(local_adapter.predict(local_task))

    pd.testing.assert_frame_equal(
        _sorted_frame(global_frame),
        _sorted_frame(pd.concat(local_frames, ignore_index=True)),
    )
    assert set(global_frame[SERIES_KEY]) == set(values_by_series)


def test_predictions_are_byte_deterministic_for_the_same_task_and_config() -> None:
    config = _config()
    task = _task(
        _history(
            {
                "sku-b": [float(value) for value in range(101, 115)],
                "sku-a": [float(value) for value in range(1, 15)],
            }
        ),
        config=config,
    )
    changed_history = _history(
        {
            "sku-b": [float(value) for value in range(101, 115)],
            "sku-a": [float(value) for value in range(1, 15)],
        }
    )
    changed_history.loc[
        (changed_history[SERIES_KEY] == "sku-a")
        & (changed_history[HISTORY_TIMESTAMP] == pd.Timestamp("2026-01-08")),
        "value",
    ] = 999.0
    changed_task = _task(changed_history, config=config)

    first = _forecast_bytes(task, config)
    second = _forecast_bytes(task, config)
    changed = _forecast_bytes(changed_task, config)

    assert first == second
    assert changed != first


def test_predictions_depend_only_on_the_retained_last_season() -> None:
    baseline = [float(value) for value in range(1, 15)]
    changed_early_history = [1001.0, 1002.0, 1003.0, 1004.0, 1005.0, 1006.0, 1007.0, *baseline[7:]]
    baseline_task = _task(_history({"sku-a": baseline}))
    changed_task = _task(_history({"sku-a": changed_early_history}))
    baseline_adapter = _adapter()
    changed_adapter = _adapter()
    baseline_adapter.fit(baseline_task)
    changed_adapter.fit(changed_task)

    pd.testing.assert_frame_equal(
        baseline_adapter.predict(baseline_task),
        changed_adapter.predict(changed_task),
    )


def test_fit_retains_exactly_the_final_season_and_no_whole_task_or_frame() -> None:
    task = _task(_history({"sku-a": [float(value) for value in range(1, 15)]}))
    adapter = SeasonalNaiveAdapter(_config())
    adapter.fit(task)

    expected = tuple((pd.Timestamp(f"2026-01-{day:02d}"), float(day)) for day in range(8, 15))
    assert adapter._season_by_series == {"sku-a": expected}
    assert not any(
        isinstance(value, (ForecastTask, pd.DataFrame)) for value in vars(adapter).values()
    )


def test_adapter_preserves_a_defensive_snapshot_of_nested_configuration() -> None:
    config = _config(quantile_levels=[])
    adapter = _adapter(config)
    task = _task(
        _history({"sku-a": [float(value) for value in range(1, 15)]}),
        config=config,
    )
    levels = config["quantile_levels"]
    assert isinstance(levels, list)
    cast(list[float], levels).append(0.5)

    adapter.fit(task)

    assert adapter.predict(task)[POINT_FORECAST].tolist() == [8.0, 9.0, 10.0, 11.0]


def test_irrelevant_json_metadata_does_not_affect_effective_config_comparison() -> None:
    config = _config(metadata={"source": "fixture", "weights": [1.0, None]})
    task = _task(
        _history({"sku-a": [float(value) for value in range(1, 15)]}),
        config=config,
    )
    adapter = _adapter(config)

    adapter.fit(task)

    assert adapter.predict(task)[POINT_FORECAST].tolist() == [8.0, 9.0, 10.0, 11.0]


def test_weekly_anchored_calendar_uses_the_same_phase_lookup() -> None:
    config = _config(m=4, model_name="weekly-snaive-4")
    history = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku-a"] * 8, dtype="string"),
            HISTORY_TIMESTAMP: pd.date_range("2026-01-05", periods=8, freq="W-MON"),
            "value": pd.Series(range(1, 9), dtype="float64"),
        }
    )
    task = Panel.from_frame(history, calendar=Calendar("W-MON")).forecast_tasks(
        horizon=3,
        origin=pd.Timestamp("2026-03-02"),
        scope=Scope.GLOBAL,
        model_config=config,
    )[0]
    adapter = _adapter(config)
    adapter.fit(task)

    frame = adapter.predict(task)

    assert frame[POINT_FORECAST].tolist() == [5.0, 6.0, 7.0]
    assert frame[TARGET_TIMESTAMP].tolist() == [
        pd.Timestamp("2026-03-02"),
        pd.Timestamp("2026-03-09"),
        pd.Timestamp("2026-03-16"),
    ]


def test_every_undeclared_operation_fails_with_a_capability_error() -> None:
    task = _task(_history({"sku-a": [float(value) for value in range(1, 15)]}))
    adapter = _adapter()
    adapter.fit(task)

    with pytest.raises(AdapterCapabilityError, match="fitted_values"):
        adapter.fitted_values(task)
    with pytest.raises(AdapterCapabilityError, match="artifact_persistence"):
        adapter.dump_state()
    with pytest.raises(AdapterCapabilityError, match="artifact_persistence"):
        adapter.load_state(b"unused")
    with pytest.raises(AdapterCapabilityError, match="incremental_update"):
        adapter.update(task)


def test_collecting_fitted_values_fails_at_fit_instead_of_degrading() -> None:
    task = _task(_history({"sku-a": [float(value) for value in range(1, 15)]}))

    with pytest.raises(AdapterCapabilityError, match="fitted_values"):
        _adapter().fit(task, collect_fitted_values=True)


def test_failed_fit_does_not_satisfy_the_predict_lifecycle() -> None:
    config = _config(quantile_levels=[0.5])
    task = _task(
        _history({"sku-a": [float(value) for value in range(1, 15)]}),
        config=config,
    )
    adapter = _adapter(config)

    with pytest.raises(AdapterCapabilityError, match="native_quantiles"):
        adapter.fit(task)
    with pytest.raises(AdapterLifecycleError, match="successful fit"):
        adapter.predict(task)


@pytest.mark.parametrize(
    ("config", "capability"),
    [
        (_config(quantile_levels=[0.5]), "native_quantiles"),
        (_config(censoring_aware=True), "censoring_aware_fit"),
    ],
)
def test_unsupported_fit_behaviors_fail_loudly(
    config: Mapping[str, object], capability: str
) -> None:
    task = _task(
        _history({"sku-a": [float(value) for value in range(1, 15)]}),
        config=config,
    )

    with pytest.raises(AdapterCapabilityError, match=capability):
        _adapter(config).fit(task)
