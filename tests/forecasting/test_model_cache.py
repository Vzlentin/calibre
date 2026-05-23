from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_base import CacheableAdapter, ModelAdapter
from calibre.forecasting.cache import ModelArtifactCache


class _CountingAdapter(CacheableAdapter, ModelAdapter):
    """Records every fit() call; pickle-based dump/load."""

    fit_calls = 0

    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._mean: float | None = None

    def fit(self, task: ForecastTask) -> None:
        type(self).fit_calls += 1
        self._mean = float(task.history[Y].mean())

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        assert self._mean is not None
        rows = []
        for h in range(1, task.horizon + 1):
            rows.append(
                {
                    UNIQUE_ID: task.unique_id,
                    DS: pd.Timestamp("2024-01-07") + pd.Timedelta(weeks=h),
                    Y_HAT: self._mean,
                    H: h,
                }
            )
        return pd.DataFrame(rows)

    def dump_state(self) -> bytes:
        return pickle.dumps(self._mean)

    def load_state(self, blob: bytes) -> None:
        self._mean = pickle.loads(blob)


def _task() -> ForecastTask:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["sku"] * 4,
            DS: pd.date_range("2024-01-01", periods=4, freq="W"),
            Y: [1.0, 2.0, 3.0, 4.0],
        }
    )
    return ForecastTask(history=history, horizon=2, model_config={"model": "Mean"})


@pytest.fixture(autouse=True)
def _reset_fit_counter():
    _CountingAdapter.fit_calls = 0
    yield
    _CountingAdapter.fit_calls = 0


def test_cache_miss_writes(tmp_path: Path) -> None:
    cache = ModelArtifactCache(str(tmp_path / "cache"))
    adapter = _CountingAdapter({"model": "Mean"})
    task = _task()

    fitted = adapter.fit_with_cache(task, cache)

    assert fitted is True
    assert _CountingAdapter.fit_calls == 1
    key = adapter.cache_key(task)
    assert cache.get(key) is not None


def test_cache_hit_skips_fit(tmp_path: Path) -> None:
    cache = ModelArtifactCache(str(tmp_path / "cache"))
    task = _task()

    first = _CountingAdapter({"model": "Mean"})
    first.fit_with_cache(task, cache)
    assert _CountingAdapter.fit_calls == 1

    second = _CountingAdapter({"model": "Mean"})
    fitted = second.fit_with_cache(task, cache)

    assert fitted is False
    assert _CountingAdapter.fit_calls == 1
    pred = second.predict(task)
    assert pred[Y_HAT].iloc[0] == pytest.approx(2.5)


def test_cache_key_changes_with_model_config(tmp_path: Path) -> None:
    task_a = _task()
    history = task_a.history
    task_b = ForecastTask(history=history, horizon=2, model_config={"model": "Other"})

    adapter = _CountingAdapter({"model": "ignored"})
    assert adapter.cache_key(task_a) != adapter.cache_key(task_b)


def test_cache_key_changes_with_history(tmp_path: Path) -> None:
    task_a = _task()
    history_b = task_a.history.assign(**{Y: [1.0, 2.0, 3.0, 99.0]})
    task_b = ForecastTask(history=history_b, horizon=2, model_config=task_a.model_config)

    adapter = _CountingAdapter({"model": "Mean"})
    assert adapter.cache_key(task_a) != adapter.cache_key(task_b)


def test_cache_key_changes_with_fit_affecting_task_fields() -> None:
    task = _task()
    adapter = _CountingAdapter({"model": "Mean"})
    future_x = pd.DataFrame(
        {
            UNIQUE_ID: ["sku", "sku"],
            DS: pd.date_range("2024-02-01", periods=2, freq="W"),
            "promo": [0.0, 1.0],
        }
    )

    variants = [
        ForecastTask(history=task.history, horizon=3, model_config=task.model_config),
        ForecastTask(
            history=task.history,
            horizon=task.horizon,
            model_config=task.model_config,
            future_x=future_x,
        ),
        ForecastTask(
            history=task.history,
            horizon=task.horizon,
            model_config=task.model_config,
            forecast_origin=pd.Timestamp("2024-02-01"),
        ),
        ForecastTask(
            history=task.history,
            horizon=task.horizon,
            model_config=task.model_config,
            task_group="group-a",
        ),
    ]

    base_key = adapter.cache_key(task)
    assert all(adapter.cache_key(variant) != base_key for variant in variants)


def test_fit_with_cache_no_cache_runs_fit() -> None:
    adapter = _CountingAdapter({"model": "Mean"})
    fitted = adapter.fit_with_cache(_task(), None)

    assert fitted is True
    assert _CountingAdapter.fit_calls == 1


def test_default_dump_state_round_trips_adapter_attributes() -> None:
    class _DefaultStateAdapter(CacheableAdapter, ModelAdapter):
        def __init__(self, model_config: dict) -> None:
            self._config = model_config
            self._seen = False

        def fit(self, task: ForecastTask) -> None:
            self._seen = True
            return None

        def predict(self, task: ForecastTask) -> pd.DataFrame:
            return pd.DataFrame()

    adapter = _DefaultStateAdapter({})
    adapter.fit(_task())
    restored = _DefaultStateAdapter({})
    restored.load_state(adapter.dump_state())

    assert restored._seen is True


def test_non_cacheable_adapter_has_no_cache_methods() -> None:
    """Adapters that do not mix in CacheableAdapter must not be cached implicitly."""

    class _NoCacheAdapter(ModelAdapter):
        def __init__(self, model_config: dict) -> None:
            self._config = model_config

        def fit(self, task: ForecastTask) -> None:
            return None

        def predict(self, task: ForecastTask) -> pd.DataFrame:
            return pd.DataFrame()

    adapter = _NoCacheAdapter({})
    assert not hasattr(adapter, "dump_state")
    assert not hasattr(adapter, "load_state")
    assert not hasattr(adapter, "fit_with_cache")


def test_cache_rejects_invalid_keys(tmp_path: Path) -> None:
    cache = ModelArtifactCache(str(tmp_path / "cache"))
    with pytest.raises(ValueError, match="Invalid cache key"):
        cache.put("../escape", b"x")
    with pytest.raises(ValueError, match="Invalid cache key"):
        cache.get("")
