from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import optuna
import pandas as pd

import calibre.tuning.optimizer as optimizer
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.tuning.optimizer import _OBJECTIVE_METRIC, _ORIGIN_INDEX, _evaluate_candidate
from calibre.tuning.task import StudyConfig, TuningCandidate, TuningTask


def _constant_space(trial: optuna.Trial) -> TuningCandidate:
    return TuningCandidate(model_config={"season_length": 1})


class _SumCostObjective:
    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        return float(frame["cost"].sum())


class _FakeLedger:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def to_df(self) -> pd.DataFrame:
        return self._frame.copy()


class _FakeResult:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.ledger = _FakeLedger(frame)


class _FakeBackendEngine:
    ledgers: list[pd.DataFrame] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __enter__(self) -> _FakeBackendEngine:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def iter_origins(self, tasks, actuals, origins):
        for frame in self.ledgers:
            yield _FakeResult(frame)

    def close(self) -> None:
        pass


def _task(origins: list[pd.Timestamp]) -> TuningTask:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["sku"],
            DS: [pd.Timestamp("2024-01-01")],
            Y: [1.0],
        }
    )
    return TuningTask(
        unique_id="sku",
        history=history,
        horizon=1,
        base_model_config={"backend": "statsforecast", "model": "SeasonalNaive"},
        search_space=_constant_space,
        actuals=history,
        origins=origins,
        objective=_SumCostObjective(),
        study_config=StudyConfig(n_trials=1, asha_grace_period=1),
    )


def _resolved_frame(costs: list[float]) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-07")
    rows = []
    for idx, cost in enumerate(costs, start=1):
        ds = start + pd.Timedelta(weeks=idx)
        rows.append(
            {
                UNIQUE_ID: "sku",
                DS: ds,
                Y: float(cost),
                Y_HAT: 0.0,
                H: idx,
                FORECAST_ORIGIN: start,
                MODEL_NAME: "stub",
                "cost": float(cost),
            }
        )
    return pd.DataFrame(rows)


def test_total_cost_accumulates_across_origins(monkeypatch) -> None:
    origins = [pd.Timestamp("2024-01-14"), pd.Timestamp("2024-01-21")]
    task = _task(origins)
    _FakeBackendEngine.ledgers = [
        _resolved_frame([10.0]),
        _resolved_frame([10.0, 20.0]),
    ]
    monkeypatch.setattr(optimizer, "BackendEngine", _FakeBackendEngine)

    value = _evaluate_candidate(task, TuningCandidate(model_config={"season_length": 1}), origins)

    assert value == 30.0


def test_intermediate_metric_matches_final(monkeypatch, tmp_path) -> None:
    origins = [pd.Timestamp("2024-01-14"), pd.Timestamp("2024-01-21")]
    reports: list[dict[str, float | int]] = []
    _FakeBackendEngine.ledgers = [
        _resolved_frame([5.0]),
        _resolved_frame([5.0, 7.0]),
    ]
    monkeypatch.setattr(optimizer, "BackendEngine", _FakeBackendEngine)
    monkeypatch.setattr(
        optimizer,
        "acquire_ray_runtime",
        lambda **kwargs: SimpleNamespace(
            ray=SimpleNamespace(put=lambda value: value),
            release=lambda: None,
        ),
    )

    from ray import tune

    class _FakeResults(list):
        def get_best_result(self, **kwargs):
            return self[0]

    class _FakeResultGridEntry:
        error = None
        config = {"season_length": 1}

        @property
        def metrics(self):
            return reports[-1]

    class _FakeTuner:
        def __init__(self, trainable, **kwargs) -> None:
            self.trainable = trainable

        def fit(self):
            self.trainable({"season_length": 1})
            return _FakeResults([_FakeResultGridEntry()])

    monkeypatch.setattr(tune, "with_resources", lambda trainable, resources: trainable)
    monkeypatch.setattr(
        tune,
        "with_parameters",
        lambda trainable, **kwargs: lambda config: trainable(config, **kwargs),
    )
    monkeypatch.setattr(tune, "report", lambda metrics: reports.append(dict(metrics)))
    monkeypatch.setattr(tune, "Tuner", _FakeTuner)

    task = _task(origins)
    task = replace(
        task,
        study_config=replace(task.study_config, tune_storage_path=str(tmp_path / "ray-tune")),
    )

    result = optimizer.optimize_task(task)

    assert result["season_length"] == 1
    assert [report[_OBJECTIVE_METRIC] for report in reports] == [5.0, 12.0]
    assert reports[-1][_OBJECTIVE_METRIC] == 12.0
    assert reports[-1][_ORIGIN_INDEX] == 2
