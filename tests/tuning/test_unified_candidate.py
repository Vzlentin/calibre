from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import optuna
import pandas as pd
import pytest

import calibre.tuning.optimizer as optimizer
from calibre.conformal import SymmetricIntervalConfig, SymmetricIntervalRuntime
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, MODEL_NAME, UNIQUE_ID, Y_HAT, H, Y
from calibre.tuning.optimizer import (
    _apply_conformal_overrides,
    _apply_ordering_overrides,
    _evaluate_candidate,
    _resolve_candidate,
)
from calibre.tuning.task import LocalTuningTask, StudyConfig, TuningCandidate


class _ConstObjective:
    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        return float(frame["cost"].sum())


@dataclass(frozen=True, slots=True)
class _DataclassObjective:
    weight: float = 1.0

    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        return float(self.weight * frame["cost"].sum())


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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
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


def _resolved_frame(cost: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                UNIQUE_ID: "sku",
                DS: pd.Timestamp("2024-01-14"),
                Y: cost,
                Y_HAT: 0.0,
                H: 1,
                FORECAST_ORIGIN: pd.Timestamp("2024-01-07"),
                MODEL_NAME: "stub",
                "cost": cost,
            }
        ]
    )


def _task(objective: Any) -> LocalTuningTask:
    history = pd.DataFrame(
        {
            UNIQUE_ID: ["sku"],
            DS: [pd.Timestamp("2024-01-01")],
            Y: [1.0],
        }
    )

    def _space(trial: optuna.Trial) -> TuningCandidate:
        return TuningCandidate(
            model_config={"season_length": trial.suggest_categorical("season_length", [1, 2])}
        )

    return LocalTuningTask(
        unique_id="sku",
        history=history,
        horizon=1,
        base_model_config={"backend": "statsforecast", "model": "SeasonalNaive"},
        search_space=_space,
        actuals=history,
        origins=[pd.Timestamp("2024-01-14")],
        objective=objective,
        study_config=StudyConfig(n_trials=1, asha_grace_period=1),
    )


def test_search_space_returns_candidate() -> None:
    task = _task(_ConstObjective())
    candidate = task.search_space(optuna.trial.FixedTrial({"season_length": 2}))
    assert isinstance(candidate, TuningCandidate)
    assert candidate.model_config == {"season_length": 2}
    assert candidate.conformal_config == {}
    assert candidate.ordering_config == {}


def test_resolve_candidate_rejects_plain_dict() -> None:
    with pytest.raises(TypeError, match="TuningCandidate"):
        _resolve_candidate({"season_length": 1})


def test_conformal_params_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[SymmetricIntervalConfig] = []

    def _spy_from_state(config: SymmetricIntervalConfig, state: dict[str, Any]):
        captured.append(config)
        return SymmetricIntervalRuntime(config)

    monkeypatch.setattr(SymmetricIntervalRuntime, "from_state", staticmethod(_spy_from_state))
    _FakeBackendEngine.ledgers = [_resolved_frame(3.0)]
    monkeypatch.setattr(optimizer, "BackendEngine", _FakeBackendEngine)

    base_config = SymmetricIntervalConfig(
        method="aci", coverage=0.9, calibration_window=4, gamma=0.05
    )

    def _factory() -> SymmetricIntervalRuntime:
        return SymmetricIntervalRuntime(base_config)

    history = pd.DataFrame({UNIQUE_ID: ["sku"], DS: [pd.Timestamp("2024-01-01")], Y: [1.0]})

    def _space(trial: optuna.Trial) -> TuningCandidate:
        return TuningCandidate(
            model_config={"season_length": 1},
            conformal_config={"coverage": 0.8, "gamma": 0.1},
        )

    task = LocalTuningTask(
        unique_id="sku",
        history=history,
        horizon=1,
        base_model_config={"backend": "statsforecast", "model": "SeasonalNaive"},
        search_space=_space,
        actuals=history,
        origins=[pd.Timestamp("2024-01-14")],
        objective=_ConstObjective(),
        conformal_runtime_factory=_factory,
        study_config=StudyConfig(n_trials=1),
    )

    candidate = task.search_space(optuna.trial.FixedTrial({}))
    _evaluate_candidate(task, candidate, [pd.Timestamp("2024-01-14")])

    assert captured, "from_state was not invoked"
    applied = captured[-1]
    assert applied.coverage == 0.8
    assert applied.gamma == 0.1
    assert applied.method == "aci"
    assert applied.calibration_window == 4


def test_ordering_overrides_replace_dataclass_fields() -> None:
    base = _DataclassObjective(weight=1.0)
    overridden = _apply_ordering_overrides(base, {"weight": 5.0})
    assert overridden.weight == 5.0


def test_ordering_overrides_reject_non_dataclass_objective() -> None:
    with pytest.raises(TypeError, match="dataclass"):
        _apply_ordering_overrides(_ConstObjective(), {"weight": 5.0})


def test_apply_conformal_overrides_returns_input_when_empty() -> None:
    base = SymmetricIntervalConfig(method="aci", coverage=0.9, calibration_window=4, gamma=0.05)
    assert _apply_conformal_overrides(base, {}) is base


def test_ordering_params_propagate_to_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    weights_seen: list[float] = []

    @dataclass(frozen=True, slots=True)
    class _RecordingObjective:
        weight: float = 1.0

        def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
            weights_seen.append(self.weight)
            return float(self.weight * frame["cost"].sum())

    _FakeBackendEngine.ledgers = [_resolved_frame(2.0)]
    monkeypatch.setattr(optimizer, "BackendEngine", _FakeBackendEngine)

    history = pd.DataFrame({UNIQUE_ID: ["sku"], DS: [pd.Timestamp("2024-01-01")], Y: [1.0]})

    def _space(trial: optuna.Trial) -> TuningCandidate:
        return TuningCandidate(
            model_config={"season_length": 1},
            ordering_config={"weight": 4.0},
        )

    task = LocalTuningTask(
        unique_id="sku",
        history=history,
        horizon=1,
        base_model_config={"backend": "statsforecast", "model": "SeasonalNaive"},
        search_space=_space,
        actuals=history,
        origins=[pd.Timestamp("2024-01-14")],
        objective=_RecordingObjective(weight=1.0),
        study_config=StudyConfig(n_trials=1),
    )

    candidate = task.search_space(optuna.trial.FixedTrial({}))
    value = _evaluate_candidate(task, candidate, [pd.Timestamp("2024-01-14")])

    assert weights_seen == [4.0]
    assert value == pytest.approx(4.0 * 2.0)
