"""Tests for ``calibre run --tune``: study-surface parity + a small real run.

The structural assertion pins the *study-determining surface* the CLI shares
with the VN2 benchmark — objective + cost fractile, sampler seed, budget,
origins-count / ASHA ``max_t``, ASHA grace period, freq — plus the
``quantile_alpha`` search dimension, without running a real search. Full
search-space equality is intentionally NOT asserted: the benchmark carries
VN2-only glue (e.g. ``lag_set_idx`` → ``HPO_LAG_SETS``) that the dataset-general
CLI does not, so the shared, runtime-coherent dimension is ``quantile_alpha``.
The small-run check drives a real tiny tune end-to-end and confirms the
discovered quantile is a configured choice (the run returned, so the study had a
finite objective). Tuning is wiring-faithful, not bit-exact: Ray + ASHA make the
discovered fractile non-reproducible even seeded, so no literal value is asserted.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from benchmarks.vn2.config import HPO_COST_OPTIMAL_TAU, HPO_SEARCH_SPACE
from calibre.cli.commands import run_tune
from calibre.cli.config import load_config_from_mapping
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter
from calibre.tuning import CumulativePinball

# The VN2 cost struct: Cu = shortage = 1.0, Co = holding = 0.2.
_VN2_UNDERAGE = 1.0
_VN2_OVERAGE = 0.2
_BENCHMARK_SEED = 42


class _TuneVN2Adapter:
    """Synthetic 2-series weekly panel carrying the VN2 cost struct.

    Long enough (and seasonal enough) to fit a small global LGBM through the
    real tune path, with critical_ratio == HPO_COST_OPTIMAL_TAU so the derived
    objective fractile matches the benchmark constant.
    """

    def name(self) -> str:
        return "tune_vn2"

    def load(self, path: str, **kwargs: Any) -> DatasetBundle:
        del path, kwargs
        dates = pd.date_range("2024-01-01", periods=20, freq="W-MON")
        history = pd.concat(
            [
                pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [10.0, 20.0, 30.0, 40.0] * 5}),
                pd.DataFrame({UNIQUE_ID: "B", DS: dates, Y: [5.0, 15.0, 25.0, 35.0] * 5}),
            ],
            ignore_index=True,
        )
        return DatasetBundle(
            history=history,
            future_x=None,
            costs=CostStruct(
                underage_cost=_VN2_UNDERAGE,
                overage_cost=_VN2_OVERAGE,
                holding_cost=_VN2_OVERAGE,
                shortage_cost=_VN2_UNDERAGE,
            ),
            hierarchy=None,
            censoring=None,
        )


register_dataset_adapter("tune_vn2")(_TuneVN2Adapter)


def _tune_config(*, search_space: dict, budget: int, origins_end: str, lags: list[int]) -> dict:
    return {
        "config_schema": "1.0",
        "dataset": {"adapter": "tune_vn2", "path": "ignored"},
        "tasks": [
            {
                "model": "lightgbm.LGBMRegressor",
                "horizon": 3,
                "config": {
                    "backend": "mlforecast",
                    "objective": "quantile",
                    "strategy": "direct",
                    "lags": lags,
                    "verbosity": -1,
                },
            }
        ],
        "origins": {"start": "2024-04-15", "end": origins_end, "freq": "W-MON"},
        "output": {"ledger_path": "ignored.parquet", "streaming": False},
        "execution": {"backend": "local", "seed": 123},
        "hpo": {
            "budget": budget,
            "seed": _BENCHMARK_SEED,
            "search_space": search_space,
            "asha_grace_period": 1,
        },
    }


def test_run_tune_matches_benchmark_study_surface(monkeypatch) -> None:
    # Three origins, matching the benchmark's HPO_N_ORIGINS=3 (so ASHA max_t agrees).
    config = load_config_from_mapping(
        _tune_config(
            search_space=HPO_SEARCH_SPACE,
            budget=25,
            origins_end="2024-04-29",
            lags=[1, 2, 3, 4],
        )
    )
    captured: dict[str, Any] = {}

    def _capture(task):
        captured["task"] = task
        return {"backend": "mlforecast", "scope": "global", "quantiles": [0.51]}

    monkeypatch.setattr("calibre.cli.commands.optimize_global_task", _capture)

    best_config = run_tune(config)
    assert best_config["quantiles"] == [0.51]

    task = captured["task"]

    # The shared, runtime-coherent search dimension is quantile_alpha; the
    # benchmark's other dims (e.g. lag_set_idx -> HPO_LAG_SETS) are VN2-only glue
    # the dataset-general CLI does not carry, so full-space equality is not pinned.
    sampled = _sampled_search_space(task)
    assert sampled["quantile_alpha"] == HPO_SEARCH_SPACE["quantile_alpha"]

    # The tuned quantile_alpha re-points BOTH the model's predicted quantile and
    # the per-trial evaluation quantile (ordering_config overrides the objective),
    # so what the search optimizes is what gets deployed.
    candidate = task.search_space(_SpecRecorder())
    alpha = sampled["quantile_alpha"]["choices"][0]
    assert candidate.ordering_config["quantile"] == alpha
    assert candidate.model_config["quantiles"] == [alpha]

    # Objective identity + tau derived from the dataset cost struct (== benchmark).
    # objective.quantile is the template placeholder (always overridden per trial
    # via ordering_config, asserted above); tau is the cost fractile, never tuned.
    assert isinstance(task.objective, CumulativePinball)
    assert task.objective.quantile == 0.5
    assert task.objective.tau == pytest.approx(_VN2_UNDERAGE / (_VN2_UNDERAGE + _VN2_OVERAGE))
    assert task.objective.tau == pytest.approx(HPO_COST_OPTIMAL_TAU)

    # Sampler seed + budget.
    assert task.study_config.seed == _BENCHMARK_SEED
    assert task.study_config.n_trials == 25

    # Origins-count (ASHA max_t = len(origins)) + grace period + freq.
    assert len(task.origins) == 3
    assert task.study_config.asha_grace_period == 1
    assert task.study_config.freq == "W-MON"

    # base_model_config drives the global study and never re-targets the fractile.
    # (The CLI's base carries the full task model config + scope=global, richer
    # than the benchmark's minimal {backend, scope}; only scope/global is shared.)
    assert task.base_model_config["scope"] == "global"


def _sampled_search_space(task) -> dict[str, dict]:
    """Recover the per-key spec the CLI search space samples from, via a recorder."""
    recorder = _SpecRecorder()
    task.search_space(recorder)
    return recorder.specs


class _SpecRecorder:
    """A trial-shaped recorder that captures the spec each suggest_* call implies."""

    def __init__(self) -> None:
        self.specs: dict[str, dict] = {}

    def suggest_categorical(self, name: str, choices: list) -> Any:
        self.specs[name] = {"type": "categorical", "choices": choices}
        return choices[0]

    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
        self.specs[name] = {"type": "int", "low": low, "high": high, "step": step}
        return low

    def suggest_float(
        self, name: str, low: float, high: float, *, step: float | None = None, log: bool = False
    ) -> float:
        self.specs[name] = {"type": "float", "low": low, "high": high, "log": log}
        if step is not None:
            self.specs[name]["step"] = step
        return low


def test_run_tune_requires_quantile_alpha_dimension() -> None:
    config = load_config_from_mapping(
        _tune_config(
            search_space={"n_estimators": {"type": "int", "low": 5, "high": 10}},
            budget=2,
            origins_end="2024-04-22",
            lags=[1, 2, 3, 4],
        )
    )
    with pytest.raises(ValueError, match="must include a 'quantile_alpha' dimension"):
        run_tune(config)


def test_run_tune_requires_hpo_block() -> None:
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "tune_vn2", "path": "ignored"},
            "tasks": [{"model": "stub", "horizon": 3, "config": {"backend": "mlforecast"}}],
            "origins": {"start": "2024-04-15", "end": "2024-04-15", "freq": "W-MON"},
            "output": {"ledger_path": "ignored.parquet", "streaming": False},
        }
    )
    with pytest.raises(ValueError, match="--tune requires an hpo config block"):
        run_tune(config)


def test_run_tune_small_real_run_completes() -> None:
    choices = [0.45, 0.51, 0.59]
    search_space = {
        "quantile_alpha": {"type": "categorical", "choices": choices},
        "n_estimators": {"type": "categorical", "choices": [5]},
    }
    config = load_config_from_mapping(
        _tune_config(
            search_space=search_space,
            budget=2,
            origins_end="2024-04-15",
            lags=[1, 2, 3, 4],
        )
    )

    # A real Ray + ASHA tune over two trials. optimize_global_task delegates to
    # _best_result_config, which RAISES unless at least one trial produced a
    # FINITE objective — so a returning call is itself the finite-objective proof
    # (no degenerate all-inf study can pass). This is a wiring proof, not a cost
    # bound: the study ran end-to-end and returned a deployable config.
    best_config = run_tune(config)

    # The discovered per-horizon quantile is one of the configured choices: the
    # search ran end-to-end and produced a valid, deployable config. Not
    # bit-exact — Ray + ASHA make the literal value non-reproducible even seeded.
    discovered_alpha = float(best_config["quantiles"][0])
    assert discovered_alpha in choices
    assert best_config["scope"] == "global"
    assert best_config["backend"] == "mlforecast"
    assert best_config["model"] == "lightgbm.LGBMRegressor"
