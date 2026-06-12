"""VN2 HPO and simulator-cost search glue."""

from __future__ import annotations

import logging
import math
import tempfile
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import optuna
import pandas as pd

from benchmarks.common.tracking import log_mlflow_params, mlflow, start_benchmark_run
from benchmarks.vn2.config import (
    BEST_CONFIG,
    DATA_DIR,
    DECISION_ROUNDS,
    DELIVERY_WEEKS,
    HORIZON,
    HPO_COST_OPTIMAL_TAU,
    HPO_LAG_SETS,
    HPO_N_ORIGINS,
    HPO_N_TRIALS,
    HPO_SEARCH_SPACE,
    HPO_TIMEOUT_SEC,
    LEAD_TIME,
    REVIEW_PERIOD,
)
from benchmarks.vn2.data import (
    as_cumulative_decision_frame,
    build_model_config,
    load_instock,
    prepare_model_history,
)
from benchmarks.vn2.replay import VN2ReplayCache, build_replay_cache, replay_cached_cost
from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig
from calibre.conformal.partitions import global_partition, series_partition
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.execution.data_loading import load_period
from calibre.tuning import (
    OBJECTIVE_METRIC,
    CumulativePinball,
    GlobalTuningTask,
    StudyConfig,
    TuningCandidate,
    optimize_global_task,
    run_optuna_study,
)

logger = logging.getLogger(__name__)

TUNE_STEP_ATTR = "tune_step"


def _cost_trial_failure_report(exc: BaseException) -> dict[str, Any]:
    """Tune-report payload for a recoverable trial failure; re-raise infra failures.

    A pruned trial or a bad-hyperparameter error (``ValueError``/``KeyError``,
    including a failed order policy now that replay fails fast) is a high-cost
    trial the optimizer should avoid. Anything else is an infrastructure failure
    (import error, Ray worker crash, misconfiguration) and must surface as a hard
    error, not masquerade as ``inf`` cost.
    """
    if isinstance(exc, optuna.TrialPruned):
        return {OBJECTIVE_METRIC: float("inf"), TUNE_STEP_ATTR: 1, "pruned": 1}
    if isinstance(exc, ValueError | KeyError):
        return {OBJECTIVE_METRIC: float("inf"), TUNE_STEP_ATTR: 1, "bad_trial": repr(exc)[:500]}
    raise exc


def _suggest_from_spec(trial: optuna.Trial, name: str, spec: dict[str, Any]) -> Any:
    """Sample a parameter from a declarative search-space spec."""
    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "float":
        return trial.suggest_float(
            name,
            spec["low"],
            spec["high"],
            step=spec.get("step"),
            log=spec.get("log", False),
        )
    raise ValueError(f"Unknown HPO spec type: {kind!r}")


def _walk_forward_origins(
    history: pd.DataFrame, n_origins: int, horizon: int
) -> list[pd.Timestamp]:
    """Pick the last `n_origins` origins from the history's tail."""
    all_dates = sorted(history[DS].unique())
    if len(all_dates) < n_origins + horizon:
        n_origins = max(1, len(all_dates) - horizon)
    if n_origins <= 0:
        return []
    return [pd.Timestamp(d) for d in all_dates[-(n_origins + horizon) : -horizon]]


def _hpo_candidate_from_params(params: dict[str, Any]) -> TuningCandidate:
    params = dict(params)
    lags = HPO_LAG_SETS[int(params.pop("lag_set_idx"))]
    quantile_alpha = float(params.pop("quantile_alpha"))
    model_config = build_model_config(quantile_alpha=quantile_alpha, lags=lags, **params)
    return TuningCandidate(
        model_config=model_config,
        ordering_config={"quantile": quantile_alpha},
    )


def _hpo_search_space(trial: optuna.Trial) -> TuningCandidate:
    params = {
        name: _suggest_from_spec(trial, name, spec) for name, spec in HPO_SEARCH_SPACE.items()
    }
    return _hpo_candidate_from_params(params)


@dataclass(frozen=True, slots=True)
class _CumulativeTerminalPinball:
    """Cumulative-target HPO objective scored on the terminal horizon only.

    With the direct cumulative target, MLForecast emits one cumulative-demand
    prediction per horizon, but only the terminal horizon estimates the whole
    protection period. Collapse non-terminal rows to zero (mirroring
    ``as_cumulative_decision_frame``) before delegating to
    :class:`CumulativePinball`, so the per-window prediction sum reduces to the
    terminal cumulative prediction instead of over-counting every horizon. The
    realised ``Y`` is left untouched so its window sum still recovers cumulative
    demand from the raw weekly actuals.
    """

    quantile: float
    tau: float
    protection_period: int

    def evaluate(self, frame: pd.DataFrame, actuals: pd.Series) -> float:
        collapsed = as_cumulative_decision_frame(frame, self.protection_period)
        return CumulativePinball(quantile=self.quantile, tau=self.tau).evaluate(collapsed, actuals)


def run_hpo(
    data_dir: Path = DATA_DIR,
    horizon: int = HORIZON,
    n_trials: int = HPO_N_TRIALS,
    n_origins: int = HPO_N_ORIGINS,
    timeout_sec: int = HPO_TIMEOUT_SEC,
    cost_optimal_tau: float = HPO_COST_OPTIMAL_TAU,
    series_filter: list[str] | None = None,
    seed: int = 42,
    verbose: bool = True,
    target_mode: str = "per_horizon",
    asha_grace_period: int = 1,
    cpu_per_trial: float = 1.0,
    max_concurrent_trials: int | None = None,
    ray_address: str | None = None,
    tune_storage_path: str | Path | None = None,
    tune_experiment_name: str | None = "vn2_hpo",
) -> dict[str, Any]:
    """Run panel-level HPO through Calibre's public GlobalTuningTask API."""
    week0 = load_period(data_dir, 0)
    if series_filter is not None:
        week0 = week0[week0[UNIQUE_ID].isin(series_filter)]

    target_mode = target_mode.lower()
    if target_mode not in {"per_horizon", "cumulative"}:
        raise ValueError("target_mode must be 'per_horizon' or 'cumulative'")

    instock = load_instock(data_dir, series_filter)
    cumulative_target = target_mode == "cumulative"
    history = prepare_model_history(
        week0,
        instock,
        protection_period=horizon,
        cumulative_target=cumulative_target,
    )
    actuals = week0[[UNIQUE_ID, DS, Y]].copy()

    origins = _walk_forward_origins(history, n_origins, horizon)
    if not origins:
        raise ValueError(f"Not enough history to build {n_origins} origins with horizon {horizon}")

    if verbose:
        logger.info(
            "Ray Tune HPO: %s trials, %s origins, timeout %ss, panel size %s series, cost-optimal tau=%.3f",
            n_trials,
            n_origins,
            timeout_sec,
            history[UNIQUE_ID].nunique(),
            cost_optimal_tau,
        )

    started = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    task = GlobalTuningTask(
        history=history,
        horizon=horizon,
        base_model_config={"backend": "mlforecast", "scope": "global"},
        search_space=_hpo_search_space,
        actuals=actuals,
        origins=origins,
        objective=(
            _CumulativeTerminalPinball(
                quantile=0.5, tau=cost_optimal_tau, protection_period=horizon
            )
            if cumulative_target
            else CumulativePinball(quantile=0.5, tau=cost_optimal_tau)
        ),
        study_config=StudyConfig(
            n_trials=n_trials,
            freq="W-MON",
            seed=seed,
            asha_grace_period=asha_grace_period,
            cpu_per_trial=cpu_per_trial,
            max_concurrent_trials=max_concurrent_trials,
            ray_address=ray_address,
            tune_storage_path=str(tune_storage_path) if tune_storage_path is not None else None,
            tune_experiment_name=tune_experiment_name,
        ),
    )
    best_config = optimize_global_task(task)
    quantile_alpha = float(best_config["quantiles"][0])
    best_config["_quantile_alpha"] = quantile_alpha
    if cumulative_target:
        best_config["_target_mode"] = "cumulative"

    if verbose:
        logger.info(
            "HPO done in %.1fs. Best alpha=%.2f lags=%s",
            time.time() - started,
            quantile_alpha,
            best_config["lags"],
        )

    if mlflow.active_run() is not None:
        mlflow.log_params({f"hpo/best_{k}": str(v)[:500] for k, v in best_config.items()})

    return best_config


def _sample_cost_search_model_config(
    trial: optuna.Trial,
    base_config: dict[str, Any],
    search_forecast: bool,
) -> dict[str, Any]:
    if not search_forecast:
        return deepcopy(base_config)

    lag_idx = trial.suggest_categorical("lag_set_idx", list(range(len(HPO_LAG_SETS))))
    target_mode = trial.suggest_categorical("target_mode", ["per_horizon", "cumulative"])
    quantile_alpha = trial.suggest_float("quantile_alpha", 0.45, 0.9)
    config = build_model_config(
        quantile_alpha=quantile_alpha,
        n_estimators=trial.suggest_int("n_estimators", 200, 800, step=50),
        learning_rate=trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        num_leaves=trial.suggest_categorical("num_leaves", [15, 31, 63, 127]),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 1.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1.0, log=True),
        lags=HPO_LAG_SETS[int(lag_idx)],
    )
    config["_quantile_alpha"] = quantile_alpha
    if target_mode == "cumulative":
        config["_target_mode"] = "cumulative"
    return config


def _sample_cost_search_crc_config(
    trial: optuna.Trial,
    protection_period: int,
    crc_partitions: list[str] | None = None,
    *,
    validate_buffer: bool = True,
) -> CumulativeConformalRiskConfig | None:
    if not trial.suggest_categorical("crc_enabled", [True, False]):
        return None

    weight_decay_choice = trial.suggest_categorical(
        "crc_weight_decay",
        ["none", 0.5, 0.7, 0.85, 0.95, 1.0],
    )
    weighted_quantile_mode = trial.suggest_categorical(
        "crc_weighted_quantile_mode",
        ["empirical", "nonexchangeable"],
    )
    buffer_min_choice = trial.suggest_categorical("crc_buffer_min", ["none", -10.0, -5.0, 0.0])
    buffer_max_choice = trial.suggest_categorical("crc_buffer_max", ["none", 0.0, 5.0, 10.0])
    buffer_min = None if buffer_min_choice == "none" else float(buffer_min_choice)
    buffer_max = None if buffer_max_choice == "none" else float(buffer_max_choice)
    if (
        validate_buffer
        and buffer_min is not None
        and buffer_max is not None
        and buffer_min > buffer_max
    ):
        raise optuna.TrialPruned("buffer_min > buffer_max")

    partition_name = trial.suggest_categorical(
        "crc_partition",
        crc_partitions or ["global", "series", "hierarchical"],
    )
    partition_key = _crc_partition_key(partition_name)

    return CumulativeConformalRiskConfig(
        coverage=trial.suggest_float("crc_coverage", 0.55, 0.9),
        calibration_window=5000,
        protection_period=protection_period,
        partition_key=partition_key,
        weight_decay=None if weight_decay_choice == "none" else float(weight_decay_choice),
        weighted_quantile_mode=weighted_quantile_mode,
        buffer_min=buffer_min,
        buffer_max=buffer_max,
        shrinkage_strength=trial.suggest_float("crc_shrinkage_strength", 0.0, 0.75),
        method_name="cost_search_crc",
    )


def _crc_partition_key(name: str) -> Callable[[Any], Any]:
    if name == "global":
        return global_partition
    if name == "series":
        return series_partition
    if name == "hierarchical":
        return lambda row: str(row[UNIQUE_ID]).split("_")[0]
    raise ValueError(f"Unknown crc partition: {name!r}")


def _cost_search_space(
    trial: optuna.Trial,
    *,
    base_config: dict[str, Any],
    search_forecast: bool,
    include_order_calibration: bool,
    protection_period: int,
    crc_partitions: list[str] | None,
) -> None:
    _sample_cost_search_model_config(trial, base_config, search_forecast)
    if include_order_calibration:
        trial.suggest_float("order_base_scale", 0.85, 1.15)
        trial.suggest_float("reorder_point_scale", 0.0, 1.0)
    _sample_cost_search_crc_config(
        trial,
        protection_period,
        crc_partitions=crc_partitions,
        validate_buffer=False,
    )
    return None


def _tune_storage_path(path: str | Path | None) -> str:
    if path is None:
        return tempfile.mkdtemp(prefix="calibre-vn2-cost-search-")
    raw = str(path)
    if "://" in raw:
        return raw
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _distribution_for_value(value: Any) -> optuna.distributions.BaseDistribution:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return optuna.distributions.CategoricalDistribution([value])
    if isinstance(value, int):
        return optuna.distributions.IntDistribution(value, value)
    if isinstance(value, float):
        return optuna.distributions.FloatDistribution(value, value)
    return optuna.distributions.CategoricalDistribution([repr(value)])


def _study_from_tune_results(results: Any) -> optuna.Study:
    study = optuna.create_study(direction="minimize")
    for result in results:
        metrics = getattr(result, "metrics", None) or {}
        if OBJECTIVE_METRIC not in metrics:
            continue
        params = dict(getattr(result, "config", {}) or {})
        value = float(metrics[OBJECTIVE_METRIC])
        study.add_trial(
            optuna.create_trial(
                value=value,
                params=params,
                distributions={
                    key: _distribution_for_value(value) for key, value in params.items()
                },
                user_attrs={
                    key: metric_value
                    for key, metric_value in metrics.items()
                    if key not in {OBJECTIVE_METRIC, TUNE_STEP_ATTR}
                },
            )
        )
    return study


def run_cost_search(
    *,
    data_dir: Path = DATA_DIR,
    model_config: dict[str, Any] | None = None,
    horizon: int = HORIZON,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    series_filter: list[str] | None = None,
    n_trials: int = 20,
    timeout_sec: int | None = None,
    seed: int = 42,
    search_forecast: bool = False,
    include_order_calibration: bool = False,
    crc_partitions: list[str] | None = None,
    log_mlflow: bool = False,
    experiment_name: str = "vn2",
    run_name: str = "cost_search",
    asha_grace_period: int = 1,
    cpu_per_trial: float = 1.0,
    max_concurrent_trials: int | None = None,
    ray_address: str | None = None,
    tune_storage_path: str | Path | None = None,
    ray_tune_experiment_name: str | None = "vn2_cost_search",
) -> optuna.Study:
    """Optimize simulator EUR cost with Calibre's public Ray Tune study core."""
    base_config = deepcopy(model_config if model_config is not None else BEST_CONFIG)
    fixed_cache: VN2ReplayCache | None = None
    if not search_forecast:
        fixed_cache = build_replay_cache(
            data_dir=data_dir,
            model_config=base_config,
            horizon=horizon,
            lead_time=lead_time,
            review_period=review_period,
            decision_rounds=decision_rounds,
            delivery_weeks=delivery_weeks,
            series_filter=series_filter,
        )

    max_t = max(1, decision_rounds + delivery_weeks)

    def _trainable(config: dict[str, Any], *, state_ref: Any | None = None) -> None:
        del state_ref
        from ray import tune

        fixed_trial = optuna.trial.FixedTrial(dict(config))
        try:
            candidate_model = _sample_cost_search_model_config(
                fixed_trial,
                base_config,
                search_forecast,
            )
            order_base_scale = (
                fixed_trial.suggest_float("order_base_scale", 0.85, 1.15)
                if include_order_calibration
                else 1.0
            )
            reorder_point_scale = (
                fixed_trial.suggest_float("reorder_point_scale", 0.0, 1.0)
                if include_order_calibration
                else None
            )
            crc_config = _sample_cost_search_crc_config(
                fixed_trial,
                lead_time + review_period,
                crc_partitions=crc_partitions,
            )
            replay_cache = (
                fixed_cache
                if fixed_cache is not None
                else build_replay_cache(
                    data_dir=data_dir,
                    model_config=candidate_model,
                    horizon=horizon,
                    lead_time=lead_time,
                    review_period=review_period,
                    decision_rounds=decision_rounds,
                    delivery_weeks=delivery_weeks,
                    series_filter=series_filter,
                )
            )

            def _report_progress(step: int, total_cost: float) -> None:
                tune.report(
                    {
                        OBJECTIVE_METRIC: float(total_cost),
                        TUNE_STEP_ATTR: step,
                    }
                )

            result = replay_cached_cost(
                replay_cache,
                order_conformal_config=crc_config,
                order_base_scale=order_base_scale,
                reorder_point_scale=reorder_point_scale,
                on_progress=_report_progress,
            )
        except Exception as exc:  # pragma: no cover - exercised by Tune on bad trials.
            tune.report(_cost_trial_failure_report(exc))
            return

        if max_t == 1 and decision_rounds + delivery_weeks == 0:
            tune.report({OBJECTIVE_METRIC: result.total_cost, TUNE_STEP_ATTR: 1})

    space = partial(
        _cost_search_space,
        base_config=base_config,
        search_forecast=search_forecast,
        include_order_calibration=include_order_calibration,
        protection_period=lead_time + review_period,
        crc_partitions=crc_partitions,
    )

    def _run() -> optuna.Study:
        outcome = run_optuna_study(
            space=space,
            trainable=_trainable,
            n_trials=n_trials,
            max_t=max_t,
            seed=seed,
            asha_grace_period=asha_grace_period,
            cpu_per_trial=cpu_per_trial,
            max_concurrent_trials=max(
                1,
                int(max_concurrent_trials) if max_concurrent_trials is not None else int(n_trials),
            ),
            ray_address=ray_address,
            tune_storage_path=_tune_storage_path(tune_storage_path),
            metric=OBJECTIVE_METRIC,
            mode="min",
            time_attr=TUNE_STEP_ATTR,
            experiment_name=ray_tune_experiment_name,
            fail_fast="raise",
        )
        return _study_from_tune_results(outcome.results)

    if log_mlflow:
        with start_benchmark_run(
            experiment_name,
            run_name,
            tags={
                "dataset": "vn2",
                "objective": "simulator_eur_cost",
                "search_forecast": str(search_forecast),
            },
        ):
            log_mlflow_params(
                {
                    "n_trials": n_trials,
                    "timeout_sec": timeout_sec,
                    "seed": seed,
                    "search_forecast": search_forecast,
                    "include_order_calibration": include_order_calibration,
                    "crc_partitions": crc_partitions,
                    "horizon": horizon,
                    "lead_time": lead_time,
                    "review_period": review_period,
                    "decision_rounds": decision_rounds,
                    "delivery_weeks": delivery_weeks,
                    "series_filter_size": len(series_filter) if series_filter is not None else None,
                }
            )
            study = _run()
            completed_trials = [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
                and math.isfinite(float(trial.value))
            ]
            if completed_trials:
                mlflow.log_metric("best/cost_total", float(study.best_value))
                log_mlflow_params(
                    {f"best.{key}": value for key, value in study.best_params.items()}
                )
                mlflow.log_dict(study.best_trial.user_attrs, "best_trial_user_attrs.json")
                with tempfile.TemporaryDirectory() as tmp:
                    trials_path = Path(tmp) / "trials.csv"
                    study.trials_dataframe().to_csv(trials_path, index=False)
                    mlflow.log_artifact(str(trials_path), artifact_path="optuna")
    else:
        study = _run()
    return study
