"""Hyperparameter optimisation for the VN2 benchmark.

Two tuning modes:
- **Per-series** (``tune_one_series`` / ``tune_all_series``): series-level
  walk-forward sweep using calibre's TuningTask API. Used for lightweight
  SeasonalNaive/similar model selection.
- **Panel HPO** (``run_hpo``): panel-level Ray Tune / Optuna search over the
  global LightGBM config. The objective is cumulative-horizon pinball loss.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd

from benchmarks.vn2.config import (
    DATA_DIR,
    HORIZON,
    HPO_COST_OPTIMAL_TAU,
    HPO_LAG_SETS,
    HPO_N_ORIGINS,
    HPO_N_TRIALS,
    HPO_SEARCH_SPACE,
    HPO_TIMEOUT_SEC,
)
from benchmarks.vn2.data import (
    _load_instock,
    _prepare_model_history,
    _prepare_policy_forecast_frame,
    _strip_private,
    _walk_forward_origins,
)
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, H, Y, quantile_column
from calibre.core.forecast_task import ForecastTask
from calibre.evaluation.point_metrics import pinball_linear, smape
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.threading import _cap_threaded_config
from calibre.tuning.objectives import Accuracy
from calibre.tuning.optimizer import create_tpe_sampler, optimize_task
from calibre.tuning.task import TuningCandidate, TuningTask

logger = logging.getLogger(__name__)
_TUNE_OBJECTIVE_METRIC = "objective"
_TUNE_STEP_ATTR = "tune_step"
_TUNE_RESULTS_PREFIX = "calibre-vn2-tune-"
ROLLING_WINDOWS = [4, 13, 26]


# ── per-series tuning ────────────────────────────────────────────────────────


def seasonal_naive_search_space(trial: optuna.Trial) -> TuningCandidate:
    """Search space for SeasonalNaive: tune season_length."""
    return TuningCandidate(
        model_config={
            "season_length": trial.suggest_categorical("season_length", [4, 13, 26, 52]),
        }
    )


def tune_one_series(
    unique_id: str,
    sales: pd.DataFrame,
    horizon: int,
    base_config: dict,
    search_space: Callable[[optuna.Trial], TuningCandidate] = seasonal_naive_search_space,
    n_trials: int = 20,
    n_origins: int = 5,
    freq: str = "W",
) -> dict:
    """Tune a single series. Returns the best model config dict."""
    series_data = sales[sales[UNIQUE_ID] == unique_id]
    all_dates = sorted(series_data[DS].unique())

    if len(all_dates) < n_origins + horizon:
        n_origins = max(1, len(all_dates) - horizon)

    origins = [pd.Timestamp(d) for d in all_dates[-(n_origins + horizon) : -horizon]]
    if not origins:
        return base_config

    history = series_data[[DS, Y]].sort_values(DS).reset_index(drop=True)
    actuals = series_data[[UNIQUE_ID, DS, Y]].copy()

    task = TuningTask(
        unique_id=unique_id,
        history=history,
        horizon=horizon,
        base_model_config=base_config,
        search_space=search_space,
        actuals=actuals,
        origins=origins,
        objective=Accuracy(metric=smape),
        n_trials=n_trials,
        freq=freq,
    )
    return optimize_task(task)


def tune_all_series(
    sales: pd.DataFrame,
    horizon: int,
    base_config: dict,
    search_space: Callable[[optuna.Trial], TuningCandidate] = seasonal_naive_search_space,
    n_trials: int = 20,
    n_origins: int = 5,
    freq: str = "W",
    max_workers: int = 4,
) -> dict[str, dict]:
    """Tune base_config per series in parallel using threads."""
    unique_ids = sorted(sales[UNIQUE_ID].unique())

    def _tune(uid: str) -> tuple[str, dict]:
        best = tune_one_series(
            uid, sales, horizon, base_config, search_space, n_trials, n_origins, freq
        )
        return uid, best

    results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_tune, uid): uid for uid in unique_ids}
        for future in concurrent.futures.as_completed(futures):
            uid, best_config = future.result()
            results[uid] = best_config

    return results


# ── panel HPO helpers ────────────────────────────────────────────────────────


@contextmanager
def _restore_cwd():
    original = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original)


@contextmanager
def _trial_thread_env(cpu_per_trial: float):
    threads = str(max(1, int(cpu_per_trial)))
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "TORCH_NUM_THREADS",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = threads
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_max_concurrent_trials(
    max_concurrent_trials: int | None,
    *,
    n_trials: int,
    cpu_per_trial: float,
) -> int:
    if max_concurrent_trials is not None:
        return max(1, min(n_trials, int(max_concurrent_trials)))
    cpus = max(1, os.cpu_count() or 1)
    by_cpu = max(1, int(cpus // max(cpu_per_trial, 1e-9)))
    return max(1, min(n_trials, by_cpu))


def _resolve_tune_storage_path(path: str | Path | None) -> str:
    if path is None:
        return tempfile.mkdtemp(prefix=_TUNE_RESULTS_PREFIX)
    raw = str(path)
    if "://" in raw:
        return raw
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


def _short_tune_trial_name(trial: Any) -> str:
    return f"trial_{trial.trial_id}"


def _best_tune_result(results: Any) -> Any:
    valid = [
        result
        for result in results
        if result.error is None
        and result.metrics is not None
        and _TUNE_OBJECTIVE_METRIC in result.metrics
        and math.isfinite(float(result.metrics[_TUNE_OBJECTIVE_METRIC]))
    ]
    if not valid:
        failed = sum(1 for result in results if result.error is not None)
        raise RuntimeError(
            "Ray Tune completed without a valid VN2 objective result "
            f"({failed} failed trial(s)). Check the trial logs and benchmark settings."
        )
    return results.get_best_result(
        metric=_TUNE_OBJECTIVE_METRIC,
        mode="min",
        filter_nan_and_inf=True,
    )


def _run_optuna_tune(
    trainable: Callable[[dict[str, Any]], None],
    search_space: Callable[[optuna.Trial], None],
    *,
    n_trials: int,
    max_t: int,
    seed: int | None,
    timeout_sec: int | None,
    asha_grace_period: int,
    cpu_per_trial: float,
    max_concurrent_trials: int | None,
    ray_address: str | None,
    ray_local_mode: bool,
    tune_storage_path: str | Path | None,
    tune_experiment_name: str | None,
) -> tuple[Any, Any]:
    """Run a VN2 Optuna search space through Ray Tune and return results + searcher."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if max_t < 1:
        raise ValueError("max_t must be at least 1")
    if cpu_per_trial <= 0:
        raise ValueError("cpu_per_trial must be positive")

    from calibre.execution.ray_runtime import acquire_ray_runtime, prepare_ray_environment

    prepare_ray_environment()
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    grace_period = max(1, min(int(asha_grace_period), max_t))
    search_alg = OptunaSearch(
        space=search_space,
        metric=_TUNE_OBJECTIVE_METRIC,
        mode="min",
        sampler=create_tpe_sampler(seed),
    )
    scheduler = ASHAScheduler(
        metric=_TUNE_OBJECTIVE_METRIC,
        mode="min",
        time_attr=_TUNE_STEP_ATTR,
        max_t=max_t,
        grace_period=grace_period,
    )
    trainable_with_resources = tune.with_resources(
        trainable,
        resources={"cpu": float(cpu_per_trial)},
    )
    tune_config_kwargs: dict[str, Any] = {
        "search_alg": search_alg,
        "scheduler": scheduler,
        "num_samples": n_trials,
        "trial_name_creator": _short_tune_trial_name,
        "trial_dirname_creator": _short_tune_trial_name,
        "max_concurrent_trials": _resolve_max_concurrent_trials(
            max_concurrent_trials,
            n_trials=n_trials,
            cpu_per_trial=cpu_per_trial,
        ),
    }
    if timeout_sec is not None:
        tune_config_kwargs["time_budget_s"] = timeout_sec
    run_config_kwargs: dict[str, Any] = {
        "storage_path": _resolve_tune_storage_path(tune_storage_path),
        "verbose": 0,
    }
    if tune_experiment_name is not None:
        run_config_kwargs["name"] = tune_experiment_name

    previous_auto_loggers = os.environ.get("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS")
    os.environ.setdefault("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")
    previous_chdir = os.environ.get("RAY_CHDIR_TO_TRIAL_DIR")
    os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = "0"
    ray_runtime = acquire_ray_runtime(address=ray_address, local_mode=ray_local_mode)
    try:
        with _restore_cwd():
            tuner = tune.Tuner(
                trainable_with_resources,
                tune_config=tune.TuneConfig(**tune_config_kwargs),
                run_config=tune.RunConfig(**run_config_kwargs),
            )
            results = tuner.fit()
    finally:
        if previous_chdir is None:
            os.environ.pop("RAY_CHDIR_TO_TRIAL_DIR", None)
        else:
            os.environ["RAY_CHDIR_TO_TRIAL_DIR"] = previous_chdir
        if previous_auto_loggers is None:
            os.environ.pop("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", None)
        else:
            os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = previous_auto_loggers
        ray_runtime.release()
    return results, search_alg


class _HpoSearchSpaceAdapter:
    """Expose VN2 panel HPO's Optuna search space to Ray Tune."""

    def __call__(self, trial: optuna.Trial) -> None:
        for name, spec in HPO_SEARCH_SPACE.items():
            _suggest_from_spec(trial, name, spec)
        return None


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


def _cumulative_pinball(
    forecast_df: pd.DataFrame,
    actuals: pd.DataFrame,
    horizon: int,
    quantile: float,
    tau: float,
) -> float:
    """Cumulative-horizon pinball loss at the cost-optimal tau."""
    qcol = quantile_column(quantile)
    if qcol not in forecast_df.columns or forecast_df.empty:
        return float("inf")

    actuals_lookup = actuals.set_index([UNIQUE_ID, DS])[Y]

    df = forecast_df[[UNIQUE_ID, DS, FORECAST_ORIGIN, H, qcol]].copy()
    df[Y] = actuals_lookup.reindex(
        pd.MultiIndex.from_arrays([df[UNIQUE_ID].values, df[DS].values])
    ).to_numpy()

    df = df[df[H] <= horizon]
    grouped = df.groupby([UNIQUE_ID, FORECAST_ORIGIN], sort=False)
    full_window = grouped[Y].transform("count") == horizon
    df = df[full_window]
    if df.empty:
        return float("inf")

    sums = df.groupby([UNIQUE_ID, FORECAST_ORIGIN], sort=False)[[Y, qcol]].sum()
    if sums.empty:
        return float("inf")

    actual_sum = sums[Y].to_numpy(dtype=float)
    pred_sum = sums[qcol].to_numpy(dtype=float)
    return float(pinball_linear(actual_sum, pred_sum, tau=tau))


def _build_model_config(
    quantile_alpha: float,
    n_estimators: int,
    learning_rate: float,
    num_leaves: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_alpha: float,
    reg_lambda: float,
    lags: list[int],
) -> dict[str, Any]:
    """Build a single-quantile global LGBM model_config for the engine."""
    return {
        "backend": "mlforecast",
        "scope": "global",
        "name": f"global_lgbm_q{quantile_column(quantile_alpha)}",
        "model": "lightgbm.LGBMRegressor",
        "objective": "quantile",
        "quantiles": [quantile_alpha],
        "strategy": "direct",
        "lags": lags,
        "lag_transforms": {
            1: [
                *(RollingMean(window_size=w) for w in ROLLING_WINDOWS),
                *(RollingStd(window_size=w) for w in ROLLING_WINDOWS),
            ]
        },
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
    }


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
    ray_local_mode: bool = False,
    tune_storage_path: str | Path | None = None,
    tune_experiment_name: str | None = "vn2_hpo",
) -> dict[str, Any]:
    """Run the panel-level Ray Tune/Optuna HPO and return the best model config."""
    from calibre.execution.data_loading import load_period

    week0 = load_period(data_dir, 0)
    if series_filter is not None:
        week0 = week0[week0[UNIQUE_ID].isin(series_filter)]

    target_mode = target_mode.lower()
    if target_mode not in {"per_horizon", "cumulative"}:
        raise ValueError("target_mode must be 'per_horizon' or 'cumulative'")

    instock = _load_instock(data_dir, series_filter)
    cumulative_target = target_mode == "cumulative"
    history = _prepare_model_history(
        week0,
        instock,
        protection_period=horizon,
        cumulative_target=cumulative_target,
    )
    actuals = week0[[UNIQUE_ID, DS, Y]].copy()

    origins = _walk_forward_origins(history, n_origins, horizon)
    if not origins:
        raise ValueError(f"Not enough history to build {n_origins} origins with horizon {horizon}")

    def _trainable(params: dict[str, Any]) -> None:
        from ray import tune

        params = dict(params)
        lags = HPO_LAG_SETS[int(params.pop("lag_set_idx"))]
        quantile_alpha = float(params.pop("quantile_alpha"))
        config = _cap_threaded_config(
            _build_model_config(quantile_alpha=quantile_alpha, lags=lags, **params),
            cpu_per_trial,
        )
        if cumulative_target:
            config["_target_mode"] = "cumulative"

        task = ForecastTask(history=history, horizon=horizon, model_config=_strip_private(config))
        engine = BackendEngine(execution=ExecutionOptions(freq="W-MON", backend="local"))
        try:
            total = 0.0
            with _trial_thread_env(cpu_per_trial):
                for origin_idx, result in enumerate(
                    engine.iter_origins([task], actuals=actuals, origins=origins),
                    start=1,
                ):
                    forecast_df = _prepare_policy_forecast_frame(
                        result.ledger.to_df(),
                        protection_period=horizon,
                        cumulative_target=cumulative_target,
                    )
                    value = _cumulative_pinball(
                        forecast_df, actuals, horizon, quantile_alpha, tau=cost_optimal_tau
                    )
                    if not math.isfinite(value):
                        tune.report(
                            {_TUNE_OBJECTIVE_METRIC: float("inf"), _TUNE_STEP_ATTR: origin_idx}
                        )
                        return
                    total += value
                    tune.report(
                        {
                            _TUNE_OBJECTIVE_METRIC: total / origin_idx,
                            "total_pinball": total,
                            _TUNE_STEP_ATTR: origin_idx,
                        }
                    )
        finally:
            engine.close()

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
    results, _ = _run_optuna_tune(
        _trainable,
        _HpoSearchSpaceAdapter(),
        n_trials=n_trials,
        max_t=len(origins),
        seed=seed,
        timeout_sec=timeout_sec,
        asha_grace_period=asha_grace_period,
        cpu_per_trial=cpu_per_trial,
        max_concurrent_trials=max_concurrent_trials,
        ray_address=ray_address,
        ray_local_mode=ray_local_mode,
        tune_storage_path=tune_storage_path,
        tune_experiment_name=tune_experiment_name,
    )
    elapsed = time.time() - started

    best_result = _best_tune_result(results)
    best_metric = float(best_result.metrics[_TUNE_OBJECTIVE_METRIC])
    best = dict(best_result.config)
    lags = HPO_LAG_SETS[int(best.pop("lag_set_idx"))]
    quantile_alpha = float(best.pop("quantile_alpha"))
    best_config = _build_model_config(quantile_alpha=quantile_alpha, lags=lags, **best)
    best_config["_quantile_alpha"] = quantile_alpha
    if cumulative_target:
        best_config["_target_mode"] = "cumulative"

    if verbose:
        logger.info(
            "HPO done in %.1fs. Best pinball=%.4f alpha=%.2f lags=%s",
            elapsed,
            best_metric,
            quantile_alpha,
            lags,
        )

    try:
        from benchmarks.common.tracking import mlflow

        if mlflow.active_run() is not None:
            mlflow.log_metric("hpo/best_pinball", best_metric)
            mlflow.log_params(
                {f"hpo/best_{k}": str(v)[:500] for k, v in best_result.config.items()}
            )
    except Exception:  # noqa: BLE001 — MLflow not always available
        pass

    return best_config
