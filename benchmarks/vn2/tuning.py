"""VN2 benchmark hyperparameter and simulator-cost tuning."""

from __future__ import annotations

import concurrent.futures
import logging
import math
import os
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import optuna
import pandas as pd

from benchmarks.common.tracking import mlflow, start_benchmark_run
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
    build_model_config,
    cumulative_pinball,
    load_instock,
    prepare_model_history,
    prepare_policy_forecast_frame,
    strip_private,
    suggest_from_spec,
    walk_forward_origins,
)
from benchmarks.vn2.replay import (
    PolicyApplicationError,
    VN2ReplayCache,
    _log_mlflow_params,
    build_replay_cache,
    replay_cached_cost,
)
from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig
from calibre.conformal.partitions import global_partition, series_partition
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask
from calibre.evaluation.point_metrics import smape
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.data_loading import load_period
from calibre.execution.threading import _cap_threaded_config
from calibre.tuning.objectives import Accuracy
from calibre.tuning.optimizer import create_tpe_sampler, optimize_task, restore_cwd
from calibre.tuning.task import TuningCandidate, TuningTask

logger = logging.getLogger(__name__)
TUNE_OBJECTIVE_METRIC = "objective"
_TUNE_STEP_ATTR = "tune_step"
_TUNE_RESULTS_PREFIX = "calibre-vn2-tune-"


def seasonal_naive_search_space(trial: optuna.Trial) -> TuningCandidate:
    """Search space for the legacy SeasonalNaive smoke benchmark."""
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
    """Tune one legacy seasonal-naive series via calibre.tuning.optimizer."""
    series_data = sales[sales[UNIQUE_ID] == unique_id]
    all_dates = sorted(series_data[DS].unique())
    if len(all_dates) < n_origins + horizon:
        n_origins = max(1, len(all_dates) - horizon)
    origins = [pd.Timestamp(d) for d in all_dates[-(n_origins + horizon) : -horizon]]
    if not origins:
        return base_config

    task = TuningTask(
        unique_id=unique_id,
        history=series_data[[UNIQUE_ID, DS, Y]].sort_values(DS).reset_index(drop=True),
        horizon=horizon,
        base_model_config=base_config,
        search_space=search_space,
        actuals=series_data[[UNIQUE_ID, DS, Y]].copy(),
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
    """Tune legacy SeasonalNaive configs per series in parallel."""
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


def _cap_threaded_model_config(config: dict[str, Any], cpu_per_trial: float) -> dict[str, Any]:
    return _cap_threaded_config(config, cpu_per_trial)


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
        and TUNE_OBJECTIVE_METRIC in result.metrics
        and math.isfinite(float(result.metrics[TUNE_OBJECTIVE_METRIC]))
    ]
    if not valid:
        failed = sum(1 for result in results if result.error is not None)
        raise RuntimeError(
            "Ray Tune completed without a valid VN2 objective result "
            f"({failed} failed trial(s)). Check the trial logs and benchmark settings."
        )
    return results.get_best_result(
        metric=TUNE_OBJECTIVE_METRIC,
        mode="min",
        filter_nan_and_inf=True,
    )


@dataclass(frozen=True)
class OptunaStudyHandle:
    study: optuna.Study


TuneRunner = Callable[..., tuple[Any, OptunaStudyHandle]]
"""Signature for an Optuna-via-Ray-Tune runner.

Accepts the trainable and search-space callables positionally, plus the
keyword-only configuration in :func:`run_optuna_tune`. Returns
    ``(results, OptunaStudyHandle)``. Used as an injection point so tests can supply
fakes without monkey-patching module state.
"""


def run_optuna_tune(
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
) -> tuple[Any, OptunaStudyHandle]:
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
        metric=TUNE_OBJECTIVE_METRIC,
        mode="min",
        sampler=create_tpe_sampler(seed),
    )
    scheduler = ASHAScheduler(
        metric=TUNE_OBJECTIVE_METRIC,
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
        with restore_cwd():
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
    return results, OptunaStudyHandle(_study_from_optuna_search(search_alg))


class _HpoSearchSpaceAdapter:
    """Expose VN2 panel HPO's Optuna search space to Ray Tune."""

    def __call__(self, trial: optuna.Trial) -> None:
        for name, spec in HPO_SEARCH_SPACE.items():
            suggest_from_spec(trial, name, spec)
        return None


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
    tune_runner: TuneRunner | None = None,
) -> dict[str, Any]:
    """Run the panel-level Ray Tune/Optuna HPO and return the best model config.

    The returned dict is a fully-formed ``model_config`` ready to feed into
    a ``ForecastTask(scope="global", strategy="direct", quantiles=[alpha])``
    via ``BackendEngine``. ``best_alpha`` is exposed under the
    ``"_quantile_alpha"`` key (a private debug field — drop before passing
    upstream if needed; the value is also recoverable from ``quantiles[0]``).

    The best HPO metric and parameters are logged to the active MLflow parent
    run when tracking is enabled.
    """
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

    origins = walk_forward_origins(history, n_origins, horizon)
    if not origins:
        raise ValueError(f"Not enough history to build {n_origins} origins with horizon {horizon}")

    def _trainable(params: dict[str, Any]) -> None:
        from ray import tune

        params = dict(params)
        lags = HPO_LAG_SETS[int(params.pop("lag_set_idx"))]
        quantile_alpha = float(params.pop("quantile_alpha"))
        config = _cap_threaded_model_config(
            build_model_config(quantile_alpha=quantile_alpha, lags=lags, **params),
            cpu_per_trial,
        )
        if cumulative_target:
            config["_target_mode"] = "cumulative"

        task = ForecastTask(history=history, horizon=horizon, model_config=strip_private(config))
        engine = BackendEngine(
            execution=ExecutionOptions(
                freq="W-MON",
                backend="local",
            )
        )
        try:
            total = 0.0
            with _trial_thread_env(cpu_per_trial):
                for origin_idx, result in enumerate(
                    engine.iter_origins([task], actuals=actuals, origins=origins),
                    start=1,
                ):
                    forecast_df = prepare_policy_forecast_frame(
                        result.ledger.to_df(),
                        protection_period=horizon,
                        cumulative_target=cumulative_target,
                    )
                    value = cumulative_pinball(
                        forecast_df,
                        actuals,
                        horizon,
                        quantile_alpha,
                        tau=cost_optimal_tau,
                    )
                    if not math.isfinite(value):
                        tune.report(
                            {
                                TUNE_OBJECTIVE_METRIC: float("inf"),
                                _TUNE_STEP_ATTR: origin_idx,
                            }
                        )
                        return
                    total += value
                    tune.report(
                        {
                            TUNE_OBJECTIVE_METRIC: total / origin_idx,
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
    runner = tune_runner if tune_runner is not None else run_optuna_tune
    results, _ = runner(
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
    best_metric = float(best_result.metrics[TUNE_OBJECTIVE_METRIC])
    best = dict(best_result.config)
    lags = HPO_LAG_SETS[int(best.pop("lag_set_idx"))]
    quantile_alpha = float(best.pop("quantile_alpha"))
    best_config = build_model_config(quantile_alpha=quantile_alpha, lags=lags, **best)
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

    if mlflow.active_run() is not None:
        mlflow.log_metric("hpo/best_pinball", best_metric)
        mlflow.log_params({f"hpo/best_{k}": str(v)[:500] for k, v in best_result.config.items()})

    return best_config


def _sample_cost_search_model_config(
    trial: Any,
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


def _optional_float_choice(value: object) -> float | None:
    if value == "none":
        return None
    if isinstance(value, str | int | float):
        return float(value)
    raise TypeError(f"Expected numeric or 'none' choice, got {type(value).__name__}")


def _sample_cost_search_crc_config(
    trial: Any,
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
    weighted_quantile_mode = cast(
        Literal["empirical", "nonexchangeable"],
        trial.suggest_categorical(
            "crc_weighted_quantile_mode",
            ["empirical", "nonexchangeable"],
        ),
    )
    buffer_min_choice = trial.suggest_categorical("crc_buffer_min", ["none", -10.0, -5.0, 0.0])
    buffer_max_choice = trial.suggest_categorical("crc_buffer_max", ["none", 0.0, 5.0, 10.0])
    buffer_min = _optional_float_choice(buffer_min_choice)
    buffer_max = _optional_float_choice(buffer_max_choice)
    if (
        validate_buffer
        and buffer_min is not None
        and buffer_max is not None
        and buffer_min > buffer_max
    ):
        # Prune rather than silently swap so trial parameters match the
        # realised config when reproducing a best trial.
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
        weight_decay=_optional_float_choice(weight_decay_choice),
        weighted_quantile_mode=weighted_quantile_mode,
        buffer_min=buffer_min,
        buffer_max=buffer_max,
        shrinkage_strength=trial.suggest_float("crc_shrinkage_strength", 0.0, 0.75),
        method_name="cost_search_crc",
    )


def _crc_partition_key(name: str):
    if name == "global":
        return global_partition
    if name == "series":
        return series_partition
    if name == "hierarchical":
        return lambda row: str(row[UNIQUE_ID]).split("_")[0]
    raise ValueError(f"Unknown crc partition: {name!r}")


class _CostSearchSpaceAdapter:
    """Expose VN2 simulator-cost search parameters to Ray Tune's OptunaSearch."""

    __slots__ = (
        "_base_config",
        "_crc_partitions",
        "_include_order_calibration",
        "_protection_period",
        "_search_forecast",
    )

    def __init__(
        self,
        *,
        base_config: dict[str, Any],
        search_forecast: bool,
        include_order_calibration: bool,
        protection_period: int,
        crc_partitions: list[str] | None,
    ) -> None:
        self._base_config = base_config
        self._search_forecast = search_forecast
        self._include_order_calibration = include_order_calibration
        self._protection_period = protection_period
        self._crc_partitions = crc_partitions

    def __call__(self, trial: optuna.Trial) -> None:
        _sample_cost_search_model_config(trial, self._base_config, self._search_forecast)
        if self._include_order_calibration:
            trial.suggest_float("order_base_scale", 0.85, 1.15)
            trial.suggest_float("reorder_point_scale", 0.0, 1.0)
        _sample_cost_search_crc_config(
            trial,
            self._protection_period,
            crc_partitions=self._crc_partitions,
            validate_buffer=False,
        )
        return None


def _study_from_optuna_search(search_alg: Any) -> optuna.Study:
    """Isolate Ray Tune's OptunaSearch study extraction at the runner boundary."""
    study = getattr(search_alg, "_ot_study", None)
    if study is None:
        raise RuntimeError("Ray Tune OptunaSearch did not expose a completed Optuna study")
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
    ray_local_mode: bool = False,
    tune_storage_path: str | Path | None = None,
    ray_tune_experiment_name: str | None = "vn2_cost_search",
    tune_runner: TuneRunner | None = None,
) -> optuna.Study:
    """Optimize simulator EUR cost with Ray Tune over cached forecast replays.

    By default the search varies CRC parameters against a fixed forecast model.
    Set ``search_forecast=True`` to include LightGBM, lag-set, quantile, and
    direct-cumulative target choices in the same objective. ``ray_local_mode``,
    ``max_concurrent_trials``, and small ``n_trials`` values keep smoke/local-dev
    runs cheap without changing the production Tune path.
    """
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

    def _trainable(config: dict[str, Any]) -> None:
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
            policy_errors: list[str] = []

            def _report_progress(step: int, total_cost: float) -> None:
                tune.report(
                    {
                        TUNE_OBJECTIVE_METRIC: float(total_cost),
                        _TUNE_STEP_ATTR: step,
                        "policy_error_count": len(policy_errors),
                    }
                )

            result = replay_cached_cost(
                replay_cache,
                order_conformal_config=crc_config,
                order_base_scale=order_base_scale,
                reorder_point_scale=reorder_point_scale,
                policy_error_mode="raise",
                on_policy_error=lambda rn, exc: policy_errors.append(f"round {rn}: {exc!r}"),
                on_progress=_report_progress,
            )
        except optuna.TrialPruned:
            tune.report(
                {
                    TUNE_OBJECTIVE_METRIC: float("inf"),
                    _TUNE_STEP_ATTR: 1,
                    "pruned": 1,
                }
            )
            return
        except PolicyApplicationError as exc:
            logger.exception("VN2 cost-search policy failure")
            tune.report(
                {
                    TUNE_OBJECTIVE_METRIC: float("inf"),
                    _TUNE_STEP_ATTR: 1,
                    "policy_failure": 1,
                    "error": repr(exc)[:500],
                }
            )
            raise
        except (ValueError, KeyError) as exc:
            tune.report(
                {
                    TUNE_OBJECTIVE_METRIC: float("inf"),
                    _TUNE_STEP_ATTR: 1,
                    "bad_trial": 1,
                    "error": repr(exc)[:500],
                }
            )
            return
        except Exception as exc:
            logger.exception("VN2 cost-search infrastructure failure")
            tune.report(
                {
                    TUNE_OBJECTIVE_METRIC: float("inf"),
                    _TUNE_STEP_ATTR: 1,
                    "infra_failure": 1,
                    "error": repr(exc)[:500],
                }
            )
            raise

        if max_t == 1 and decision_rounds + delivery_weeks == 0:
            tune.report({TUNE_OBJECTIVE_METRIC: result.total_cost, _TUNE_STEP_ATTR: 1})

    search_space = _CostSearchSpaceAdapter(
        base_config=base_config,
        search_forecast=search_forecast,
        include_order_calibration=include_order_calibration,
        protection_period=lead_time + review_period,
        crc_partitions=crc_partitions,
    )

    runner = tune_runner if tune_runner is not None else run_optuna_tune

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
            _log_mlflow_params(
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
            results, study_handle = runner(
                _trainable,
                search_space,
                n_trials=n_trials,
                timeout_sec=timeout_sec,
                max_t=max_t,
                seed=seed,
                asha_grace_period=asha_grace_period,
                cpu_per_trial=cpu_per_trial,
                max_concurrent_trials=max_concurrent_trials,
                ray_address=ray_address,
                ray_local_mode=ray_local_mode,
                tune_storage_path=tune_storage_path,
                tune_experiment_name=ray_tune_experiment_name,
            )
            _best_tune_result(results)
            study = study_handle.study
            completed_trials = [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE
                and trial.value is not None
                and math.isfinite(float(trial.value))
            ]
            if completed_trials:
                mlflow.log_metric("best/cost_total", float(study.best_value))
                _log_mlflow_params(
                    {f"best.{key}": value for key, value in study.best_params.items()}
                )
                mlflow.log_dict(study.best_trial.user_attrs, "best_trial_user_attrs.json")
                with tempfile.TemporaryDirectory() as tmp:
                    trials_path = Path(tmp) / "trials.csv"
                    study.trials_dataframe().to_csv(trials_path, index=False)
                    mlflow.log_artifact(str(trials_path), artifact_path="optuna")
    else:
        results, study_handle = runner(
            _trainable,
            search_space,
            n_trials=n_trials,
            timeout_sec=timeout_sec,
            max_t=max_t,
            seed=seed,
            asha_grace_period=asha_grace_period,
            cpu_per_trial=cpu_per_trial,
            max_concurrent_trials=max_concurrent_trials,
            ray_address=ray_address,
            ray_local_mode=ray_local_mode,
            tune_storage_path=tune_storage_path,
            tune_experiment_name=ray_tune_experiment_name,
        )
        _best_tune_result(results)
        study = study_handle.study
    return study
