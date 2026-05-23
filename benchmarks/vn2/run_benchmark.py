"""Calibre's tuned VN2 benchmark — global LGBM + panel-level HPO + R,S.

This is the flagship Calibre entry: a single global LightGBM quantile
regressor exercised through ``MLForecastAdapter`` with ``strategy="direct"``,
hyper-tuned via a thin panel-level Optuna sweep, then driven through a
one-sided cumulative conformal order target.

Pipeline:

1. **Model config**: use the committed HPO-best ``BEST_CONFIG`` by default,
   or rerun the panel-level Optuna sweep over ``week_0_sales.csv`` when
   ``tune=True``. The HPO objective is cumulative-horizon pinball loss at
   the chosen quantile; the conformal order runtime then calibrates a signed
   residual around that cumulative base forecast.
2. **Decision loop** (rounds 1..N): refit the global LGBM with the best
   config at every round, apply/observe a cumulative conformal risk runtime,
   use its ``hi_*`` bound as the R,S target, then step the ``VN2Simulator``.
3. **Delivery weeks**: zero orders, just simulator.
"""

from __future__ import annotations

import logging
import math
import sys
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import benchmarks.vn2.config as _vn2_config
from benchmarks.common.tracking import (
    log_config_module,
    log_costs_dataframe,
    mlflow,
    start_benchmark_run,
)
from benchmarks.vn2.config import (
    BEST_CONFIG,
    CONFORMAL_ORDER_CONFIG,
    DATA_DIR,
    DECISION_ROUNDS,
    DELIVERY_WEEKS,
    HORIZON,
    HPO_LAG_SETS,
    HPO_N_ORIGINS,
    HPO_N_TRIALS,
    HPO_TIMEOUT_SEC,
    LEAD_TIME,
    REVIEW_PERIOD,
)
from benchmarks.vn2.data import (
    _as_cumulative_decision_frame,  # noqa: F401 – re-exported for test backward compat
    _load_instock,
    _model_uses_cumulative_target,
    _prepare_cumulative_target_history,  # noqa: F401 – re-exported for test backward compat
    _prepare_model_history,
    _prepare_policy_forecast_frame,
    _round_actuals,
    _strip_private,
)
from benchmarks.vn2.diagnostics import (
    _log_mlflow_params,
    _optimal_order_path_for_sku,  # noqa: F401 – re-exported for test backward compat
)
from benchmarks.vn2.replay import (
    VN2ReplayCache,
    _build_rs_params,
    _orders_from_policy_result,
    _run_order_conformal_warmup,
    _summary_from_simulator,
    build_replay_cache,
    replay_cached_cost,
)
from benchmarks.vn2.simulator import (
    VN2Simulator,
    extract_new_actuals,
    load_initial_states,
)
from benchmarks.vn2.tuning import (
    _TUNE_OBJECTIVE_METRIC,
    _TUNE_STEP_ATTR,
    _best_tune_result,
    _build_model_config,
    _run_optuna_tune,
    run_hpo,
)
from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig, CumulativeRiskRuntime
from calibre.conformal.partitions import global_partition, series_partition
from calibre.conformal.runtime import (
    ConformalRuntime,
    SymmetricIntervalConfig,
    build_symmetric_interval_runtime,
)
from calibre.core.forecast_frame import DS, UNIQUE_ID, quantile_column
from calibre.core.forecast_task import ForecastTask
from calibre.execution import (
    DecisionLoop,
    DecisionLoopConfig,
    RoundResult,
    observe_cumulative,
    observe_per_horizon,
)
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.data_loading import load_period
from calibre.execution.io import join_uri
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy

logger = logging.getLogger(__name__)


def _optuna_study_from_search_alg(search_alg: Any) -> optuna.Study:
    study = getattr(search_alg, "_ot_study", None)
    if study is None:
        raise RuntimeError("Ray Tune OptunaSearch did not expose a completed Optuna study")
    return study


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
    config = _build_model_config(
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


def _crc_partition_key(name: str) -> Callable:
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
) -> optuna.Study:
    """Optimize simulator EUR cost with Ray Tune over cached forecast replays."""
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
                fixed_trial, base_config, search_forecast
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
                        _TUNE_OBJECTIVE_METRIC: float(total_cost),
                        _TUNE_STEP_ATTR: step,
                        "policy_error_count": len(policy_errors),
                    }
                )

            result = replay_cached_cost(
                replay_cache,
                order_conformal_config=crc_config,
                order_base_scale=order_base_scale,
                reorder_point_scale=reorder_point_scale,
                on_policy_error=lambda rn, exc: policy_errors.append(f"round {rn}: {exc!r}"),
                on_progress=_report_progress,
                raise_on_policy_error=True,  # policy failure = bad trial, not cheap trial
            )
        except optuna.TrialPruned:
            tune.report({_TUNE_OBJECTIVE_METRIC: float("inf"), _TUNE_STEP_ATTR: 1, "pruned": 1})
            return
        except (ValueError, KeyError) as exc:
            # Bad trial (bad config or policy error) — report high cost, do not re-raise.
            tune.report(
                {_TUNE_OBJECTIVE_METRIC: float("inf"), _TUNE_STEP_ATTR: 1, "error": repr(exc)[:500]}
            )
            return
        # Infrastructure failures (Ray worker crash, import error, etc.) propagate.
        # They are NOT converted to inf; the trial fails visibly.

        if max_t == 1 and decision_rounds + delivery_weeks == 0:
            tune.report({_TUNE_OBJECTIVE_METRIC: result.total_cost, _TUNE_STEP_ATTR: 1})

    search_space = _CostSearchSpaceAdapter(
        base_config=base_config,
        search_forecast=search_forecast,
        include_order_calibration=include_order_calibration,
        protection_period=lead_time + review_period,
        crc_partitions=crc_partitions,
    )

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
            results, search_alg = _run_optuna_tune(
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
            study = _optuna_study_from_search_alg(search_alg)
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
        results, search_alg = _run_optuna_tune(
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
        study = _optuna_study_from_search_alg(search_alg)
    return study


def run_benchmark(
    data_dir: str | Path = DATA_DIR,
    horizon: int = HORIZON,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    series_filter: list[str] | None = None,
    results_dir: Path | None = None,
    verbose: bool = True,
    hpo_n_trials: int = HPO_N_TRIALS,
    hpo_n_origins: int = HPO_N_ORIGINS,
    hpo_timeout_sec: int = HPO_TIMEOUT_SEC,
    hpo_seed: int = 42,
    tune: bool = False,
    best_config: dict[str, Any] | None = None,
    conformal_config: SymmetricIntervalConfig | None = None,
    order_conformal_config: CumulativeConformalRiskConfig | None = CONFORMAL_ORDER_CONFIG,
    order_conformal_warmup_origins: int = HPO_N_ORIGINS,
    execution_backend: Literal["local", "ray", "auto"] = "auto",
    ray_address: str | None = None,
    staging_uri: str | None = None,
    ray_threshold: int = 10,
    max_concurrency: int | None = None,
    cpu_per_task: float | None = None,
) -> pd.DataFrame:
    """Run Calibre's tuned VN2 benchmark and return per-product cost summary."""
    with start_benchmark_run(
        "vn2",
        "tuned",
        tags={
            "dataset": "vn2",
            "policy": (
                "rs-capped-crc"
                if order_conformal_config is not None
                else "rs-conformal-quantile"
                if conformal_config is not None
                else "rs"
            ),
            "model_family": "lgbm",
            "horizon": str(horizon),
        },
    ):
        log_config_module(_vn2_config)
        mlflow.log_param("hpo_seed", hpo_seed)
        mlflow.log_param("tune", tune)

        initial_states = load_initial_states(join_uri(data_dir, "week_0_initial_state.csv"))
        if series_filter is not None:
            initial_states = {uid: s for uid, s in initial_states.items() if uid in series_filter}

        instock = _load_instock(data_dir, series_filter)

        if verbose:
            logger.info("Loaded %s products.", len(initial_states))

        if best_config is None:
            if tune:
                best_config = run_hpo(
                    data_dir=Path(data_dir),
                    horizon=horizon,
                    n_trials=hpo_n_trials,
                    n_origins=hpo_n_origins,
                    timeout_sec=hpo_timeout_sec,
                    series_filter=series_filter,
                    seed=hpo_seed,
                    verbose=verbose,
                )
            else:
                best_config = deepcopy(BEST_CONFIG)
        quantile_alpha = float(best_config.get("_quantile_alpha", best_config["quantiles"][0]))
        cumulative_target = _model_uses_cumulative_target(best_config)
        engine_config = _strip_private(best_config)

        if verbose:
            logger.info("Best alpha: %.3f", quantile_alpha)

        mlflow.log_param("quantile_alpha", quantile_alpha)
        mlflow.log_param("target_mode", "cumulative" if cumulative_target else "per_horizon")

        simulator = VN2Simulator(initial_states)
        target_quantile_col = quantile_column(quantile_alpha)
        order_conformal_runtime: CumulativeRiskRuntime | None = None
        conformal_runtime_obj: ConformalRuntime | None = None
        observe_fn = None

        if order_conformal_config is not None:
            resolved_order_config = replace(
                order_conformal_config,
                base_column=target_quantile_col,
                protection_period=lead_time + review_period,
            )
            order_conformal_runtime = CumulativeRiskRuntime(resolved_order_config)
            week0_sales = load_period(data_dir, 0)
            if series_filter is not None:
                week0_sales = week0_sales[week0_sales[UNIQUE_ID].isin(initial_states)]
            _run_order_conformal_warmup(
                sales=week0_sales,
                instock=instock,
                model_config=engine_config,
                horizon=horizon,
                warmup_origins=order_conformal_warmup_origins,
                runtime=order_conformal_runtime,
                series_filter=list(initial_states),
                cumulative_target=cumulative_target,
                execution_backend=execution_backend,
                ray_address=ray_address,
                staging_uri=staging_uri,
                ray_threshold=ray_threshold,
                max_concurrency=max_concurrency,
                cpu_per_task=cpu_per_task,
            )
            conformal_runtime_obj = order_conformal_runtime
            observe_fn = observe_cumulative
            mlflow.log_param("order_conformal_method", resolved_order_config.method_name)
            mlflow.log_param("order_conformal_coverage", resolved_order_config.coverage)
            mlflow.log_param("order_conformal_weight_decay", resolved_order_config.weight_decay)
            mlflow.log_param(
                "order_conformal_weighted_quantile_mode",
                resolved_order_config.weighted_quantile_mode,
            )
            mlflow.log_param("order_conformal_warmup_origins", order_conformal_warmup_origins)
        elif conformal_config is not None:
            conformal_runtime_obj = build_symmetric_interval_runtime(conformal_config)
            if conformal_config.mode == "cumulative":
                observe_fn = observe_cumulative
            else:
                lower_col, upper_col = conformal_config.interval_columns
                observe_fn = partial(observe_per_horizon, lower_col=lower_col, upper_col=upper_col)

        engine = BackendEngine(
            execution=ExecutionOptions(
                freq="W-MON",
                backend=execution_backend,
                ray_address=ray_address,
                staging_uri=staging_uri,
                ray_threshold=ray_threshold,
                max_concurrency=max_concurrency,
                cpu_per_task=cpu_per_task,
            )
        )

        def _build_round(rn: int) -> tuple[list[ForecastTask], pd.Timestamp, pd.DataFrame]:
            if verbose:
                logger.info("\n--- Decision round %s ---", rn)
            round_sales = load_period(data_dir, rn - 1)
            if series_filter is not None:
                round_sales = round_sales[round_sales[UNIQUE_ID].isin(initial_states)]
            history = _prepare_model_history(
                round_sales, instock, protection_period=horizon, cumulative_target=cumulative_target
            )
            origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)
            return (
                [ForecastTask(history=history, horizon=horizon, model_config=engine_config)],
                origin,
                round_sales,
            )

        def _policy(frame: pd.DataFrame) -> dict[str, float]:
            if frame.empty:
                if verbose:
                    logger.info("  Empty forecast, using zero orders.")
                return dict.fromkeys(initial_states, 0.0)
            try:
                if order_conformal_runtime is not None:
                    order_config = OrderPolicyConfig(
                        policy="rs",
                        params=_build_rs_params(simulator, lead_time, review_period),
                        coverage=order_conformal_runtime.config.coverage,
                    )
                elif target_quantile_col not in frame.columns:
                    if verbose:
                        logger.info("  Missing quantile column, using zero orders.")
                    return dict.fromkeys(initial_states, 0.0)
                else:
                    order_config = OrderPolicyConfig(
                        policy="rs",
                        params=_build_rs_params(simulator, lead_time, review_period),
                        quantile=quantile_alpha,
                    )
                order_result = apply_order_policy(frame, order_config)
                return _orders_from_policy_result(order_result, initial_states)
            except (ValueError, KeyError) as exc:
                if verbose:
                    logger.info("  Order computation failed: %s. Using zero orders.", exc)
                # zero orders: degraded-mode fallback for the live benchmark loop only
                return dict.fromkeys(initial_states, 0.0)

        def _get_actuals(rn: int) -> dict[str, float]:
            if rn <= decision_rounds:
                return _round_actuals(data_dir, rn, initial_states)
            try:
                actuals = extract_new_actuals(data_dir, rn)
                return {uid: actuals.get(uid, 0.0) for uid in initial_states}
            except (FileNotFoundError, ValueError):
                return dict.fromkeys(initial_states, 0.0)

        def _on_round(rr: RoundResult) -> None:
            if verbose:
                logger.info(
                    "  Origin: %s  Total order qty: %.0f",
                    rr.origin.date(),
                    sum(rr.orders.values()),
                )
            holding_cum = sum(s.cumulative_holding_cost for s in simulator.states.values())
            shortage_cum = sum(s.cumulative_shortage_cost for s in simulator.states.values())
            mlflow.log_metric("cost/holding", holding_cum, step=rr.round_num)
            mlflow.log_metric("cost/shortage", shortage_cum, step=rr.round_num)
            mlflow.log_metric("cost/total", holding_cum + shortage_cum, step=rr.round_num)

        try:
            DecisionLoop(
                engine=engine,
                simulator=simulator,
                build_round_tasks=_build_round,
                policy=_policy,
                get_actuals=_get_actuals,
                config=DecisionLoopConfig(
                    n_rounds=decision_rounds,
                    n_delivery_rounds=delivery_weeks,
                    on_round=_on_round,
                ),
                runtime=conformal_runtime_obj,
                observe_fn=observe_fn,
                ensemble=partial(
                    _prepare_policy_forecast_frame,
                    protection_period=horizon,
                    cumulative_target=cumulative_target,
                )
                if cumulative_target
                else None,
            ).run()
        finally:
            engine.close()

        summary_df = _summary_from_simulator(simulator)

        if verbose:
            total_holding = summary_df["holding_cost"].sum()
            total_shortage = summary_df["shortage_cost"].sum()
            total_cost = summary_df["total_cost"].sum()
            logger.info("\n%s", "=" * 50)
            logger.info("VN2 TUNED BENCHMARK RESULTS")
            logger.info("%s", "=" * 50)
            logger.info("Products:        %s", len(summary_df))
            logger.info("Holding cost:    EUR %s", f"{total_holding:,.2f}")
            logger.info("Shortage cost:   EUR %s", f"{total_shortage:,.2f}")
            logger.info("TOTAL COST:      EUR %s", f"{total_cost:,.2f}")
            logger.info("%s", "=" * 50)

        log_costs_dataframe(summary_df)

        if results_dir is not None:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            out_path = results_dir / "per_product_costs.csv"
            summary_df.to_csv(out_path, index=False)
            if verbose:
                logger.info("\nPer-product costs saved to: %s", out_path)

    return summary_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_benchmark(
        results_dir=Path(__file__).parent / "results",
        verbose=True,
    )
