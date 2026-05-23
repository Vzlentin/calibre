"""Calibre's tuned VN2 benchmark orchestration shell.

Phase-4 split status:
- Data preparation and cumulative target shaping live in ``benchmarks.vn2.data``.
- Panel HPO and simulator-cost search live in ``benchmarks.vn2.tuning`` and reuse
  ``calibre.tuning.optimizer`` helpers for Optuna/Ray Tune setup and thread caps.
- Cached replay/order-policy application lives in ``benchmarks.vn2.replay``.
- Oracle and cost-attribution diagnostics live in ``benchmarks.vn2.diagnostics``.

The remaining benchmark-specific feature gaps are explicit: MLForecast
date/static feature support and exogenous ``future_x`` plumbing are still out of
scope for this wave, so the benchmark continues to use lag-based seasonal
aggregations and no future exogenous inputs.
"""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import benchmarks.vn2.config as _vn2_config
import benchmarks.vn2.tuning as _tuning
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
    HPO_N_ORIGINS,
    HPO_N_TRIALS,
    HPO_TIMEOUT_SEC,
    LEAD_TIME,
    REVIEW_PERIOD,
)
from benchmarks.vn2.data import (
    _as_cumulative_decision_frame,
    _load_instock,
    _model_uses_cumulative_target,
    _prepare_cumulative_target_history,
    _prepare_model_history,
    _prepare_policy_forecast_frame,
    _strip_private,
)
from benchmarks.vn2.diagnostics import _optimal_order_path_for_sku, oracle_diagnostic
from benchmarks.vn2.replay import (
    ReplayResult,
    VN2ReplayCache,
    _build_rs_params,
    _orders_from_policy_result,
    _round_actuals,
    _run_order_conformal_warmup,
    _summary_from_simulator,
    build_replay_cache,
    log_cached_replay_run,
    replay_cached_cost,
)
from benchmarks.vn2.simulator import VN2Simulator, extract_new_actuals, load_initial_states
from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig, CumulativeRiskRuntime
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
_TUNE_OBJECTIVE_METRIC = _tuning._TUNE_OBJECTIVE_METRIC
_run_optuna_tune = _tuning._run_optuna_tune
run_hpo = _tuning.run_hpo

__all__ = [
    "_TUNE_OBJECTIVE_METRIC",
    "_as_cumulative_decision_frame",
    "_optimal_order_path_for_sku",
    "_prepare_cumulative_target_history",
    "_round_actuals",
    "_run_order_conformal_warmup",
    "ReplayResult",
    "VN2ReplayCache",
    "build_replay_cache",
    "log_cached_replay_run",
    "oracle_diagnostic",
    "replay_cached_cost",
    "run_benchmark",
    "run_cost_search",
    "run_hpo",
]


def run_cost_search(*args: Any, **kwargs: Any):
    original = _tuning._run_optuna_tune
    _tuning._run_optuna_tune = globals().get("_run_optuna_tune", original)
    try:
        return _tuning.run_cost_search(*args, **kwargs)
    finally:
        _tuning._run_optuna_tune = original


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
    policy_error_mode: Literal["raise", "zero"] = "raise",
) -> pd.DataFrame:
    """Run Calibre's tuned VN2 benchmark and return per-product cost summary.

    Args:
        data_dir: Directory containing week_*_sales.csv and week_0_initial_state.csv.
        horizon: Forecast horizon (= lead_time + review_period).
        lead_time: Order lead time in weeks.
        review_period: Review period in weeks.
        decision_rounds: Number of active ordering rounds (1 to N).
        delivery_weeks: Number of weeks after last order (no new orders).
        series_filter: Optional list of unique_ids to restrict the benchmark.
        results_dir: If provided, save per-product CSV here.
        verbose: Print progress and cost summary.
        hpo_n_trials: Optuna trial count for the pre-HPO phase.
        hpo_n_origins: Walk-forward origins per HPO trial.
        hpo_timeout_sec: Wall-clock cap for the HPO phase.
        tune: If True, run live HPO when ``best_config`` is not provided.
            If False, use the committed ``BEST_CONFIG`` by default.
        best_config: Pre-computed model config. When provided, overrides both
            ``tune`` and the committed ``BEST_CONFIG``.
        conformal_config: Optional legacy symmetric conformal runtime config.
            When provided without ``order_conformal_config``, forecasts are
            enriched/observed online but orders still use the cost-tuned
            quantile target.
        order_conformal_config: Optional one-sided cumulative conformal risk
            config. When provided, orders are generated from the emitted
            conformal ``hi_*`` bound rather than the direct quantile path.
        order_conformal_warmup_origins: Resolved week_0 walk-forward origins
            used to seed the one-sided order conformal residual pool.
        execution_backend: Forecast scheduler backend: ``local``, ``ray``, or ``auto``.
        ray_address: Optional Ray cluster address. ``None`` starts local Ray when needed.
        staging_uri: Shared task staging URI required for remote Ray clusters.
        ray_threshold: Minimum local task count before ``auto`` uses Ray.
        max_concurrency: Optional cap on concurrent uid tasks.
        cpu_per_task: Optional CPU resources requested by each Ray worker task.
        policy_error_mode: ``"raise"`` (default) fails fast when the policy
            frame is structurally invalid or order computation raises. ``"zero"``
            is the legacy diagnostic mode that emits zero orders on these
            errors; choose it deliberately when running a degraded replay so
            broken wiring is not silently masked.

    Returns:
        DataFrame with columns: unique_id, holding_cost, shortage_cost, total_cost.
    """
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

        # ------------------------------------------------------------------ #
        # Phase 1: choose model config
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # Phase 2: Decision loop — refit each round, conformal-driven R,S
        # ------------------------------------------------------------------ #
        simulator = VN2Simulator(initial_states)
        target_quantile_col = quantile_column(quantile_alpha)
        order_conformal_runtime: CumulativeRiskRuntime | None = None
        conformal_runtime: ConformalRuntime | None = None
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
            conformal_runtime = order_conformal_runtime
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
            conformal_runtime = build_symmetric_interval_runtime(conformal_config)
            if conformal_config.mode == "cumulative":
                observe_fn = observe_cumulative
            else:
                lower_col, upper_col = conformal_config.interval_columns
                observe_fn = partial(observe_per_horizon, lower_col=lower_col, upper_col=upper_col)
        else:
            conformal_runtime = None
            observe_fn = None

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
            # Round inputs are the previous week's resolved sales (week_{rn-1});
            # round_num itself indexes the upcoming actuals via _round_actuals.
            round_sales = load_period(data_dir, rn - 1)
            if series_filter is not None:
                round_sales = round_sales[round_sales[UNIQUE_ID].isin(initial_states)]
            history = _prepare_model_history(
                round_sales,
                instock,
                protection_period=horizon,
                cumulative_target=cumulative_target,
            )
            # +1 week so the engine's strict `<` filter keeps the latest observation
            origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)
            return (
                [ForecastTask(history=history, horizon=horizon, model_config=engine_config)],
                origin,
                round_sales,
            )

        def _policy(frame: pd.DataFrame) -> dict[str, float]:
            if frame.empty:
                if policy_error_mode == "zero":
                    logger.warning(
                        "  Empty forecast frame; emitting zero orders (policy_error_mode=zero)."
                    )
                    return dict.fromkeys(initial_states, 0.0)
                raise ValueError(
                    "VN2 policy received an empty forecast frame; set "
                    "policy_error_mode='zero' to keep diagnostic replay behavior."
                )
            try:
                if order_conformal_runtime is not None:
                    order_config = OrderPolicyConfig(
                        policy="rs",
                        params=_build_rs_params(simulator, lead_time, review_period),
                        coverage=order_conformal_runtime.config.coverage,
                    )
                elif target_quantile_col not in frame.columns:
                    if policy_error_mode == "zero":
                        logger.warning(
                            "  Missing quantile column %s; emitting zero orders "
                            "(policy_error_mode=zero).",
                            target_quantile_col,
                        )
                        return dict.fromkeys(initial_states, 0.0)
                    raise KeyError(
                        f"VN2 policy frame is missing quantile column {target_quantile_col!r}; "
                        "set policy_error_mode='zero' to keep diagnostic replay behavior."
                    )
                else:
                    order_config = OrderPolicyConfig(
                        policy="rs",
                        params=_build_rs_params(simulator, lead_time, review_period),
                        quantile=quantile_alpha,
                    )
                order_result = apply_order_policy(frame, order_config)
                return _orders_from_policy_result(order_result, initial_states)
            except (ValueError, KeyError) as exc:
                if policy_error_mode == "zero":
                    logger.warning(
                        "  Order computation failed: %s. Emitting zero orders "
                        "(policy_error_mode=zero).",
                        exc,
                    )
                    return dict.fromkeys(initial_states, 0.0)
                raise

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
                runtime=conformal_runtime,
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

        # ------------------------------------------------------------------ #
        # Results
        # ------------------------------------------------------------------ #
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
