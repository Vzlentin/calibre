"""Thin runner for Calibre's tuned VN2 benchmark."""

from __future__ import annotations

import logging
import sys
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import benchmarks.vn2.config as _vn2_config
from benchmarks.common.tracking import (
    load_dotenv,
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
    load_instock,
    model_uses_cumulative_target,
    prepare_model_history,
    prepare_policy_forecast_frame,
    strip_private,
)
from benchmarks.vn2.replay import (
    build_rs_params,
    order_conformal_warmup_frames,
    orders_from_policy_result,
    round_actuals,
    summary_from_simulator,
)
from benchmarks.vn2.search import run_hpo
from benchmarks.vn2.simulator import VN2Simulator, extract_new_actuals, load_initial_states
from calibre.cli.config import BackendConfig
from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
)
from calibre.conformal.runtime import (
    ConformalRuntime,
    SymmetricIntervalConfig,
    build_symmetric_interval_runtime,
)
from calibre.core.forecast_frame import DS, UNIQUE_ID, quantile_column
from calibre.core.forecast_task import ForecastTask, TaskGroups
from calibre.core.io import join_uri, write_parquet
from calibre.execution import (
    DecisionLoop,
    DecisionLoopConfig,
    RoundResult,
    warmup_cumulative,
)
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.data_loading import load_period
from calibre.execution.task_builder import partition_tasks
from calibre.ordering.policy_config import RsConfig, apply_order_policy

logger = logging.getLogger(__name__)


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

        instock = load_instock(data_dir, series_filter)

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
        cumulative_target = model_uses_cumulative_target(best_config)
        engine_config = strip_private(best_config)

        if verbose:
            logger.info("Best alpha: %.3f", quantile_alpha)

        mlflow.log_param("quantile_alpha", quantile_alpha)
        mlflow.log_param("target_mode", "cumulative" if cumulative_target else "per_horizon")

        simulator = VN2Simulator(initial_states)
        target_quantile_col = quantile_column(quantile_alpha)
        order_conformal_runtime: CumulativeRiskRuntime | None = None
        conformal_runtime: ConformalRuntime | None = None
        if order_conformal_config is not None:
            from dataclasses import replace

            resolved_order_config = replace(
                order_conformal_config,
                base_column=target_quantile_col,
                protection_period=lead_time + review_period,
            )
            order_conformal_runtime = CumulativeRiskRuntime(resolved_order_config)
            week0_sales = load_period(data_dir, 0)
            if series_filter is not None:
                week0_sales = week0_sales[week0_sales[UNIQUE_ID].isin(initial_states)]
            warmup_cumulative(
                order_conformal_runtime,
                order_conformal_warmup_frames(
                    sales=week0_sales,
                    instock=instock,
                    model_config=engine_config,
                    horizon=horizon,
                    warmup_origins=order_conformal_warmup_origins,
                    series_filter=list(initial_states),
                    cumulative_target=cumulative_target,
                    execution_backend=execution_backend,
                    ray_address=ray_address,
                    staging_uri=staging_uri,
                    ray_threshold=ray_threshold,
                    max_concurrency=max_concurrency,
                    cpu_per_task=cpu_per_task,
                ),
            )
            conformal_runtime = order_conformal_runtime
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
        else:
            conformal_runtime = None

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

        def build_round(rn: int) -> tuple[TaskGroups, pd.Timestamp, pd.DataFrame]:
            if verbose:
                logger.info("\n--- Decision round %s ---", rn)
            round_sales = load_period(data_dir, rn - 1)
            if series_filter is not None:
                round_sales = round_sales[round_sales[UNIQUE_ID].isin(initial_states)]
            history = prepare_model_history(
                round_sales,
                instock,
                protection_period=horizon,
                cumulative_target=cumulative_target,
            )
            origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)
            return (
                partition_tasks(
                    [ForecastTask(history=history, horizon=horizon, model_config=engine_config)]
                ),
                origin,
                round_sales,
            )

        def policy(frame: pd.DataFrame) -> dict[str, float]:
            if frame.empty:
                if verbose:
                    logger.info("  Empty forecast, using zero orders.")
                return dict.fromkeys(initial_states, 0.0)
            try:
                if order_conformal_runtime is not None:
                    order_config = RsConfig(
                        params=build_rs_params(simulator, lead_time, review_period),
                        coverage=order_conformal_runtime.config.coverage,
                    )
                elif target_quantile_col not in frame.columns:
                    if verbose:
                        logger.info("  Missing quantile column, using zero orders.")
                    return dict.fromkeys(initial_states, 0.0)
                else:
                    order_config = RsConfig(
                        params=build_rs_params(simulator, lead_time, review_period),
                        quantile=quantile_alpha,
                    )
                order_result = apply_order_policy(frame, order_config)
                return orders_from_policy_result(order_result, initial_states)
            except (ValueError, KeyError) as exc:
                if verbose:
                    logger.info("  Order computation failed: %s. Using zero orders.", exc)
                return dict.fromkeys(initial_states, 0.0)

        def get_actuals(rn: int) -> dict[str, float]:
            if rn <= decision_rounds:
                return round_actuals(data_dir, rn, initial_states)
            try:
                actuals = extract_new_actuals(data_dir, rn)
                return {uid: actuals.get(uid, 0.0) for uid in initial_states}
            except (FileNotFoundError, ValueError):
                return dict.fromkeys(initial_states, 0.0)

        def on_round(rr: RoundResult) -> None:
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
                build_round_tasks=build_round,
                policy=policy,
                get_actuals=get_actuals,
                config=DecisionLoopConfig(
                    n_rounds=decision_rounds,
                    n_delivery_rounds=delivery_weeks,
                    on_round=on_round,
                ),
                runtime=conformal_runtime,
                ensemble=partial(
                    prepare_policy_forecast_frame,
                    protection_period=horizon,
                    cumulative_target=cumulative_target,
                )
                if cumulative_target
                else None,
            ).run()
        finally:
            engine.close()

        summary_df = summary_from_simulator(simulator)

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


def run_from_config(config: BackendConfig, *, tune: bool = False) -> pd.DataFrame:
    """Run the VN2 benchmark from a parsed BackendConfig (CLI mapping)."""
    summary = run_benchmark(
        data_dir=config.dataset.path,
        horizon=config.tasks[0].horizon,
        tune=tune,
        results_dir=None,
        verbose=True,
        execution_backend=config.execution.backend,
        ray_address=config.execution.ray_address,
        staging_uri=config.execution.staging_uri,
        ray_threshold=config.execution.ray_threshold,
        max_concurrency=config.execution.max_concurrency,
        cpu_per_task=config.execution.cpu_per_task,
    )
    if config.output.ledger_path is not None:
        write_parquet(summary, config.output.ledger_path)
    total_cost = float(summary["total_cost"].sum()) if "total_cost" in summary else float("nan")
    logger.info("benchmark complete", extra={"rows": len(summary), "total_cost": total_cost})
    return summary


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_benchmark(
        results_dir=Path(__file__).parent / "results",
        verbose=True,
    )
