"""VN2 seasonal-naive smoke / API-validation pipeline.

This is the legacy `run_benchmark.py` body, kept around as a reference
exercise for the parts of Calibre that the tuned pipeline bypasses:

- `ConformalRuntime` (ACI / MSCP per-horizon),
- MSCP cumulative mode,
- `apply_rs_policy` default summed-conformal / cumulative path,
- `VN2Simulator` end-to-end,
- `TuningTask` wiring (kept under the optional ``tune=True`` flag — a
  stale interface we keep alive so Phase 4's panel-level rework has a
  live caller; per-series season-length tuning is not useful in itself).

End-to-end pipeline:
    1. (optional) TUNING — per-series Optuna sweep over SeasonalNaive's
       ``season_length`` (off by default).
    2. WARMUP — walk-forward forecasts on week_0 history to calibrate
       the conformal runtime.
    3. DECISION ROUNDS (1-6) — load updated sales, forecast, conformal
       intervals, R,S order, simulator step, observe.
    4. DELIVERY WEEKS (7-8) — no new orders; step simulator with zero
       orders.
    5. RESULTS — print cost breakdown, optionally save per-product CSV.
"""

from __future__ import annotations

import contextlib
import math
import sys
from functools import partial
from pathlib import Path

import mlflow
import pandas as pd

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import benchmarks.vn2.config as _vn2_config
from benchmarks.common.tracking import (
    log_config_module,
    log_costs_dataframe,
    start_benchmark_run,
)
from benchmarks.vn2.config import (
    CONFORMAL_CONFIG,
    DATA_DIR,
    DECISION_ROUNDS,
    DELIVERY_WEEKS,
    HORIZON,
    LEAD_TIME,
    MODEL_CONFIGS,
    REVIEW_PERIOD,
    TUNE_BASE_CONFIG,
    TUNE_MAX_WORKERS,
    TUNE_N_ORIGINS,
    TUNE_N_TRIALS,
    WARMUP_ORIGINS,
)
from benchmarks.vn2.simulator import (
    VN2Simulator,
    extract_new_actuals,
    load_initial_states,
)
from benchmarks.vn2.tuning import seasonal_naive_search_space, tune_all_series
from calibre.conformal.runtime import ConformalPolicyConfig, ConformalRuntime
from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y,
    interval_column_names,
)
from calibre.engine.backend import BackendEngine
from calibre.ensemble.median import ensemble_median
from calibre.orchestration import (
    DecisionLoop,
    DecisionLoopConfig,
    RoundResult,
    observe_cumulative,
    observe_per_horizon,
)
from calibre.order.config import OrderPolicyConfig, apply_order_policy
from calibre.order.types import RsPolicyParameters
from calibre.pipeline.loading import load_period
from calibre.pipeline.tasks import build_tasks
from calibre.tasks.forecast_task import ForecastTask


def _build_rs_params(
    simulator: VN2Simulator,
    lead_time: int,
    review_period: int,
) -> list[RsPolicyParameters]:
    return [
        RsPolicyParameters(
            unique_id=uid,
            inventory_position=s.end_inventory + s.in_transit_w1 + s.in_transit_w2,
            lead_time=lead_time,
            review_period=review_period,
        )
        for uid, s in simulator.states.items()
    ]


def _split_ready(
    frame: pd.DataFrame,
    mode: str,
    lower_col: str,
    upper_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a pending-forecast frame into (to_observe, still_unresolved).

    Cumulative mode: a row is ready iff every row in its
    (uid, model, origin) window has a resolved actual.
    Per-horizon mode: a row is ready iff it has both an actual and intervals.
    """
    if mode == "cumulative":
        group_keys = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]
        grouped_y = frame.groupby(group_keys, sort=False)[Y]
        window_complete = grouped_y.transform("count").eq(grouped_y.transform("size"))
        return frame[window_complete], frame[~window_complete]
    resolved = frame[Y].notna() & frame[lower_col].notna() & frame[upper_col].notna()
    return frame[resolved], frame[~resolved]


def _fill_actuals(
    frame: pd.DataFrame,
    lookup: pd.Series,
) -> pd.DataFrame:
    """Vectorized fill of NaN y from a (uid, ds)-indexed Series."""
    if frame.empty or not frame[Y].isna().any():
        return frame
    keys = pd.MultiIndex.from_arrays([frame[UNIQUE_ID].values, frame[DS].values])
    filled = lookup.reindex(keys).to_numpy()
    result = frame.copy()
    missing_mask = result[Y].isna().to_numpy()
    result.loc[missing_mask, Y] = filled[missing_mask]
    return result


def _run_warmup(
    sales: pd.DataFrame,
    model_configs: list[dict],
    horizon: int,
    warmup_origins: int,
    conformal_config: ConformalPolicyConfig,
    series_filter: list[str] | None,
) -> ConformalRuntime:
    """Run walk-forward warmup to calibrate the conformal runtime.

    Generates raw multi-model forecasts across the last `warmup_origins`
    walk-forward origins, ensembles them, and feeds each resolved prediction
    through apply/observe to build calibration state.

    Returns a ConformalRuntime with calibration history accumulated.
    """
    conformal_runtime = ConformalRuntime(conformal_config)

    if warmup_origins == 0:
        return conformal_runtime

    all_dates = sorted(sales[DS].unique())
    if len(all_dates) < warmup_origins + horizon:
        warmup_origins = max(1, len(all_dates) - horizon)

    origin_dates = [pd.Timestamp(d) for d in all_dates[-(warmup_origins + horizon) : -horizon]]
    if not origin_dates:
        return conformal_runtime

    tasks = build_tasks(
        sales,
        model_configs,
        horizon=horizon,
        series_filter=series_filter,
    )

    engine = BackendEngine(freq="W-MON")
    result = engine.execute(tasks, actuals=sales, origins=origin_dates)
    ledger_df = result.ledger.to_df()

    if ledger_df.empty:
        return conformal_runtime

    ensemble_df = ensemble_median(ledger_df)

    lower_col, upper_col = interval_column_names(conformal_config.coverage)
    actuals_lookup = sales.drop_duplicates(subset=[UNIQUE_ID, DS]).set_index([UNIQUE_ID, DS])[Y]

    pending_applied: list[pd.DataFrame] = []

    for origin in origin_dates:
        origin_rows = ensemble_df[ensemble_df[FORECAST_ORIGIN] == origin]
        if origin_rows.empty:
            continue

        applied = conformal_runtime.apply(origin_rows)
        if lower_col not in applied.columns:
            continue

        still_pending: list[pd.DataFrame] = []
        for prev_applied in pending_applied:
            if not (prev_applied[DS] <= origin).any():
                still_pending.append(prev_applied)
                continue

            updated = _fill_actuals(prev_applied, actuals_lookup)
            to_observe, unresolved = _split_ready(
                updated, conformal_config.mode, lower_col, upper_col
            )
            if not to_observe.empty:
                with contextlib.suppress(ValueError):
                    conformal_runtime.observe(to_observe)
            if not unresolved.empty:
                still_pending.append(unresolved)

        pending_applied = still_pending + [applied]

    return conformal_runtime


def run_seasonal(
    data_dir: Path = DATA_DIR,
    model_configs: list[dict] = MODEL_CONFIGS,
    conformal_config: ConformalPolicyConfig = CONFORMAL_CONFIG,
    horizon: int = HORIZON,
    warmup_origins: int = WARMUP_ORIGINS,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    series_filter: list[str] | None = None,
    results_dir: Path | None = None,
    verbose: bool = True,
    tune: bool = False,
    tune_base_config: dict = TUNE_BASE_CONFIG,
    tune_n_trials: int = TUNE_N_TRIALS,
    tune_n_origins: int = TUNE_N_ORIGINS,
    tune_max_workers: int = TUNE_MAX_WORKERS,
) -> pd.DataFrame:
    """Run the seasonal-naive smoke benchmark and return per-product cost summary.

    Args:
        data_dir: Directory containing week_*_sales.csv and week_0_initial_state.csv.
        model_configs: List of model config dicts for BackendEngine tasks.
            Defaults to a single SeasonalNaive (no ensemble).
        conformal_config: ConformalPolicyConfig for ACI/MSCP.
        horizon: Forecast horizon (= lead_time + review_period).
        warmup_origins: Number of walk-forward origins to use for conformal warmup.
        lead_time: Order lead time in weeks.
        review_period: Review period in weeks.
        decision_rounds: Number of active ordering rounds (1 to N).
        delivery_weeks: Number of weeks after last order (no new orders, just simulation).
        series_filter: Optional list of unique_ids to restrict the benchmark.
        results_dir: If provided, save per-product CSV here.
        verbose: Print progress and cost summary.
        tune: If True, run per-series HPO and add the tuned model to the ensemble.
            Off by default — `season_length=52` is a given for VN2; the flag stays
            so the `TuningTask` wiring keeps a live caller.
        tune_base_config: Base model config to tune (default: SeasonalNaive).
        tune_n_trials: Number of Optuna trials per series.
        tune_n_origins: Number of walk-forward origins per tuning trial.
        tune_max_workers: Maximum parallel threads for tuning.

    Returns:
        DataFrame with columns: unique_id, holding_cost, shortage_cost, total_cost.
    """
    with start_benchmark_run(
        "vn2",
        "seasonal",
        tags={
            "dataset": "vn2",
            "policy": "rs-conformal",
            "model_family": "seasonal_naive",
            "horizon": str(horizon),
        },
    ):
        log_config_module(_vn2_config)
        lower_col, upper_col = interval_column_names(conformal_config.coverage)

        initial_state_path = data_dir / "week_0_initial_state.csv"
        all_states = load_initial_states(initial_state_path)

        if series_filter is not None:
            states = {uid: s for uid, s in all_states.items() if uid in series_filter}
        else:
            states = all_states

        simulator = VN2Simulator(states)

        if verbose:
            print(f"Loaded {len(states)} products from initial state.")

        week0_sales = load_period(data_dir, 0)
        if series_filter is not None:
            week0_sales = week0_sales[week0_sales[UNIQUE_ID].isin(series_filter)]

        tuned_configs: dict[str, dict] = {}
        if tune:
            if verbose:
                print(
                    f"Tuning {len(states)} series "
                    f"({tune_n_trials} trials x {tune_n_origins} origins, "
                    f"{tune_max_workers} workers)..."
                )
            tuned_configs = tune_all_series(
                sales=week0_sales,
                horizon=horizon,
                base_config=tune_base_config,
                search_space=seasonal_naive_search_space,
                n_trials=tune_n_trials,
                n_origins=tune_n_origins,
                max_workers=tune_max_workers,
            )
            if verbose:
                print(f"Tuning complete. {len(tuned_configs)} series tuned.")

        if verbose:
            print(f"Running warmup with {warmup_origins} origins...")

        conformal_runtime = _run_warmup(
            sales=week0_sales,
            model_configs=model_configs,
            horizon=horizon,
            warmup_origins=warmup_origins,
            conformal_config=conformal_config,
            series_filter=series_filter,
        )

        if verbose:
            n_calibrated = len(conformal_runtime._policies)
            print(f"Warmup complete. Calibrated {n_calibrated} conformal policies.")

        engine = BackendEngine(freq="W-MON")

        if conformal_config.mode == "cumulative":
            observe_fn = observe_cumulative
        else:
            observe_fn = partial(observe_per_horizon, lower_col=lower_col, upper_col=upper_col)

        def _build_round(rn: int) -> tuple[list[ForecastTask], pd.Timestamp, pd.DataFrame]:
            if verbose:
                print(f"\n--- Decision round {rn} ---")
            round_sales = load_period(data_dir, rn - 1)
            if series_filter is not None:
                round_sales = round_sales[round_sales[UNIQUE_ID].isin(series_filter)]
            # +1 week so the engine's strict `<` filter keeps the latest observation.
            origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)
            tasks = build_tasks(
                round_sales, model_configs, horizon=horizon, series_filter=list(states.keys())
            )
            if tuned_configs:
                sorted_sales = round_sales.sort_values([UNIQUE_ID, DS])
                for uid_key, series_data in sorted_sales.groupby(UNIQUE_ID, sort=False):
                    uid = str(uid_key)
                    if uid not in tuned_configs or series_data.empty:
                        continue
                    history = series_data[[UNIQUE_ID, DS, Y]].reset_index(drop=True)
                    tasks.append(
                        ForecastTask(
                            history=history, horizon=horizon, model_config=tuned_configs[uid]
                        )
                    )
            return tasks, origin, round_sales

        def _policy(frame: pd.DataFrame) -> dict[str, float]:
            if lower_col not in frame.columns or upper_col not in frame.columns:
                if verbose:
                    print("  No interval columns yet (warmup still needed). Using zero orders.")
                return {uid: 0.0 for uid in states}
            try:
                order_result = apply_order_policy(
                    frame,
                    OrderPolicyConfig(
                        policy="rs",
                        params=_build_rs_params(
                            simulator, lead_time=lead_time, review_period=review_period
                        ),
                        coverage=conformal_config.coverage,
                    ),
                )
                orders: dict[str, float] = {}
                for uid, qty in zip(
                    order_result[UNIQUE_ID].astype(str),
                    order_result["order_qty"].astype(float),
                    strict=False,
                ):
                    orders[uid] = float(max(math.ceil(qty), 0))
                return orders
            except (ValueError, KeyError) as exc:
                if verbose:
                    print(f"  Order policy failed: {exc}. Using zero orders.")
                return {uid: 0.0 for uid in states}

        def _get_actuals(rn: int) -> dict[str, float]:
            week = rn
            try:
                actuals = extract_new_actuals(data_dir, week)
                return {uid: actuals.get(uid, 0.0) for uid in states}
            except (FileNotFoundError, ValueError):
                if rn <= decision_rounds:
                    round_raw = pd.read_csv(data_dir / f"week_{rn}_sales.csv")
                    date_cols = [c for c in round_raw.columns if c not in ("Store", "Product")]
                    last_col = date_cols[-1]
                    unique_ids = (
                        round_raw["Store"].astype(int).astype(str)
                        + "_"
                        + round_raw["Product"].astype(int).astype(str)
                    )
                    fallback = dict(
                        zip(unique_ids, round_raw[last_col].fillna(0.0).astype(float), strict=False)
                    )
                    return {uid: fallback.get(uid, 0.0) for uid in states}
                return {uid: 0.0 for uid in states}

        def _on_round(rr: RoundResult) -> None:
            if verbose:
                print(
                    f"  Origin: {rr.origin.date()}  Total order qty: {sum(rr.orders.values()):.0f}"
                )
            holding_cum = sum(s.cumulative_holding_cost for s in simulator.states.values())
            shortage_cum = sum(s.cumulative_shortage_cost for s in simulator.states.values())
            mlflow.log_metric("cost/holding", holding_cum, step=rr.round_num)
            mlflow.log_metric("cost/shortage", shortage_cum, step=rr.round_num)
            mlflow.log_metric("cost/total", holding_cum + shortage_cum, step=rr.round_num)

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
            ensemble=ensemble_median,
            observe_fn=observe_fn,
        ).run()

        rows = []
        for uid, state in simulator.states.items():
            rows.append(
                {
                    "unique_id": uid,
                    "holding_cost": state.cumulative_holding_cost,
                    "shortage_cost": state.cumulative_shortage_cost,
                    "total_cost": state.cumulative_holding_cost + state.cumulative_shortage_cost,
                }
            )
        summary_df = pd.DataFrame(rows).sort_values("unique_id").reset_index(drop=True)

        if verbose:
            total_holding = summary_df["holding_cost"].sum()
            total_shortage = summary_df["shortage_cost"].sum()
            total_cost = summary_df["total_cost"].sum()
            print("\n" + "=" * 50)
            print("VN2 SEASONAL-NAIVE SMOKE RESULTS")
            print("=" * 50)
            print(f"Products:        {len(summary_df)}")
            print(f"Holding cost:    EUR {total_holding:,.2f}")
            print(f"Shortage cost:   EUR {total_shortage:,.2f}")
            print(f"TOTAL COST:      EUR {total_cost:,.2f}")
            print("=" * 50)

        log_costs_dataframe(summary_df)

        if results_dir is not None:
            results_dir = Path(results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            out_path = results_dir / "per_product_costs_seasonal.csv"
            summary_df.to_csv(out_path, index=False)
            if verbose:
                print(f"\nPer-product costs saved to: {out_path}")

        return summary_df


if __name__ == "__main__":
    run_seasonal(
        results_dir=Path(__file__).parent / "results",
        verbose=True,
    )
