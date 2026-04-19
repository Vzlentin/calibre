"""VN2 inventory planning benchmark orchestrator.

End-to-end pipeline:
    1. WARMUP  — run walk-forward forecasts on week_0 history to calibrate
                 the conformal runtime (ACI per series).
    2. DECISION ROUNDS (1-6) — for each round:
         a. Load updated sales data.
         b. Forecast with multi-model ensemble.
         c. Apply conformal intervals.
         d. Compute R,S order quantities from current inventory position.
         e. Step the simulator with realised actuals.
         f. Feed resolved h=1 forecasts back to conformal runtime.
    3. DELIVERY WEEKS (7-8) — no new orders; step simulator with zero orders.
    4. RESULTS — print cost breakdown, save per-product CSV.
"""

from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path

import pandas as pd

# Allow running as a script from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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
    """Build R,S policy parameters from current simulator state."""
    params = []
    for uid, state in simulator.states.items():
        inventory_position = state.end_inventory + state.in_transit_w1 + state.in_transit_w2
        params.append(
            RsPolicyParameters(
                unique_id=uid,
                inventory_position=inventory_position,
                lead_time=lead_time,
                review_period=review_period,
            )
        )
    return params


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

    # Identify walk-forward origins from week_0 history
    all_dates = sorted(sales[DS].unique())
    if len(all_dates) < warmup_origins + horizon:
        warmup_origins = max(1, len(all_dates) - horizon)

    # Use the last warmup_origins dates as forecast origins (leave horizon for actuals)
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

    # Ensemble the multi-model predictions
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


def run_benchmark(
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
    tune: bool = True,
    tune_base_config: dict = TUNE_BASE_CONFIG,
    tune_n_trials: int = TUNE_N_TRIALS,
    tune_n_origins: int = TUNE_N_ORIGINS,
    tune_max_workers: int = TUNE_MAX_WORKERS,
) -> pd.DataFrame:
    """Run the full VN2 benchmark and return per-product cost summary.

    Args:
        data_dir: Directory containing week_*_sales.csv and week_0_initial_state.csv.
        model_configs: List of model config dicts for BackendEngine tasks.
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
        tune_base_config: Base model config to tune (default: SeasonalNaive).
        tune_n_trials: Number of Optuna trials per series.
        tune_n_origins: Number of walk-forward origins per tuning trial.
        tune_max_workers: Maximum parallel threads for tuning.

    Returns:
        DataFrame with columns: unique_id, holding_cost, shortage_cost, total_cost.
    """
    lower_col, upper_col = interval_column_names(conformal_config.coverage)

    # ------------------------------------------------------------------ #
    # Load initial inventory state
    # ------------------------------------------------------------------ #
    initial_state_path = data_dir / "week_0_initial_state.csv"
    all_states = load_initial_states(initial_state_path)

    if series_filter is not None:
        states = {uid: s for uid, s in all_states.items() if uid in series_filter}
    else:
        states = all_states

    simulator = VN2Simulator(states)

    if verbose:
        print(f"Loaded {len(states)} products from initial state.")

    # ------------------------------------------------------------------ #
    # Phase 0: TUNING — per-series HPO on historical data
    # ------------------------------------------------------------------ #
    week0_sales = load_period(data_dir, 0)
    if series_filter is not None:
        week0_sales = week0_sales[week0_sales[UNIQUE_ID].isin(series_filter)]

    # tuned_configs maps unique_id → best model config; empty dict if tune=False
    tuned_configs: dict[str, dict] = {}
    if tune:
        if verbose:
            print(
                f"Tuning {len(states)} series "
                f"({tune_n_trials} trials × {tune_n_origins} origins, "
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

    # ------------------------------------------------------------------ #
    # Phase 1: WARMUP — calibrate conformal runtime
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Phase 2: DECISION ROUNDS
    # ------------------------------------------------------------------ #
    # Track the most recent ensemble+interval predictions per (uid, origin)
    # so we can fill in actuals and call observe() when they resolve. Rows
    # are dropped once their actuals are filled in (per-horizon scores) or
    # once the full protection-period window is resolved (cumulative).
    pending_forecasts: list[pd.DataFrame] = []
    actuals_lookup: dict[tuple[str, pd.Timestamp], float] = {}

    engine = BackendEngine(freq="W-MON")

    for round_num in range(1, decision_rounds + 1):
        if verbose:
            print(f"\n--- Decision round {round_num} ---")

        # Load updated sales data for this round
        round_sales = load_period(data_dir, round_num)
        if series_filter is not None:
            round_sales = round_sales[round_sales[UNIQUE_ID].isin(series_filter)]

        # Forecast from the latest date in round sales
        latest_date = round_sales[DS].max()
        origin = pd.Timestamp(latest_date)

        tasks = build_tasks(
            round_sales,
            model_configs,
            horizon=horizon,
            series_filter=list(states.keys()),
        )
        if tuned_configs:
            sorted_sales = round_sales.sort_values([UNIQUE_ID, DS])
            for uid, series_data in sorted_sales.groupby(UNIQUE_ID, sort=False):
                if uid not in tuned_configs or series_data.empty:
                    continue
                history = series_data[[UNIQUE_ID, DS, Y]].reset_index(drop=True)
                tasks.append(
                    ForecastTask(
                        history=history,
                        horizon=horizon,
                        model_config=tuned_configs[uid],
                    )
                )

        result = engine.execute(tasks, actuals=round_sales, origins=[origin])
        raw_ledger = result.ledger.to_df()

        if raw_ledger.empty:
            if verbose:
                print(f"  Round {round_num}: empty forecast ledger, skipping.")
            continue

        # Ensemble the multi-model predictions
        ensemble_df = ensemble_median(raw_ledger)

        # Apply conformal intervals
        applied_df = conformal_runtime.apply(ensemble_df)

        # Store for later observe()
        if lower_col in applied_df.columns and upper_col in applied_df.columns:
            pending_forecasts.append(applied_df.copy())

        # ------------------------------------------------------------------ #
        # Get actual demand for this round
        # ------------------------------------------------------------------ #
        # The next week's sales file contains the new column for this round's actuals
        next_week = round_num + 1
        try:
            actual_demand = extract_new_actuals(data_dir, next_week)
        except (FileNotFoundError, ValueError):
            # Fall back to the last column of the round's own sales file
            round_raw = pd.read_csv(data_dir / f"week_{round_num}_sales.csv")
            date_cols = [c for c in round_raw.columns if c not in ("Store", "Product")]
            last_col = date_cols[-1]
            unique_ids = (
                round_raw["Store"].astype(int).astype(str)
                + "_"
                + round_raw["Product"].astype(int).astype(str)
            )
            actual_demand = dict(
                zip(unique_ids, round_raw[last_col].fillna(0.0).astype(float), strict=False)
            )

        # Filter demand to our products
        actual_demand = {uid: actual_demand.get(uid, 0.0) for uid in states}

        # ------------------------------------------------------------------ #
        # Compute R,S orders from current inventory position
        # ------------------------------------------------------------------ #
        rs_params = _build_rs_params(simulator, lead_time=lead_time, review_period=review_period)
        order_config = OrderPolicyConfig(
            policy="rs",
            params=rs_params,
            coverage=conformal_config.coverage,
        )

        if lower_col in applied_df.columns and upper_col in applied_df.columns:
            try:
                order_result = apply_order_policy(applied_df, order_config)
                orders: dict[str, float] = {}
                for _, row in order_result.iterrows():
                    uid = str(row[UNIQUE_ID])
                    qty = math.ceil(float(row.get("order_qty", 0.0)))
                    orders[uid] = max(float(qty), 0.0)
            except (ValueError, KeyError) as exc:
                if verbose:
                    print(f"  Order policy failed: {exc}. Using zero orders.")
                orders = {uid: 0.0 for uid in states}
        else:
            if verbose:
                print("  No interval columns yet (warmup still needed). Using zero orders.")
            orders = {uid: 0.0 for uid in states}

        if verbose:
            total_order = sum(orders.values())
            print(f"  Origin: {origin.date()}  Total order qty: {total_order:.0f}")

        # ------------------------------------------------------------------ #
        # Step simulator
        # ------------------------------------------------------------------ #
        simulator.step(round_num, orders=orders, actual_demand=actual_demand)

        # ------------------------------------------------------------------ #
        # Feed resolved forecasts to conformal observe()
        # ------------------------------------------------------------------ #
        # Each round delivers actuals for one new period; record them keyed
        # by the date they apply to (origin + 1 week == ds for h=1 of the
        # forecast just issued at this origin).
        actuals_ds = pd.Timestamp(origin) + pd.Timedelta(weeks=1)
        for uid, demand in actual_demand.items():
            actuals_lookup[(uid, actuals_ds)] = float(demand)

        lookup_series = pd.Series(actuals_lookup, dtype=float)
        if not lookup_series.empty:
            lookup_series.index = pd.MultiIndex.from_tuples(lookup_series.index)

        still_pending: list[pd.DataFrame] = []
        for prev_forecast in pending_forecasts:
            updated = _fill_actuals(prev_forecast, lookup_series)
            to_observe, still_unresolved = _split_ready(
                updated, conformal_config.mode, lower_col, upper_col
            )
            if not to_observe.empty:
                with contextlib.suppress(ValueError):
                    conformal_runtime.observe(to_observe)
            if not still_unresolved.empty:
                still_pending.append(still_unresolved)

        pending_forecasts = still_pending

    # ------------------------------------------------------------------ #
    # Phase 3: DELIVERY WEEKS — no orders, just simulation
    # ------------------------------------------------------------------ #
    for week_offset in range(1, delivery_weeks + 1):
        week = decision_rounds + week_offset
        if verbose:
            print(f"\n--- Delivery week {week} (no orders) ---")
        try:
            actual_demand = extract_new_actuals(data_dir, week)
            actual_demand = {uid: actual_demand.get(uid, 0.0) for uid in states}
        except (FileNotFoundError, ValueError):
            actual_demand = {uid: 0.0 for uid in states}

        simulator.step(week, orders={uid: 0.0 for uid in states}, actual_demand=actual_demand)

    # ------------------------------------------------------------------ #
    # Phase 4: RESULTS
    # ------------------------------------------------------------------ #
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
        print("VN2 BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Products:        {len(summary_df)}")
        print(f"Holding cost:    €{total_holding:,.2f}")
        print(f"Shortage cost:   €{total_shortage:,.2f}")
        print(f"TOTAL COST:      €{total_cost:,.2f}")
        print("=" * 50)

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "per_product_costs.csv"
        summary_df.to_csv(out_path, index=False)
        if verbose:
            print(f"\nPer-product costs saved to: {out_path}")

    return summary_df


if __name__ == "__main__":
    results = run_benchmark(
        results_dir=Path(__file__).parent / "results",
        verbose=True,
    )
