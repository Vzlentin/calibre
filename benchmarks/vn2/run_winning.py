"""VN2 winning approach expressed entirely through Calibre's public API.

The pipeline is now:
  - ``calibre.forecasting.features.add_stockout_features`` for censored-demand imputation
  - A single ``ForecastTask`` with ``scope="global"``, ``quantiles=[0.52]``,
    ``strategy="direct"``, run through ``BackendEngine`` with the
    ``mlforecast`` backend wrapping LightGBM quantile regression.
  - ``apply_rs_policy(..., quantile=0.52)`` — order quantity is computed
    by summing the per-horizon q_0p52 forecasts across the protection
    period, bypassing the conformal upper-bound summing problem.

Usage:
    uv run python benchmarks/vn2/run_winning.py
"""

from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path

import mlflow
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import benchmarks.vn2.config as _vn2_config
from benchmarks.common.tracking import (
    log_config_module,
    log_costs_dataframe,
    start_benchmark_run,
)
from benchmarks.vn2.config import (
    DATA_DIR,
    DECISION_ROUNDS,
    DELIVERY_WEEKS,
    HORIZON,
    LEAD_TIME,
    REVIEW_PERIOD,
)
from benchmarks.vn2.simulator import (
    VN2Simulator,
    extract_new_actuals,
    load_initial_states,
)
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import RsPolicyParameters
from calibre.execution.backend import BackendEngine
from calibre.execution.data_loading import load_period, melt_wide_instock
from calibre.forecasting.features import add_stockout_features
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy

# Cost-aligned quantile: Cu / (Cu + Co) = 1.0 / (1.0 + 0.2) ≈ 0.833.
# 0.52 is the empirically-tuned per-horizon level used in the VN2 winner;
# summing 3 horizons of q_0p52 approximates the cumulative q_0p833.
QUANTILE_ALPHA = 0.52

LAGS = [1, 2, 3, 4, 13, 26, 52]
ROLLING_WINDOWS = [4, 13, 26]

MODEL_CONFIG: dict = {
    "backend": "mlforecast",
    "scope": "global",
    "name": "global_lgbm",
    "model": "lightgbm.LGBMRegressor",
    "objective": "quantile",
    "quantiles": [QUANTILE_ALPHA],
    "strategy": "direct",
    "lags": LAGS,
    "lag_transforms": {
        1: [
            *(RollingMean(window_size=w) for w in ROLLING_WINDOWS),
            *(RollingStd(window_size=w) for w in ROLLING_WINDOWS),
        ]
    },
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}


def _prepare_history(sales: pd.DataFrame, instock: pd.DataFrame | None) -> pd.DataFrame:
    """Replace observed sales with censored-demand imputed values."""
    df = add_stockout_features(sales, instock)
    return df[[UNIQUE_ID, DS, "y_uncensored"]].rename(columns={"y_uncensored": Y})


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


def _round_actuals(
    data_dir: Path,
    round_num: int,
    state_keys: Mapping[str, object],
) -> dict[str, float]:
    # round_num indexes the resolved-actuals week directly: round 1's demand
    # is week_1_sales' last column. Earlier revisions used round_num + 1.
    try:
        actuals = extract_new_actuals(data_dir, round_num)
    except (FileNotFoundError, ValueError):
        actuals = {}
    return {uid: actuals.get(uid, 0.0) for uid in state_keys}


def run_winning(
    data_dir: Path = DATA_DIR,
    horizon: int = HORIZON,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    series_filter: list[str] | None = None,
    results_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the VN2 benchmark via Calibre's MLForecastAdapter quantile path."""
    with start_benchmark_run(
        "vn2",
        "winning",
        tags={
            "dataset": "vn2",
            "policy": "rs",
            "model_family": "lgbm",
            "horizon": str(horizon),
        },
    ):
        log_config_module(_vn2_config)

        initial_states = load_initial_states(data_dir / "week_0_initial_state.csv")

        instock = None
        instock_path = data_dir / "week_0_in_stock.csv"
        if instock_path.exists():
            instock = melt_wide_instock(instock_path)

        if series_filter is not None:
            initial_states = {uid: s for uid, s in initial_states.items() if uid in series_filter}
            if instock is not None:
                instock = instock[instock[UNIQUE_ID].isin(series_filter)]

        simulator = VN2Simulator(initial_states)
        engine = BackendEngine(freq="W-MON")

        if verbose:
            print(f"Loaded {len(initial_states)} products.")

        for round_num in range(1, decision_rounds + 1):
            if verbose:
                print(f"\n--- Decision round {round_num} ---")

            # Round inputs are the previous week's resolved sales (week_{rn-1});
            # round_num itself indexes the upcoming actuals via _round_actuals.
            round_sales = load_period(data_dir, round_num - 1)
            if series_filter is not None:
                round_sales = round_sales[round_sales[UNIQUE_ID].isin(initial_states)]

            history = _prepare_history(round_sales, instock)
            # +1 week so the engine's strict `<` filter keeps the latest observation
            origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)

            task = ForecastTask(history=history, horizon=horizon, model_config=MODEL_CONFIG)
            result = engine.execute([task], actuals=round_sales, origins=[origin])
            forecast_df = result.ledger.to_df()

            actual_demand = _round_actuals(data_dir, round_num, initial_states)

            order_config = OrderPolicyConfig(
                policy="rs",
                params=_build_rs_params(simulator, lead_time, review_period),
                quantile=QUANTILE_ALPHA,
            )

            try:
                order_result = apply_order_policy(forecast_df, order_config)
                orders: dict[str, float] = dict.fromkeys(initial_states, 0.0)
                for uid, qty in zip(
                    order_result[UNIQUE_ID].astype(str),
                    order_result["order_qty"].astype(float),
                    strict=False,
                ):
                    orders[uid] = float(max(math.ceil(qty), 0))
            except (ValueError, KeyError) as exc:
                if verbose:
                    print(f"  Order computation failed: {exc}. Using zero orders.")
                orders = dict.fromkeys(initial_states, 0.0)

            if verbose:
                total_order = sum(orders.values())
                print(f"  Origin: {origin.date()}  Total order qty: {total_order:.0f}")

            simulator.step(round_num, orders=orders, actual_demand=actual_demand)

            holding_cum = shortage_cum = 0.0
            for s in simulator.states.values():
                holding_cum += s.cumulative_holding_cost
                shortage_cum += s.cumulative_shortage_cost
            mlflow.log_metric("cost/holding", holding_cum, step=round_num)
            mlflow.log_metric("cost/shortage", shortage_cum, step=round_num)
            mlflow.log_metric("cost/total", holding_cum + shortage_cum, step=round_num)

        for week_offset in range(1, delivery_weeks + 1):
            week = decision_rounds + week_offset
            if verbose:
                print(f"\n--- Delivery week {week} (no orders) ---")
            try:
                actual_demand = extract_new_actuals(data_dir, week)
                actual_demand = {uid: actual_demand.get(uid, 0.0) for uid in initial_states}
            except (FileNotFoundError, ValueError):
                actual_demand = {uid: 0.0 for uid in initial_states}
            simulator.step(
                week,
                orders={uid: 0.0 for uid in initial_states},
                actual_demand=actual_demand,
            )

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
            print("VN2 WINNING APPROACH (CALIBRE MIGRATION) RESULTS")
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
            out_path = results_dir / "per_product_costs_winning.csv"
            summary_df.to_csv(out_path, index=False)
            if verbose:
                print(f"\nPer-product costs saved to: {out_path}")

        return summary_df


if __name__ == "__main__":
    run_winning(
        results_dir=Path(__file__).parent / "results",
        verbose=True,
    )
