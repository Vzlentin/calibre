"""VN2 winning approach: global LightGBM + Calibre conformal + cost-aware ordering.

Reproduces the key elements of the VN2 1st-place solution within Calibre:

  Forecasting stage:
    - Single global LightGBM model trained on all 599 series jointly
    - Stockout-aware feature engineering (censored demand imputation)
    - Per-series dynamic scaling to reduce magnitude imbalance
    - Rich lag/rolling/calendar features
    - Direct multi-horizon strategy (one model per horizon step)
    - Time-decayed sample weights

  Decision stage:
    - Calibre's Adaptive Conformal Inference for uncertainty quantification
    - Cost-aware ordering with critical fractile Cu/(Cu+Co) = 0.833
    - Inventory projection over the protection period (lead_time + review)

  This combines the winner's forecasting approach with Calibre's conformal
  prediction — adding distribution-free, adaptive uncertainty bands on top
  of strong point forecasts.

Usage:
    uv run python benchmarks/vn2/run_winning.py
"""

from __future__ import annotations

import contextlib
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from benchmarks.vn2.features import build_training_frame
from benchmarks.vn2.simulator import (
    VN2Simulator,
    extract_new_actuals,
    load_initial_states,
)
from calibre.conformal.runtime import ConformalPolicyConfig, ConformalRuntime
from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)
from calibre.order.config import OrderPolicyConfig, apply_order_policy
from calibre.order.types import RsPolicyParameters
from calibre.pipeline.loading import load_master, load_period, melt_wide_instock

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vn2"

HORIZON = 3  # lead_time(2) + review_period(1)
LEAD_TIME = 2
REVIEW_PERIOD = 1
DECISION_ROUNDS = 6
DELIVERY_WEEKS = 2
WARMUP_ORIGINS = 20

# Cost-optimal service level: Cu/(Cu+Co) = 1.0/(1.0+0.2) ≈ 0.833
COVERAGE = 0.833

CONFORMAL_CONFIG = ConformalPolicyConfig(
    method="aci",
    coverage=COVERAGE,
    gamma=0.05,
    calibration_window=50,
)

# LightGBM hyperparameters (aligned with VN2 winner's GBDT approach)
LGB_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

# Feature configuration
LAGS = [1, 2, 3, 4, 13, 26, 52]
ROLLING_WINDOWS = [4, 13, 26]
HALF_LIFE_WEEKS = 52

# Columns that are features for the model (everything except identifiers/targets/weights)
_ID_COLS = {
    UNIQUE_ID,
    DS,
    Y,
    "y_uncensored",
    "y_scaled",
    "sample_weight",
    "series_mean",
    "series_std",
}


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return feature column names from a training frame."""
    return [c for c in df.columns if c not in _ID_COLS]


# ---------------------------------------------------------------------------
# Global LightGBM with direct multi-horizon
# ---------------------------------------------------------------------------


def _prepare_horizon_target(
    df: pd.DataFrame,
    h: int,
    target_col: str = "y_scaled",
) -> pd.DataFrame:
    """Shift target by h steps to create the direct-horizon training target.

    For horizon h, the target at time t is the scaled demand at time t+h.
    The model learns normalised patterns; predictions are unscaled at inference.
    """
    out = df.copy()
    out[f"target_h{h}"] = out.groupby(UNIQUE_ID, sort=False)[target_col].shift(-h)
    return out


def train_global_models(
    training_frame: pd.DataFrame,
    horizon: int = HORIZON,
    lgb_params: dict = LGB_PARAMS,
) -> list[lgb.LGBMRegressor]:
    """Train one LightGBM model per horizon step (direct multi-horizon strategy).

    Returns a list of H fitted models where models[h-1] predicts demand at t+h.
    """
    feature_cols = _get_feature_cols(training_frame)
    models = []

    for h in range(1, horizon + 1):
        df_h = _prepare_horizon_target(training_frame, h)

        target_col = f"target_h{h}"
        valid = df_h[target_col].notna() & df_h[feature_cols].notna().all(axis=1)
        train = df_h[valid]

        if train.empty:
            raise ValueError(f"No valid training rows for horizon h={h}")

        X = train[feature_cols]
        y_target = train[target_col]
        weights = train["sample_weight"] if "sample_weight" in train.columns else None

        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, y_target, sample_weight=weights)
        models.append(model)

    return models


def predict_global(
    models: list[lgb.LGBMRegressor],
    latest_frame: pd.DataFrame,
    origin: pd.Timestamp,
) -> pd.DataFrame:
    """Generate forecasts from the latest observation for all series.

    Uses the most recent row per series as the feature vector, then predicts
    each horizon with its dedicated model.

    Returns a forecast-frame-compatible DataFrame.
    """
    feature_cols = _get_feature_cols(latest_frame)

    # Use the most recent row per series as the prediction input
    latest = (
        latest_frame.sort_values([UNIQUE_ID, DS])
        .groupby(UNIQUE_ID, sort=False)
        .last()
        .reset_index()
    )

    rows = []
    for h_idx, model in enumerate(models):
        h = h_idx + 1
        X = latest[feature_cols]
        preds = model.predict(X)

        # Unscale predictions: y_hat = pred * series_std + series_mean
        if "series_mean" in latest.columns and "series_std" in latest.columns:
            preds_unscaled = preds * latest["series_std"].values + latest["series_mean"].values
        else:
            preds_unscaled = preds

        # Clip to non-negative (demand can't be negative)
        preds_unscaled = np.maximum(preds_unscaled, 0.0)

        # Compute forecast dates (origin + h weeks)
        forecast_dates = origin + pd.Timedelta(weeks=h)

        for i, uid in enumerate(latest[UNIQUE_ID].values):
            rows.append(
                {
                    UNIQUE_ID: str(uid),
                    DS: forecast_dates,
                    Y: np.nan,
                    Y_HAT: float(preds_unscaled[i]),
                    H: h,
                    FORECAST_ORIGIN: origin,
                    MODEL_NAME: "global_lgbm",
                }
            )

    result = pd.DataFrame(rows)
    result[DS] = pd.to_datetime(result[DS])
    result[FORECAST_ORIGIN] = pd.to_datetime(result[FORECAST_ORIGIN])
    result[Y_HAT] = result[Y_HAT].astype("float64")
    result[Y] = result[Y].astype("float64")
    result[H] = result[H].astype("int64")
    result[UNIQUE_ID] = result[UNIQUE_ID].astype("object")
    result[MODEL_NAME] = result[MODEL_NAME].astype("object")

    return result


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Warmup: calibrate conformal runtime on historical walk-forward forecasts
# ---------------------------------------------------------------------------


def _run_warmup(
    models: list[lgb.LGBMRegressor],
    training_frame: pd.DataFrame,
    sales: pd.DataFrame,
    conformal_config: ConformalPolicyConfig,
    warmup_origins: int,
    horizon: int,
    series_filter: list[str] | None,
) -> ConformalRuntime:
    """Walk-forward warmup to calibrate conformal intervals on historical data."""
    conformal_runtime = ConformalRuntime(conformal_config)
    if warmup_origins == 0:
        return conformal_runtime

    lower_col, upper_col = interval_column_names(conformal_config.coverage)

    all_dates = sorted(sales[DS].unique())
    if len(all_dates) < warmup_origins + horizon:
        warmup_origins = max(1, len(all_dates) - horizon)

    origin_dates = [pd.Timestamp(d) for d in all_dates[-(warmup_origins + horizon) : -horizon]]
    if not origin_dates:
        return conformal_runtime

    # Actuals lookup for resolving predictions
    actuals_lookup = sales.drop_duplicates(subset=[UNIQUE_ID, DS]).set_index([UNIQUE_ID, DS])[Y]

    pending_applied: list[pd.DataFrame] = []

    for origin in origin_dates:
        # Build feature frame up to this origin
        origin_frame = training_frame[training_frame[DS] <= origin].copy()
        if series_filter:
            origin_frame = origin_frame[origin_frame[UNIQUE_ID].isin(series_filter)]

        if origin_frame.empty:
            continue

        # Predict from this origin
        forecast_df = predict_global(models, origin_frame, origin)

        # Apply conformal intervals
        applied = conformal_runtime.apply(forecast_df)
        if lower_col not in applied.columns:
            continue

        pending_applied.append(applied)

        # Resolve previously applied predictions whose ds <= current origin
        still_pending = []
        for prev_applied in pending_applied[:-1]:
            resolved_mask = prev_applied[DS] <= origin
            if not resolved_mask.any():
                still_pending.append(prev_applied)
                continue

            updated = prev_applied.copy()
            resolved_idx = updated.index[resolved_mask & updated[Y].isna()]
            if not resolved_idx.empty:
                keys = pd.MultiIndex.from_arrays(
                    [
                        updated.loc[resolved_idx, UNIQUE_ID].values,
                        updated.loc[resolved_idx, DS].values,
                    ]
                )
                y_vals = actuals_lookup.reindex(keys).values
                updated.loc[resolved_idx, Y] = y_vals

            has_y = updated[Y].notna()
            has_intervals = updated[lower_col].notna() & updated[upper_col].notna()
            to_observe = updated[has_y & has_intervals]
            if not to_observe.empty:
                with contextlib.suppress(ValueError):
                    conformal_runtime.observe(to_observe)

            unresolved = updated[~(has_y & has_intervals)]
            if not unresolved.empty:
                still_pending.append(unresolved)

        pending_applied = still_pending + [applied]

    return conformal_runtime


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def run_winning(
    data_dir: Path = DATA_DIR,
    horizon: int = HORIZON,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    warmup_origins: int = WARMUP_ORIGINS,
    conformal_config: ConformalPolicyConfig = CONFORMAL_CONFIG,
    lgb_params: dict = LGB_PARAMS,
    series_filter: list[str] | None = None,
    results_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the VN2 benchmark using the winning approach + Calibre conformal.

    Pipeline:
      0. Load data & build feature-engineered training frame
      1. Train global direct multi-horizon LightGBM models
      2. Warmup conformal runtime on historical walk-forward predictions
      3. Decision rounds: predict → conformal intervals → cost-aware orders
      4. Delivery weeks (no orders)
      5. Summarise costs

    Returns per-product cost DataFrame.
    """
    lower_col, upper_col = interval_column_names(conformal_config.coverage)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    initial_states = load_initial_states(data_dir / "week_0_initial_state.csv")

    # Optional data sources (may not exist)
    instock = None
    instock_path = data_dir / "week_0_in_stock.csv"
    if instock_path.exists():
        instock = melt_wide_instock(instock_path)

    master = None
    master_path = data_dir / "week_0_master.csv"
    if master_path.exists():
        master = load_master(master_path)

    week0_sales = load_period(data_dir, 0)

    if series_filter is not None:
        initial_states = {uid: s for uid, s in initial_states.items() if uid in series_filter}
        week0_sales = week0_sales[week0_sales[UNIQUE_ID].isin(series_filter)]
        if instock is not None:
            instock = instock[instock[UNIQUE_ID].isin(series_filter)]

    simulator = VN2Simulator(initial_states)

    if verbose:
        print(f"Loaded {len(initial_states)} products.")

    # ------------------------------------------------------------------
    # Phase 0: Feature engineering on week_0 history
    # ------------------------------------------------------------------
    if verbose:
        print("Building feature-engineered training frame...")

    training_frame = build_training_frame(
        sales=week0_sales,
        instock=instock,
        master=master,
        lags=LAGS,
        rolling_windows=ROLLING_WINDOWS,
        half_life_weeks=HALF_LIFE_WEEKS,
    )

    if verbose:
        feature_cols = _get_feature_cols(training_frame)
        print(f"  Features: {len(feature_cols)} columns")
        print(f"  Training rows: {len(training_frame)}")

    # ------------------------------------------------------------------
    # Phase 1: Train global direct multi-horizon LightGBM
    # ------------------------------------------------------------------
    if verbose:
        print(f"Training {horizon} global LightGBM models (direct multi-horizon)...")

    # Train on scaled target so model learns patterns not magnitudes
    feature_cols = _get_feature_cols(training_frame)
    models = train_global_models(training_frame, horizon=horizon, lgb_params=lgb_params)

    if verbose:
        print(f"  Models trained: {len(models)}")

    # ------------------------------------------------------------------
    # Phase 2: Warmup conformal runtime
    # ------------------------------------------------------------------
    if verbose:
        print(f"Running conformal warmup with {warmup_origins} origins...")

    conformal_runtime = _run_warmup(
        models=models,
        training_frame=training_frame,
        sales=week0_sales,
        conformal_config=conformal_config,
        warmup_origins=warmup_origins,
        horizon=horizon,
        series_filter=list(initial_states.keys()) if series_filter else None,
    )

    if verbose:
        n_calibrated = len(conformal_runtime._policies)
        print(f"  Calibrated {n_calibrated} conformal policies.")

    # ------------------------------------------------------------------
    # Phase 3: Decision rounds
    # ------------------------------------------------------------------
    pending_forecasts: list[pd.DataFrame] = []

    for round_num in range(1, decision_rounds + 1):
        if verbose:
            print(f"\n--- Decision round {round_num} ---")

        # Load updated sales
        round_sales = load_period(data_dir, round_num)
        if series_filter is not None:
            round_sales = round_sales[round_sales[UNIQUE_ID].isin(initial_states.keys())]
        # Rebuild features with new data
        round_frame = build_training_frame(
            sales=round_sales,
            instock=instock,
            master=master,
            lags=LAGS,
            rolling_windows=ROLLING_WINDOWS,
            half_life_weeks=HALF_LIFE_WEEKS,
        )

        # Retrain models on expanded data (online learning)
        models = train_global_models(
            round_frame,
            horizon=horizon,
            lgb_params=lgb_params,
        )

        # Predict from latest date
        latest_date = round_sales[DS].max()
        origin = pd.Timestamp(latest_date)

        forecast_df = predict_global(models, round_frame, origin)

        # Apply conformal intervals
        applied_df = conformal_runtime.apply(forecast_df)

        if lower_col in applied_df.columns and upper_col in applied_df.columns:
            pending_forecasts.append(applied_df.copy())

        # Get actual demand
        try:
            actual_demand = extract_new_actuals(data_dir, round_num + 1)
        except (FileNotFoundError, ValueError):
            round_raw = pd.read_csv(data_dir / f"week_{round_num}_sales.csv")
            date_cols = [c for c in round_raw.columns if c not in ("Store", "Product")]
            last_col = date_cols[-1]
            unique_ids = (
                round_raw["Store"].astype(int).astype(str)
                + "_"
                + round_raw["Product"].astype(int).astype(str)
            )
            actual_demand = dict(
                zip(
                    unique_ids,
                    round_raw[last_col].fillna(0.0).astype(float),
                    strict=False,
                )
            )

        actual_demand = {uid: actual_demand.get(uid, 0.0) for uid in initial_states}

        # Compute orders using R,S policy with cost-aligned conformal intervals
        rs_params = _build_rs_params(
            simulator,
            lead_time=lead_time,
            review_period=review_period,
        )
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
                    print(f"  Order policy failed: {exc}. Zero orders.")
                orders = {uid: 0.0 for uid in initial_states}
        else:
            if verbose:
                print("  No interval columns. Zero orders.")
            orders = {uid: 0.0 for uid in initial_states}

        if verbose:
            total_order = sum(orders.values())
            print(f"  Origin: {origin.date()}  Total order qty: {total_order:.0f}")

        # Step simulator
        simulator.step(round_num, orders=orders, actual_demand=actual_demand)

        # Feed resolved h=1 forecasts to conformal observe()
        still_pending: list[pd.DataFrame] = []
        for prev_forecast in pending_forecasts:
            h1_mask = (
                (prev_forecast[H] == 1)
                & (prev_forecast[FORECAST_ORIGIN] == origin)
                & prev_forecast[Y].isna()
            )
            if not h1_mask.any():
                still_pending.append(prev_forecast)
                continue

            updated = prev_forecast.copy()
            for idx in updated.index[h1_mask]:
                uid = str(updated.at[idx, UNIQUE_ID])
                updated.at[idx, Y] = float(actual_demand.get(uid, np.nan))

            h1_valid = (
                h1_mask
                & updated[Y].notna()
                & updated[lower_col].notna()
                & updated[upper_col].notna()
            )
            if h1_valid.any():
                with contextlib.suppress(ValueError):
                    conformal_runtime.observe(updated.loc[h1_valid])

            other = ~h1_mask
            if other.any():
                still_pending.append(prev_forecast.loc[other])

        pending_forecasts = still_pending

    # ------------------------------------------------------------------
    # Phase 4: Delivery weeks
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Phase 5: Results
    # ------------------------------------------------------------------
    rows = []
    for uid, state in simulator.states.items():
        rows.append(
            {
                "unique_id": uid,
                "holding_cost": state.cumulative_holding_cost,
                "shortage_cost": state.cumulative_shortage_cost,
                "total_cost": (state.cumulative_holding_cost + state.cumulative_shortage_cost),
            }
        )
    summary_df = pd.DataFrame(rows).sort_values("unique_id").reset_index(drop=True)

    if verbose:
        total_holding = summary_df["holding_cost"].sum()
        total_shortage = summary_df["shortage_cost"].sum()
        total_cost = summary_df["total_cost"].sum()
        print("\n" + "=" * 50)
        print("VN2 WINNING APPROACH + CONFORMAL RESULTS")
        print("=" * 50)
        print(f"Products:        {len(summary_df)}")
        print(f"Holding cost:    €{total_holding:,.2f}")
        print(f"Shortage cost:   €{total_shortage:,.2f}")
        print(f"TOTAL COST:      €{total_cost:,.2f}")
        print("=" * 50)

    if results_dir is not None:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out_path = results_dir / "per_product_costs_winning.csv"
        summary_df.to_csv(out_path, index=False)
        if verbose:
            print(f"\nPer-product costs saved to: {out_path}")

    return summary_df


if __name__ == "__main__":
    results = run_winning(
        results_dir=Path(__file__).parent / "results",
        verbose=True,
    )
