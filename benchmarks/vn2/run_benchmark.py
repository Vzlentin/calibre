"""Calibre's tuned VN2 benchmark — global LGBM + panel-level HPO + R,S.

This is the flagship Calibre entry: a single global LightGBM quantile
regressor exercised through ``MLForecastAdapter`` with ``strategy="direct"``,
hyper-tuned via a thin panel-level Optuna sweep, then driven through the
standard VN2 decision loop with ``apply_rs_policy(..., quantile=alpha)``.

Pipeline:

1. **Pre-HPO** (once, on ``week_0_sales.csv``): N walk-forward origins,
   panel-level Optuna sweep over LightGBM hyper-parameters and lag sets.
   Objective is cumulative-horizon pinball loss at the chosen quantile —
   the same statistic ``apply_rs_policy(..., quantile=p)`` sums at decision
   time, so the HPO optimises what we deploy.
2. **Decision loop** (rounds 1..N): refit the global LGBM with the best
   config at every round, ``apply_rs_policy`` with ``quantile=best_alpha``,
   step the ``VN2Simulator``.
3. **Delivery weeks**: zero orders, just simulator.

Documented gaps that this script works around (Phase-4 material):

- ``MLForecastAdapter`` silently drops ``date_features`` / ``static_features``
  (`calibre/models/mlforecast.py:111`), so calendar/static signal lives only
  in lag-based seasonal aggregations.
- ``TuningTask`` is per-series and point-metric only
  (`calibre/tasks/tuning_task.py`); the panel + quantile + cumulative-cost
  HPO is therefore inlined here.
- ``ensemble_median`` ignores ``q_*``, so the optional multi-alpha ensemble
  averages quantile columns inline.
- No orchestration layer for the VN2 decision loop — it is still inlined.
- Exogenous ``future_x`` is dead end-to-end and not used.
"""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import optuna
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import benchmarks.vn2.config as _vn2_config
from benchmarks.common.tracking import (
    log_config_module,
    log_costs_dataframe,
    optuna_mlflow_callback,
    start_benchmark_run,
)
from benchmarks.vn2.config import (
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
from benchmarks.vn2.simulator import (
    VN2Simulator,
    extract_new_actuals,
    load_initial_states,
)
from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    UNIQUE_ID,
    H,
    Y,
    quantile_column,
)
from calibre.engine.backend import BackendEngine
from calibre.features import add_stockout_features
from calibre.metrics import pinball_linear
from calibre.order.config import OrderPolicyConfig, apply_order_policy
from calibre.order.types import RsPolicyParameters
from calibre.pipeline.loading import load_period, melt_wide_instock
from calibre.tasks.forecast_task import ForecastTask

# Default rolling-mean / rolling-std windows applied at lag 1; these
# carry the seasonal signal MLForecastAdapter would otherwise drop.
ROLLING_WINDOWS = [4, 13, 26]


# ------------------------------------------------------------------ #
# Data preparation
# ------------------------------------------------------------------ #
def _prepare_history(sales: pd.DataFrame, instock: pd.DataFrame | None) -> pd.DataFrame:
    """Replace observed sales with censored-demand imputed values."""
    df = add_stockout_features(sales, instock)
    return df[[UNIQUE_ID, DS, "y_uncensored"]].rename(columns={"y_uncensored": Y})


def _load_instock(data_dir: Path, series_filter: list[str] | None) -> pd.DataFrame | None:
    instock_path = data_dir / "week_0_in_stock.csv"
    if not instock_path.exists():
        return None
    instock = melt_wide_instock(instock_path)
    if series_filter is not None:
        instock = instock[instock[UNIQUE_ID].isin(series_filter)]
    return instock


# ------------------------------------------------------------------ #
# Model config builder
# ------------------------------------------------------------------ #
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


# ------------------------------------------------------------------ #
# HPO
# ------------------------------------------------------------------ #
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
    """Pick the last `n_origins` origins from the history's tail.

    Each origin must leave at least `horizon` periods of future actuals
    so we can score the cumulative pinball loss.
    """
    all_dates = sorted(history[DS].unique())
    if len(all_dates) < n_origins + horizon:
        n_origins = max(1, len(all_dates) - horizon)
    if n_origins <= 0:
        return []
    return [pd.Timestamp(d) for d in all_dates[-(n_origins + horizon) : -horizon]]


def _cumulative_pinball(
    forecast_df: pd.DataFrame,
    actuals: pd.DataFrame,
    horizon: int,
    quantile: float,
    tau: float,
) -> float:
    """Cumulative-horizon pinball loss at the cost-optimal tau.

    For each window ``(uid, origin)`` with all ``h=1..horizon`` resolved,
    compute ``pinball(Σy_actual, Σq_<quantile>[h], tau=tau)`` and average
    over windows. ``tau`` is the cost-optimal cumulative quantile
    (``Cu / (Cu + Co)``); ``quantile`` is the per-horizon model knob.
    Pinball at tau is, up to a constant ``Cu + Co``, the newsvendor cost
    on cumulative demand — so this matches what
    ``apply_rs_policy(..., quantile=p)`` deploys.
    """
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
    mlflow_callbacks: list | None = None,
) -> dict[str, Any]:
    """Run the panel-level Optuna HPO and return the best model config.

    The returned dict is a fully-formed ``model_config`` ready to feed into
    a ``ForecastTask(scope="global", strategy="direct", quantiles=[alpha])``
    via ``BackendEngine``. ``best_alpha`` is exposed under the
    ``"_quantile_alpha"`` key (a private debug field — drop before passing
    upstream if needed; the value is also recoverable from ``quantiles[0]``).

    If ``mlflow_callbacks`` is provided, it is passed to ``study.optimize``
    so each trial is logged as a nested MLflow run under the active parent.
    """
    week0 = load_period(data_dir, 0)
    if series_filter is not None:
        week0 = week0[week0[UNIQUE_ID].isin(series_filter)]

    instock = _load_instock(data_dir, series_filter)
    history = _prepare_history(week0, instock)
    actuals = week0[[UNIQUE_ID, DS, Y]].copy()

    origins = _walk_forward_origins(history, n_origins, horizon)
    if not origins:
        raise ValueError(f"Not enough history to build {n_origins} origins with horizon {horizon}")

    engine = BackendEngine(freq="W-MON")

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            name: _suggest_from_spec(trial, name, spec) for name, spec in HPO_SEARCH_SPACE.items()
        }
        lags = HPO_LAG_SETS[int(params.pop("lag_set_idx"))]
        quantile_alpha = float(params.pop("quantile_alpha"))
        config = _build_model_config(quantile_alpha=quantile_alpha, lags=lags, **params)

        task = ForecastTask(history=history, horizon=horizon, model_config=config)
        result = engine.execute([task], actuals=actuals, origins=origins)
        forecast_df = result.ledger.to_df()

        return _cumulative_pinball(
            forecast_df, actuals, horizon, quantile_alpha, tau=cost_optimal_tau
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    if verbose:
        print(
            f"HPO: {n_trials} trials, {n_origins} origins, "
            f"timeout {timeout_sec}s, panel size {history[UNIQUE_ID].nunique()} series, "
            f"cost-optimal tau={cost_optimal_tau:.3f}"
        )

    started = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        gc_after_trial=True,
        callbacks=mlflow_callbacks or [],
    )
    elapsed = time.time() - started

    best = dict(study.best_trial.params)
    lags = HPO_LAG_SETS[int(best.pop("lag_set_idx"))]
    quantile_alpha = float(best.pop("quantile_alpha"))
    best_config = _build_model_config(quantile_alpha=quantile_alpha, lags=lags, **best)
    best_config["_quantile_alpha"] = quantile_alpha

    if verbose:
        print(
            f"HPO done in {elapsed:.1f}s. Best pinball={study.best_value:.4f} "
            f"alpha={quantile_alpha:.2f} lags={lags}"
        )

    if mlflow.active_run() is not None:
        mlflow.log_metric("hpo/best_pinball", study.best_value)
        mlflow.log_params(
            {f"hpo/best_{k}": str(v)[:500] for k, v in study.best_trial.params.items()}
        )

    return best_config


# ------------------------------------------------------------------ #
# Decision loop helpers
# ------------------------------------------------------------------ #
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
    try:
        actuals = extract_new_actuals(data_dir, round_num + 1)
    except (FileNotFoundError, ValueError):
        # Fall back to the last date column of the current round's sales file.
        round_raw = pd.read_csv(data_dir / f"week_{round_num}_sales.csv")
        date_cols = [c for c in round_raw.columns if c not in ("Store", "Product")]
        last_col = date_cols[-1]
        unique_ids = (
            round_raw["Store"].astype(int).astype(str)
            + "_"
            + round_raw["Product"].astype(int).astype(str)
        )
        actuals = dict(zip(unique_ids, round_raw[last_col].fillna(0.0).astype(float), strict=False))
    return {uid: actuals.get(uid, 0.0) for uid in state_keys}


def _strip_private(config: dict[str, Any]) -> dict[str, Any]:
    """Drop debug ``_*`` keys before handing the config to the engine."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
def run_benchmark(
    data_dir: Path = DATA_DIR,
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
    best_config: dict[str, Any] | None = None,
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
        best_config: Pre-computed model config (skips HPO if provided).

    Returns:
        DataFrame with columns: unique_id, holding_cost, shortage_cost, total_cost.
    """
    with start_benchmark_run(
        "vn2",
        "tuned",
        tags={
            "dataset": "vn2",
            "policy": "rs",
            "model_family": "lgbm",
            "horizon": str(horizon),
        },
    ):
        log_config_module(_vn2_config)
        mlflow.log_param("hpo_seed", hpo_seed)

        initial_states = load_initial_states(data_dir / "week_0_initial_state.csv")
        if series_filter is not None:
            initial_states = {uid: s for uid, s in initial_states.items() if uid in series_filter}

        instock = _load_instock(data_dir, series_filter)

        if verbose:
            print(f"Loaded {len(initial_states)} products.")

        # ------------------------------------------------------------------ #
        # Phase 1: HPO on week_0 (cumulative pinball at chosen quantile)
        # ------------------------------------------------------------------ #
        if best_config is None:
            best_config = run_hpo(
                data_dir=data_dir,
                horizon=horizon,
                n_trials=hpo_n_trials,
                n_origins=hpo_n_origins,
                timeout_sec=hpo_timeout_sec,
                series_filter=series_filter,
                seed=hpo_seed,
                verbose=verbose,
                mlflow_callbacks=[optuna_mlflow_callback("vn2", metric_name="pinball_cumulative")],
            )
        quantile_alpha = float(best_config.get("_quantile_alpha", best_config["quantiles"][0]))
        engine_config = _strip_private(best_config)

        if verbose:
            print(f"Best alpha: {quantile_alpha:.3f}")

        mlflow.log_param("quantile_alpha", quantile_alpha)

        # ------------------------------------------------------------------ #
        # Phase 2: Decision loop — refit each round, R,S with quantile path
        # ------------------------------------------------------------------ #
        simulator = VN2Simulator(initial_states)
        engine = BackendEngine(freq="W-MON")
        target_quantile_col = quantile_column(quantile_alpha)

        for round_num in range(1, decision_rounds + 1):
            if verbose:
                print(f"\n--- Decision round {round_num} ---")

            round_sales = load_period(data_dir, round_num)
            if series_filter is not None:
                round_sales = round_sales[round_sales[UNIQUE_ID].isin(initial_states)]

            history = _prepare_history(round_sales, instock)
            # +1 week so the engine's strict `<` filter keeps the latest observation
            origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)

            task = ForecastTask(history=history, horizon=horizon, model_config=engine_config)
            result = engine.execute([task], actuals=round_sales, origins=[origin])
            forecast_df = result.ledger.to_df()

            if forecast_df.empty or target_quantile_col not in forecast_df.columns:
                if verbose:
                    print("  Empty forecast or missing quantile column, using zero orders.")
                orders: dict[str, float] = dict.fromkeys(initial_states, 0.0)
            else:
                order_config = OrderPolicyConfig(
                    policy="rs",
                    params=_build_rs_params(simulator, lead_time, review_period),
                    quantile=quantile_alpha,
                )
                try:
                    order_result = apply_order_policy(forecast_df, order_config)
                    orders = dict.fromkeys(initial_states, 0.0)
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

            actual_demand = _round_actuals(data_dir, round_num, initial_states)

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

        # ------------------------------------------------------------------ #
        # Phase 3: Delivery weeks (no orders)
        # ------------------------------------------------------------------ #
        for week_offset in range(1, delivery_weeks + 1):
            week = decision_rounds + week_offset
            if verbose:
                print(f"\n--- Delivery week {week} (no orders) ---")
            try:
                actual_demand = extract_new_actuals(data_dir, week)
                actual_demand = {uid: actual_demand.get(uid, 0.0) for uid in initial_states}
            except (FileNotFoundError, ValueError):
                actual_demand = dict.fromkeys(initial_states, 0.0)
            simulator.step(
                week,
                orders=dict.fromkeys(initial_states, 0.0),
                actual_demand=actual_demand,
            )

        # ------------------------------------------------------------------ #
        # Phase 4: Results
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
            print("VN2 TUNED BENCHMARK RESULTS")
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
            out_path = results_dir / "per_product_costs.csv"
            summary_df.to_csv(out_path, index=False)
            if verbose:
                print(f"\nPer-product costs saved to: {out_path}")

    return summary_df


if __name__ == "__main__":
    run_benchmark(
        results_dir=Path(__file__).parent / "results",
        verbose=True,
    )
