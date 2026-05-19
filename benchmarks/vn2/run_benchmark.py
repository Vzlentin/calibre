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

Documented gaps that this script works around (Phase-4 material):

- ``MLForecastAdapter`` silently drops ``date_features`` / ``static_features``
  (`calibre/models/mlforecast.py:111`), so calendar/static signal lives only
  in lag-based seasonal aggregations.
- ``TuningTask`` is per-series and point-metric only
  (`calibre/tasks/tuning_task.py`); the panel + quantile + cumulative-cost
  HPO is therefore inlined here.
- ``ensemble_median`` ignores ``q_*``, so the optional multi-alpha ensemble
  averages quantile columns inline.
- Decision loop now delegated to ``calibre.execution.DecisionLoop``.
- Exogenous ``future_x`` is dead end-to-end and not used.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass, replace
from functools import cache, partial
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd

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
    ProductState,
    VN2Simulator,
    extract_new_actuals,
    load_initial_states,
)
from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
)
from calibre.conformal.partitions import global_partition, series_partition
from calibre.conformal.runtime import (
    ConformalRuntime,
    SymmetricIntervalConfig,
    build_symmetric_interval_runtime,
)
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    is_quantile_column,
    quantile_column,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import RsPolicyParameters
from calibre.evaluation.point_metrics import pinball_linear
from calibre.execution import (
    DecisionLoop,
    DecisionLoopConfig,
    RoundResult,
    observe_cumulative,
    observe_per_horizon,
)
from calibre.execution.backend import BackendEngine, ExecutionOptions
from calibre.execution.data_loading import load_period, melt_wide_instock
from calibre.execution.io import exists, join_uri
from calibre.forecasting.features import add_stockout_features
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy
from calibre.tuning.optimizer import create_tpe_sampler

# Default rolling-mean / rolling-std windows applied at lag 1; these
# carry the seasonal signal MLForecastAdapter would otherwise drop.
ROLLING_WINDOWS = [4, 13, 26]
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data preparation
# ------------------------------------------------------------------ #
def _prepare_history(sales: pd.DataFrame, instock: pd.DataFrame | None) -> pd.DataFrame:
    """Replace observed sales with censored-demand imputed values."""
    df = add_stockout_features(sales, instock)
    return df[[UNIQUE_ID, DS, "y_uncensored"]].rename(columns={"y_uncensored": Y})


def _prepare_cumulative_target_history(
    sales: pd.DataFrame,
    instock: pd.DataFrame | None,
    protection_period: int,
) -> pd.DataFrame:
    """Build a leakage-free direct cumulative-demand target frame.

    The target at timestamp ``t`` is the trailing sum ending at ``t`` over the
    protection period. A forecast made at origin ``o`` can therefore use the
    model's terminal-horizon prediction at ``o + protection_period`` as the
    direct estimate of demand over ``h=1..protection_period``.

    Invariant: the rolled target is used only to fit the model. The decision
    ledger's ``Y`` column is later refilled from the raw weekly ``sales`` frame
    passed into ``engine.execute(actuals=sales, ...)``, so ``window[Y].sum()``
    inside ``CumulativeRiskRuntime.observe`` recovers the cumulative
    realised demand via summation. ``_as_cumulative_decision_frame`` zeroes
    non-terminal-horizon ``Y_HAT``/quantile rows so the matching ``base_sum``
    reduces to the terminal cumulative prediction.
    """
    if protection_period < 1:
        raise ValueError("protection_period must be at least 1")

    history = _prepare_history(sales, instock).sort_values([UNIQUE_ID, DS]).copy()
    history[Y] = (
        history.groupby(UNIQUE_ID, sort=False)[Y]
        .transform(
            lambda values: values.rolling(
                protection_period,
                min_periods=protection_period,
            ).sum()
        )
        .astype("float64")
    )
    return history.dropna(subset=[Y]).reset_index(drop=True)


def _load_instock(data_dir: str | Path, series_filter: list[str] | None) -> pd.DataFrame | None:
    instock_path = join_uri(data_dir, "week_0_in_stock.csv")
    if not exists(instock_path):
        return None
    instock = melt_wide_instock(instock_path)
    if series_filter is not None:
        instock = instock[instock[UNIQUE_ID].isin(series_filter)]
    return instock


def _model_uses_cumulative_target(model_config: Mapping[str, Any]) -> bool:
    return str(model_config.get("_target_mode", "")).lower() == "cumulative"


def _prepare_model_history(
    sales: pd.DataFrame,
    instock: pd.DataFrame | None,
    protection_period: int,
    cumulative_target: bool,
) -> pd.DataFrame:
    if cumulative_target:
        return _prepare_cumulative_target_history(sales, instock, protection_period)
    return _prepare_history(sales, instock)


def _as_cumulative_decision_frame(
    frame: pd.DataFrame,
    protection_period: int,
) -> pd.DataFrame:
    """Move direct cumulative predictions onto the terminal decision row.

    MLForecast emits one prediction per horizon. With the cumulative target,
    only the terminal horizon estimates the whole protection period. The R,S
    policy and CRC runtime still expect ``h=1..K`` weekly actuals, so the base
    forecast columns are zeroed before ``h=K`` and retained at ``h=K``.
    """
    if frame.empty:
        return frame.copy()

    value_cols = [Y_HAT, *(col for col in frame.columns if is_quantile_column(col))]
    result = frame.copy()
    group_cols = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]
    for _, group in result.groupby(group_cols, sort=False):
        ordered = group.sort_values(H)
        terminal = ordered[ordered[H].astype(int) == protection_period]
        if terminal.empty:
            continue
        terminal_idx = terminal.index[-1]
        terminal_values = result.loc[terminal_idx, value_cols].copy()
        within_window = ordered[ordered[H].astype(int) <= protection_period].index
        result.loc[within_window, value_cols] = 0.0
        result.loc[terminal_idx, value_cols] = terminal_values
    return result


def _prepare_policy_forecast_frame(
    frame: pd.DataFrame,
    protection_period: int,
    cumulative_target: bool,
) -> pd.DataFrame:
    if cumulative_target:
        return _as_cumulative_decision_frame(frame, protection_period)
    return frame


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
    target_mode: str = "per_horizon",
) -> dict[str, Any]:
    """Run the panel-level Optuna HPO and return the best model config.

    The returned dict is a fully-formed ``model_config`` ready to feed into
    a ``ForecastTask(scope="global", strategy="direct", quantiles=[alpha])``
    via ``BackendEngine``. ``best_alpha`` is exposed under the
    ``"_quantile_alpha"`` key (a private debug field — drop before passing
    upstream if needed; the value is also recoverable from ``quantiles[0]``).

    HPO summary metrics and artifacts are logged to the active MLflow parent
    run when tracking is enabled.
    """
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

    engine = BackendEngine(execution=ExecutionOptions(freq="W-MON"))

    def _objective(trial: optuna.Trial) -> float:
        params: dict[str, Any] = {
            name: _suggest_from_spec(trial, name, spec) for name, spec in HPO_SEARCH_SPACE.items()
        }
        lags = HPO_LAG_SETS[int(params.pop("lag_set_idx"))]
        quantile_alpha = float(params.pop("quantile_alpha"))
        config = _build_model_config(quantile_alpha=quantile_alpha, lags=lags, **params)
        if cumulative_target:
            config["_target_mode"] = "cumulative"

        task = ForecastTask(history=history, horizon=horizon, model_config=_strip_private(config))
        result = engine.execute([task], actuals=actuals, origins=origins)
        forecast_df = _prepare_policy_forecast_frame(
            result.ledger.to_df(),
            protection_period=horizon,
            cumulative_target=cumulative_target,
        )

        return _cumulative_pinball(
            forecast_df, actuals, horizon, quantile_alpha, tau=cost_optimal_tau
        )

    study = optuna.create_study(
        direction="minimize",
        sampler=create_tpe_sampler(seed),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    if verbose:
        logger.info(
            "HPO: %s trials, %s origins, timeout %ss, panel size %s series, cost-optimal tau=%.3f",
            n_trials,
            n_origins,
            timeout_sec,
            history[UNIQUE_ID].nunique(),
            cost_optimal_tau,
        )

    started = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        _objective,
        n_trials=n_trials,
        timeout=timeout_sec,
        gc_after_trial=True,
    )
    elapsed = time.time() - started

    best = dict(study.best_trial.params)
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
            study.best_value,
            quantile_alpha,
            lags,
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
    data_dir: str | Path,
    round_num: int,
    state_keys: Mapping[str, object],
) -> dict[str, float]:
    # round_num indexes the resolved-actuals week directly: round 1's demand
    # is week_1_sales' last column. Earlier revisions used round_num + 1.
    try:
        actuals = extract_new_actuals(data_dir, round_num)
    except (FileNotFoundError, ValueError):
        # Fall back to the last date column of the current round's sales file.
        round_raw = pd.read_csv(join_uri(data_dir, f"week_{round_num}_sales.csv"))
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


def _run_order_conformal_warmup(
    *,
    sales: pd.DataFrame,
    instock: pd.DataFrame | None,
    model_config: dict[str, Any],
    horizon: int,
    warmup_origins: int,
    runtime: CumulativeRiskRuntime,
    series_filter: list[str] | None,
    cumulative_target: bool = False,
    execution_backend: str = "auto",
    ray_address: str | None = None,
    ray_threshold: int = 10,
    max_concurrency: int | None = None,
) -> None:
    """Calibrate the cumulative order conformal runtime on resolved origins."""
    for frame in _order_conformal_warmup_frames(
        sales=sales,
        instock=instock,
        model_config=model_config,
        horizon=horizon,
        warmup_origins=warmup_origins,
        series_filter=series_filter,
        cumulative_target=cumulative_target,
        execution_backend=execution_backend,
        ray_address=ray_address,
        ray_threshold=ray_threshold,
        max_concurrency=max_concurrency,
    ):
        runtime.observe(runtime.apply(frame))


def _order_conformal_warmup_frames(
    *,
    sales: pd.DataFrame,
    instock: pd.DataFrame | None,
    model_config: dict[str, Any],
    horizon: int,
    warmup_origins: int,
    series_filter: list[str] | None,
    cumulative_target: bool = False,
    execution_backend: str = "auto",
    ray_address: str | None = None,
    ray_threshold: int = 10,
    max_concurrency: int | None = None,
) -> list[pd.DataFrame]:
    """Return resolved warmup forecast frames for CRC calibration."""
    if warmup_origins <= 0:
        return []

    history = _prepare_model_history(
        sales,
        instock,
        protection_period=horizon,
        cumulative_target=cumulative_target,
    )
    all_dates = sorted(history[DS].unique())
    if len(all_dates) < warmup_origins + horizon:
        warmup_origins = max(1, len(all_dates) - horizon)
    origin_dates = [pd.Timestamp(d) for d in all_dates[-(warmup_origins + horizon) : -horizon]]
    if not origin_dates:
        return []

    engine = BackendEngine(
        execution=ExecutionOptions(
            freq="W-MON",
            backend=execution_backend,  # type: ignore[arg-type]
            ray_address=ray_address,
            ray_threshold=ray_threshold,
            max_concurrency=max_concurrency,
        )
    )
    task = ForecastTask(history=history, horizon=horizon, model_config=model_config)
    ledger_df = _prepare_policy_forecast_frame(
        engine.execute([task], actuals=sales, origins=origin_dates).ledger.to_df(),
        protection_period=horizon,
        cumulative_target=cumulative_target,
    )
    if ledger_df.empty:
        return []

    actuals_lookup = sales.drop_duplicates(subset=[UNIQUE_ID, DS]).set_index([UNIQUE_ID, DS])[Y]
    unresolved = ledger_df[Y].isna()
    if unresolved.any():
        keys = pd.MultiIndex.from_arrays(
            [ledger_df.loc[unresolved, UNIQUE_ID].values, ledger_df.loc[unresolved, DS].values]
        )
        ledger_df.loc[unresolved, Y] = actuals_lookup.reindex(keys).to_numpy()

    if series_filter is not None:
        ledger_df = ledger_df[ledger_df[UNIQUE_ID].isin(series_filter)]

    frames: list[pd.DataFrame] = []
    for origin in origin_dates:
        origin_rows = ledger_df[ledger_df[FORECAST_ORIGIN] == origin]
        if not origin_rows.empty:
            frames.append(origin_rows)
    return frames


def _summary_from_simulator(simulator: VN2Simulator) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values("unique_id").reset_index(drop=True)


@dataclass
class CachedRound:
    round_num: int
    origin: pd.Timestamp
    frame: pd.DataFrame


@dataclass
class VN2ReplayCache:
    """Forecast and actual-demand cache for simulator-cost experiments."""

    initial_states: dict[str, ProductState]
    warmup_frames: list[pd.DataFrame]
    rounds: dict[int, CachedRound]
    actuals_by_round: dict[int, dict[str, float]]
    model_config: dict[str, Any]
    quantile_alpha: float
    horizon: int
    lead_time: int
    review_period: int
    decision_rounds: int
    delivery_weeks: int
    cumulative_target: bool


@dataclass
class ReplayResult:
    summary: pd.DataFrame
    orders_by_round: dict[int, dict[str, float]]
    history: pd.DataFrame

    @property
    def total_cost(self) -> float:
        return float(self.summary["total_cost"].sum())


def _scale_base_forecasts(frame: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Apply an explicit base-forecast calibration scale for ablation searches."""
    if scale == 1.0 or frame.empty:
        return frame.copy()
    result = frame.copy()
    for col in [Y_HAT, *(c for c in result.columns if is_quantile_column(c))]:
        result[col] = result[col].astype(float) * float(scale)
    return result


def _orders_from_policy_result(
    order_result: pd.DataFrame,
    state_keys: Mapping[str, object],
    reorder_point_scale: float | None = None,
) -> dict[str, float]:
    adjusted = order_result.copy()
    if reorder_point_scale is not None:
        reorder_point = adjusted["target_stock_level"].astype(float) * float(reorder_point_scale)
        inventory_position = adjusted["inventory_position"].astype(float)
        adjusted.loc[inventory_position >= reorder_point, "order_qty"] = 0.0

    orders: dict[str, float] = dict.fromkeys(state_keys, 0.0)
    for uid, qty in zip(
        adjusted[UNIQUE_ID].astype(str),
        adjusted["order_qty"].astype(float),
        strict=False,
    ):
        orders[uid] = float(max(math.ceil(qty), 0))
    return orders


def _actuals_for_replay_round(
    data_dir: str | Path,
    round_num: int,
    decision_rounds: int,
    state_keys: Mapping[str, object],
) -> dict[str, float]:
    if round_num <= decision_rounds:
        return _round_actuals(data_dir, round_num, state_keys)
    try:
        actuals = extract_new_actuals(data_dir, round_num)
        return {uid: actuals.get(uid, 0.0) for uid in state_keys}
    except (FileNotFoundError, ValueError):
        return dict.fromkeys(state_keys, 0.0)


def build_replay_cache(
    *,
    data_dir: str | Path = DATA_DIR,
    model_config: dict[str, Any] | None = None,
    horizon: int = HORIZON,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    series_filter: list[str] | None = None,
    order_conformal_warmup_origins: int = HPO_N_ORIGINS,
) -> VN2ReplayCache:
    """Fit/cache benchmark forecasts and realised demand for cost replay."""
    model_config = deepcopy(model_config if model_config is not None else BEST_CONFIG)
    quantile_alpha = float(model_config.get("_quantile_alpha", model_config["quantiles"][0]))
    cumulative_target = _model_uses_cumulative_target(model_config)
    engine_config = _strip_private(model_config)

    initial_states = load_initial_states(join_uri(data_dir, "week_0_initial_state.csv"))
    if series_filter is not None:
        initial_states = {uid: s for uid, s in initial_states.items() if uid in series_filter}

    instock = _load_instock(data_dir, series_filter)
    week0_sales = load_period(data_dir, 0)
    if series_filter is not None:
        week0_sales = week0_sales[week0_sales[UNIQUE_ID].isin(initial_states)]

    warmup_frames = _order_conformal_warmup_frames(
        sales=week0_sales,
        instock=instock,
        model_config=engine_config,
        horizon=horizon,
        warmup_origins=order_conformal_warmup_origins,
        series_filter=list(initial_states),
        cumulative_target=cumulative_target,
    )

    engine = BackendEngine(execution=ExecutionOptions(freq="W-MON"))
    rounds: dict[int, CachedRound] = {}
    for rn in range(1, decision_rounds + 1):
        round_sales = load_period(data_dir, rn - 1)
        if series_filter is not None:
            round_sales = round_sales[round_sales[UNIQUE_ID].isin(initial_states)]
        history = _prepare_model_history(
            round_sales,
            instock,
            protection_period=horizon,
            cumulative_target=cumulative_target,
        )
        origin = pd.Timestamp(round_sales[DS].max()) + pd.Timedelta(weeks=1)
        task = ForecastTask(history=history, horizon=horizon, model_config=engine_config)
        frame = _prepare_policy_forecast_frame(
            engine.execute([task], actuals=round_sales, origins=[origin]).ledger.to_df(),
            protection_period=horizon,
            cumulative_target=cumulative_target,
        )
        rounds[rn] = CachedRound(round_num=rn, origin=origin, frame=frame)

    actuals_by_round = {
        rn: _actuals_for_replay_round(data_dir, rn, decision_rounds, initial_states)
        for rn in range(1, decision_rounds + delivery_weeks + 1)
    }

    return VN2ReplayCache(
        initial_states=initial_states,
        warmup_frames=warmup_frames,
        rounds=rounds,
        actuals_by_round=actuals_by_round,
        model_config=model_config,
        quantile_alpha=quantile_alpha,
        horizon=horizon,
        lead_time=lead_time,
        review_period=review_period,
        decision_rounds=decision_rounds,
        delivery_weeks=delivery_weeks,
        cumulative_target=cumulative_target,
    )


def replay_cached_cost(
    cache: VN2ReplayCache,
    *,
    order_conformal_config: CumulativeConformalRiskConfig | None = CONFORMAL_ORDER_CONFIG,
    order_base_scale: float = 1.0,
    reorder_point_scale: float | None = None,
    on_policy_error: Callable[[int, Exception], None] | None = None,
) -> ReplayResult:
    """Replay cached forecasts through the exact VN2 simulator.

    A failure in ``apply_order_policy`` for a single round falls back to zero
    orders so the cost trajectory remains comparable across rounds; pass
    ``on_policy_error`` to surface the underlying exception (e.g. a print or
    logger.warning), otherwise the failure is silent.
    """
    simulator = VN2Simulator(cache.initial_states)
    target_quantile_col = quantile_column(cache.quantile_alpha)

    runtime: CumulativeRiskRuntime | None = None
    if order_conformal_config is not None:
        resolved_config = replace(
            order_conformal_config,
            base_column=target_quantile_col,
            protection_period=cache.lead_time + cache.review_period,
        )
        runtime = CumulativeRiskRuntime(resolved_config)
        for frame in cache.warmup_frames:
            runtime.observe(runtime.apply(_scale_base_forecasts(frame, order_base_scale)))

    orders_by_round: dict[int, dict[str, float]] = {}
    pending: list[pd.DataFrame] = []
    actuals_cache: dict[tuple[str, pd.Timestamp], float] = {}
    freq_offset = pd.tseries.frequencies.to_offset("W-MON")

    for rn in range(1, cache.decision_rounds + 1):
        cached_round = cache.rounds[rn]
        frame = _scale_base_forecasts(cached_round.frame, order_base_scale)
        try:
            if runtime is not None:
                policy_frame = runtime.apply(frame)
                pending.append(policy_frame.copy())
                order_config = OrderPolicyConfig(
                    policy="rs",
                    params=_build_rs_params(simulator, cache.lead_time, cache.review_period),
                    coverage=runtime.config.coverage,
                )
            else:
                policy_frame = frame
                order_config = OrderPolicyConfig(
                    policy="rs",
                    params=_build_rs_params(simulator, cache.lead_time, cache.review_period),
                    quantile=cache.quantile_alpha,
                )
            order_result = apply_order_policy(policy_frame, order_config)
            orders = _orders_from_policy_result(
                order_result,
                cache.initial_states,
                reorder_point_scale=reorder_point_scale,
            )
        except (ValueError, KeyError) as exc:
            if on_policy_error is not None:
                on_policy_error(rn, exc)
            orders = dict.fromkeys(cache.initial_states, 0.0)

        actual_demand = cache.actuals_by_round.get(rn, dict.fromkeys(cache.initial_states, 0.0))
        simulator.step(rn, orders=orders, actual_demand=actual_demand)
        orders_by_round[rn] = orders

        if runtime is not None and actual_demand:
            actuals_ds = cached_round.origin + freq_offset
            for uid, demand in actual_demand.items():
                actuals_cache[(uid, actuals_ds)] = demand
            lookup = pd.Series(actuals_cache, dtype=float)
            if not lookup.empty:
                lookup.index = pd.MultiIndex.from_tuples(lookup.index)
            pending = observe_cumulative(runtime, pending, lookup)

    for week_offset in range(1, cache.delivery_weeks + 1):
        delivery_num = cache.decision_rounds + week_offset
        actual_demand = cache.actuals_by_round.get(
            delivery_num,
            dict.fromkeys(cache.initial_states, 0.0),
        )
        simulator.step(
            delivery_num,
            orders={uid: 0.0 for uid in actual_demand},
            actual_demand=actual_demand,
        )

    return ReplayResult(
        summary=_summary_from_simulator(simulator),
        orders_by_round=orders_by_round,
        history=simulator.to_dataframe(),
    )


def _stable_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _stable_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_stable_value(v) for v in value]
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    state = getattr(value, "__dict__", None)
    if state is not None:
        return {
            "class": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "state": _stable_value(state),
        }
    return repr(value)


def _stable_config_key(config: dict[str, Any]) -> str:
    return json.dumps(_stable_value(config), sort_keys=True)


def _log_mlflow_params(params: Mapping[str, Any]) -> None:
    scalar_params: dict[str, str] = {}
    non_scalar: dict[str, Any] = {}

    for key, value in params.items():
        if is_dataclass(value) and not isinstance(value, type):
            for field_name, field_value in asdict(value).items():
                scalar_params[f"{key}.{field_name}"[:250]] = str(field_value)[:500]
            continue

        if isinstance(value, Path | bool | int | float | str) or value is None:
            scalar_params[str(key)[:250]] = str(value)[:500]
        else:
            non_scalar[str(key)] = _stable_value(value)

    if scalar_params:
        mlflow.log_params(scalar_params)
    if non_scalar:
        mlflow.log_dict(non_scalar, "params_non_scalar.json")


def log_cached_replay_run(
    result: ReplayResult,
    *,
    run_name: str = "cached_replay",
    experiment_name: str = "vn2",
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
) -> None:
    """Log a cached simulator replay to MLflow."""
    resolved_tags = {"dataset": "vn2", "source": "cached_replay"}
    if tags:
        resolved_tags.update({str(key): str(value) for key, value in tags.items()})

    with start_benchmark_run(experiment_name, run_name, tags=resolved_tags):
        if params:
            _log_mlflow_params(params)
        log_costs_dataframe(result.summary)
        mlflow.log_dict(
            {str(round_num): orders for round_num, orders in result.orders_by_round.items()},
            "replay/orders_by_round.json",
        )
        with tempfile.TemporaryDirectory() as tmp:
            history_path = Path(tmp) / "history.csv"
            result.history.to_csv(history_path, index=False)
            mlflow.log_artifact(str(history_path), artifact_path="replay")


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
    if buffer_min is not None and buffer_max is not None and buffer_min > buffer_max:
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
        weight_decay=None if weight_decay_choice == "none" else float(weight_decay_choice),
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
) -> optuna.Study:
    """Optimize simulator EUR cost with cached forecast replays.

    By default the search varies CRC parameters against a fixed forecast model.
    Set ``search_forecast=True`` to include LightGBM, lag-set, quantile, and
    direct-cumulative target choices in the same objective.
    """
    base_config = deepcopy(model_config if model_config is not None else BEST_CONFIG)
    forecast_cache: dict[str, VN2ReplayCache] = {}
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

    def _objective(trial: optuna.Trial) -> float:
        candidate_model = _sample_cost_search_model_config(trial, base_config, search_forecast)
        order_base_scale = (
            trial.suggest_float("order_base_scale", 0.85, 1.15)
            if include_order_calibration
            else 1.0
        )
        reorder_point_scale = (
            trial.suggest_float("reorder_point_scale", 0.0, 1.0)
            if include_order_calibration
            else None
        )
        crc_config = _sample_cost_search_crc_config(
            trial,
            lead_time + review_period,
            crc_partitions=crc_partitions,
        )
        cache_key = _stable_config_key(
            {
                "model": candidate_model,
                "horizon": horizon,
                "lead_time": lead_time,
                "review_period": review_period,
                "decision_rounds": decision_rounds,
                "delivery_weeks": delivery_weeks,
                "series_filter": series_filter,
            }
        )

        try:
            if fixed_cache is not None:
                forecast_cache[cache_key] = fixed_cache
            elif cache_key not in forecast_cache:
                forecast_cache[cache_key] = build_replay_cache(
                    data_dir=data_dir,
                    model_config=candidate_model,
                    horizon=horizon,
                    lead_time=lead_time,
                    review_period=review_period,
                    decision_rounds=decision_rounds,
                    delivery_weeks=delivery_weeks,
                    series_filter=series_filter,
                )
            policy_errors: list[str] = []
            result = replay_cached_cost(
                forecast_cache[cache_key],
                order_conformal_config=crc_config,
                order_base_scale=order_base_scale,
                reorder_point_scale=reorder_point_scale,
                on_policy_error=lambda rn, exc: policy_errors.append(f"round {rn}: {exc!r}"),
            )
            if policy_errors:
                trial.set_user_attr("policy_errors", policy_errors)
        except optuna.TrialPruned:
            raise
        except Exception as exc:  # pragma: no cover - exercised by Optuna on bad trials.
            trial.set_user_attr("error", repr(exc))
            raise

        trial.set_user_attr("total_holding_cost", float(result.summary["holding_cost"].sum()))
        trial.set_user_attr("total_shortage_cost", float(result.summary["shortage_cost"].sum()))
        trial.set_user_attr("order_base_scale", order_base_scale)
        trial.set_user_attr("reorder_point_scale", reorder_point_scale)
        return result.total_cost

    study = optuna.create_study(
        direction="minimize",
        sampler=create_tpe_sampler(seed),
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
            study.optimize(
                _objective,
                n_trials=n_trials,
                timeout=timeout_sec,
                gc_after_trial=True,
                catch=(Exception,),
            )
            completed_trials = [
                trial
                for trial in study.trials
                if trial.state == optuna.trial.TrialState.COMPLETE and trial.value is not None
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
        study.optimize(
            _objective,
            n_trials=n_trials,
            timeout=timeout_sec,
            gc_after_trial=True,
            catch=(Exception,),
        )
    return study


def _optimal_order_path_for_sku(
    initial_state: ProductState,
    demand_by_week: Mapping[int, float],
    *,
    decision_rounds: int,
    total_weeks: int,
    order_step: float = 1.0,
    lead_time: int = LEAD_TIME,
) -> dict[int, float]:
    """Exact finite-horizon oracle for one SKU under VN2 lead-time mechanics."""
    if order_step <= 0:
        raise ValueError("order_step must be positive")
    if lead_time != LEAD_TIME:
        raise ValueError(f"VN2 oracle currently supports lead_time={LEAD_TIME}, got {lead_time}")

    if total_weeks - decision_rounds <= lead_time:
        return _just_in_time_order_path_for_sku(
            initial_state,
            demand_by_week,
            decision_rounds=decision_rounds,
            total_weeks=total_weeks,
            order_step=order_step,
            lead_time=lead_time,
        )

    demands = tuple(float(demand_by_week.get(week, 0.0)) for week in range(1, total_weeks + 1))
    choices: dict[tuple[int, float, float, float], float] = {}

    def _key(
        week: int, end_inventory: float, p1: float, p2: float
    ) -> tuple[int, float, float, float]:
        return (week, round(end_inventory, 6), round(p1, 6), round(p2, 6))

    @cache
    def _dp(week: int, end_inventory: float, p1: float, p2: float) -> float:
        if week > total_weeks:
            return 0.0

        demand = demands[week - 1]
        if week <= decision_rounds:
            arrival_week = week + 2
            remaining_after_arrival = (
                sum(demands[arrival_week - 1 :]) if arrival_week <= total_weeks else 0.0
            )
            max_units = int(math.ceil(max(0.0, remaining_after_arrival) / order_step))
            order_values: Iterable[float] = (unit * order_step for unit in range(max_units + 1))
        else:
            order_values = (0.0,)

        best_cost = float("inf")
        best_order = 0.0
        for order in order_values:
            arrivals = p1
            start_inventory = end_inventory + arrivals
            sales = min(start_inventory, demand)
            missed_sales = demand - sales
            next_end_inventory = start_inventory - sales
            period_cost = (
                VN2Simulator.HOLDING_COST_RATE * next_end_inventory
                + VN2Simulator.SHORTAGE_COST_RATE * missed_sales
            )
            future_cost = _dp(
                week + 1,
                round(next_end_inventory, 6),
                round(p2, 6),
                round(order, 6),
            )
            total_cost = period_cost + future_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_order = float(order)

        choices[_key(week, end_inventory, p1, p2)] = best_order
        return best_cost

    week = 1
    end_inventory = float(initial_state.end_inventory)
    p1 = float(initial_state.in_transit_w1)
    p2 = float(initial_state.in_transit_w2)
    _dp(week, round(end_inventory, 6), round(p1, 6), round(p2, 6))

    orders: dict[int, float] = {}
    while week <= total_weeks:
        order = choices.get(_key(week, end_inventory, p1, p2), 0.0)
        if week <= decision_rounds:
            orders[week] = order

        demand = demands[week - 1]
        arrivals = p1
        start_inventory = end_inventory + arrivals
        sales = min(start_inventory, demand)
        end_inventory = round(start_inventory - sales, 6)
        next_p2 = round(order, 6) if week <= decision_rounds else 0.0
        p1, p2 = round(p2, 6), next_p2
        week += 1

    return orders


def _advance_state_values(
    end_inventory: float,
    p1: float,
    p2: float,
    demand: float,
    order: float,
) -> tuple[float, float, float]:
    start_inventory = float(end_inventory) + float(p1)
    sales = min(start_inventory, float(demand))
    next_end_inventory = start_inventory - sales
    return next_end_inventory, float(p2), float(order)


def _round_order_to_step(quantity: float, order_step: float) -> float:
    if quantity <= 0.0:
        return 0.0
    return float(math.ceil(quantity / order_step) * order_step)


def _just_in_time_order_path_for_sku(
    initial_state: ProductState,
    demand_by_week: Mapping[int, float],
    *,
    decision_rounds: int,
    total_weeks: int,
    order_step: float,
    lead_time: int,
) -> dict[int, float]:
    """Fast exact VN2 oracle when delivery horizon is no longer than lead time.

    With weekly review, no order cost, and ``delivery_weeks <= lead_time``, an
    order placed in round ``t`` only needs to cover demand in the first week it
    can arrive, ``t + lead_time``. Later demand can be served by later orders
    just in time, so carrying extra inventory is never better than waiting.
    """
    end_inventory = float(initial_state.end_inventory)
    p1 = float(initial_state.in_transit_w1)
    p2 = float(initial_state.in_transit_w2)
    orders: dict[int, float] = {}

    for week in range(1, total_weeks + 1):
        if week <= decision_rounds:
            arrival_week = week + lead_time
            if arrival_week <= total_weeks:
                projected_end = end_inventory
                projected_p1 = p1
                projected_p2 = p2
                for projected_week in range(week, arrival_week):
                    projected_end, projected_p1, projected_p2 = _advance_state_values(
                        projected_end,
                        projected_p1,
                        projected_p2,
                        demand_by_week.get(projected_week, 0.0),
                        order=0.0,
                    )
                available_at_arrival = projected_end + projected_p1
                target = float(demand_by_week.get(arrival_week, 0.0))
                order = _round_order_to_step(target - available_at_arrival, order_step)
            else:
                order = 0.0
            orders[week] = order
        else:
            order = 0.0

        end_inventory, p1, p2 = _advance_state_values(
            end_inventory,
            p1,
            p2,
            demand_by_week.get(week, 0.0),
            order=order,
        )

    return orders


def _simulate_orders(
    initial_states: dict[str, ProductState],
    actuals_by_round: Mapping[int, dict[str, float]],
    orders_by_round: Mapping[int, dict[str, float]],
    *,
    decision_rounds: int,
    delivery_weeks: int,
) -> VN2Simulator:
    simulator = VN2Simulator(initial_states)
    for week in range(1, decision_rounds + delivery_weeks + 1):
        actuals = actuals_by_round.get(week, dict.fromkeys(initial_states, 0.0))
        orders = orders_by_round.get(week, dict.fromkeys(initial_states, 0.0))
        if week > decision_rounds:
            orders = {uid: 0.0 for uid in actuals}
        simulator.step(week, orders=orders, actual_demand=actuals)
    return simulator


def _cost_diagnostic_tables(
    actual: ReplayResult,
    oracle_summary: pd.DataFrame,
    oracle_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    actual_history = actual.history.rename(
        columns={
            "holding_cost": "actual_holding_cost",
            "shortage_cost": "actual_shortage_cost",
            "end_inventory": "actual_end_inventory",
        }
    )
    oracle_history = oracle_history.rename(
        columns={
            "holding_cost": "oracle_holding_cost",
            "shortage_cost": "oracle_shortage_cost",
            "end_inventory": "oracle_end_inventory",
        }
    )
    merged = actual_history.merge(
        oracle_history[
            [
                UNIQUE_ID,
                "week",
                "oracle_holding_cost",
                "oracle_shortage_cost",
                "oracle_end_inventory",
            ]
        ],
        on=[UNIQUE_ID, "week"],
        how="left",
    )
    merged["avoidable_overstock_cost"] = (
        merged["actual_holding_cost"] - merged["oracle_holding_cost"]
    ).clip(lower=0.0)
    merged["avoidable_shortage_cost"] = (
        merged["actual_shortage_cost"] - merged["oracle_shortage_cost"]
    ).clip(lower=0.0)
    actual_total = merged["actual_holding_cost"] + merged["actual_shortage_cost"]
    oracle_total = merged["oracle_holding_cost"] + merged["oracle_shortage_cost"]
    merged["avoidable_total_cost"] = actual_total - oracle_total
    merged["timing_residual_cost"] = (
        merged["avoidable_total_cost"]
        - merged["avoidable_overstock_cost"]
        - merged["avoidable_shortage_cost"]
    )

    by_sku = (
        actual.summary.merge(
            oracle_summary,
            on=UNIQUE_ID,
            suffixes=("_actual", "_oracle"),
        )
        .assign(
            avoidable_total_cost=lambda df: df["total_cost_actual"] - df["total_cost_oracle"],
            avoidable_holding_cost=lambda df: df["holding_cost_actual"] - df["holding_cost_oracle"],
            avoidable_shortage_cost=lambda df: (
                df["shortage_cost_actual"] - df["shortage_cost_oracle"]
            ),
        )
        .sort_values("avoidable_total_cost", ascending=False)
        .reset_index(drop=True)
    )
    by_round = (
        merged.groupby("week", as_index=False)[
            [
                "avoidable_overstock_cost",
                "avoidable_shortage_cost",
                "timing_residual_cost",
                "avoidable_total_cost",
            ]
        ]
        .sum()
        .rename(columns={"week": "round"})
    )
    totals = {
        "actual_total_cost": float(actual.summary["total_cost"].sum()),
        "oracle_total_cost": float(oracle_summary["total_cost"].sum()),
        "avoidable_total_cost": float(
            actual.summary["total_cost"].sum() - oracle_summary["total_cost"].sum()
        ),
        "avoidable_overstock_cost": float(merged["avoidable_overstock_cost"].sum()),
        "avoidable_shortage_cost": float(merged["avoidable_shortage_cost"].sum()),
        "timing_residual_cost": float(merged["timing_residual_cost"].sum()),
    }
    return by_sku, by_round, totals


def oracle_diagnostic(
    *,
    data_dir: Path = DATA_DIR,
    model_config: dict[str, Any] | None = None,
    order_conformal_config: CumulativeConformalRiskConfig | None = CONFORMAL_ORDER_CONFIG,
    horizon: int = HORIZON,
    lead_time: int = LEAD_TIME,
    review_period: int = REVIEW_PERIOD,
    decision_rounds: int = DECISION_ROUNDS,
    delivery_weeks: int = DELIVERY_WEEKS,
    series_filter: list[str] | None = None,
    order_base_scale: float = 1.0,
    order_step: float = 1.0,
) -> dict[str, Any]:
    """Compare current benchmark orders with finite-horizon oracle orders.

    The oracle uses realised future demand and exact VN2 lead-time/cost
    mechanics. It is diagnostic only; no oracle information is used by the
    deployed benchmark policy.
    """
    cache = build_replay_cache(
        data_dir=data_dir,
        model_config=model_config,
        horizon=horizon,
        lead_time=lead_time,
        review_period=review_period,
        decision_rounds=decision_rounds,
        delivery_weeks=delivery_weeks,
        series_filter=series_filter,
    )
    actual = replay_cached_cost(
        cache,
        order_conformal_config=order_conformal_config,
        order_base_scale=order_base_scale,
    )

    total_weeks = decision_rounds + delivery_weeks
    oracle_orders_by_round: dict[int, dict[str, float]] = {
        week: {} for week in range(1, decision_rounds + 1)
    }
    for uid, state in cache.initial_states.items():
        sku_demands = {
            week: cache.actuals_by_round.get(week, {}).get(uid, 0.0)
            for week in range(1, total_weeks + 1)
        }
        sku_orders = _optimal_order_path_for_sku(
            state,
            sku_demands,
            decision_rounds=decision_rounds,
            total_weeks=total_weeks,
            order_step=order_step,
            lead_time=lead_time,
        )
        for week, order in sku_orders.items():
            oracle_orders_by_round[week][uid] = order

    oracle_simulator = _simulate_orders(
        cache.initial_states,
        cache.actuals_by_round,
        oracle_orders_by_round,
        decision_rounds=decision_rounds,
        delivery_weeks=delivery_weeks,
    )
    oracle_summary = _summary_from_simulator(oracle_simulator)
    oracle_history = oracle_simulator.to_dataframe()

    order_rows = []
    for week in range(1, decision_rounds + 1):
        actual_orders = actual.orders_by_round.get(week, {})
        oracle_orders = oracle_orders_by_round.get(week, {})
        for uid in cache.initial_states:
            actual_order = float(actual_orders.get(uid, 0.0))
            oracle_order = float(oracle_orders.get(uid, 0.0))
            order_rows.append(
                {
                    "round": week,
                    UNIQUE_ID: uid,
                    "actual_order_qty": actual_order,
                    "oracle_order_qty": oracle_order,
                    "order_gap": actual_order - oracle_order,
                    "order_bias": "over"
                    if actual_order > oracle_order
                    else "under"
                    if actual_order < oracle_order
                    else "match",
                }
            )
    orders = pd.DataFrame(order_rows)
    by_sku, by_round, totals = _cost_diagnostic_tables(actual, oracle_summary, oracle_history)

    return {
        "totals": totals,
        "orders": orders,
        "by_sku": by_sku,
        "by_round": by_round,
        "actual_summary": actual.summary,
        "oracle_summary": oracle_summary,
        "actual_history": actual.history,
        "oracle_history": oracle_history,
    }


# ------------------------------------------------------------------ #
# Main entry point
# ------------------------------------------------------------------ #
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
    execution_backend: str = "auto",
    ray_address: str | None = None,
    ray_threshold: int = 10,
    max_concurrency: int | None = None,
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
        ray_threshold: Minimum local task count before ``auto`` uses Ray.
        max_concurrency: Optional cap on concurrent uid tasks.

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
        engine = BackendEngine(
            execution=ExecutionOptions(
                freq="W-MON",
                backend=execution_backend,  # type: ignore[arg-type]
                ray_address=ray_address,
                ray_threshold=ray_threshold,
                max_concurrency=max_concurrency,
            )
        )
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
                ray_threshold=ray_threshold,
                max_concurrency=max_concurrency,
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
