"""VN2 benchmark data preparation and model-configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import optuna
import pandas as pd
from mlforecast.lag_transforms import RollingMean, RollingStd

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
from calibre.evaluation.point_metrics import pinball_linear
from calibre.execution.data_loading import melt_wide_instock
from calibre.execution.io import exists, join_uri
from calibre.forecasting.features import add_stockout_features

# Default rolling-mean / rolling-std windows applied at lag 1; these
# carry the seasonal signal MLForecastAdapter would otherwise drop.
ROLLING_WINDOWS = [4, 13, 26]


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


def _strip_private(config: dict[str, Any]) -> dict[str, Any]:
    """Drop debug ``_*`` keys before handing the config to the engine."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


as_cumulative_decision_frame = _as_cumulative_decision_frame
build_model_config = _build_model_config
cumulative_pinball = _cumulative_pinball
load_instock = _load_instock
model_uses_cumulative_target = _model_uses_cumulative_target
prepare_cumulative_target_history = _prepare_cumulative_target_history
prepare_model_history = _prepare_model_history
prepare_policy_forecast_frame = _prepare_policy_forecast_frame
strip_private = _strip_private
suggest_from_spec = _suggest_from_spec
walk_forward_origins = _walk_forward_origins

__all__ = [
    "ROLLING_WINDOWS",
    "as_cumulative_decision_frame",
    "build_model_config",
    "cumulative_pinball",
    "load_instock",
    "model_uses_cumulative_target",
    "prepare_cumulative_target_history",
    "prepare_model_history",
    "prepare_policy_forecast_frame",
    "strip_private",
    "suggest_from_spec",
    "walk_forward_origins",
]
