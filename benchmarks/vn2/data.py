"""VN2 data shaping and model-config helpers."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

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
from calibre.core.io import exists, join_uri
from calibre.execution.data_loading import melt_wide_instock
from calibre.forecasting.features import add_stockout_features

# Default rolling-mean / rolling-std windows applied at lag 1; these carry the
# seasonal signal MLForecastAdapter would otherwise drop.
ROLLING_WINDOWS = [4, 13, 26]


def _prepare_history(sales: pd.DataFrame, instock: pd.DataFrame | None) -> pd.DataFrame:
    """Replace observed sales with censored-demand imputed values."""
    df = add_stockout_features(sales, instock)
    return df[[UNIQUE_ID, DS, "y_uncensored"]].rename(columns={"y_uncensored": Y})


def prepare_cumulative_target_history(
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
    realised demand via summation. ``as_cumulative_decision_frame`` zeroes
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


def load_instock(data_dir: str | Path, series_filter: list[str] | None) -> pd.DataFrame | None:
    """Load the in-stock frame, or ``None`` when the file is absent."""
    instock_path = join_uri(data_dir, "week_0_in_stock.csv")
    if not exists(instock_path):
        return None
    instock = melt_wide_instock(instock_path)
    if series_filter is not None:
        instock = instock[instock[UNIQUE_ID].isin(series_filter)]
    return instock


def model_uses_cumulative_target(model_config: Mapping[str, Any]) -> bool:
    """Return whether a model config requests the cumulative target mode."""
    return str(model_config.get("_target_mode", "")).lower() == "cumulative"


def prepare_model_history(
    sales: pd.DataFrame,
    instock: pd.DataFrame | None,
    protection_period: int,
    cumulative_target: bool,
) -> pd.DataFrame:
    """Shape model-training history, branching on the cumulative-target mode."""
    if cumulative_target:
        return prepare_cumulative_target_history(sales, instock, protection_period)
    return _prepare_history(sales, instock)


def as_cumulative_decision_frame(
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


def prepare_policy_forecast_frame(
    frame: pd.DataFrame,
    protection_period: int,
    cumulative_target: bool,
) -> pd.DataFrame:
    """Shape a policy forecast frame, branching on the cumulative-target mode."""
    if cumulative_target:
        return as_cumulative_decision_frame(frame, protection_period)
    return frame


def build_model_config(
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


def strip_private(config: dict[str, Any]) -> dict[str, Any]:
    """Drop debug ``_*`` keys before handing the config to the engine."""
    return {k: v for k, v in config.items() if not k.startswith("_")}
