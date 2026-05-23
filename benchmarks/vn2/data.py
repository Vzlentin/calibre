"""Data-loading and windowing utilities shared by VN2 benchmark modules."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from benchmarks.vn2.simulator import extract_new_actuals
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    is_quantile_column,
)
from calibre.execution.data_loading import melt_wide_instock
from calibre.execution.io import exists, join_uri
from calibre.forecasting.features import add_stockout_features


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


def _walk_forward_origins(
    history: pd.DataFrame, n_origins: int, horizon: int
) -> list[pd.Timestamp]:
    """Pick the last `n_origins` origins from the history's tail."""
    all_dates = sorted(history[DS].unique())
    if len(all_dates) < n_origins + horizon:
        n_origins = max(1, len(all_dates) - horizon)
    if n_origins <= 0:
        return []
    return [pd.Timestamp(d) for d in all_dates[-(n_origins + horizon) : -horizon]]


def _round_actuals(
    data_dir: str | Path,
    round_num: int,
    state_keys: Mapping[str, object],
) -> dict[str, float]:
    try:
        actuals = extract_new_actuals(data_dir, round_num)
    except (FileNotFoundError, ValueError):
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


def _strip_private(config: dict[str, Any]) -> dict[str, Any]:
    """Drop debug ``_*`` keys before handing the config to the engine."""
    return {k: v for k, v in config.items() if not k.startswith("_")}


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
