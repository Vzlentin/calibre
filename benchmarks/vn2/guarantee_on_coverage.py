"""Post-hoc realized-coverage analysis for the guarantee-on VN2 run (#286).

The guarantee-on variant calibrates a one-sided decision bound at tau; nothing
in the engine computes *realized* coverage of that bound, so this module does it
post hoc from run artifacts: per ``(unique_id, forecast_origin)``, the event
"realized protection-window demand sum <= calibrated bound". Coverage is
reported against both the raw sales series and the censoring-aware series
(``y_uncensored``), because the old engine's live scoring consumes censored
sales and raw-only coverage would not transfer to a demand-honest engine.
"""

from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.forecasting.features import add_stockout_features

H = "h"
FORECAST_ORIGIN = "forecast_origin"
BOUND = "bound"
DEMAND = "demand"
COVERED = "covered"


def detect_bound_column(frame: pd.DataFrame) -> str:
    """Return the single one-sided upper-bound column (``hi_<coverage>``).

    Raises:
        ValueError: if the frame carries zero or several ``hi_`` columns.
    """
    candidates = [c for c in frame.columns if c.startswith("hi_")]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one hi_<coverage> column, found {candidates}")
    return candidates[0]


def terminal_bounds(
    conformal_frame: pd.DataFrame,
    protection_period: int,
    bound_col: str | None = None,
) -> pd.DataFrame:
    """Extract the decision bound per ``(unique_id, forecast_origin)``.

    In cumulative conformal mode the order-up-to level is the ``hi_<coverage>``
    value on the terminal-horizon row (``h == protection_period``); earlier-h
    rows carry NaN.

    Returns:
        Frame ``[unique_id, forecast_origin, bound]``, one row per pair.
    """
    col = bound_col or detect_bound_column(conformal_frame)
    terminal = conformal_frame.loc[conformal_frame[H] == protection_period]
    out = terminal[[UNIQUE_ID, FORECAST_ORIGIN, col]].rename(columns={col: BOUND})
    if out[BOUND].isna().any():
        raise ValueError("NaN bound on a terminal-horizon row; frame is not a settled run")
    return out.reset_index(drop=True)


def window_demand(
    panel: pd.DataFrame,
    origins: list[pd.Timestamp],
    protection_period: int,
    value_col: str = Y,
) -> pd.DataFrame:
    """Sum realized demand per series over each origin's protection window.

    The window for an origin is the ``protection_period`` weekly steps
    ``origin, origin+1w, ..., origin+(P-1)w`` — the h=1..P weeks the cumulative
    bound covers.

    Raises:
        ValueError: if any window week is missing from the panel (a truncated
            window would silently understate demand and overstate coverage).

    Returns:
        Frame ``[unique_id, forecast_origin, demand]``.
    """
    frames = []
    available = set(panel[DS].unique())
    for origin in origins:
        weeks = [pd.Timestamp(origin) + pd.Timedelta(weeks=k) for k in range(protection_period)]
        missing = [w for w in weeks if w not in available]
        if missing:
            raise ValueError(f"origin {origin}: window weeks missing from panel: {missing}")
        window = panel.loc[panel[DS].isin(weeks)]
        summed = window.groupby(UNIQUE_ID, sort=True)[value_col].sum().reset_index()
        summed = summed.rename(columns={value_col: DEMAND})
        summed[FORECAST_ORIGIN] = pd.Timestamp(origin)
        frames.append(summed[[UNIQUE_ID, FORECAST_ORIGIN, DEMAND]])
    return pd.concat(frames, ignore_index=True)


def censoring_aware_panel(sales: pd.DataFrame, instock: pd.DataFrame | None) -> pd.DataFrame:
    """Build the censoring-aware demand panel ``[unique_id, ds, y_uncensored]``.

    Reuses the engine's own imputation (:func:`add_stockout_features`): out-of-
    stock weeks get ``max(observed sales, expanding median of in-stock sales)``.
    This is an imputed *estimate* of demand, not ground truth — the memo must
    say so.
    """
    enriched = add_stockout_features(sales, instock)
    return enriched[[UNIQUE_ID, DS, "y_uncensored"]]


def realized_coverage(bounds: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Join bounds and window demand; report coverage per origin plus overall.

    Coverage is the mean of the one-sided event ``demand <= bound`` over series.
    The join must be exact: any ``(unique_id, forecast_origin)`` present on one
    side only raises.

    Returns:
        Frame ``[forecast_origin, n, covered, coverage]`` with a trailing
        ``overall`` row (``forecast_origin`` = NaT).
    """
    merged = bounds.merge(demand, on=[UNIQUE_ID, FORECAST_ORIGIN], how="outer", indicator=True)
    if (merged["_merge"] != "both").any():
        bad = merged.loc[merged["_merge"] != "both", [UNIQUE_ID, FORECAST_ORIGIN, "_merge"]]
        raise ValueError(f"bounds/demand join mismatch:\n{bad.head(10)}")
    merged[COVERED] = merged[DEMAND] <= merged[BOUND]

    per_origin = (
        merged.groupby(FORECAST_ORIGIN, sort=True)[COVERED]
        .agg(n="count", covered="sum")
        .reset_index()
    )
    per_origin["coverage"] = per_origin["covered"] / per_origin["n"]
    overall = pd.DataFrame(
        {
            FORECAST_ORIGIN: [pd.NaT],
            "n": [len(merged)],
            "covered": [int(merged[COVERED].sum())],
            "coverage": [float(merged[COVERED].mean())],
        }
    )
    per_origin["covered"] = per_origin["covered"].astype(int)
    return pd.concat([per_origin, overall], ignore_index=True)
