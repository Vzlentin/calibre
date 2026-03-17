from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from calibre.contracts.forecast_frame import (
    DS,
    UNIQUE_ID,
    Y,
    Y_HAT,
    H,
)


def resolve_actuals(
    ledger_df: pd.DataFrame,
    actuals: pd.DataFrame,
    current_origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill y where ds <= current_origin and y is currently NaN.

    Returns (updated_ledger, newly_resolved_rows).
    """
    updated = ledger_df.copy()

    mask_pending = updated[Y].isna() & (updated[DS] <= current_origin)
    if not mask_pending.any():
        return updated, pd.DataFrame(columns=updated.columns)

    lookup = actuals.drop_duplicates(subset=[UNIQUE_ID, DS]).set_index([UNIQUE_ID, DS])[Y]

    pending_idx = updated.index[mask_pending]
    pending_keys = pd.MultiIndex.from_arrays(
        [updated.loc[pending_idx, UNIQUE_ID].values, updated.loc[pending_idx, DS].values]
    )
    resolved_y = lookup.reindex(pending_keys).values
    updated.loc[pending_idx, Y] = resolved_y

    newly_resolved = updated.loc[pending_idx[updated.loc[pending_idx, Y].notna()]].copy()

    return updated, newly_resolved


def compute_row_errors(resolved_df: pd.DataFrame) -> pd.DataFrame:
    """Add error, abs_error, pct_error columns to resolved rows."""
    df = resolved_df.copy()
    df["error"] = df[Y] - df[Y_HAT]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["error"] / df[Y].replace(0, np.nan)
    return df


def compute_metrics(
    ledger_df: pd.DataFrame,
    metrics: list[Callable],
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """Compute aggregate metrics on resolved rows, grouped by specified columns."""
    if group_by is None:
        group_by = [UNIQUE_ID, H]

    resolved = ledger_df.dropna(subset=[Y, Y_HAT])
    if resolved.empty:
        return pd.DataFrame()

    results = []
    for keys, group in resolved.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = group[Y].to_numpy()
        predicted = group[Y_HAT].to_numpy()

        row = dict(zip(group_by, keys))
        for metric_fn in metrics:
            name = getattr(metric_fn, "__name__", None)
            if name is None:
                name = getattr(metric_fn.func, "__name__", str(metric_fn))
            row[name] = metric_fn(actual, predicted)
        results.append(row)

    return pd.DataFrame(results)
