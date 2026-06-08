from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    DS,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)

COVERAGE_DIAGNOSTIC_COLUMNS = [
    "target_coverage",
    "total_rows",
    "resolved_rows",
    "unresolved_rows",
    "scored_rows",
    "unscored_rows",
    "coverage",
    "mean_interval_width",
    "median_interval_width",
    "min_interval_width",
    "max_interval_width",
]


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
    interval_bounds: tuple[str, str] | None = None,
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

        row = dict(zip(group_by, keys, strict=False))
        for metric_fn in metrics:
            name: str | None = getattr(metric_fn, "__name__", None)
            if name is None:
                if isinstance(metric_fn, functools.partial):
                    name = getattr(metric_fn.func, "__name__", str(metric_fn))
                else:
                    name = str(metric_fn)
            row[name] = metric_fn(actual, predicted)

        if interval_bounds is not None:
            lower_col, upper_col = interval_bounds
            if lower_col in group.columns and upper_col in group.columns:
                interval_group = group.dropna(subset=[lower_col, upper_col, Y])
                if not interval_group.empty:
                    covered = (interval_group[lower_col] <= interval_group[Y]) & (
                        interval_group[Y] <= interval_group[upper_col]
                    )
                    row["coverage"] = float(covered.mean())
                    row["mean_interval_width"] = float(
                        (interval_group[upper_col] - interval_group[lower_col]).mean()
                    )
        results.append(row)

    return pd.DataFrame(results)


def compute_interval_coverage(
    ledger_df: pd.DataFrame,
    *,
    coverage: float,
    group_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute realized interval coverage with explicit scored denominators.

    Unresolved rows are counted but excluded from the scored denominator. Rows
    with resolved actuals but missing or non-finite interval bounds are reported
    as ``unscored_rows`` so warmup gaps do not masquerade as miscoverage.
    """

    if group_by is None:
        group_by = [UNIQUE_ID, H, MODEL_NAME]
    group_cols = list(group_by)
    lower_col, upper_col = interval_column_names(coverage)
    required = [Y, lower_col, upper_col, *group_cols]
    missing = [column for column in required if column not in ledger_df.columns]
    if missing:
        raise ValueError(f"coverage input missing required column(s): {missing}")

    columns = [*group_cols, *COVERAGE_DIAGNOSTIC_COLUMNS]
    if ledger_df.empty and group_cols:
        return pd.DataFrame(columns=columns)

    y_values = pd.to_numeric(ledger_df[Y], errors="coerce")
    lower_values = pd.to_numeric(ledger_df[lower_col], errors="coerce")
    upper_values = pd.to_numeric(ledger_df[upper_col], errors="coerce")
    resolved_mask = y_values.notna() & np.isfinite(y_values)
    finite_bounds_mask = (
        lower_values.notna()
        & upper_values.notna()
        & np.isfinite(lower_values)
        & np.isfinite(upper_values)
    )
    scored_mask = resolved_mask & finite_bounds_mask
    covered_mask = scored_mask & (lower_values <= y_values) & (y_values <= upper_values)
    widths = upper_values - lower_values

    if group_cols:
        grouped = ledger_df.groupby(group_cols, dropna=False, sort=False)
        iterator = grouped
    else:
        iterator = [((), ledger_df)]

    rows: list[dict[str, object]] = []
    for keys, group in iterator:
        if not isinstance(keys, tuple):
            keys = (keys,)
        idx = group.index
        total_rows = int(len(group))
        resolved_rows = int(resolved_mask.loc[idx].sum())
        scored_rows = int(scored_mask.loc[idx].sum())
        scored_idx = idx[scored_mask.loc[idx].to_numpy()]
        scored_widths = widths.loc[scored_idx]

        row: dict[str, object] = dict(zip(group_cols, keys, strict=False))
        row.update(
            {
                "target_coverage": float(coverage),
                "total_rows": total_rows,
                "resolved_rows": resolved_rows,
                "unresolved_rows": total_rows - resolved_rows,
                "scored_rows": scored_rows,
                "unscored_rows": resolved_rows - scored_rows,
                "coverage": (
                    float(covered_mask.loc[idx].sum() / scored_rows) if scored_rows > 0 else np.nan
                ),
                "mean_interval_width": (float(scored_widths.mean()) if scored_rows > 0 else np.nan),
                "median_interval_width": (
                    float(scored_widths.median()) if scored_rows > 0 else np.nan
                ),
                "min_interval_width": (float(scored_widths.min()) if scored_rows > 0 else np.nan),
                "max_interval_width": (float(scored_widths.max()) if scored_rows > 0 else np.nan),
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=columns)
