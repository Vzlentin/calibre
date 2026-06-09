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
    if not (ledger_df[Y].isna() & (ledger_df[DS] <= current_origin)).any():
        updated = ledger_df.copy()
        return updated, pd.DataFrame(columns=updated.columns)

    # Keep this import local: importing execution.actuals at module load creates
    # an evaluation -> execution -> backend -> evaluation cycle.
    from calibre.execution.actuals import FrameActualsSource

    return FrameActualsSource(actuals).resolve(ledger_df, current_origin)


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

    prepared = prepare_interval_coverage_frame(
        ledger_df,
        coverage=coverage,
        group_by=group_by,
    )
    return summarize_interval_coverage(prepared, coverage=coverage, group_by=group_by)


def prepare_interval_coverage_frame(
    ledger_df: pd.DataFrame,
    *,
    coverage: float,
    group_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Prepare interval-coverage masks once for one or more aggregations."""

    group_cols = _coverage_group_columns(group_by)
    lower_col, upper_col = interval_column_names(coverage)
    required = [Y, lower_col, upper_col, *group_cols]
    missing = [column for column in required if column not in ledger_df.columns]
    if missing:
        raise ValueError(f"coverage input missing required column(s): {missing}")

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

    scoring = (
        ledger_df.loc[:, group_cols].copy() if group_cols else pd.DataFrame(index=ledger_df.index)
    )
    scoring["_resolved"] = resolved_mask
    scoring["_scored"] = scored_mask
    scoring["_covered"] = covered_mask
    scoring["_width"] = widths.where(scored_mask)
    return scoring


def summarize_interval_coverage(
    scoring: pd.DataFrame,
    *,
    coverage: float,
    group_by: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Aggregate a prepared interval-coverage frame."""

    group_cols = _coverage_group_columns(group_by)
    missing = [
        column
        for column in [*group_cols, "_resolved", "_scored", "_covered", "_width"]
        if column not in scoring.columns
    ]
    if missing:
        raise ValueError(f"coverage scoring input missing required column(s): {missing}")

    if group_cols:
        result = (
            scoring.groupby(group_cols, dropna=False, sort=False)
            .agg(
                total_rows=("_resolved", "size"),
                resolved_rows=("_resolved", "sum"),
                scored_rows=("_scored", "sum"),
                covered_rows=("_covered", "sum"),
                mean_interval_width=("_width", "mean"),
                median_interval_width=("_width", "median"),
                min_interval_width=("_width", "min"),
                max_interval_width=("_width", "max"),
            )
            .reset_index()
        )
    else:
        result = pd.DataFrame(
            [
                {
                    "total_rows": int(len(scoring)),
                    "resolved_rows": int(scoring["_resolved"].sum()),
                    "scored_rows": int(scoring["_scored"].sum()),
                    "covered_rows": int(scoring["_covered"].sum()),
                    "mean_interval_width": float(scoring["_width"].mean()),
                    "median_interval_width": float(scoring["_width"].median()),
                    "min_interval_width": float(scoring["_width"].min()),
                    "max_interval_width": float(scoring["_width"].max()),
                }
            ]
        )

    result["target_coverage"] = float(coverage)
    result["unresolved_rows"] = result["total_rows"] - result["resolved_rows"]
    result["unscored_rows"] = result["resolved_rows"] - result["scored_rows"]
    result["coverage"] = np.where(
        result["scored_rows"].gt(0),
        result["covered_rows"] / result["scored_rows"],
        np.nan,
    )
    columns = [*group_cols, *COVERAGE_DIAGNOSTIC_COLUMNS]
    return result.drop(columns="covered_rows")[columns]


def _coverage_group_columns(group_by: Sequence[str] | None) -> list[str]:
    if group_by is None:
        group_by = [UNIQUE_ID, H, MODEL_NAME]
    return list(group_by)
