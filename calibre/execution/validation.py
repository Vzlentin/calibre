"""Structural validation of dataset bundles and cost-file loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from calibre.core.forecast_frame import DS, IN_STOCK, UNIQUE_ID, Y, validate_actuals_frame
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetBundle


def _validate_monotonic_per_series(df: pd.DataFrame, *, name: str) -> None:
    for uid, group in df.groupby(UNIQUE_ID, sort=False):
        if not group[DS].is_monotonic_increasing:
            raise ValueError(f"{name} ds values must be monotonic for unique_id={uid!r}")


def _validate_hierarchy(bundle: DatasetBundle, *, history_uids: set[str]) -> None:
    hierarchy = bundle.hierarchy
    if hierarchy is None:
        return
    if UNIQUE_ID not in hierarchy.columns:
        raise ValueError("hierarchy missing required column: unique_id")
    if hierarchy[UNIQUE_ID].isna().any():
        raise ValueError("hierarchy has null unique_id values")
    if hierarchy[UNIQUE_ID].duplicated().any():
        duplicates = hierarchy.loc[hierarchy[UNIQUE_ID].duplicated(), UNIQUE_ID].astype(str)
        raise ValueError(f"hierarchy has duplicate unique_id rows: {sorted(duplicates.unique())}")
    hierarchy_uids = set(hierarchy[UNIQUE_ID].astype(str).unique())
    missing = history_uids - hierarchy_uids
    if missing:
        raise ValueError(f"hierarchy missing unique_id values from history: {sorted(missing)}")


def _validate_key_frame(df: pd.DataFrame, *, name: str) -> None:
    missing = {UNIQUE_ID, DS} - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")
    if df[[UNIQUE_ID, DS]].isna().any().any():
        raise ValueError(f"{name} has null values in key columns")
    if not pd.api.types.is_datetime64_any_dtype(df[DS]):
        raise ValueError(f"{name}.{DS} expected datetime64, got {df[DS].dtype}")
    _validate_monotonic_per_series(df, name=name)


def validate_dataset_bundle(bundle: DatasetBundle) -> None:
    """Validate a :class:`DatasetBundle`'s history, hierarchy, future_x, censoring, and costs.

    Raises:
        ValueError: If any frame violates its structural contract — e.g. null
            actuals, non-monotonic dates, a hierarchy missing history series, or
            costs missing a ``unique_id``.
    """
    validate_actuals_frame(bundle.history)
    if bundle.history[Y].isna().any():
        raise ValueError("history.y must not contain nulls")
    _validate_monotonic_per_series(bundle.history, name="history")

    history_uids = set(bundle.history[UNIQUE_ID].astype(str).unique())
    _validate_hierarchy(bundle, history_uids=history_uids)

    if bundle.future_x is not None:
        _validate_key_frame(bundle.future_x, name="future_x")
        future_uids = set(bundle.future_x[UNIQUE_ID].astype(str).unique())
        unknown = future_uids - history_uids
        if unknown:
            raise ValueError(f"future_x contains unknown unique_id values: {sorted(unknown)}")

    if bundle.censoring is not None:
        _validate_key_frame(bundle.censoring, name="censoring")
        if IN_STOCK not in bundle.censoring.columns:
            raise ValueError(f"censoring missing required column: {IN_STOCK}")
        if bundle.censoring[IN_STOCK].dtype != bool:
            raise ValueError(
                f"censoring.{IN_STOCK} expected bool, got {bundle.censoring[IN_STOCK].dtype}"
            )

    if isinstance(bundle.costs, CostStruct):
        return
    if not isinstance(bundle.costs, dict):
        raise ValueError("costs must be a CostStruct or dict[str, CostStruct]")
    missing_costs = history_uids - set(str(uid) for uid in bundle.costs)
    if missing_costs:
        raise ValueError(f"costs missing unique_id values: {sorted(missing_costs)}")
    for uid, cost in bundle.costs.items():
        if not isinstance(cost, CostStruct):
            raise ValueError(f"cost for unique_id={uid!r} must be CostStruct")


def _read_cost_frame(uri: str | Path) -> pd.DataFrame:
    text = str(uri)
    if text.endswith(".parquet"):
        return pd.read_parquet(text)
    return pd.read_csv(text)


def load_costs(uri: str | Path) -> dict[str, CostStruct]:
    """Load a per-``unique_id`` cost table (CSV or parquet) into a :class:`CostStruct` map.

    Raises:
        ValueError: If required columns are missing or a ``unique_id`` is
            duplicated.
    """
    frame = _read_cost_frame(uri)
    required = {UNIQUE_ID, "underage_cost", "overage_cost"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"cost file missing required columns: {sorted(missing)}")
    if frame[UNIQUE_ID].duplicated().any():
        duplicates = frame.loc[frame[UNIQUE_ID].duplicated(), UNIQUE_ID].astype(str).tolist()
        raise ValueError(f"duplicate cost rows for unique_id values: {duplicates}")

    costs: dict[str, CostStruct] = {}
    for _, row in frame.iterrows():
        values: dict[str, Any] = {
            "underage_cost": float(row["underage_cost"]),
            "overage_cost": float(row["overage_cost"]),
        }
        if "holding_cost" in frame.columns and pd.notna(row["holding_cost"]):
            values["holding_cost"] = float(row["holding_cost"])
        if "shortage_cost" in frame.columns and pd.notna(row["shortage_cost"]):
            values["shortage_cost"] = float(row["shortage_cost"])
        costs[str(row[UNIQUE_ID])] = CostStruct(**values)
    return costs
