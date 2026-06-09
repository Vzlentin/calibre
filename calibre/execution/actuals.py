"""Actual lookup sources for delayed ledger resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y, validate_actuals_frame
from calibre.reconciliation.summing import TOTAL_LABEL, build_hierarchy_index


class ActualsSource(Protocol):
    """Source of actual values for due ledger rows."""

    def resolve(
        self,
        ledger_df: pd.DataFrame,
        current_origin: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(updated_ledger, newly_resolved_rows)``."""


@dataclass(frozen=True)
class FrameActualsSource:
    """Resolve actuals from an already materialized actuals frame."""

    actuals: pd.DataFrame
    _lookup: pd.Series = field(init=False, repr=False)

    def __post_init__(self) -> None:
        actuals = _normalize_actuals_frame(self.actuals)
        lookup = actuals.drop_duplicates(subset=[UNIQUE_ID, DS]).set_index([UNIQUE_ID, DS])[Y]
        object.__setattr__(self, "actuals", actuals)
        object.__setattr__(self, "_lookup", lookup)

    def resolve(
        self,
        ledger_df: pd.DataFrame,
        current_origin: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        updated = ledger_df.copy()
        pending_idx = _pending_index(updated, current_origin)
        if len(pending_idx) == 0:
            return updated, pd.DataFrame(columns=updated.columns)

        pending_keys = _pending_keys(updated, pending_idx)
        resolved_y = self._lookup.reindex(pending_keys).to_numpy()
        updated.loc[pending_idx, Y] = resolved_y
        return _resolution_result(updated, pending_idx)


class HierarchyActualsSource:
    """Resolve hierarchy node actuals lazily from bottom-level actual history."""

    def __init__(self, bottom_actuals: pd.DataFrame, hierarchy: pd.DataFrame) -> None:
        self._hierarchy_index = build_hierarchy_index(hierarchy)
        data = _normalize_actuals_frame(bottom_actuals)

        bottom_ids = set(self._hierarchy_index.bottom_ids)
        unknown = sorted(set(data[UNIQUE_ID].unique()) - bottom_ids)
        if unknown:
            raise ValueError(
                f"bottom actuals contain unique_id values not present in hierarchy: {unknown}"
            )

        duplicates = data[data.duplicated([UNIQUE_ID, DS], keep=False)]
        if not duplicates.empty:
            keys = (
                duplicates[[UNIQUE_ID, DS]]
                .drop_duplicates()
                .sort_values([UNIQUE_ID, DS], kind="stable")
            )
            values = [tuple(row) for row in keys.itertuples(index=False, name=None)]
            raise ValueError(f"bottom actuals contain duplicate (unique_id, ds) keys: {values}")

        self._lookup = data.set_index([UNIQUE_ID, DS])[Y].sort_index()
        self._members_by_label = self._build_members_by_label()
        self._node_labels = frozenset(self._members_by_label)

    def resolve(
        self,
        ledger_df: pd.DataFrame,
        current_origin: pd.Timestamp,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        updated = ledger_df.copy()
        pending_idx = _pending_index(updated, current_origin)
        if len(pending_idx) == 0:
            return updated, pd.DataFrame(columns=updated.columns)

        requested = updated.loc[pending_idx, UNIQUE_ID].astype(str)
        unknown = sorted(set(requested.unique()) - self._node_labels)
        if unknown:
            raise ValueError(
                f"requested hierarchy node labels are not present in hierarchy: {unknown}"
            )

        resolved_cache: dict[tuple[str, pd.Timestamp], float] = {}
        resolved_y = [
            self._resolve_label(str(label), pd.Timestamp(ds), resolved_cache)
            for label, ds in zip(
                requested,
                pd.to_datetime(updated.loc[pending_idx, DS]),
                strict=True,
            )
        ]
        updated.loc[pending_idx, Y] = resolved_y
        return _resolution_result(updated, pending_idx)

    def _build_members_by_label(self) -> dict[str, tuple[str, ...]]:
        frame = self._hierarchy_index.frame
        members: dict[str, tuple[str, ...]] = {
            uid: (uid,) for uid in self._hierarchy_index.bottom_ids
        }
        for col in self._hierarchy_index.attr_cols:
            values = frame[col].astype(str)
            for value, group in frame.groupby(values, sort=True):
                label = f"{col}={value}"
                members[label] = tuple(group[UNIQUE_ID].astype(str))
        members[TOTAL_LABEL] = self._hierarchy_index.bottom_ids
        return members

    def _resolve_label(
        self,
        label: str,
        ds: pd.Timestamp,
        resolved_cache: dict[tuple[str, pd.Timestamp], float],
    ) -> float:
        cache_key = (label, ds)
        if cache_key in resolved_cache:
            return resolved_cache[cache_key]

        members = self._members_by_label[label]
        keys = pd.MultiIndex.from_arrays([list(members), [ds] * len(members)])
        if not keys.isin(self._lookup.index).all():
            resolved_cache[cache_key] = np.nan
            return np.nan
        values = self._lookup.reindex(keys)
        if len(members) == 1:
            resolved = float(values.iloc[0]) if pd.notna(values.iloc[0]) else np.nan
        else:
            resolved = float(values.sum())
        resolved_cache[cache_key] = resolved
        return resolved


def ensure_actuals_source(actuals: pd.DataFrame | ActualsSource) -> ActualsSource:
    if isinstance(actuals, pd.DataFrame):
        return FrameActualsSource(actuals)
    return actuals


def _normalize_actuals_frame(actuals: pd.DataFrame) -> pd.DataFrame:
    data = actuals[[UNIQUE_ID, DS, Y]].copy()
    data[UNIQUE_ID] = data[UNIQUE_ID].astype(str)
    data[DS] = pd.to_datetime(data[DS]).astype("datetime64[ns]")
    data[Y] = data[Y].astype("float64")
    validate_actuals_frame(data)
    return data


def _pending_index(ledger_df: pd.DataFrame, current_origin: pd.Timestamp) -> pd.Index:
    mask_pending = ledger_df[Y].isna() & (
        pd.to_datetime(ledger_df[DS]) <= pd.Timestamp(current_origin)
    )
    return ledger_df.index[mask_pending]


def _pending_keys(ledger_df: pd.DataFrame, pending_idx: pd.Index) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [
            ledger_df.loc[pending_idx, UNIQUE_ID].astype(str).to_numpy(),
            pd.to_datetime(ledger_df.loc[pending_idx, DS]).to_numpy(),
        ]
    )


def _resolution_result(
    updated: pd.DataFrame,
    pending_idx: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    newly_resolved_idx = pending_idx[updated.loc[pending_idx, Y].notna()]
    newly_resolved = updated.loc[newly_resolved_idx].copy()
    return updated, newly_resolved
