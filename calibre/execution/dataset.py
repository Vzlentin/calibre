"""Dataset/inventory/sales ingestion seams and their built-in adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID
from calibre.core.io import read_parquet
from calibre.core.order_types import CostStruct
from calibre.ordering.simulation.state import ProductState, make_pipeline


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """All inputs a dataset adapter resolves for a run.

    Carries history, future regressors, costs, optional hierarchy/censoring
    frames, and optional initial inventory state keyed by ``unique_id`` (set by
    adapters that drive the ordering cost tally; ``None`` for diagnostic-only
    datasets).
    """

    history: pd.DataFrame
    future_x: pd.DataFrame | None
    costs: CostStruct | dict[str, CostStruct]
    hierarchy: pd.DataFrame | None
    censoring: pd.DataFrame | None
    inventory: dict[str, ProductState] | None = None


class DatasetAdapter(Protocol):
    """Load a named dataset into a :class:`DatasetBundle`."""

    def load(self, path: str | Path, **kwargs) -> DatasetBundle: ...

    def name(self) -> str: ...


class InventoryAdapter(Protocol):
    """Supply per-SKU inventory state and lead times to the ordering simulation."""

    def load_state(self, unique_id: str, at_origin: pd.Timestamp) -> ProductState: ...

    def load_lead_times(self) -> dict[str, int]: ...


class SyntheticInventoryAdapter:
    """In-memory inventory state for tests, keyed by ``unique_id``."""

    def __init__(
        self,
        states: dict[str, ProductState],
        *,
        lead_times: dict[str, int] | None = None,
    ) -> None:
        self._states = {uid: state.copy() for uid, state in states.items()}
        self._lead_times = (
            dict(lead_times)
            if lead_times is not None
            else {uid: state.lead_time_depth for uid, state in self._states.items()}
        )

    def load_state(self, unique_id: str, at_origin: pd.Timestamp) -> ProductState:
        del at_origin
        try:
            return self._states[unique_id].copy()
        except KeyError as err:
            raise KeyError(f"Unknown inventory state unique_id: {unique_id}") from err

    def load_lead_times(self) -> dict[str, int]:
        return dict(self._lead_times)


class SnapshotInventoryAdapter:
    """Parquet/fsspec-backed inventory state with optional point-in-time ``as_of`` selection."""

    def __init__(self, snapshot_uri: str | Path, *, default_lead_time: int = 0) -> None:
        self.snapshot_uri = str(snapshot_uri)
        self.default_lead_time = int(default_lead_time)
        self._snapshot: pd.DataFrame | None = None

    def load_state(self, unique_id: str, at_origin: pd.Timestamp) -> ProductState:
        row = self._row_for(unique_id, at_origin)
        lead_time_depth = int(row.get("lead_time_depth", self.default_lead_time))
        return ProductState(
            unique_id=unique_id,
            end_inventory=float(row["end_inventory"]),
            pipeline=make_pipeline(_pipeline_values(row), lead_time_depth),
            cumulative_costs={
                "holding": float(row.get("cumulative_holding_cost", 0.0)),
                "shortage": float(row.get("cumulative_shortage_cost", 0.0)),
            },
        )

    def load_lead_times(self) -> dict[str, int]:
        snapshot = self._load_snapshot()
        if "lead_time_depth" not in snapshot.columns:
            return {
                str(uid): self.default_lead_time
                for uid in snapshot["unique_id"].drop_duplicates().tolist()
            }
        return {
            str(row["unique_id"]): int(row["lead_time_depth"])
            for _, row in snapshot.drop_duplicates("unique_id", keep="last").iterrows()
        }

    def _row_for(self, unique_id: str, at_origin: pd.Timestamp) -> pd.Series:
        snapshot = self._load_snapshot()
        rows = snapshot[snapshot["unique_id"].astype(str) == str(unique_id)]
        if rows.empty:
            raise KeyError(f"Unknown inventory state unique_id: {unique_id}")
        if "as_of" in rows.columns:
            rows = rows.copy()
            rows["as_of"] = pd.to_datetime(rows["as_of"])
            rows = rows[rows["as_of"] <= pd.Timestamp(at_origin)]
            if rows.empty:
                raise KeyError(
                    f"No inventory snapshot for unique_id={unique_id!r} at or before {at_origin}"
                )
            rows = rows.sort_values("as_of")
        return rows.iloc[-1]

    def _load_snapshot(self) -> pd.DataFrame:
        if self._snapshot is None:
            self._snapshot = read_parquet(self.snapshot_uri)
        return self._snapshot


class ErpInventoryAdapter:
    """Placeholder ERP inventory seam — client code implements the lookups."""

    def load_state(self, unique_id: str, at_origin: pd.Timestamp) -> ProductState:
        raise NotImplementedError("Implement ERP inventory state loading in client code")

    def load_lead_times(self) -> dict[str, int]:
        raise NotImplementedError("Implement ERP lead-time loading in client code")


class SalesAdapter(Protocol):
    """Sales-history ingestion seam, mirroring ``InventoryAdapter``.

    ``load_sales`` returns the ``(unique_id, ds, y[, regressors])`` history frame
    the forecaster consumes — the same shape the inline ``/fit`` JSON body used
    to carry — so the API can swap inline history for a parquet/SQL URI without
    touching the downstream fit/predict path.

    Point-in-time contract: when the source carries an ``as_of`` revision marker,
    every adapter collapses to the latest revision per ``(unique_id, ds)`` as of
    the cutoff, and rows whose ``as_of`` is NULL/missing are ALWAYS visible —
    treated as the latest revision regardless of the cutoff. Both the snapshot
    and SQL adapters share ``_latest_revision`` so this NULL semantics never
    diverges between backends."""

    def load_sales(self, sku_set: list[str], as_of: pd.Timestamp | None) -> pd.DataFrame: ...


def _filter_sku_set(frame: pd.DataFrame, sku_set: list[str]) -> pd.DataFrame:
    wanted = {str(s) for s in sku_set}
    return frame[frame[UNIQUE_ID].astype(str).isin(wanted)].copy()


def _latest_revision(frame: pd.DataFrame, as_of: pd.Timestamp | None) -> pd.DataFrame:
    """Collapse a revision-stamped sales frame to the latest revision per key.

    When ``frame`` has no ``as_of`` column it is returned unchanged. Otherwise
    ``as_of`` is coerced to datetime and, when a cutoff is supplied, rows are kept
    where ``as_of`` is NULL/NaT OR ``as_of <= cutoff`` — NULL/missing ``as_of`` is
    the canonical "always visible, latest revision" marker. The remaining rows are
    sorted by ``as_of`` (NaT last, so NULL wins ties) and reduced to the last
    revision per ``(unique_id, ds)``, then the marker column is dropped so it never
    reaches the forecaster as a spurious regressor."""
    if "as_of" not in frame.columns:
        return frame
    frame = frame.copy()
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    if as_of is not None:
        asof = frame["as_of"]
        frame = frame[asof.isna() | (asof <= pd.Timestamp(as_of))]
    return (
        frame.sort_values("as_of")
        .drop_duplicates([UNIQUE_ID, DS], keep="last")
        .drop(columns="as_of")
    )


class SnapshotSalesAdapter:
    """Parquet/fsspec-backed sales history with point-in-time ``as_of`` support.

    When the snapshot carries an ``as_of`` column we treat it as a revision
    marker via the shared ``_latest_revision`` helper: the latest revision per
    ``(unique_id, ds)`` as of the cutoff is kept, NULL/missing ``as_of`` rows stay
    visible as the latest revision, and the column is removed before the frame
    reaches the forecaster (so it is never mistaken for an exogenous regressor)."""

    def __init__(self, sales_uri: str | Path) -> None:
        self.sales_uri = str(sales_uri)
        self._snapshot: pd.DataFrame | None = None

    def load_sales(self, sku_set: list[str], as_of: pd.Timestamp | None) -> pd.DataFrame:
        frame = _filter_sku_set(self._load_snapshot(), sku_set)
        frame = _latest_revision(frame, as_of)
        if DS in frame.columns:
            frame[DS] = pd.to_datetime(frame[DS])
        return frame.reset_index(drop=True)

    def _load_snapshot(self) -> pd.DataFrame:
        if self._snapshot is None:
            self._snapshot = read_parquet(self.sales_uri)
        return self._snapshot


class SyntheticSalesAdapter:
    """In-memory sales history for tests, wrapping a frame or per-SKU dict."""

    def __init__(self, frame_or_dict: pd.DataFrame | dict[str, pd.DataFrame]) -> None:
        if isinstance(frame_or_dict, dict):
            self._frame = (
                pd.concat(frame_or_dict.values(), ignore_index=True)
                if frame_or_dict
                else pd.DataFrame()
            )
        else:
            self._frame = frame_or_dict.copy()

    def load_sales(self, sku_set: list[str], as_of: pd.Timestamp | None) -> pd.DataFrame:
        del as_of
        if self._frame.empty:
            return self._frame.copy()
        return _filter_sku_set(self._frame, sku_set).reset_index(drop=True)


def _pipeline_values(row: pd.Series) -> list[float]:
    pipeline_cols = sorted(
        [col for col in row.index if str(col).startswith("pipeline_")],
        key=lambda col: int(str(col).split("_", 1)[1]),
    )
    if pipeline_cols:
        return [float(row[col]) for col in pipeline_cols]
    in_transit_cols = sorted(
        [col for col in row.index if str(col).startswith("in_transit_w")],
        key=lambda col: int(str(col).rsplit("w", 1)[1]),
    )
    if in_transit_cols:
        return [float(row[col]) for col in in_transit_cols]
    return []
