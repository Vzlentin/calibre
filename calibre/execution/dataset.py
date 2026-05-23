from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from calibre.core.order_types import CostStruct
from calibre.execution.io import read_parquet
from calibre.ordering.simulation.state import ProductState, make_pipeline


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    history: pd.DataFrame
    future_x: pd.DataFrame | None
    costs: CostStruct | dict[str, CostStruct]
    hierarchy: pd.DataFrame | None
    censoring: pd.DataFrame | None


class DatasetAdapter(Protocol):
    def load(self, path: str | Path, **kwargs) -> DatasetBundle: ...

    def name(self) -> str: ...


class SalesAdapter(Protocol):
    def load_history(self, source: str | Path | None = None) -> pd.DataFrame: ...


class InventoryAdapter(Protocol):
    def load_state(self, unique_id: str, at_origin: pd.Timestamp) -> ProductState: ...

    def load_lead_times(self) -> dict[str, int]: ...


class SyntheticInventoryAdapter:
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
    def load_state(self, unique_id: str, at_origin: pd.Timestamp) -> ProductState:
        raise NotImplementedError("Implement ERP inventory state loading in client code")

    def load_lead_times(self) -> dict[str, int]:
        raise NotImplementedError("Implement ERP lead-time loading in client code")


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
