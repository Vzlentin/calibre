"""Project driver-independent durable domain state for Tier-2 comparisons."""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Protocol

import numpy as np
import pandas as pd

from newcalibre.conformal import ObserveAnnotation
from newcalibre.domain import Calendar, SessionIdentity
from newcalibre.engine import SettlementSnapshot
from newcalibre.engine.run_store import CommitKey, CommitReceipt
from newcalibre.ledger import ForecastRow, OrderRow, SettlementRecord
from newcalibre.observe import ObservationResolution, ObservedActual, PendingObservation

_Normalized = object


class _DurableRunStore(Protocol):
    """Expose only the durable store facts consumed by the projection."""

    @property
    def session(self) -> SessionIdentity: ...

    @property
    def calendar(self) -> Calendar: ...

    @property
    def latest_origin(self) -> pd.Timestamp | None: ...

    @property
    def forecasts(self) -> tuple[ForecastRow, ...]: ...

    @property
    def orders(self) -> tuple[OrderRow, ...]: ...

    @property
    def settlements(self) -> tuple[SettlementRecord, ...]: ...

    @property
    def observed_history(self) -> tuple[ObservedActual, ...]: ...

    @property
    def pending_observations(self) -> tuple[PendingObservation, ...]: ...

    @property
    def observation_resolutions(self) -> tuple[ObservationResolution, ...]: ...

    @property
    def observe_annotations(self) -> tuple[ObserveAnnotation, ...]: ...

    def settlement_snapshot(self, periods: Sequence[pd.Timestamp]) -> SettlementSnapshot: ...

    @property
    def states(self) -> Mapping[str, bytes]: ...

    @property
    def checkpoints(self) -> Mapping[str, bytes]: ...

    @property
    def checkpoint_indexes(self) -> Mapping[str, bytes]: ...

    @property
    def revision(self) -> int: ...

    @property
    def resume_marker(self) -> pd.Timestamp | None: ...

    @property
    def receipts(self) -> Mapping[CommitKey, CommitReceipt]: ...

    @property
    def settlement_receipts(self) -> Mapping[pd.Timestamp, CommitReceipt | None]: ...


@dataclass(frozen=True, slots=True)
class DurableState:
    """Hold the canonical durable facts shared by both engine drivers."""

    session: str
    forecasts: tuple[_Normalized, ...]
    orders: tuple[_Normalized, ...]
    settlements: tuple[_Normalized, ...]
    observed_history: tuple[_Normalized, ...]
    pending_observations: tuple[_Normalized, ...]
    observation_resolutions: tuple[_Normalized, ...]
    observe_annotations: tuple[_Normalized, ...]
    conformal_states: tuple[_Normalized, ...]
    checkpoints: tuple[_Normalized, ...]
    checkpoint_indexes: tuple[_Normalized, ...]
    revision: int
    receipts: tuple[_Normalized, ...]
    settlement_receipts: tuple[_Normalized, ...]
    resume_marker: _Normalized
    inventory_positions: tuple[_Normalized, ...]
    open_orders: tuple[_Normalized, ...]
    booked_costs: tuple[_Normalized, ...]


def project_durable_state(
    store: _DurableRunStore,
    *,
    include_journal: bool = True,
) -> DurableState:
    """Return a key-sorted typed projection without timing or audit counters."""
    forecasts = tuple(
        _forecast(row)
        for row in sorted(
            store.forecasts,
            key=lambda row: (
                row.series_key.encode(),
                _timestamp_key(row.origin),
                row.horizon_step,
                row.model_name.encode(),
            ),
        )
    )
    orders = tuple(
        _order(row)
        for row in sorted(
            store.orders,
            key=lambda row: (
                row.session.value,
                row.series_key.encode(),
                _timestamp_key(row.origin),
                row.model_name.encode(),
            ),
        )
    )
    settlements = tuple(
        _settlement(row)
        for row in sorted(
            store.settlements,
            key=lambda row: (
                row.session.value,
                row.series_key.encode(),
                _timestamp_key(row.period),
            ),
        )
    )
    observed = tuple(
        _normalize(value)
        for value in sorted(
            store.observed_history,
            key=lambda value: (value.series_key.encode(), _timestamp_key(value.timestamp)),
        )
    )
    pending = tuple(
        _normalize(value)
        for value in sorted(
            store.pending_observations,
            key=lambda value: _forecast_key(value.forecast_key),
        )
    )
    resolutions = tuple(
        _normalize(value)
        for value in sorted(
            store.observation_resolutions,
            key=lambda value: _forecast_key(value.forecast_key),
        )
    )
    annotations = tuple(
        _normalize(value)
        for value in sorted(
            store.observe_annotations,
            key=lambda value: _forecast_key(value.forecast_key),
        )
    )
    state_rows = tuple(
        (label, _normalize(value))
        for label, value in sorted(
            store.states.items(),
            key=lambda item: item[0].encode(),
        )
    )
    checkpoint_rows = tuple(
        (key, _normalize(value))
        for key, value in sorted(
            store.checkpoints.items(),
            key=lambda item: item[0].encode(),
        )
    )
    checkpoint_index_rows = tuple(
        (key, _normalize(value))
        for key, value in sorted(
            store.checkpoint_indexes.items(),
            key=lambda item: item[0].encode(),
        )
    )
    receipt_rows = (
        tuple(
            (_normalize(key), _normalize(receipt))
            for key, receipt in sorted(
                store.receipts.items(),
                key=lambda item: repr(_normalize(item[0])),
            )
        )
        if include_journal
        else ()
    )
    settlement_receipt_rows = (
        tuple(
            (_normalize(period), _normalize(receipt))
            for period, receipt in sorted(store.settlement_receipts.items())
        )
        if include_journal
        else ()
    )
    positions, open_orders = _inventory_state(store)
    holding = math.fsum(row.holding.amount for row in store.settlements)
    shortage = math.fsum(row.shortage.amount for row in store.settlements)
    total = math.fsum((holding, shortage))
    return DurableState(
        session=store.session.value,
        forecasts=forecasts,
        orders=orders,
        settlements=settlements,
        observed_history=observed,
        pending_observations=pending,
        observation_resolutions=resolutions,
        observe_annotations=annotations,
        conformal_states=state_rows,
        checkpoints=checkpoint_rows,
        checkpoint_indexes=checkpoint_index_rows,
        revision=store.revision if include_journal else 0,
        receipts=receipt_rows,
        settlement_receipts=settlement_receipt_rows,
        resume_marker=_normalize(store.resume_marker if include_journal else None),
        inventory_positions=positions,
        open_orders=open_orders,
        booked_costs=(
            ("holding", _normalize(holding)),
            ("shortage", _normalize(shortage)),
            ("total", _normalize(total)),
        ),
    )


def _forecast(row: ForecastRow) -> _Normalized:
    values = tuple(
        (name, _normalize(value))
        for name, value in sorted(row.values.items(), key=lambda item: item[0].encode())
    )
    issuances = tuple(
        (_normalize(bound_key), _normalize(issuance))
        for bound_key, issuance in sorted(
            row.issuances.items(),
            key=lambda item: tuple(column.encode() for column in item[0]),
        )
    )
    return (
        "forecast",
        values,
        issuances,
        _normalize(row.observation_issuance),
    )


def _order(row: OrderRow) -> _Normalized:
    return (
        "order",
        _normalize(row.session),
        _normalize(row.series_key),
        _normalize(row.origin),
        _normalize(row.model_name),
        _normalize(row.quantity),
        _normalize(row.arrival_period),
        _normalize(row.evidence),
    )


def _settlement(row: SettlementRecord) -> _Normalized:
    return (
        "settlement",
        _normalize(row.session),
        _normalize(row.series_key),
        _normalize(row.period),
        _normalize(row.arrivals),
        _normalize(row.actuals_semantics),
        _normalize(row.transition),
        _normalize(row.inventory_position),
        _normalize(row.holding),
        _normalize(row.shortage),
        _normalize(row.realized_cost),
    )


def _inventory_state(
    store: _DurableRunStore,
) -> tuple[tuple[_Normalized, ...], tuple[_Normalized, ...]]:
    if store.settlements:
        frontier = max(row.period for row in store.settlements)
        probe = store.calendar.advance(frontier, 1)
    elif store.latest_origin is not None:
        probe = store.latest_origin
    else:
        return (), ()
    snapshot = store.settlement_snapshot((probe,))
    positions = tuple(
        (series_key, _normalize(position))
        for series_key, position in sorted(
            snapshot.current_positions.items(),
            key=lambda item: item[0].encode(),
        )
    )
    open_orders = tuple(
        (series_key, _normalize(quantity))
        for series_key, quantity in sorted(
            snapshot.open_order_quantities.items(),
            key=lambda item: item[0].encode(),
        )
    )
    return positions, open_orders


def _normalize(value: object) -> _Normalized:
    if value is None:
        return ("none",)
    if value is pd.NA:
        return ("pandas-na",)
    if value is pd.NaT:
        return ("pandas-nat",)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, pd.Timestamp):
        return ("timestamp", value.unit, _timestamp_key(value))
    if isinstance(value, SessionIdentity):
        return ("session", value.value)
    if isinstance(value, Enum):
        value_type = type(value)
        return (
            "enum",
            value_type.__module__,
            value_type.__qualname__,
            value.name,
            _normalize(value.value),
        )
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, Integral):
        return ("int", int(value))
    if isinstance(value, Real):
        normalized = float(value)
        return ("float64", struct.pack(">d", normalized).hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, Mapping):
        entries = [(_normalize(key), _normalize(item)) for key, item in value.items()]
        return ("mapping", tuple(sorted(entries, key=repr)))
    if isinstance(value, (tuple, list)):
        return ("tuple", tuple(_normalize(item) for item in value))
    if is_dataclass(value) and not isinstance(value, type):
        value_type = type(value)
        return (
            "dataclass",
            value_type.__module__,
            value_type.__qualname__,
            tuple(
                (field.name, _normalize(getattr(value, field.name)))
                for field in fields(value)
                if not field.name.startswith("_")
            ),
        )
    raise TypeError(f"unsupported durable-state value: {type(value).__name__}")


def _forecast_key(value: object) -> tuple[bytes, tuple[str, int], int, bytes]:
    return (
        value.series_key.encode(),
        _timestamp_key(value.origin),
        value.horizon_step,
        value.model_name.encode(),
    )


def _timestamp_key(value: pd.Timestamp) -> tuple[str, int]:
    return value.unit, int(value.asm8.view("i8"))


__all__ = ["DurableState", "project_durable_state"]
