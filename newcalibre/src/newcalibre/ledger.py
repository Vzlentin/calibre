"""Record immutable forecast, order, and settlement facts in one ledger."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd

from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    Calendar,
    ForecastFrameError,
    GuaranteeClaim,
    GuaranteeDescriptor,
    SessionIdentity,
    interval_columns,
    quantile_column,
    validate_forecast_frame,
)

type ForecastKey = tuple[str, pd.Timestamp, int, str]
type OrderKey = tuple[SessionIdentity, str, pd.Timestamp, str]
type SettlementKey = tuple[SessionIdentity, str, pd.Timestamp]


class LedgerError(ValueError):
    """Report a row or append operation that violates the ledger contract."""


class GuaranteedSide(StrEnum):
    """Name the bound side asserted by a one-sided coverage claim."""

    LOWER = "lower"
    UPPER = "upper"


@dataclass(frozen=True, slots=True)
class ForecastIssuance:
    """Carry the declared guarantee and issued bound-readiness facts."""

    descriptor: GuaranteeDescriptor
    guaranteed_side: GuaranteedSide | None
    calibration_ready: bool
    bounds_finite: bool
    bounds_null_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, GuaranteeDescriptor):
            raise LedgerError("forecast issuance descriptor must be a GuaranteeDescriptor")
        if not isinstance(self.calibration_ready, bool):
            raise LedgerError("calibration readiness must be a boolean")
        if not isinstance(self.bounds_finite, bool):
            raise LedgerError("bounds finiteness must be a boolean")

        if self.bounds_finite:
            if self.bounds_null_reason is not None:
                raise LedgerError("finite bounds cannot carry a bounds null reason")
        else:
            _require_text(self.bounds_null_reason, name="bounds null reason")

        one_sided = self.descriptor.type.claim is GuaranteeClaim.ONE_SIDED_COVERAGE
        if one_sided and not isinstance(self.guaranteed_side, GuaranteedSide):
            raise LedgerError("one-sided coverage requires a guaranteed side")
        if not one_sided and self.guaranteed_side is not None:
            raise LedgerError("only one-sided coverage may declare a guaranteed side")


@dataclass(frozen=True, slots=True, init=False)
class ForecastRow:
    """Expose one validated forecast-frame row as a deeply immutable fact."""

    series_key: str
    target_timestamp: pd.Timestamp
    actual_value: float | None
    point_forecast: float
    horizon_step: int
    origin: pd.Timestamp
    model_name: str
    issuance: ForecastIssuance
    _values: tuple[tuple[str, object], ...]

    def __init__(self) -> None:
        raise TypeError("ForecastRow instances are created by Ledger.append_forecasts()")

    @classmethod
    def _from_validated_values(
        cls,
        values: Mapping[str, object],
        *,
        issuance: ForecastIssuance,
    ) -> ForecastRow:
        snapshot = tuple((name, _snapshot_scalar(value)) for name, value in values.items())
        by_name = dict(snapshot)
        instance = object.__new__(cls)
        object.__setattr__(instance, "series_key", by_name[SERIES_KEY])
        object.__setattr__(instance, "target_timestamp", by_name[TARGET_TIMESTAMP])
        object.__setattr__(instance, "actual_value", by_name[ACTUAL_VALUE])
        object.__setattr__(instance, "point_forecast", by_name[POINT_FORECAST])
        object.__setattr__(instance, "horizon_step", by_name[HORIZON_STEP])
        object.__setattr__(instance, "origin", by_name[ORIGIN])
        object.__setattr__(instance, "model_name", by_name[MODEL_NAME])
        object.__setattr__(instance, "issuance", issuance)
        object.__setattr__(instance, "_values", snapshot)
        return instance

    @property
    def key(self) -> ForecastKey:
        """Return the exact immutable forecast-row key."""
        return (self.series_key, self.origin, self.horizon_step, self.model_name)

    @property
    def values(self) -> Mapping[str, object]:
        """Return a read-only snapshot of every validated frame value."""
        return MappingProxyType(dict(self._values))

    @property
    def descriptor(self) -> GuaranteeDescriptor:
        """Return the guarantee declared at issuance."""
        return self.issuance.descriptor

    @property
    def guaranteed_side(self) -> GuaranteedSide | None:
        """Return the asserted side for a one-sided guarantee, when applicable."""
        return self.issuance.guaranteed_side

    @property
    def calibration_ready(self) -> bool:
        """Return the calibrator readiness decision recorded at issuance."""
        return self.issuance.calibration_ready

    @property
    def bounds_finite(self) -> bool:
        """Return whether the issued bound payload was finite."""
        return self.issuance.bounds_finite

    @property
    def bounds_null_reason(self) -> str | None:
        """Return the recorded cause of a non-finite bound payload."""
        return self.issuance.bounds_null_reason


@dataclass(frozen=True, slots=True)
class OrderRow:
    """Record one immutable non-negative order decision."""

    session: SessionIdentity
    series_key: str
    origin: pd.Timestamp
    model_name: str
    quantity: float
    arrival_period: pd.Timestamp

    def __post_init__(self) -> None:
        _require_session(self.session)
        _require_text(self.series_key, name="series key")
        _require_timestamp(self.origin, name="order origin")
        _require_text(self.model_name, name="model name")
        object.__setattr__(
            self,
            "quantity",
            _finite_nonnegative(self.quantity, name="order quantity"),
        )
        _require_timestamp(self.arrival_period, name="arrival period")

    @property
    def key(self) -> OrderKey:
        """Return the exact immutable order-row key."""
        return (self.session, self.series_key, self.origin, self.model_name)


@dataclass(frozen=True, slots=True)
class StockoutTransition:
    """Record demand consumption and the configured stock-out transition result."""

    rule: str
    demand: float
    fulfilled_demand: float
    unmet_demand: float
    closing_on_hand: float
    closing_backorders: float

    def __post_init__(self) -> None:
        _require_text(self.rule, name="stock-out transition rule")
        for name in (
            "demand",
            "fulfilled_demand",
            "unmet_demand",
            "closing_on_hand",
            "closing_backorders",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name.replace("_", " ")),
            )
        if math.fsum((self.fulfilled_demand, self.unmet_demand)) != self.demand:
            raise LedgerError("fulfilled demand plus unmet demand must equal demand")


@dataclass(frozen=True, slots=True)
class BookedCost:
    """Trace one booked cost amount to its non-negative rate and basis."""

    rate: float
    basis: float
    amount: float

    def __post_init__(self) -> None:
        rate = _finite_nonnegative(self.rate, name="cost rate")
        basis = _finite_nonnegative(self.basis, name="cost basis")
        amount = _finite_nonnegative(self.amount, name="cost amount")
        try:
            recomputed = rate * basis
        except OverflowError as error:
            raise LedgerError("cost rate times basis must be finite") from error
        if not math.isfinite(recomputed):
            raise LedgerError("cost rate times basis must be finite")
        if amount != recomputed:
            raise LedgerError("cost amount must equal rate times basis")
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "amount", amount)

    @property
    def recomputed_amount(self) -> float:
        """Recompute the booked amount from its public trace."""
        return self.rate * self.basis


@dataclass(frozen=True, slots=True)
class SettlementRecord:
    """Book one period's arrivals, demand transition, and realized costs."""

    session: SessionIdentity
    series_key: str
    period: pd.Timestamp
    arrivals: float
    transition: StockoutTransition
    holding: BookedCost
    shortage: BookedCost

    def __post_init__(self) -> None:
        _require_session(self.session)
        _require_text(self.series_key, name="series key")
        _require_timestamp(self.period, name="settlement period")
        object.__setattr__(
            self,
            "arrivals",
            _finite_nonnegative(self.arrivals, name="settlement arrivals"),
        )
        if not isinstance(self.transition, StockoutTransition):
            raise LedgerError("settlement transition must be a StockoutTransition")
        if not isinstance(self.holding, BookedCost):
            raise LedgerError("settlement holding cost must be a BookedCost")
        if not isinstance(self.shortage, BookedCost):
            raise LedgerError("settlement shortage cost must be a BookedCost")
        if not math.isfinite(math.fsum((self.holding.amount, self.shortage.amount))):
            raise LedgerError("settlement realized cost must be finite")

    @property
    def key(self) -> SettlementKey:
        """Return the exact immutable settlement-record key."""
        return (self.session, self.series_key, self.period)

    @property
    def realized_cost(self) -> float:
        """Return holding plus shortage cost booked by this record."""
        return math.fsum((self.holding.amount, self.shortage.amount))


class Ledger:
    """Append and expose three immutable row families for one session."""

    __slots__ = (
        "_forecast_keys",
        "_forecasts",
        "_order_keys",
        "_orders",
        "_session",
        "_settlement_keys",
        "_settlements",
    )

    def __init__(self, *, session: SessionIdentity) -> None:
        _require_session(session)
        self._session = session
        self._forecasts: list[ForecastRow] = []
        self._orders: list[OrderRow] = []
        self._settlements: list[SettlementRecord] = []
        self._forecast_keys: set[ForecastKey] = set()
        self._order_keys: set[OrderKey] = set()
        self._settlement_keys: set[SettlementKey] = set()

    @property
    def session(self) -> SessionIdentity:
        """Return the session that scopes order and settlement rows."""
        return self._session

    @property
    def forecasts(self) -> tuple[ForecastRow, ...]:
        """Return forecast rows in stable append order."""
        return tuple(self._forecasts)

    @property
    def orders(self) -> tuple[OrderRow, ...]:
        """Return order rows in stable append order."""
        return tuple(self._orders)

    @property
    def settlements(self) -> tuple[SettlementRecord, ...]:
        """Return settlement records in stable append order."""
        return tuple(self._settlements)

    def append_forecasts(
        self,
        frame: pd.DataFrame,
        *,
        calendar: Calendar,
        issuances: Mapping[ForecastKey, ForecastIssuance],
    ) -> None:
        """Validate and atomically append one pending forecast-frame chunk."""
        try:
            validated = validate_forecast_frame(frame, calendar=calendar)
        except ForecastFrameError as error:
            raise LedgerError(str(error)) from error
        if not validated[ACTUAL_VALUE].isna().all():
            raise LedgerError("forecast rows must be pending when appended")
        if not isinstance(issuances, Mapping):
            raise LedgerError("forecast issuances must be keyed by forecast row key")

        columns = tuple(validated.columns)
        staged_values: list[dict[str, object]] = []
        staged_keys: list[ForecastKey] = []
        for values in validated.itertuples(index=False, name=None):
            by_name = dict(zip(columns, values, strict=True))
            staged_values.append(by_name)
            staged_keys.append(_forecast_key(by_name))

        try:
            issuance_keys = set(issuances)
        except (TypeError, ValueError) as error:
            raise LedgerError("forecast issuance keys must be hashable row keys") from error
        if issuance_keys != set(staged_keys):
            raise LedgerError("forecast issuance keys must exactly match forecast frame keys")
        duplicate = next((key for key in staged_keys if key in self._forecast_keys), None)
        if duplicate is not None:
            raise LedgerError(f"duplicate forecast key: {duplicate!r}")

        staged_rows: list[ForecastRow] = []
        for key, values in zip(staged_keys, staged_values, strict=True):
            issuance = issuances[key]
            if not isinstance(issuance, ForecastIssuance):
                raise LedgerError("each forecast issuance must be a ForecastIssuance")
            payload_is_finite = _bound_payload_is_finite(values, issuance)
            if payload_is_finite is not issuance.bounds_finite:
                raise LedgerError("issued bounds finiteness does not match the forecast payload")
            staged_rows.append(ForecastRow._from_validated_values(values, issuance=issuance))

        self._forecasts.extend(staged_rows)
        self._forecast_keys.update(staged_keys)

    def append_orders(self, rows: Iterable[OrderRow]) -> None:
        """Validate one input chunk once, then append it atomically."""
        staged = _stage_rows(rows, row_type=OrderRow, family="order")
        staged_keys: set[OrderKey] = set()
        for row in staged:
            if row.session != self._session:
                raise LedgerError("order row session does not match the ledger session")
            if row.key in self._order_keys or row.key in staged_keys:
                raise LedgerError(f"duplicate order key: {row.key!r}")
            staged_keys.add(row.key)
        self._orders.extend(staged)
        self._order_keys.update(staged_keys)

    def append_settlements(self, rows: Iterable[SettlementRecord]) -> None:
        """Validate one input chunk once, then append it atomically."""
        staged = _stage_rows(rows, row_type=SettlementRecord, family="settlement")
        staged_keys: set[SettlementKey] = set()
        for row in staged:
            if row.session != self._session:
                raise LedgerError("settlement row session does not match the ledger session")
            if row.key in self._settlement_keys or row.key in staged_keys:
                raise LedgerError(f"duplicate settlement key: {row.key!r}")
            staged_keys.add(row.key)
        self._settlements.extend(staged)
        self._settlement_keys.update(staged_keys)


def _stage_rows[T](
    rows: Iterable[T],
    *,
    row_type: type[T],
    family: str,
) -> list[T]:
    if isinstance(rows, (str, bytes)):
        raise LedgerError(f"{family} rows must be an iterable of {row_type.__name__}")
    try:
        staged = list(rows)
    except TypeError as error:
        raise LedgerError(f"{family} rows must be iterable") from error
    if any(not isinstance(row, row_type) for row in staged):
        raise LedgerError(f"every {family} row must be a {row_type.__name__}")
    return staged


def _forecast_key(values: Mapping[str, object]) -> ForecastKey:
    return (
        cast(str, values[SERIES_KEY]),
        cast(pd.Timestamp, values[ORIGIN]),
        cast(int, values[HORIZON_STEP]),
        cast(str, values[MODEL_NAME]),
    )


def _bound_payload_is_finite(
    values: Mapping[str, object],
    issuance: ForecastIssuance,
) -> bool:
    lower, upper = interval_columns(issuance.descriptor.level)
    quantile = quantile_column(issuance.descriptor.level)
    claim = issuance.descriptor.type.claim

    if claim is GuaranteeClaim.ONE_SIDED_COVERAGE:
        column = lower if issuance.guaranteed_side is GuaranteedSide.LOWER else upper
        bound_columns = (column,)
    elif claim is GuaranteeClaim.TWO_SIDED_COVERAGE:
        bound_columns = (lower, upper)
    elif quantile in values:
        bound_columns = (quantile,)
    elif lower in values and upper in values:
        bound_columns = (lower, upper)
    else:
        bound_columns = ()
    return bool(bound_columns) and all(
        column in values and _is_finite_real(values[column]) for column in bound_columns
    )


def _snapshot_scalar(value: object) -> object:
    if value is pd.NA or value is pd.NaT or value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _is_finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LedgerError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise LedgerError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise LedgerError(f"{name} must be a finite real number")
    if normalized < 0.0:
        raise LedgerError(f"{name} must be non-negative")
    return 0.0 if normalized == 0.0 else normalized


def _require_session(session: object) -> None:
    if not isinstance(session, SessionIdentity):
        raise LedgerError("session must be a SessionIdentity")


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise LedgerError(f"{name} must be valid UTF-8") from error
    return value


def _require_timestamp(value: object, *, name: str) -> None:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise LedgerError(f"{name} must be a non-missing pandas Timestamp")
    if value.tz is not None:
        raise LedgerError(f"{name} must be timezone-naive")


__all__ = [
    "BookedCost",
    "ForecastIssuance",
    "ForecastRow",
    "GuaranteedSide",
    "Ledger",
    "LedgerError",
    "OrderRow",
    "SettlementRecord",
    "StockoutTransition",
]
