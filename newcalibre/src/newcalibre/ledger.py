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
    FRAME_KEY_COLUMNS,
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
    forecast_bound_groups,
    interval_columns,
    quantile_column,
    validate_forecast_frame,
)

type ForecastKey = tuple[str, pd.Timestamp, int, str]
type BoundKey = tuple[str, ...]
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
            if self.descriptor.type.claim is not GuaranteeClaim.NONE and not self.calibration_ready:
                raise LedgerError("finite engine-calibrated bounds require calibration readiness")
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
    issuances: Mapping[BoundKey, ForecastIssuance]
    _values: tuple[tuple[str, object], ...]

    def __init__(self) -> None:
        raise TypeError("ForecastRow instances are created by Ledger.append_forecasts()")

    @classmethod
    def _from_validated_values(
        cls,
        values: Mapping[str, object],
        *,
        issuances: Mapping[BoundKey, ForecastIssuance],
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
        object.__setattr__(instance, "issuances", MappingProxyType(dict(issuances)))
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
        _require_identifier(self.series_key, name="series key")
        _require_timestamp(self.origin, name="order origin")
        _require_identifier(self.model_name, name="model name")
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
        _require_identifier(self.series_key, name="series key")
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
        if self.holding.basis != self.transition.closing_on_hand:
            raise LedgerError("settlement holding cost basis must equal transition closing on hand")
        if self.shortage.basis != self.transition.unmet_demand:
            raise LedgerError("settlement shortage cost basis must equal transition unmet demand")
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
        "_calendar",
        "_forecasts",
        "_orders",
        "_session",
        "_settlements",
    )

    def __init__(self, *, session: SessionIdentity, calendar: Calendar) -> None:
        _require_session(session)
        _require_bound_calendar(calendar)
        self._session = session
        self._calendar = calendar
        self._forecasts: dict[ForecastKey, ForecastRow] = {}
        self._orders: dict[OrderKey, OrderRow] = {}
        self._settlements: dict[SettlementKey, SettlementRecord] = {}

    @property
    def session(self) -> SessionIdentity:
        """Return the session that scopes order and settlement rows."""
        return self._session

    @property
    def calendar(self) -> Calendar:
        """Return the one bound calendar used for every forecast append."""
        return self._calendar

    @property
    def forecasts(self) -> tuple[ForecastRow, ...]:
        """Return forecast rows in stable append order."""
        return tuple(self._forecasts.values())

    @property
    def orders(self) -> tuple[OrderRow, ...]:
        """Return order rows in stable append order."""
        return tuple(self._orders.values())

    @property
    def settlements(self) -> tuple[SettlementRecord, ...]:
        """Return settlement records in stable append order."""
        return tuple(self._settlements.values())

    def append_forecasts(
        self,
        frame: pd.DataFrame,
        *,
        issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]],
    ) -> None:
        """Validate and atomically append one pending forecast-frame chunk."""
        try:
            validated = validate_forecast_frame(frame, calendar=self._calendar)
        except ForecastFrameError as error:
            raise LedgerError(str(error)) from error
        if not validated[ACTUAL_VALUE].isna().all():
            raise LedgerError("forecast rows must be pending when appended")
        if not isinstance(issuances, Mapping):
            raise LedgerError("forecast issuances must be keyed by forecast row key")

        columns = tuple(validated.columns)
        staged: dict[ForecastKey, dict[str, object]] = {}
        for values in validated.itertuples(index=False, name=None):
            by_name = dict(zip(columns, values, strict=True))
            staged[_forecast_key(by_name)] = by_name

        try:
            issuance_keys = set(issuances)
        except (TypeError, ValueError) as error:
            raise LedgerError("forecast issuance keys must be hashable row keys") from error
        staged_keys = set(staged)
        if issuance_keys != staged_keys:
            raise LedgerError("forecast issuance keys must exactly match forecast frame keys")
        duplicate = next((key for key in staged if key in self._forecasts), None)
        if duplicate is not None:
            raise LedgerError(f"duplicate forecast key: {duplicate!r}")

        bound_groups = forecast_bound_groups(columns)
        staged_rows: dict[ForecastKey, ForecastRow] = {}
        for key, values in staged.items():
            if not _is_finite_real(values[POINT_FORECAST]):
                raise LedgerError("point forecast must be a finite real number")
            row_issuances = _validate_row_issuances(
                values,
                bound_groups=bound_groups,
                issuances=issuances[key],
            )
            staged_rows[key] = ForecastRow._from_validated_values(
                values,
                issuances=row_issuances,
            )

        self._forecasts.update(staged_rows)

    def append_orders(self, rows: Iterable[OrderRow]) -> None:
        """Validate one input chunk once, then append it atomically."""
        staged = _stage_rows(rows, row_type=OrderRow, family="order")
        staged_rows: dict[OrderKey, OrderRow] = {}
        for row in staged:
            if row.session != self._session:
                raise LedgerError("order row session does not match the ledger session")
            key = row.key
            if key in self._orders or key in staged_rows:
                raise LedgerError(f"duplicate order key: {key!r}")
            staged_rows[key] = row
        self._orders.update(staged_rows)

    def append_settlements(self, rows: Iterable[SettlementRecord]) -> None:
        """Validate one input chunk once, then append it atomically."""
        staged = _stage_rows(rows, row_type=SettlementRecord, family="settlement")
        staged_rows: dict[SettlementKey, SettlementRecord] = {}
        for row in staged:
            if row.session != self._session:
                raise LedgerError("settlement row session does not match the ledger session")
            key = row.key
            if key in self._settlements or key in staged_rows:
                raise LedgerError(f"duplicate settlement key: {key!r}")
            staged_rows[key] = row
        self._settlements.update(staged_rows)


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
    return cast(ForecastKey, tuple(values[column] for column in FRAME_KEY_COLUMNS))


def _validate_row_issuances(
    values: Mapping[str, object],
    *,
    bound_groups: tuple[BoundKey, ...],
    issuances: Mapping[BoundKey, ForecastIssuance],
) -> dict[BoundKey, ForecastIssuance]:
    if not isinstance(issuances, Mapping):
        raise LedgerError("each forecast row's issuances must be keyed by bound key")
    try:
        snapshot = dict(issuances)
    except (TypeError, ValueError) as error:
        raise LedgerError("forecast bound issuance keys must be hashable") from error

    group_by_column = {column: group for group in bound_groups for column in group}
    expected_columns = set(group_by_column)
    accounted_columns: set[str] = set()
    for bound_key, issuance in snapshot.items():
        _validate_bound_key(bound_key, group_by_column=group_by_column)
        overlap = accounted_columns.intersection(bound_key)
        if overlap:
            raise LedgerError(
                f"forecast bound columns may be issued only once: {sorted(overlap)!r}"
            )
        accounted_columns.update(bound_key)
        if not isinstance(issuance, ForecastIssuance):
            raise LedgerError("each bound issuance must be a ForecastIssuance")
        _validate_bound_descriptor(bound_key, issuance)
        payload_is_finite = all(_is_finite_real(values[column]) for column in bound_key)
        if payload_is_finite is not issuance.bounds_finite:
            raise LedgerError("issued bounds finiteness does not match the forecast payload")

    if accounted_columns != expected_columns:
        raise LedgerError(
            "forecast bound issuance keys must exactly account for every bound column"
        )
    return snapshot


def _validate_bound_key(
    bound_key: BoundKey,
    *,
    group_by_column: Mapping[str, BoundKey],
) -> None:
    if (
        not isinstance(bound_key, tuple)
        or not 1 <= len(bound_key) <= 2
        or any(not isinstance(column, str) for column in bound_key)
    ):
        raise LedgerError("forecast bound key must be a one- or two-column tuple")
    if len(set(bound_key)) != len(bound_key):
        raise LedgerError("forecast bound key cannot repeat a column")
    try:
        groups = {group_by_column[column] for column in bound_key}
    except KeyError as error:
        raise LedgerError(
            f"forecast bound key names a missing or non-bound column: {error.args[0]!r}"
        ) from error
    if len(groups) != 1:
        raise LedgerError("forecast bound key cannot combine distinct forecast levels")
    group = groups.pop()
    if len(group) == 1 and bound_key != group:
        raise LedgerError("a quantile bound key must name its exact canonical column")
    if len(bound_key) == 2 and bound_key != group:
        raise LedgerError("an interval bound key must be the canonical lower/upper pair")


def _validate_bound_descriptor(
    bound_key: BoundKey,
    issuance: ForecastIssuance,
) -> None:
    claim = issuance.descriptor.type.claim

    if claim is GuaranteeClaim.RISK_CONTROL:
        # A risk-control level is the claimed realized-loss limit, not a
        # FRA-2 quantile or nominal-coverage suffix. The explicit bound key
        # identifies the payload without conflating those two quantities.
        return

    quantile = (quantile_column(issuance.descriptor.level),)
    interval = interval_columns(issuance.descriptor.level)

    if claim is GuaranteeClaim.ONE_SIDED_COVERAGE:
        column = interval[0] if issuance.guaranteed_side is GuaranteedSide.LOWER else interval[1]
        if bound_key != (column,):
            raise LedgerError(
                "one-sided descriptor must match its canonical guaranteed-side column"
            )
        return
    if claim is GuaranteeClaim.TWO_SIDED_COVERAGE:
        if bound_key != interval:
            raise LedgerError("two-sided descriptor must match its canonical lower/upper pair")
        return

    if bound_key not in (quantile, interval, (interval[0],), (interval[1],)):
        raise LedgerError("bound key level does not match its guarantee descriptor")


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


def _require_bound_calendar(calendar: object) -> None:
    if not isinstance(calendar, Calendar) or calendar.phase is None:
        raise LedgerError("ledger calendar must be a bound Calendar")


def _require_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise LedgerError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise LedgerError(f"{name} must be valid UTF-8") from error


def _require_text(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise LedgerError(f"{name} must be valid UTF-8") from error


def _require_timestamp(value: object, *, name: str) -> None:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise LedgerError(f"{name} must be a non-missing pandas Timestamp")
    if value.tz is not None:
        raise LedgerError(f"{name} must be timezone-naive")


__all__ = [
    "BoundKey",
    "BookedCost",
    "ForecastKey",
    "ForecastIssuance",
    "ForecastRow",
    "GuaranteedSide",
    "Ledger",
    "LedgerError",
    "OrderKey",
    "OrderRow",
    "SettlementKey",
    "SettlementRecord",
    "StockoutTransition",
]
