"""Reduce immutable decision and settlement facts into realized-cost objectives."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Real
from types import MappingProxyType

import pandas as pd

from newcalibre.domain import (
    ActualsSemantics,
    CostStructure,
    EmissionScope,
    SessionIdentity,
)
from newcalibre.ledger import SettlementRecord

DecisionCostKey = tuple[str, pd.Timestamp]


class ObjectiveError(ValueError):
    """Report objective input that cannot produce honest realized-cost facts."""


@dataclass(frozen=True, slots=True)
class CostValue:
    """Carry one non-negative cost together with its actuals meaning."""

    value: float
    actuals_semantics: ActualsSemantics

    def __post_init__(self) -> None:
        if not isinstance(self.actuals_semantics, ActualsSemantics):
            raise TypeError("cost value actuals semantics must be ActualsSemantics")
        value = _nonnegative_cost(self.value, name="cost value", allow_infinity=True)
        object.__setattr__(self, "value", value)

    @property
    def is_infeasible(self) -> bool:
        """Return whether this value is the explicit positive-infinity score."""
        return math.isinf(self.value)


@dataclass(frozen=True, slots=True)
class CostComponents:
    """Expose one decision's independently recomputable booked components."""

    holding: CostValue
    shortage: CostValue
    total: CostValue

    def __post_init__(self) -> None:
        values = (self.holding, self.shortage, self.total)
        if any(not isinstance(value, CostValue) for value in values):
            raise TypeError("cost components must contain CostValue values")
        semantics = {value.actuals_semantics for value in values}
        if len(semantics) != 1:
            raise ObjectiveError("cost component semantics must match")
        if any(value.is_infeasible for value in values):
            raise ObjectiveError("booked cost components must be finite")
        expected = _finite_sum(
            (self.holding.value, self.shortage.value),
            name="booked component total",
        )
        if self.total.value != expected:
            raise ObjectiveError("cost total must equal holding plus shortage")


@dataclass(frozen=True, slots=True)
class OriginPartial:
    """Label the cumulative objective after one canonically ordered origin."""

    origin: pd.Timestamp
    cost: CostValue

    def __post_init__(self) -> None:
        _require_origin(self.origin)
        if not isinstance(self.cost, CostValue):
            raise TypeError("origin partial cost must be a CostValue")
        if self.cost.is_infeasible:
            raise ObjectiveError("feasible partial costs must be finite")


@dataclass(frozen=True, slots=True)
class DiagnosticWindow:
    """Declare one diagnostic decision window without settlement state."""

    series_key: str
    origin: pd.Timestamp
    mode: EmissionScope
    quantities: tuple[float, ...]
    demands: tuple[float, ...]
    costs: CostStructure
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND

    def __post_init__(self) -> None:
        _require_identifier(self.series_key, name="diagnostic series key")
        _require_origin(self.origin)
        if not isinstance(self.mode, EmissionScope):
            raise TypeError("diagnostic mode must be an EmissionScope")
        quantities = _finite_vector(self.quantities, name="diagnostic quantity")
        demands = _finite_vector(self.demands, name="diagnostic demand")
        if not quantities or not demands:
            raise ObjectiveError("diagnostic windows must not be empty")
        if self.mode is EmissionScope.PER_STEP:
            if len(quantities) != len(demands):
                raise ObjectiveError("per-step diagnostic quantities must match demand rows")
        elif self.mode is EmissionScope.WINDOW_SUM:
            if len(quantities) != 1:
                raise ObjectiveError("window-sum diagnostics require exactly one decision")
        else:  # pragma: no cover - EmissionScope owns the closed mode set.
            raise ObjectiveError("unsupported diagnostic emission scope")
        if not isinstance(self.costs, CostStructure):
            raise TypeError("diagnostic costs must be a CostStructure")
        if not isinstance(self.actuals_semantics, ActualsSemantics):
            raise TypeError("diagnostic actuals semantics must be ActualsSemantics")
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "demands", demands)

    @property
    def key(self) -> DecisionCostKey:
        """Return the exact per-decision diagnostic key."""
        return (self.series_key, self.origin)


@dataclass(frozen=True, slots=True)
class DiagnosticObjective:
    """Return diagnostic OBJ-1 costs without claiming settle-path equivalence."""

    mode: EmissionScope
    actuals_semantics: ActualsSemantics
    by_decision: Mapping[DecisionCostKey, CostValue] = field(repr=False)
    total: CostValue

    def __post_init__(self) -> None:
        if not isinstance(self.mode, EmissionScope):
            raise TypeError("diagnostic objective mode must be an EmissionScope")
        _require_semantics(self.actuals_semantics)
        by_decision = dict(self.by_decision)
        _validate_cost_mapping(
            by_decision,
            actuals_semantics=self.actuals_semantics,
            name="diagnostic decision costs",
        )
        if not isinstance(self.total, CostValue):
            raise TypeError("diagnostic objective total must be a CostValue")
        if self.total.actuals_semantics is not self.actuals_semantics:
            raise ObjectiveError("diagnostic total semantics must match the objective")
        expected = _finite_sum(
            (value.value for value in by_decision.values()),
            name="diagnostic objective total",
        )
        if self.total.value != expected:
            raise ObjectiveError("diagnostic total must equal its decision costs")
        object.__setattr__(self, "by_decision", MappingProxyType(by_decision))


@dataclass(frozen=True, slots=True)
class SettlementObjective:
    """Return the exported OBJ-2 settle-path objective and bookkeeping totals."""

    session: SessionIdentity | None
    actuals_semantics: ActualsSemantics
    by_decision: Mapping[DecisionCostKey, CostComponents] = field(repr=False)
    by_origin: Mapping[pd.Timestamp, CostValue] = field(repr=False)
    by_series: Mapping[str, CostValue] = field(repr=False)
    partials: tuple[OriginPartial, ...]
    holding: CostValue
    shortage: CostValue
    total: CostValue
    feasible: bool
    infeasible_reason: str | None = None

    def __post_init__(self) -> None:
        _require_semantics(self.actuals_semantics)
        by_decision = dict(self.by_decision)
        by_origin = dict(self.by_origin)
        by_series = dict(self.by_series)
        partials = tuple(self.partials)
        if any(not isinstance(value, CostComponents) for value in by_decision.values()):
            raise TypeError("settlement decision costs must contain CostComponents")
        for key in by_decision:
            _require_decision_key(key, name="settlement decision cost key")
        for origin in by_origin:
            _require_origin(origin, name="settlement origin cost key")
        _validate_cost_values(
            by_origin.values(),
            actuals_semantics=self.actuals_semantics,
            name="settlement origin costs",
        )
        for series_key in by_series:
            _require_identifier(series_key, name="settlement series cost key")
        _validate_cost_values(
            by_series.values(),
            actuals_semantics=self.actuals_semantics,
            name="settlement series costs",
        )
        if any(not isinstance(partial, OriginPartial) for partial in partials):
            raise TypeError("settlement partials must contain OriginPartial values")
        for value in (*by_decision.values(), *partials):
            semantics = (
                value.total.actuals_semantics
                if isinstance(value, CostComponents)
                else value.cost.actuals_semantics
            )
            if semantics is not self.actuals_semantics:
                raise ObjectiveError("settlement derived costs must preserve actuals semantics")
        for value in (self.holding, self.shortage, self.total):
            if not isinstance(value, CostValue):
                raise TypeError("settlement totals must be CostValue values")
            if value.actuals_semantics is not self.actuals_semantics:
                raise ObjectiveError("settlement total semantics must match the objective")
        if not isinstance(self.feasible, bool):
            raise TypeError("settlement objective feasible must be a boolean")
        if self.feasible:
            if not isinstance(self.session, SessionIdentity):
                raise TypeError("a feasible settlement objective requires one session")
            if not by_decision:
                raise ObjectiveError("a feasible settlement objective requires records")
            if self.infeasible_reason is not None:
                raise ObjectiveError("a feasible settlement objective has no failure reason")
            if any(value.is_infeasible for value in (self.holding, self.shortage, self.total)):
                raise ObjectiveError("a feasible settlement objective must be finite")
            decision_origins = {origin for _series_key, origin in by_decision}
            decision_series = {series_key for series_key, _origin in by_decision}
            if set(by_origin) != decision_origins:
                raise ObjectiveError("settlement origin totals must exactly cover decision costs")
            if set(by_series) != decision_series:
                raise ObjectiveError("settlement series totals must exactly cover decision costs")
            for origin, origin_cost in by_origin.items():
                expected_origin = _finite_sum(
                    (
                        component
                        for (series_key, decision_origin), costs in by_decision.items()
                        if decision_origin == origin
                        for component in (costs.holding.value, costs.shortage.value)
                    ),
                    name="settlement origin cost",
                )
                if origin_cost.value != expected_origin:
                    raise ObjectiveError("settlement origin total must equal its decision costs")
            for series_key, series_cost in by_series.items():
                expected_series = _finite_sum(
                    (
                        component
                        for (decision_series_key, _origin), costs in by_decision.items()
                        if decision_series_key == series_key
                        for component in (costs.holding.value, costs.shortage.value)
                    ),
                    name="settlement series cost",
                )
                if series_cost.value != expected_series:
                    raise ObjectiveError("settlement series total must equal its decision costs")
            expected_holding = _finite_sum(
                (component.holding.value for component in by_decision.values()),
                name="settlement holding total",
            )
            expected_shortage = _finite_sum(
                (component.shortage.value for component in by_decision.values()),
                name="settlement shortage total",
            )
            if self.holding.value != expected_holding or self.shortage.value != expected_shortage:
                raise ObjectiveError("settlement component totals must equal decision components")
            expected = _finite_sum(
                (component.total.value for component in by_decision.values()),
                name="settlement objective total",
            )
            if self.total.value != expected:
                raise ObjectiveError("settlement total must equal its decision costs")
            if tuple(partial.origin for partial in partials) != tuple(sorted(by_origin)):
                raise ObjectiveError("settlement partials must exactly follow ordered origins")
            cumulative_components: list[float] = []
            previous = 0.0
            for partial in partials:
                cumulative_components.extend(
                    component
                    for (_series_key, origin), costs in by_decision.items()
                    if origin == partial.origin
                    for component in (costs.holding.value, costs.shortage.value)
                )
                expected_partial = _finite_sum(
                    cumulative_components,
                    name="settlement partial cost",
                )
                if partial.cost.value != expected_partial or partial.cost.value < previous:
                    raise ObjectiveError("settlement partial costs must be exact and monotone")
                previous = partial.cost.value
            if partials[-1].cost != self.total:
                raise ObjectiveError("the final partial must equal the settlement total")
        else:
            if self.session is not None or by_decision or by_origin or by_series or partials:
                raise ObjectiveError("an infeasible empty objective cannot carry settled facts")
            if not self.total.is_infeasible:
                raise ObjectiveError("an infeasible objective must score positive infinity")
            if self.holding.value != 0.0 or self.shortage.value != 0.0:
                raise ObjectiveError("an infeasible empty objective has zero booked components")
            _require_reason(self.infeasible_reason)
        object.__setattr__(self, "by_decision", MappingProxyType(by_decision))
        object.__setattr__(self, "by_origin", MappingProxyType(by_origin))
        object.__setattr__(self, "by_series", MappingProxyType(by_series))
        object.__setattr__(self, "partials", partials)


@dataclass(frozen=True, slots=True)
class RegretObjective:
    """Return non-negative candidate regret aligned by exact decision key."""

    actuals_semantics: ActualsSemantics
    by_decision: Mapping[DecisionCostKey, CostValue] = field(repr=False)
    total: CostValue

    def __post_init__(self) -> None:
        _require_semantics(self.actuals_semantics)
        by_decision = dict(self.by_decision)
        _validate_cost_mapping(
            by_decision,
            actuals_semantics=self.actuals_semantics,
            name="regret decision costs",
        )
        if any(value.is_infeasible for value in by_decision.values()):
            raise ObjectiveError("regret decision costs must be finite")
        if not isinstance(self.total, CostValue):
            raise TypeError("regret total must be a CostValue")
        if self.total.actuals_semantics is not self.actuals_semantics:
            raise ObjectiveError("regret total semantics must match")
        expected = _finite_sum(
            (value.value for value in by_decision.values()),
            name="regret total",
        )
        if self.total.value != expected:
            raise ObjectiveError("regret total must equal its aligned decision costs")
        object.__setattr__(self, "by_decision", MappingProxyType(by_decision))


def diagnostic_cost(
    windows: Iterable[DiagnosticWindow],
    *,
    mode: EmissionScope,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> DiagnosticObjective:
    """Evaluate OBJ-1 over explicit diagnostic windows in exactly one mode."""
    if not isinstance(mode, EmissionScope):
        raise TypeError("diagnostic objective mode must be an EmissionScope")
    _require_semantics(actuals_semantics)
    staged = _snapshot_iterable(windows, name="diagnostic windows")
    if not staged:
        raise ObjectiveError("diagnostic objective requires at least one window")
    if any(not isinstance(window, DiagnosticWindow) for window in staged):
        raise TypeError("diagnostic objective requires DiagnosticWindow values")
    typed = tuple(staged)
    if any(window.mode is not mode for window in typed):
        raise ObjectiveError("diagnostic objective mode mismatches or mixes window modes")
    if any(window.actuals_semantics is not actuals_semantics for window in typed):
        raise ObjectiveError("diagnostic objective actuals semantics must match every window")
    if mode is EmissionScope.WINDOW_SUM and len(typed) != 1:
        raise ObjectiveError("window-sum diagnostic evaluation requires exactly one window")
    keys = [window.key for window in typed]
    if len(set(keys)) != len(keys):
        raise ObjectiveError("diagnostic objective decision keys must be unique")

    by_decision: dict[DecisionCostKey, CostValue] = {}
    for window in sorted(typed, key=lambda value: _decision_sort_key(value.key)):
        if mode is EmissionScope.PER_STEP:
            value = _finite_sum(
                (
                    _diagnostic_row_cost(quantity, demand, window.costs)
                    for quantity, demand in zip(
                        window.quantities,
                        window.demands,
                        strict=True,
                    )
                ),
                name="per-step diagnostic cost",
            )
        else:
            demand = _finite_sum(window.demands, name="window-sum diagnostic demand")
            value = _diagnostic_row_cost(window.quantities[0], demand, window.costs)
        by_decision[window.key] = CostValue(value, actuals_semantics)

    return DiagnosticObjective(
        mode=mode,
        actuals_semantics=actuals_semantics,
        by_decision=by_decision,
        total=CostValue(
            _finite_sum(
                (value.value for value in by_decision.values()),
                name="diagnostic objective total",
            ),
            actuals_semantics,
        ),
    )


def settle_path_cost(
    records: Iterable[SettlementRecord],
    *,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> SettlementObjective:
    """Reduce U5 settlement records into the exported OBJ-2 default objective."""
    _require_semantics(actuals_semantics)
    staged = _snapshot_iterable(records, name="settlement records")
    if not staged:
        zero = CostValue(0.0, actuals_semantics)
        return SettlementObjective(
            session=None,
            actuals_semantics=actuals_semantics,
            by_decision={},
            by_origin={},
            by_series={},
            partials=(),
            holding=zero,
            shortage=zero,
            total=CostValue(math.inf, actuals_semantics),
            feasible=False,
            infeasible_reason="candidate emitted no settlement records",
        )
    if any(not isinstance(record, SettlementRecord) for record in staged):
        raise TypeError("settle-path objective requires SettlementRecord values")
    typed = tuple(staged)
    session = typed[0].session
    if any(record.session != session for record in typed):
        raise ObjectiveError("settle-path objective records must share one session")
    if any(record.actuals_semantics is not actuals_semantics for record in typed):
        raise ObjectiveError(
            "settle-path actuals semantics must match the explicit objective binding"
        )
    keys = [(record.series_key, record.period) for record in typed]
    if len(set(keys)) != len(keys):
        raise ObjectiveError("settle-path objective records must have unique decision keys")

    ordered = tuple(
        sorted(
            typed,
            key=lambda record: _decision_sort_key((record.series_key, record.period)),
        )
    )
    by_decision: dict[DecisionCostKey, CostComponents] = {}
    origin_components: dict[pd.Timestamp, list[float]] = {}
    series_components: dict[str, list[float]] = {}
    holding_values: list[float] = []
    shortage_values: list[float] = []
    for record in ordered:
        holding = CostValue(record.holding.amount, actuals_semantics)
        shortage = CostValue(record.shortage.amount, actuals_semantics)
        total = CostValue(
            _finite_sum(
                (holding.value, shortage.value),
                name="settlement decision cost",
            ),
            actuals_semantics,
        )
        key = (record.series_key, record.period)
        by_decision[key] = CostComponents(holding=holding, shortage=shortage, total=total)
        origin_components.setdefault(record.period, []).extend((holding.value, shortage.value))
        series_components.setdefault(record.series_key, []).extend((holding.value, shortage.value))
        holding_values.append(holding.value)
        shortage_values.append(shortage.value)

    by_origin = {
        origin: CostValue(
            _finite_sum(origin_components[origin], name="settlement origin cost"),
            actuals_semantics,
        )
        for origin in sorted(origin_components)
    }
    by_series = {
        series_key: CostValue(
            _finite_sum(series_components[series_key], name="settlement series cost"),
            actuals_semantics,
        )
        for series_key in sorted(series_components, key=str.encode)
    }
    partials: list[OriginPartial] = []
    cumulative_components: list[float] = []
    for origin in sorted(origin_components):
        cumulative_components.extend(origin_components[origin])
        partials.append(
            OriginPartial(
                origin=origin,
                cost=CostValue(
                    _finite_sum(cumulative_components, name="settlement partial cost"),
                    actuals_semantics,
                ),
            )
        )
    holding_total = _finite_sum(holding_values, name="settlement holding total")
    shortage_total = _finite_sum(shortage_values, name="settlement shortage total")
    total = _finite_sum(
        (component for values in origin_components.values() for component in values),
        name="settlement objective total",
    )
    return SettlementObjective(
        session=session,
        actuals_semantics=actuals_semantics,
        by_decision=by_decision,
        by_origin=by_origin,
        by_series=by_series,
        partials=tuple(partials),
        holding=CostValue(holding_total, actuals_semantics),
        shortage=CostValue(shortage_total, actuals_semantics),
        total=CostValue(total, actuals_semantics),
        feasible=True,
    )


def key_aligned_regret(
    candidate: Mapping[DecisionCostKey, CostValue],
    oracle: Mapping[DecisionCostKey, CostValue],
    *,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> RegretObjective:
    """Compute OBJ-7 over the exact key intersection, never mapping order."""
    _require_semantics(actuals_semantics)
    candidate_values = _snapshot_cost_stream(
        candidate,
        actuals_semantics=actuals_semantics,
        name="candidate cost stream",
    )
    oracle_values = _snapshot_cost_stream(
        oracle,
        actuals_semantics=actuals_semantics,
        name="oracle cost stream",
    )
    aligned_keys = sorted(
        set(candidate_values) & set(oracle_values),
        key=_decision_sort_key,
    )
    by_decision = {
        key: CostValue(
            max(candidate_values[key].value - oracle_values[key].value, 0.0),
            actuals_semantics,
        )
        for key in aligned_keys
    }
    return RegretObjective(
        actuals_semantics=actuals_semantics,
        by_decision=by_decision,
        total=CostValue(
            _finite_sum(
                (value.value for value in by_decision.values()),
                name="regret total",
            ),
            actuals_semantics,
        ),
    )


def _snapshot_cost_stream(
    value: Mapping[DecisionCostKey, CostValue],
    *,
    actuals_semantics: ActualsSemantics,
    name: str,
) -> dict[DecisionCostKey, CostValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    snapshot = dict(value)
    _validate_cost_mapping(snapshot, actuals_semantics=actuals_semantics, name=name)
    if any(cost.is_infeasible for cost in snapshot.values()):
        raise ObjectiveError(f"{name} values must be finite")
    return snapshot


def _validate_cost_mapping(
    value: Mapping[DecisionCostKey, CostValue],
    *,
    actuals_semantics: ActualsSemantics,
    name: str,
) -> None:
    for key in value:
        _require_decision_key(key, name=f"{name} key")
    _validate_cost_values(
        value.values(),
        actuals_semantics=actuals_semantics,
        name=name,
    )


def _validate_cost_values(
    values: Iterable[CostValue],
    *,
    actuals_semantics: ActualsSemantics,
    name: str,
) -> None:
    for cost in values:
        if not isinstance(cost, CostValue):
            raise TypeError(f"{name} must contain CostValue values")
        if cost.actuals_semantics is not actuals_semantics:
            raise ObjectiveError(f"{name} actuals semantics must match")


def _diagnostic_row_cost(quantity: float, demand: float, costs: CostStructure) -> float:
    overage_basis = max(quantity - demand, 0.0)
    underage_basis = max(demand - quantity, 0.0)
    try:
        terms = (costs.overage * overage_basis, costs.underage * underage_basis)
    except OverflowError as error:
        raise ObjectiveError("diagnostic cost exceeds finite float range") from error
    return _finite_sum(terms, name="diagnostic decision cost")


def _snapshot_iterable[T](value: Iterable[T], *, name: str) -> tuple[T, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of values")
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of values") from error


def _finite_vector(value: Iterable[object], *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} values must be iterable")
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} values must be iterable") from error
    return tuple(_finite_nonnegative(item, name=name) for item in values)


def _finite_nonnegative(value: object, *, name: str) -> float:
    normalized = _nonnegative_cost(value, name=name, allow_infinity=False)
    return normalized


def _nonnegative_cost(value: object, *, name: str, allow_infinity: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ObjectiveError(f"{name} must be non-negative and finite") from error
    if math.isnan(normalized) or normalized < 0.0 or normalized == -math.inf:
        raise ObjectiveError(f"{name} must be non-negative and finite")
    if math.isinf(normalized) and not allow_infinity:
        raise ObjectiveError(f"{name} must be non-negative and finite")
    return 0.0 if normalized == 0.0 else normalized


def _finite_sum(values: Iterable[float], *, name: str) -> float:
    try:
        total = math.fsum(values)
    except OverflowError as error:
        raise ObjectiveError(f"{name} exceeds finite float range") from error
    if not math.isfinite(total) or total < 0.0:
        raise ObjectiveError(f"{name} exceeds finite float range")
    return 0.0 if total == 0.0 else total


def _require_decision_key(value: object, *, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ObjectiveError(f"{name} must be a (series key, origin) tuple")
    series_key, origin = value
    _require_identifier(series_key, name=f"{name} series key")
    _require_origin(origin, name=f"{name} origin")


def _decision_sort_key(value: DecisionCostKey) -> tuple[pd.Timestamp, bytes]:
    return (value[1], value[0].encode())


def _require_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ObjectiveError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ObjectiveError(f"{name} must be valid UTF-8") from error


def _require_origin(value: object, *, name: str = "diagnostic origin") -> None:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise ObjectiveError(f"{name} must be a non-missing pandas Timestamp")
    if value.tz is not None:
        raise ObjectiveError(f"{name} must be timezone-naive")


def _require_semantics(value: object) -> None:
    if not isinstance(value, ActualsSemantics):
        raise TypeError("objective actuals semantics must be ActualsSemantics")


def _require_reason(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ObjectiveError("an infeasible objective requires a non-empty reason")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise ObjectiveError("infeasible objective reason must be valid UTF-8") from error


DEFAULT_OBJECTIVE = settle_path_cost


__all__ = [
    "DEFAULT_OBJECTIVE",
    "CostComponents",
    "CostValue",
    "DecisionCostKey",
    "DiagnosticObjective",
    "DiagnosticWindow",
    "ObjectiveError",
    "OriginPartial",
    "RegretObjective",
    "SettlementObjective",
    "diagnostic_cost",
    "key_aligned_regret",
    "settle_path_cost",
]
