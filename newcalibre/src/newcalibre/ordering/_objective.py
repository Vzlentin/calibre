"""Reduce immutable decision and settlement facts into realized-cost objectives."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
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


class CandidateInfeasible(Exception):
    """Signal a statistically infeasible or degenerate candidate evaluation."""

    def __init__(self, reason: str) -> None:
        _require_reason(reason)
        super().__init__(reason)


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
    total: CostValue = field(init=False)

    def __post_init__(self) -> None:
        values = (self.holding, self.shortage)
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
        object.__setattr__(self, "total", CostValue(expected, self.holding.actuals_semantics))


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
class _SettlementAggregates:
    """Hold one internally derived, immutable settlement reduction."""

    by_origin: Mapping[pd.Timestamp, CostValue]
    by_series: Mapping[str, CostValue]
    partials: tuple[OriginPartial, ...]
    holding: CostValue
    shortage: CostValue
    total: CostValue


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
    total: CostValue = field(init=False)

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
        object.__setattr__(self, "by_decision", MappingProxyType(by_decision))
        object.__setattr__(
            self,
            "total",
            CostValue(
                _finite_sum(
                    (value.value for value in by_decision.values()),
                    name="diagnostic objective total",
                ),
                self.actuals_semantics,
            ),
        )


@dataclass(frozen=True, slots=True)
class SettlementObjective:
    """Return the exported OBJ-2 settle-path objective and bookkeeping totals."""

    session: SessionIdentity | None
    actuals_semantics: ActualsSemantics
    components_by_decision: Mapping[DecisionCostKey, CostComponents] = field(repr=False)
    infeasible_reason: str | None = None
    by_decision: Mapping[DecisionCostKey, CostValue] = field(init=False, repr=False)
    by_origin: Mapping[pd.Timestamp, CostValue] = field(init=False, repr=False)
    by_series: Mapping[str, CostValue] = field(init=False, repr=False)
    partials: tuple[OriginPartial, ...] = field(init=False)
    holding: CostValue = field(init=False)
    shortage: CostValue = field(init=False)
    total: CostValue = field(init=False)
    feasible: bool = field(init=False)

    def __post_init__(self) -> None:
        _require_semantics(self.actuals_semantics)
        components_by_decision = dict(self.components_by_decision)
        for key in components_by_decision:
            _require_decision_key(key, name="settlement decision cost key")
        components_by_decision = dict(
            sorted(
                components_by_decision.items(),
                key=lambda item: _decision_sort_key(item[0]),
            )
        )
        if any(not isinstance(value, CostComponents) for value in components_by_decision.values()):
            raise TypeError("settlement decision costs must contain CostComponents")
        for components in components_by_decision.values():
            if components.total.actuals_semantics is not self.actuals_semantics:
                raise ObjectiveError("settlement derived costs must preserve actuals semantics")
        by_decision = {key: components.total for key, components in components_by_decision.items()}
        if components_by_decision:
            if not isinstance(self.session, SessionIdentity):
                raise TypeError("a feasible settlement objective requires one session")
            if self.infeasible_reason is not None:
                raise ObjectiveError("a feasible settlement objective has no failure reason")
            aggregates = _aggregate_settlement_costs(
                components_by_decision,
                actuals_semantics=self.actuals_semantics,
            )
            feasible = True
        else:
            if self.session is not None:
                raise ObjectiveError("an infeasible empty objective cannot carry settled facts")
            _require_reason(self.infeasible_reason)
            zero = CostValue(0.0, self.actuals_semantics)
            aggregates = _SettlementAggregates(
                by_origin=MappingProxyType({}),
                by_series=MappingProxyType({}),
                partials=(),
                holding=zero,
                shortage=zero,
                total=CostValue(math.inf, self.actuals_semantics),
            )
            feasible = False
        object.__setattr__(
            self,
            "components_by_decision",
            MappingProxyType(components_by_decision),
        )
        object.__setattr__(self, "by_decision", MappingProxyType(by_decision))
        object.__setattr__(self, "by_origin", aggregates.by_origin)
        object.__setattr__(self, "by_series", aggregates.by_series)
        object.__setattr__(self, "partials", aggregates.partials)
        object.__setattr__(self, "holding", aggregates.holding)
        object.__setattr__(self, "shortage", aggregates.shortage)
        object.__setattr__(self, "total", aggregates.total)
        object.__setattr__(self, "feasible", feasible)


@dataclass(frozen=True, slots=True)
class RegretObjective:
    """Return non-negative candidate regret aligned by exact decision key."""

    actuals_semantics: ActualsSemantics
    by_decision: Mapping[DecisionCostKey, CostValue] = field(repr=False)
    total: CostValue = field(init=False)

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
        object.__setattr__(self, "by_decision", MappingProxyType(by_decision))
        object.__setattr__(
            self,
            "total",
            CostValue(
                _finite_sum(
                    (value.value for value in by_decision.values()),
                    name="regret total",
                ),
                self.actuals_semantics,
            ),
        )


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
    if any(window.mode is not mode for window in staged):
        raise ObjectiveError("diagnostic objective mode mismatches or mixes window modes")
    if any(window.actuals_semantics is not actuals_semantics for window in staged):
        raise ObjectiveError("diagnostic objective actuals semantics must match every window")
    if mode is EmissionScope.WINDOW_SUM and len(staged) != 1:
        raise ObjectiveError("window-sum diagnostic evaluation requires exactly one window")
    if len({window.key for window in staged}) != len(staged):
        raise ObjectiveError("diagnostic objective decision keys must be unique")

    by_decision: dict[DecisionCostKey, CostValue] = {}
    for window in sorted(staged, key=lambda value: _decision_sort_key(value.key)):
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
        return SettlementObjective(
            session=None,
            actuals_semantics=actuals_semantics,
            components_by_decision={},
            infeasible_reason="candidate emitted no settlement records",
        )
    if any(not isinstance(record, SettlementRecord) for record in staged):
        raise TypeError("settle-path objective requires SettlementRecord values")
    session = staged[0].session
    if any(record.session != session for record in staged):
        raise ObjectiveError("settle-path objective records must share one session")
    if any(record.actuals_semantics is not actuals_semantics for record in staged):
        raise ObjectiveError(
            "settle-path actuals semantics must match the explicit objective binding"
        )
    if len({record.key for record in staged}) != len(staged):
        raise ObjectiveError("settle-path objective records must have unique decision keys")

    ordered = sorted(
        staged,
        key=lambda record: _decision_sort_key((record.series_key, record.period)),
    )
    by_decision: dict[DecisionCostKey, CostComponents] = {}
    for record in ordered:
        holding = CostValue(record.holding.amount, actuals_semantics)
        shortage = CostValue(record.shortage.amount, actuals_semantics)
        key = (record.series_key, record.period)
        by_decision[key] = CostComponents(holding=holding, shortage=shortage)

    return SettlementObjective(
        session=session,
        actuals_semantics=actuals_semantics,
        components_by_decision=by_decision,
    )


def evaluate_settlement_candidate(
    evaluate: Callable[[], Iterable[SettlementRecord]],
    *,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> SettlementObjective:
    """Evaluate one settlement-record callback with explicit candidate failures."""
    _require_semantics(actuals_semantics)
    if not callable(evaluate):
        raise TypeError("settlement candidate evaluator must be callable")
    try:
        records = evaluate()
        return settle_path_cost(records, actuals_semantics=actuals_semantics)
    except CandidateInfeasible as error:
        return SettlementObjective(
            session=None,
            actuals_semantics=actuals_semantics,
            components_by_decision={},
            infeasible_reason=str(error),
        )


def realized_cost(
    records: Iterable[SettlementRecord],
    *,
    actuals_semantics: ActualsSemantics = ActualsSemantics.DEMAND,
) -> CostValue:
    """Return the labeled scalar realized cost used by the tuning default."""
    return settle_path_cost(records, actuals_semantics=actuals_semantics).total


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


def _aggregate_settlement_costs(
    by_decision: Mapping[DecisionCostKey, CostComponents],
    *,
    actuals_semantics: ActualsSemantics,
) -> _SettlementAggregates:
    origin_totals: dict[pd.Timestamp, list[float]] = {}
    series_totals: dict[str, list[float]] = {}
    holding_values: list[float] = []
    shortage_values: list[float] = []
    for (series_key, origin), costs in sorted(
        by_decision.items(),
        key=lambda item: _decision_sort_key(item[0]),
    ):
        origin_totals.setdefault(origin, []).append(costs.total.value)
        series_totals.setdefault(series_key, []).append(costs.total.value)
        holding_values.append(costs.holding.value)
        shortage_values.append(costs.shortage.value)

    by_origin = {
        origin: CostValue(
            _finite_sum(values, name="settlement origin cost"),
            actuals_semantics,
        )
        for origin, values in sorted(origin_totals.items())
    }
    by_series = {
        series_key: CostValue(
            _finite_sum(values, name="settlement series cost"),
            actuals_semantics,
        )
        for series_key, values in sorted(
            series_totals.items(),
            key=lambda item: item[0].encode(),
        )
    }
    partials: list[OriginPartial] = []
    cumulative = 0.0
    for origin, cost in by_origin.items():
        cumulative = _finite_sum(
            (cumulative, cost.value),
            name="settlement partial cost",
        )
        partials.append(
            OriginPartial(
                origin=origin,
                cost=CostValue(cumulative, actuals_semantics),
            )
        )

    return _SettlementAggregates(
        by_origin=MappingProxyType(by_origin),
        by_series=MappingProxyType(by_series),
        partials=tuple(partials),
        holding=CostValue(
            _finite_sum(holding_values, name="settlement holding total"),
            actuals_semantics,
        ),
        shortage=CostValue(
            _finite_sum(shortage_values, name="settlement shortage total"),
            actuals_semantics,
        ),
        total=partials[-1].cost,
    )


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
        iterator = iter(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of values") from error
    return tuple(iterator)


def _finite_vector(value: Iterable[object], *, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{name} values must be iterable")
    try:
        values = tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} values must be iterable") from error
    return tuple(_nonnegative_cost(item, name=name, allow_infinity=False) for item in values)


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


DEFAULT_OBJECTIVE = realized_cost


__all__ = [
    "DEFAULT_OBJECTIVE",
    "CandidateInfeasible",
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
    "evaluate_settlement_candidate",
    "key_aligned_regret",
    "realized_cost",
    "settle_path_cost",
]
