"""Record immutable forecast, order, and settlement facts in one ledger."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from types import MappingProxyType
from typing import cast

import numpy as np
import pandas as pd

from newcalibre.conformal import ForecastKey as ConformalForecastKey
from newcalibre.conformal import IssuedBoundFacts, ObserveAnnotation, ResolvedObservation
from newcalibre.domain import (
    ACTUAL_VALUE,
    FRAME_KEY_COLUMNS,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    REQUIRED_FRAME_COLUMNS,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    ActualsSemantics,
    Calendar,
    CalendarError,
    DecisionEvidence,
    EmissionScope,
    ForecastFrameError,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    InventoryPosition,
    SessionIdentity,
    StockoutRule,
    forecast_bound_groups,
    interval_columns,
    quantile_column,
    validate_forecast_frame,
)
from newcalibre.observe.state import (
    ObservationResolution,
    ObserveCycle,
    ObservedActual,
    PendingObservation,
)

type ForecastKey = tuple[str, pd.Timestamp, int, str]
type BoundKey = tuple[str, ...]
type OrderKey = tuple[SessionIdentity, str, pd.Timestamp, str]
type SettlementKey = tuple[SessionIdentity, str, pd.Timestamp]
type PredicateKey = tuple[GuaranteeClaim, GuaranteeCurrency | None]

_ROW_EVENT_PREDICATE_KEYS: frozenset[PredicateKey] = frozenset(
    {
        (
            GuaranteeClaim.ONE_SIDED_COVERAGE,
            GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        ),
        (
            GuaranteeClaim.TWO_SIDED_COVERAGE,
            GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        ),
    }
)


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


@dataclass(frozen=True, slots=True)
class PredicateResult:
    """Return one finite numeric predicate value and optional coverage event."""

    value: float
    covered: bool | None

    def __post_init__(self) -> None:
        value = _finite_real(self.value, name="predicate result value")
        if self.covered is not None and not isinstance(self.covered, bool):
            raise LedgerError("predicate result covered must be a boolean or not applicable")
        if self.covered is not None and value != float(self.covered):
            raise LedgerError("a coverage predicate value must equal its boolean indicator")
        object.__setattr__(self, "value", value)


type Predicate = Callable[
    [float, tuple[float, ...], ForecastIssuance],
    PredicateResult,
]


@dataclass(frozen=True, slots=True)
class PredicateRegistration:
    """Bind one supported finite-sample row-event pair to a predicate."""

    key: PredicateKey
    predicate: Predicate

    def __post_init__(self) -> None:
        _validate_predicate_key(self.key)
        if self.key not in _ROW_EVENT_PREDICATE_KEYS:
            raise LedgerError(
                "callable row-event predicates support only finite-sample "
                "one- and two-sided coverage"
            )
        if not callable(self.predicate):
            raise LedgerError("registered predicate must be callable")


@dataclass(frozen=True, slots=True, init=False)
class PredicateRegistry:
    """Hold one immutable predicate per supported finite-sample row-event pair."""

    _registrations: tuple[PredicateRegistration, ...]
    _by_key: Mapping[PredicateKey, PredicateRegistration] = field(repr=False)

    def __init__(self, registrations: Iterable[PredicateRegistration]) -> None:
        if isinstance(registrations, (str, bytes)):
            raise LedgerError("predicate registrations must be an iterable")
        try:
            iterator = iter(registrations)
        except TypeError as error:
            raise LedgerError("predicate registrations must be iterable") from error

        staged: list[PredicateRegistration] = []
        by_key: dict[PredicateKey, PredicateRegistration] = {}
        for registration in iterator:
            if not isinstance(registration, PredicateRegistration):
                raise LedgerError("every predicate registration must be a PredicateRegistration")
            if registration.key in by_key:
                raise LedgerError(f"duplicate predicate key: {registration.key!r}")
            staged.append(registration)
            by_key[registration.key] = registration

        object.__setattr__(self, "_registrations", tuple(staged))
        object.__setattr__(self, "_by_key", MappingProxyType(by_key))

    @property
    def registrations(self) -> tuple[PredicateRegistration, ...]:
        """Return registrations in their declared order."""
        return self._registrations

    @classmethod
    def gate_a(cls) -> PredicateRegistry:
        """Return the complete Gate-A row-event predicate registry."""
        return cls(
            (
                PredicateRegistration(
                    key=(
                        GuaranteeClaim.ONE_SIDED_COVERAGE,
                        GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                    ),
                    predicate=_one_sided_coverage_predicate,
                ),
                PredicateRegistration(
                    key=(
                        GuaranteeClaim.TWO_SIDED_COVERAGE,
                        GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                    ),
                    predicate=_two_sided_coverage_predicate,
                ),
            )
        )

    def _registration_for(self, key: PredicateKey) -> PredicateRegistration | None:
        return self._by_key.get(key)


@dataclass(frozen=True, slots=True)
class CoverageTarget:
    """Identify one descriptor-and-bound scoring denominator."""

    descriptor: GuaranteeDescriptor
    guaranteed_side: GuaranteedSide | None
    bound_key: BoundKey

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, GuaranteeDescriptor):
            raise LedgerError("coverage target descriptor must be a GuaranteeDescriptor")
        if self.guaranteed_side is not None and not isinstance(
            self.guaranteed_side, GuaranteedSide
        ):
            raise LedgerError("coverage target side must be a GuaranteedSide or not applicable")
        if (
            not isinstance(self.bound_key, tuple)
            or not self.bound_key
            or any(not isinstance(column, str) or not column for column in self.bound_key)
        ):
            raise LedgerError("coverage target bound key must name forecast bound columns")
        object.__setattr__(self, "bound_key", tuple(self.bound_key))


@dataclass(frozen=True, slots=True)
class ScoreOutcome:
    """Record one row-and-bound evaluation with total unscored attribution."""

    forecast_key: ForecastKey
    target: CoverageTarget
    resolved: bool
    scored: bool
    value: float | None
    covered: bool | None
    unscored_reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.target, CoverageTarget):
            raise LedgerError("score outcome target must be a CoverageTarget")
        if not isinstance(self.resolved, bool) or not isinstance(self.scored, bool):
            raise LedgerError("score outcome state flags must be boolean")
        if not self.resolved:
            if self.scored or any(
                value is not None for value in (self.value, self.covered, self.unscored_reason)
            ):
                raise LedgerError("pending score outcomes cannot carry scoring results")
            return
        if self.scored:
            value = _finite_real(self.value, name="scored outcome value")
            if self.covered is not None and not isinstance(self.covered, bool):
                raise LedgerError("scored outcome coverage must be boolean or not applicable")
            if self.unscored_reason is not None:
                raise LedgerError("scored outcomes cannot carry an unscored reason")
            object.__setattr__(self, "value", value)
            return
        if self.value is not None or self.covered is not None:
            raise LedgerError("unscored outcomes cannot carry predicate results")
        _require_text(self.unscored_reason, name="resolved unscored reason")

    @property
    def bound_key(self) -> BoundKey:
        """Return the explicit bound key evaluated by this outcome."""
        return self.target.bound_key


@dataclass(frozen=True, slots=True)
class CoverageSummary:
    """Summarize one coverage target with scored-only denominator discipline."""

    total: int
    pending: int
    resolved: int
    scored: int
    covered: int
    unscored: int
    unscored_by_reason: Mapping[str, int]
    coverage_ratio: float | None

    def __post_init__(self) -> None:
        counts = {
            name: _nonnegative_count(getattr(self, name), name=name)
            for name in ("total", "pending", "resolved", "scored", "covered", "unscored")
        }
        if counts["total"] != counts["pending"] + counts["resolved"]:
            raise LedgerError("coverage summary total must equal pending plus resolved")
        if counts["resolved"] != counts["scored"] + counts["unscored"]:
            raise LedgerError("coverage summary resolved must equal scored plus unscored")
        if counts["covered"] > counts["scored"]:
            raise LedgerError("coverage summary covered cannot exceed scored")
        reasons = _validated_reason_counts(self.unscored_by_reason)
        if sum(reasons.values()) != counts["unscored"]:
            raise LedgerError("coverage summary reason counts must equal unscored")

        ratio = self.coverage_ratio
        if ratio is not None:
            normalized_ratio = _finite_real(ratio, name="coverage ratio")
            if not 0.0 <= normalized_ratio <= 1.0:
                raise LedgerError("coverage ratio must lie between zero and one")
            if not counts["scored"]:
                raise LedgerError("coverage ratio requires a scored denominator")
            if normalized_ratio != counts["covered"] / counts["scored"]:
                raise LedgerError("coverage ratio must equal covered divided by scored")
            object.__setattr__(self, "coverage_ratio", normalized_ratio)

        object.__setattr__(
            self,
            "unscored_by_reason",
            MappingProxyType(reasons),
        )


@dataclass(frozen=True, slots=True)
class CoverageReport:
    """Expose immutable per-bound outcomes, target summaries, and global counts."""

    outcomes: tuple[ScoreOutcome, ...]
    summaries: Mapping[CoverageTarget, CoverageSummary]
    unscored_by_reason: Mapping[str, int]
    bound_count: int
    pending_bound_count: int
    resolved_bound_count: int
    scored_bound_count: int
    covered_bound_count: int
    unscored_bound_count: int

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        if any(not isinstance(outcome, ScoreOutcome) for outcome in outcomes):
            raise LedgerError("coverage report outcomes must be ScoreOutcome values")
        if not isinstance(self.summaries, Mapping):
            raise LedgerError("coverage report summaries must be a mapping")
        summaries = dict(self.summaries)
        if any(
            not isinstance(target, CoverageTarget) or not isinstance(summary, CoverageSummary)
            for target, summary in summaries.items()
        ):
            raise LedgerError("coverage report summaries must map targets to summaries")

        count_names = (
            "bound_count",
            "pending_bound_count",
            "resolved_bound_count",
            "scored_bound_count",
            "covered_bound_count",
            "unscored_bound_count",
        )
        counts = {name: _nonnegative_count(getattr(self, name), name=name) for name in count_names}
        if counts["bound_count"] != (
            counts["pending_bound_count"] + counts["resolved_bound_count"]
        ):
            raise LedgerError("coverage report bound count must equal pending plus resolved")
        if counts["resolved_bound_count"] != (
            counts["scored_bound_count"] + counts["unscored_bound_count"]
        ):
            raise LedgerError("coverage report resolved count must equal scored plus unscored")
        if counts["covered_bound_count"] > counts["scored_bound_count"]:
            raise LedgerError("coverage report covered count cannot exceed scored")
        reasons = _validated_reason_counts(self.unscored_by_reason)
        if sum(reasons.values()) != counts["unscored_bound_count"]:
            raise LedgerError("coverage report reason counts must equal unscored")

        derived = _derive_report_facts(outcomes)
        if counts != derived.counts:
            raise LedgerError("coverage report counts must match its outcomes")
        if reasons != derived.unscored_by_reason:
            raise LedgerError("coverage report reasons must match its outcomes")
        if summaries != derived.summaries:
            raise LedgerError("coverage report summaries must match its outcomes")

        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "summaries", MappingProxyType(summaries))
        object.__setattr__(
            self,
            "unscored_by_reason",
            MappingProxyType(reasons),
        )


@dataclass(slots=True)
class _SummaryAccumulator:
    total: int = 0
    pending: int = 0
    resolved: int = 0
    scored: int = 0
    covered: int = 0
    unscored: int = 0
    coverage_observations: int = 0
    unscored_by_reason: Counter[str] = field(default_factory=Counter)

    def add(self, outcome: ScoreOutcome) -> None:
        self.total += 1
        if not outcome.resolved:
            self.pending += 1
            return
        self.resolved += 1
        if outcome.scored:
            self.scored += 1
            if outcome.covered is not None:
                self.coverage_observations += 1
                self.covered += int(outcome.covered)
            return
        self.unscored += 1
        reason = cast(str, outcome.unscored_reason)
        self.unscored_by_reason[reason] += 1

    def freeze(self) -> CoverageSummary:
        ratio = (
            self.covered / self.scored
            if self.scored and self.coverage_observations == self.scored
            else None
        )
        return CoverageSummary(
            total=self.total,
            pending=self.pending,
            resolved=self.resolved,
            scored=self.scored,
            covered=self.covered,
            unscored=self.unscored,
            unscored_by_reason=self.unscored_by_reason,
            coverage_ratio=ratio,
        )


@dataclass(frozen=True, slots=True)
class _DerivedReportFacts:
    counts: dict[str, int]
    summaries: dict[CoverageTarget, CoverageSummary]
    unscored_by_reason: dict[str, int]


def _derive_report_facts(outcomes: tuple[ScoreOutcome, ...]) -> _DerivedReportFacts:
    accumulators: dict[CoverageTarget, _SummaryAccumulator] = {}
    overall = _SummaryAccumulator()
    for outcome in outcomes:
        overall.add(outcome)
        accumulator = accumulators.get(outcome.target)
        if accumulator is None:
            accumulator = _SummaryAccumulator()
            accumulators[outcome.target] = accumulator
        accumulator.add(outcome)
    return _DerivedReportFacts(
        counts={
            "bound_count": overall.total,
            "pending_bound_count": overall.pending,
            "resolved_bound_count": overall.resolved,
            "scored_bound_count": overall.scored,
            "covered_bound_count": overall.covered,
            "unscored_bound_count": overall.unscored,
        },
        summaries={target: summary.freeze() for target, summary in accumulators.items()},
        unscored_by_reason=dict(overall.unscored_by_reason),
    )


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
    observation_issuance: IssuedBoundFacts | None
    _values: tuple[tuple[str, object], ...]

    def __init__(self) -> None:
        raise TypeError("ForecastRow instances are created by Ledger.append_forecasts()")

    @classmethod
    def _from_validated_values(
        cls,
        values: Mapping[str, object],
        *,
        issuances: Mapping[BoundKey, ForecastIssuance],
        observation_issuance: IssuedBoundFacts | None = None,
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
        object.__setattr__(
            instance,
            "observation_issuance",
            None
            if observation_issuance is None
            else IssuedBoundFacts.snapshot(observation_issuance),
        )
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

    def _with_actual_value(self, actual_value: float) -> ForecastRow:
        values = dict(self._values)
        values[ACTUAL_VALUE] = actual_value
        return self._from_validated_values(
            values,
            issuances=self.issuances,
            observation_issuance=self.observation_issuance,
        )


@dataclass(frozen=True, slots=True)
class OrderRow:
    """Record one immutable non-negative order decision."""

    session: SessionIdentity
    series_key: str
    origin: pd.Timestamp
    model_name: str
    quantity: float
    arrival_period: pd.Timestamp
    evidence: DecisionEvidence | None = None

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
        if not self.quantity.is_integer():
            raise LedgerError("order quantity must be recorded in whole units")
        _require_timestamp(self.arrival_period, name="arrival period")
        if self.evidence is not None and not isinstance(self.evidence, DecisionEvidence):
            raise LedgerError("order evidence must be DecisionEvidence or omitted")

    @property
    def key(self) -> OrderKey:
        """Return the exact immutable order-row key."""
        return (self.session, self.series_key, self.origin, self.model_name)


@dataclass(frozen=True, slots=True)
class StockoutTransition:
    """Record demand consumption and the configured stock-out transition result."""

    rule: StockoutRule
    demand: float
    fulfilled_demand: float
    unmet_demand: float
    closing_on_hand: float
    closing_backorders: float

    def __post_init__(self) -> None:
        if not isinstance(self.rule, StockoutRule):
            raise LedgerError("stock-out transition rule must be a StockoutRule")
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
        try:
            accounted_demand = math.fsum((self.fulfilled_demand, self.unmet_demand))
        except OverflowError as error:
            raise LedgerError("fulfilled demand plus unmet demand must be finite") from error
        if not math.isclose(
            accounted_demand,
            self.demand,
            rel_tol=0.0,
            abs_tol=math.ulp(self.demand),
        ):
            raise LedgerError("fulfilled demand plus unmet demand must equal demand")

    @property
    def available_inventory(self) -> float:
        """Return the engine-booked inventory available before demand consumption."""
        try:
            available = math.fsum((self.fulfilled_demand, self.closing_on_hand))
        except OverflowError as error:
            raise LedgerError("available settlement inventory must be finite") from error
        if not math.isfinite(available):
            raise LedgerError("available settlement inventory must be finite")
        return available


def lost_sales_transition(
    *,
    opening: InventoryPosition,
    arrivals: float,
    demand: float,
) -> StockoutTransition:
    """Apply the single lost-sales inventory transition."""
    if not isinstance(opening, InventoryPosition):
        raise LedgerError("lost-sales opening state must be an InventoryPosition")
    if opening.backorders != 0.0:
        raise LedgerError("lost-sales settlement requires zero opening backorders")
    normalized_arrivals = _finite_nonnegative(arrivals, name="settlement arrivals")
    normalized_demand = _finite_nonnegative(demand, name="settlement demand")
    try:
        available = math.fsum((opening.on_hand, normalized_arrivals))
    except OverflowError as error:
        raise LedgerError("available settlement inventory must be finite") from error
    if not math.isfinite(available):
        raise LedgerError("available settlement inventory must be finite")
    fulfilled = min(available, normalized_demand)
    return StockoutTransition(
        rule=StockoutRule.LOST_SALES,
        demand=normalized_demand,
        fulfilled_demand=fulfilled,
        unmet_demand=normalized_demand - fulfilled,
        closing_on_hand=available - fulfilled,
        closing_backorders=0.0,
    )


def validate_lost_sales_transition(
    *,
    transition: StockoutTransition,
    arrivals: float,
    opening: InventoryPosition | None = None,
) -> None:
    """Prove a transition is reachable from a non-negative lost-sales opening."""
    if not isinstance(transition, StockoutTransition):
        raise LedgerError("lost-sales transition must be a StockoutTransition")
    if transition.rule is not StockoutRule.LOST_SALES:
        raise LedgerError("lost-sales transition must use the lost-sales rule")
    if transition.closing_backorders != 0.0:
        raise LedgerError("lost-sales transition must close with zero backorders")
    normalized_arrivals = _finite_nonnegative(arrivals, name="settlement arrivals")
    available = transition.available_inventory
    if available < normalized_arrivals and not _quantities_equal(
        available,
        normalized_arrivals,
    ):
        raise LedgerError("settlement arrivals cannot disappear from inventory")
    implied_opening = max(0.0, available - normalized_arrivals)
    expected_fulfilled = min(available, transition.demand)
    if not _quantities_equal(transition.fulfilled_demand, expected_fulfilled):
        raise LedgerError("lost-sales transition must fulfill all available demand")
    if opening is None:
        return
    if not isinstance(opening, InventoryPosition):
        raise LedgerError("lost-sales opening state must be an InventoryPosition")
    if opening.backorders != 0.0:
        raise LedgerError("lost-sales settlement requires zero opening backorders")
    if not _quantities_equal(implied_opening, opening.on_hand):
        raise LedgerError("lost-sales transition does not match opening on-hand inventory")


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
    actuals_semantics: ActualsSemantics
    transition: StockoutTransition
    inventory_position: InventoryPosition
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
        if not isinstance(self.actuals_semantics, ActualsSemantics):
            raise LedgerError("settlement actuals semantics must be ActualsSemantics")
        if not isinstance(self.transition, StockoutTransition):
            raise LedgerError("settlement transition must be a StockoutTransition")
        if not isinstance(self.inventory_position, InventoryPosition):
            raise LedgerError("settlement inventory position must be an InventoryPosition")
        if self.inventory_position.on_hand != self.transition.closing_on_hand:
            raise LedgerError("settlement inventory on_hand must equal transition closing on hand")
        if self.inventory_position.backorders != self.transition.closing_backorders:
            raise LedgerError(
                "settlement inventory backorders must equal transition closing backorders"
            )
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
        "_annotations",
        "_calendar",
        "_forecasts",
        "_observed_history",
        "_orders",
        "_pending_forecasts",
        "_resolutions",
        "_session",
        "_settlements",
    )

    def __init__(self, *, session: SessionIdentity, calendar: Calendar) -> None:
        _require_session(session)
        _require_bound_calendar(calendar)
        self._session = session
        self._calendar = calendar
        self._forecasts: dict[ForecastKey, ForecastRow] = {}
        self._pending_forecasts: dict[ForecastKey, PendingObservation] = {}
        self._observed_history: dict[tuple[str, pd.Timestamp], ObservedActual] = {}
        self._resolutions: dict[ForecastKey, ObservationResolution] = {}
        self._annotations: dict[ForecastKey, ObserveAnnotation] = {}
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
    def observed_history(self) -> tuple[ObservedActual, ...]:
        """Return a fresh observed-actual snapshot in durable append order."""
        return tuple(
            ObservedActual(
                value.series_key,
                value.timestamp,
                value.recorded_value,
                value.censoring_assertion,
                value.availability_bound,
            )
            for value in self._observed_history.values()
        )

    @property
    def pending_observations(self) -> tuple[PendingObservation, ...]:
        """Return a fresh typed snapshot of pending rows in append order."""
        return tuple(
            PendingObservation(
                forecast_key=value.forecast_key,
                target_timestamp=value.target_timestamp,
                point_forecast=value.point_forecast,
                issued=value.issued,
                resolution=value.resolution,
            )
            for value in self._pending_forecasts.values()
        )

    @property
    def pending_observation_count(self) -> int:
        """Return the number of pending observations without materializing them."""
        return len(self._pending_forecasts)

    @property
    def observation_resolutions(self) -> tuple[ObservationResolution, ...]:
        """Return delivered censoring-aware row resolutions in commit order."""
        return tuple(self._resolutions.values())

    @property
    def observe_annotations(self) -> tuple[ObserveAnnotation, ...]:
        """Return durable observe annotations in commit order."""
        return tuple(self._annotations.values())

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
        observation_issuances: Mapping[ForecastKey, IssuedBoundFacts] | None = None,
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
        observation_facts = _observation_issuance_snapshot(
            observation_issuances,
            staged_keys=staged_keys,
        )

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
            observation_issuance = observation_facts.get(key)
            if observation_issuance is not None:
                observation_issuance = _validate_observation_issuance(
                    values,
                    observation_issuance,
                )
            staged_rows[key] = ForecastRow._from_validated_values(
                values,
                issuances=row_issuances,
                observation_issuance=observation_issuance,
            )

        for quantile_group in (group for group in bound_groups if len(group) == 1):
            issued = [quantile_group in row.issuances for row in staged_rows.values()]
            if any(issued) and not all(issued):
                raise LedgerError(
                    "forecast bound issuance keys must exactly account for every supplied "
                    "quantile group"
                )

        self._forecasts.update(staged_rows)
        self._pending_forecasts.update(
            (
                key,
                PendingObservation(
                    forecast_key=ConformalForecastKey(
                        row.series_key,
                        row.origin,
                        row.horizon_step,
                        row.model_name,
                    ),
                    target_timestamp=row.target_timestamp,
                    point_forecast=row.point_forecast,
                    issued=row.observation_issuance,
                ),
            )
            for key, row in staged_rows.items()
        )

    def append_orders(self, rows: Iterable[OrderRow]) -> None:
        """Validate one input chunk once, then append it atomically."""
        staged = _stage_rows(rows, row_type=OrderRow, family="order")
        staged_rows: dict[OrderKey, OrderRow] = {}
        for row in staged:
            if row.session != self._session:
                raise LedgerError("order row session does not match the ledger session")
            try:
                self._calendar.require_member(row.origin, name="order origin")
                self._calendar.require_member(row.arrival_period, name="order arrival period")
            except CalendarError as error:
                raise LedgerError(str(error)) from error
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
            try:
                self._calendar.require_member(row.period, name="settlement period")
            except CalendarError as error:
                raise LedgerError(str(error)) from error
            key = row.key
            if key in self._settlements or key in staged_rows:
                raise LedgerError(f"duplicate settlement key: {key!r}")
            staged_rows[key] = row
        self._settlements.update(staged_rows)

    def _publish_staged_rows(self, staged: Ledger) -> None:
        """Publish rows already validated by an isolated sibling ledger."""
        if not isinstance(staged, Ledger):
            raise TypeError("staged ledger rows require a Ledger")
        if staged.session != self._session or staged.calendar != self._calendar:
            raise LedgerError("staged ledger rows do not match the owned ledger")
        self._forecasts.update(staged._forecasts)
        self._pending_forecasts.update(staged._pending_forecasts)
        self._orders.update(staged._orders)
        self._settlements.update(staged._settlements)

    def due_frame(self, origin: pd.Timestamp) -> pd.DataFrame:
        """Return a fresh append-ordered snapshot of pending rows due before origin."""
        self._require_calendar_origin(origin)
        due_values: list[dict[str, object]] = []
        for pending in self._pending_forecasts.values():
            if pending.target_timestamp < origin:
                key = _ledger_forecast_key(pending.forecast_key)
                due_values.append(dict(self._forecasts[key]._values))
        if not due_values:
            return _empty_forecast_frame()

        due_frame = pd.DataFrame(due_values)
        due_frame[SERIES_KEY] = due_frame[SERIES_KEY].astype("string")
        due_frame[MODEL_NAME] = due_frame[MODEL_NAME].astype("string")
        due_frame[ACTUAL_VALUE] = due_frame[ACTUAL_VALUE].astype("float64")
        return due_frame

    def apply_observe_cycle(self, cycle: ObserveCycle, *, origin: pd.Timestamp) -> None:
        """Validate then atomically materialize one complete observe-cycle delta."""
        staged = self._validated_observe_cycle(cycle, origin=origin)
        history, retained, removal_keys, resolutions, annotations = staged
        self._observed_history.update(history)
        for key in removal_keys:
            self._pending_forecasts.pop(key)
        self._pending_forecasts.update(retained)
        self._resolutions.update(resolutions)
        self._annotations.update(annotations)

    def _validated_observe_cycle(
        self,
        cycle: ObserveCycle,
        *,
        origin: pd.Timestamp,
    ) -> tuple[
        dict[tuple[str, pd.Timestamp], ObservedActual],
        dict[ForecastKey, PendingObservation],
        set[ForecastKey],
        dict[ForecastKey, ObservationResolution],
        dict[ForecastKey, ObserveAnnotation],
    ]:
        self._require_calendar_origin(origin)
        if not isinstance(cycle, ObserveCycle):
            raise LedgerError("ledger observation requires an ObserveCycle")

        history: dict[tuple[str, pd.Timestamp], ObservedActual] = {}
        for value in cycle.history_appends:
            if value.timestamp >= origin:
                raise LedgerError(f"observed actual is not admissible at origin: {value.key!r}")
            previous = self._observed_history.get(value.key)
            if previous is None:
                previous = history.get(value.key)
            if previous is not None:
                if previous.recorded_fact != value.recorded_fact:
                    raise LedgerError(f"conflicting observed actual: {value.key!r}")
                raise LedgerError(f"observed actual is already recorded: {value.key!r}")
            history[value.key] = value

        current = self._pending_forecasts
        removal_keys = {_ledger_forecast_key(key) for key in cycle.pending_removals}
        unknown_removal = next((key for key in removal_keys if key not in current), None)
        if unknown_removal is not None:
            raise LedgerError(f"unknown pending forecast key: {unknown_removal!r}")
        retention_by_key = {
            _ledger_forecast_key(value.forecast_key): value for value in cycle.pending_retentions
        }
        unknown_retention = (
            set(retention_by_key)
            .difference(current)
            .union(set(retention_by_key).intersection(removal_keys))
        )
        if unknown_retention:
            raise LedgerError(
                "observe pending retentions must address known non-removed rows; "
                f"invalid={sorted(unknown_retention, key=repr)!r}"
            )
        retained_rows: dict[ForecastKey, PendingObservation] = {}
        for key, retained in retention_by_key.items():
            prior = current[key]
            if (
                retained.forecast_key != prior.forecast_key
                or retained.target_timestamp != prior.target_timestamp
                or retained.point_forecast != prior.point_forecast
                or retained.issued != prior.issued
            ):
                raise LedgerError(f"pending retention changed issued row facts: {key!r}")
            if prior.resolution is not None and retained.resolution != prior.resolution:
                raise LedgerError(f"pending retention changed a resolved row: {key!r}")
            if retained.resolution is not None and retained.target_timestamp >= origin:
                raise LedgerError(f"pending retention resolved a row that is not due: {key!r}")
            retained_rows[key] = retained

        resolution_by_key = {
            _ledger_forecast_key(value.forecast_key): value for value in cycle.resolutions
        }
        if set(resolution_by_key) != removal_keys:
            raise LedgerError("observe resolutions must exactly match pending removals")
        for key, resolution in resolution_by_key.items():
            prior = current[key]
            if prior.target_timestamp >= origin:
                raise LedgerError(f"forecast row is not yet due: {key!r}")
            if resolution.target_timestamp != prior.target_timestamp:
                raise LedgerError(f"resolution target does not match pending row: {key!r}")
            if prior.resolution is not None and prior.resolution != resolution:
                raise LedgerError(f"resolution conflicts with retained pending state: {key!r}")
            row = self._forecasts[key]
            if row.actual_value is not None or key in self._resolutions:
                raise LedgerError(f"forecast row is already resolved: {key!r}")
            _finite_real(resolution.actual, name="resolved actual value")

        delivered_sequence = tuple(
            _ledger_forecast_key(observation.forecast_key)
            for observation in cycle.deliveries.observations
        )
        delivered_keys = set(delivered_sequence)
        if len(delivered_keys) != len(delivered_sequence):
            raise LedgerError("observe deliveries contain a duplicate forecast key")
        conformal_removal_keys = {key for key in removal_keys if current[key].issued is not None}
        if delivered_keys != conformal_removal_keys:
            raise LedgerError(
                "observe deliveries must exactly match conformal-issued pending removals"
            )
        for observation in cycle.deliveries.observations:
            key = _ledger_forecast_key(observation.forecast_key)
            if not _delivery_matches_staged_resolution(
                observation,
                pending=current[key],
                resolution=resolution_by_key[key],
            ):
                raise LedgerError(f"observe delivery facts do not match staged resolution: {key!r}")
        annotation_by_key = {
            _ledger_forecast_key(value.forecast_key): value for value in cycle.annotations
        }
        if set(annotation_by_key) != delivered_keys:
            raise LedgerError("observe annotations must exactly match delivered rows")
        duplicate_annotation = next(
            (key for key in annotation_by_key if key in self._annotations),
            None,
        )
        if duplicate_annotation is not None:
            raise LedgerError(f"duplicate observe annotation key: {duplicate_annotation!r}")

        return (
            history,
            retained_rows,
            removal_keys,
            resolution_by_key,
            annotation_by_key,
        )

    def coverage_report(self, registry: PredicateRegistry) -> CoverageReport:
        """Score each forecast-bound fact once under its registered descriptor pair."""
        if not isinstance(registry, PredicateRegistry):
            raise LedgerError("coverage reporting requires a PredicateRegistry")

        outcomes: list[ScoreOutcome] = []
        for forecast_key, row in self._forecasts.items():
            resolution = self._resolutions.get(forecast_key)
            row_actual = None if resolution is None else resolution.actual
            window_sum_actual = (
                _resolved_window_sum(
                    self._forecasts,
                    forecast_key=forecast_key,
                    resolutions=self._resolutions,
                )
                if any(
                    issuance.bounds_finite
                    and issuance.descriptor.window is EmissionScope.WINDOW_SUM
                    for issuance in row.issuances.values()
                )
                else None
            )
            annotation = self._annotations.get(forecast_key)
            for bound_key, issuance in row.issuances.items():
                target = CoverageTarget(
                    descriptor=issuance.descriptor,
                    guaranteed_side=issuance.guaranteed_side,
                    bound_key=bound_key,
                )
                actual_value = row_actual
                if issuance.bounds_finite and target.descriptor.window is EmissionScope.WINDOW_SUM:
                    actual_value = window_sum_actual
                outcome = _score_bound(
                    forecast_key=forecast_key,
                    row=row,
                    actual_value=actual_value,
                    target=target,
                    issuance=issuance,
                    annotation=annotation,
                    registry=registry,
                )
                outcomes.append(outcome)

        frozen_outcomes = tuple(outcomes)
        derived = _derive_report_facts(frozen_outcomes)
        return CoverageReport(
            outcomes=frozen_outcomes,
            summaries=derived.summaries,
            unscored_by_reason=derived.unscored_by_reason,
            **derived.counts,
        )

    def _require_calendar_origin(self, origin: pd.Timestamp) -> None:
        try:
            self._calendar.require_member(origin, name="ledger origin")
        except CalendarError as error:
            raise LedgerError(f"ledger origin must lie on the owned calendar: {error}") from error


def _delivery_matches_staged_resolution(
    observation: ResolvedObservation,
    *,
    pending: PendingObservation,
    resolution: ObservationResolution,
) -> bool:
    """Return whether a delivery repeats its pending and resolution facts exactly."""
    issued = pending.issued
    return (
        issued is not None
        and observation.forecast_key == pending.forecast_key
        and observation.target_timestamp == resolution.target_timestamp
        and observation.actual == resolution.actual
        and observation.point_forecast == pending.point_forecast
        and observation.censoring_assertion is resolution.censoring_assertion
        and observation.availability_bound == resolution.availability_bound
        and _issued_facts_match(observation.issued, issued)
    )


def _issued_facts_match(left: IssuedBoundFacts, right: IssuedBoundFacts) -> bool:
    """Compare issued facts while treating paired cold-start NaNs as equal."""
    return (
        left.method_name == right.method_name
        and left.emission_form is right.emission_form
        and left.emission_scope is right.emission_scope
        and left.partition_label == right.partition_label
        and left.working_level == right.working_level
        and left.state_reference == right.state_reference
        and _nan_equal(left.lower_bound, right.lower_bound)
        and _nan_equal(left.upper_bound, right.upper_bound)
        and left.calibration_ready is right.calibration_ready
        and left.bounds_null_reason == right.bounds_null_reason
        and left.effective_descriptor == right.effective_descriptor
        and left.bindings == right.bindings
    )


def _nan_equal(left: float, right: float) -> bool:
    return left == right or (math.isnan(left) and math.isnan(right))


def _resolved_window_sum(
    forecasts: Mapping[ForecastKey, ForecastRow],
    *,
    forecast_key: ForecastKey,
    resolutions: Mapping[ForecastKey, ObservationResolution],
) -> float | None:
    series_key, origin, terminal_step, model_name = forecast_key
    members: list[ForecastRow] = []
    for step in range(1, terminal_step + 1):
        key = (series_key, origin, step, model_name)
        try:
            members.append(forecasts[key])
        except KeyError as error:
            raise LedgerError(
                "window-sum coverage requires every leading protection-window member"
            ) from error
    actuals = tuple(
        None if (resolution := resolutions.get(member.key)) is None else resolution.actual
        for member in members
    )
    if any(actual is None for actual in actuals):
        return None
    return _finite_real(
        math.fsum(cast(float, actual) for actual in actuals),
        name="resolved window-sum actual",
    )


def _score_bound(
    *,
    forecast_key: ForecastKey,
    row: ForecastRow,
    actual_value: float | None,
    target: CoverageTarget,
    issuance: ForecastIssuance,
    annotation: ObserveAnnotation | None,
    registry: PredicateRegistry,
) -> ScoreOutcome:
    if actual_value is None:
        return ScoreOutcome(
            forecast_key=forecast_key,
            target=target,
            resolved=False,
            scored=False,
            value=None,
            covered=None,
            unscored_reason=None,
        )

    claim = issuance.descriptor.type.claim
    if claim is not GuaranteeClaim.NONE and not issuance.calibration_ready:
        return _unscored_outcome(forecast_key, target, reason="warm-up")
    if not issuance.bounds_finite:
        return _unscored_outcome(
            forecast_key,
            target,
            reason=cast(str, issuance.bounds_null_reason),
        )
    if claim is GuaranteeClaim.NONE:
        return _unscored_outcome(
            forecast_key,
            target,
            reason="not-engine-calibrated",
        )
    if (
        annotation is not None
        and not annotation.advanced_delivered_score
        and annotation.exclusion_cause in {"declared-censored", "declared-censored-window"}
    ):
        return _unscored_outcome(
            forecast_key,
            target,
            reason=annotation.exclusion_cause,
        )

    predicate_key: PredicateKey = (
        claim,
        issuance.descriptor.type.currency,
    )
    registration = registry._registration_for(predicate_key)
    if registration is None:
        return _unscored_outcome(
            forecast_key,
            target,
            reason="predicate-unregistered",
        )

    values = dict(row._values)
    bound_values = tuple(
        _finite_real(values[column], name="issued bound value") for column in target.bound_key
    )
    result = registration.predicate(actual_value, bound_values, issuance)
    if not isinstance(result, PredicateResult):
        raise LedgerError("registered predicate must return a PredicateResult")
    return ScoreOutcome(
        forecast_key=forecast_key,
        target=target,
        resolved=True,
        scored=True,
        value=result.value,
        covered=result.covered,
        unscored_reason=None,
    )


def _unscored_outcome(
    forecast_key: ForecastKey,
    target: CoverageTarget,
    *,
    reason: str,
) -> ScoreOutcome:
    return ScoreOutcome(
        forecast_key=forecast_key,
        target=target,
        resolved=True,
        scored=False,
        value=None,
        covered=None,
        unscored_reason=reason,
    )


def _one_sided_coverage_predicate(
    actual_value: float,
    bound_values: tuple[float, ...],
    issuance: ForecastIssuance,
) -> PredicateResult:
    if len(bound_values) != 1:
        raise LedgerError("one-sided predicate requires exactly one bound value")
    if issuance.guaranteed_side is GuaranteedSide.LOWER:
        covered = bound_values[0] <= actual_value
    elif issuance.guaranteed_side is GuaranteedSide.UPPER:
        covered = actual_value <= bound_values[0]
    else:
        raise LedgerError("one-sided predicate requires a guaranteed side")
    return PredicateResult(value=float(covered), covered=covered)


def _two_sided_coverage_predicate(
    actual_value: float,
    bound_values: tuple[float, ...],
    issuance: ForecastIssuance,
) -> PredicateResult:
    if len(bound_values) != 2:
        raise LedgerError("two-sided predicate requires a lower/upper bound pair")
    if issuance.descriptor.type.claim is not GuaranteeClaim.TWO_SIDED_COVERAGE:
        raise LedgerError("two-sided predicate requires a two-sided descriptor")
    covered = bound_values[0] <= actual_value <= bound_values[1]
    return PredicateResult(value=float(covered), covered=covered)


def _validate_predicate_key(key: object) -> None:
    if not isinstance(key, tuple) or len(key) != 2:
        raise LedgerError("predicate key must be an exact (claim, currency) pair")
    claim, currency = key
    if not isinstance(claim, GuaranteeClaim):
        raise LedgerError("predicate key claim must be a GuaranteeClaim")
    if claim is GuaranteeClaim.NONE:
        if currency is not None:
            raise LedgerError("the none claim cannot carry a predicate currency")
        raise LedgerError("the none claim cannot register a scoring predicate")
    if not isinstance(currency, GuaranteeCurrency):
        raise LedgerError("a non-none predicate key requires a GuaranteeCurrency")


def _empty_forecast_frame() -> pd.DataFrame:
    columns = {
        SERIES_KEY: pd.Series(dtype="string"),
        TARGET_TIMESTAMP: pd.Series(dtype="datetime64[ns]"),
        ACTUAL_VALUE: pd.Series(dtype="float64"),
        POINT_FORECAST: pd.Series(dtype="float64"),
        HORIZON_STEP: pd.Series(dtype="int64"),
        ORIGIN: pd.Series(dtype="datetime64[ns]"),
        MODEL_NAME: pd.Series(dtype="string"),
    }
    return pd.DataFrame({column: columns[column] for column in REQUIRED_FRAME_COLUMNS})


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


def _ledger_forecast_key(value: ConformalForecastKey) -> ForecastKey:
    return (
        value.series_key,
        value.origin,
        value.horizon_step,
        value.model_name,
    )


def _observation_issuance_snapshot(
    values: Mapping[ForecastKey, IssuedBoundFacts] | None,
    *,
    staged_keys: set[ForecastKey],
) -> dict[ForecastKey, IssuedBoundFacts]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise LedgerError("observation issuances must be keyed by forecast row key")
    try:
        snapshot = dict(values)
    except (TypeError, ValueError) as error:
        raise LedgerError("observation issuance keys must be hashable forecast keys") from error
    unknown = next((key for key in snapshot if key not in staged_keys), None)
    if unknown is not None:
        raise LedgerError(f"observation issuance names an unknown forecast key: {unknown!r}")
    frozen: dict[ForecastKey, IssuedBoundFacts] = {}
    for key, facts in snapshot.items():
        try:
            frozen[key] = IssuedBoundFacts.snapshot(facts)
        except ValueError as error:
            raise LedgerError(str(error)) from error
    return frozen


def _validate_observation_issuance(
    values: Mapping[str, object],
    facts: IssuedBoundFacts,
) -> IssuedBoundFacts:
    lower_column, upper_column = interval_columns(facts.effective_descriptor.level)
    if lower_column not in values or upper_column not in values:
        raise LedgerError(
            "observation issuance descriptor level must identify forecast bound columns"
        )
    lower = values[lower_column]
    upper = values[upper_column]
    if not _same_optional_bound(lower, facts.lower_bound) or not _same_optional_bound(
        upper,
        facts.upper_bound,
    ):
        raise LedgerError("observation issuance bounds must equal the forecast payload")
    return facts


def _same_optional_bound(value: object, expected: float) -> bool:
    if _is_missing_scalar(value):
        return math.isnan(expected)
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return False
    return math.isfinite(normalized) and normalized == expected


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
    interval_groups = tuple(group for group in bound_groups if len(group) == 2)
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
        if issuance.bounds_finite:
            payload_matches = all(_is_finite_real(values[column]) for column in bound_key)
        else:
            payload_matches = all(_is_missing_scalar(values[column]) for column in bound_key)
        if not payload_matches:
            raise LedgerError(
                "issued bounds finiteness/nullability does not match the forecast payload"
            )

    if any(not accounted_columns.intersection(group) for group in interval_groups):
        raise LedgerError("forecast bound issuance keys must account for every interval group")
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
    if _is_missing_scalar(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, np.generic):
        value = value.item()
    return isinstance(value, float) and math.isnan(value)


def _is_finite_real(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError):
        return False


def _nonnegative_count(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerError(f"{name.replace('_', ' ')} must be a non-negative integer")
    return value


def _validated_reason_counts(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise LedgerError("unscored reason counts must be a mapping")
    reasons: dict[str, int] = {}
    for reason, count in value.items():
        _require_text(reason, name="unscored reason")
        normalized_count = _nonnegative_count(count, name="unscored reason count")
        if normalized_count == 0:
            raise LedgerError("unscored reason counts must be positive")
        reasons[cast(str, reason)] = normalized_count
    return reasons


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LedgerError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise LedgerError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise LedgerError(f"{name} must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized


def _finite_nonnegative(value: object, *, name: str) -> float:
    normalized = _finite_real(value, name=name)
    if normalized < 0.0:
        raise LedgerError(f"{name} must be non-negative")
    return normalized


def _quantities_equal(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=0.0,
        abs_tol=max(math.ulp(left), math.ulp(right)),
    )


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
    "CoverageReport",
    "CoverageSummary",
    "CoverageTarget",
    "ForecastKey",
    "ForecastIssuance",
    "ForecastRow",
    "GuaranteedSide",
    "Ledger",
    "LedgerError",
    "OrderKey",
    "OrderRow",
    "Predicate",
    "PredicateKey",
    "PredicateRegistration",
    "PredicateRegistry",
    "PredicateResult",
    "ScoreOutcome",
    "SettlementKey",
    "SettlementRecord",
    "StockoutTransition",
    "lost_sales_transition",
    "validate_lost_sales_transition",
]
