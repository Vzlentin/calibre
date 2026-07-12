"""Validate policy output and materialize durable whole-unit orders."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Real

import pandas as pd

from newcalibre.domain import Calendar, SessionIdentity
from newcalibre.engine.errors import EngineError
from newcalibre.ledger import OrderRow
from newcalibre.ordering import OrderingConfiguration

type DecisionKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class OrderProposal:
    """Propose one finite real-valued quantity for a decision group."""

    series_key: str
    model_name: str
    quantity: float

    def __post_init__(self) -> None:
        _require_utf8_identifier(self.series_key, name="proposal series key")
        _require_utf8_identifier(self.model_name, name="proposal model name")
        object.__setattr__(self, "quantity", _finite_real(self.quantity, name="proposal quantity"))

    @property
    def key(self) -> DecisionKey:
        """Return the exact decision group addressed by this proposal."""
        return self.series_key, self.model_name


@dataclass(frozen=True, slots=True)
class DecisionBatch:
    """Carry one atomically validated policy result into Commit."""

    session: SessionIdentity
    origin: pd.Timestamp
    requested: tuple[DecisionKey, ...]
    proposals: tuple[OrderProposal, ...]
    _engine_token: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("decision batch session must be a SessionIdentity")
        _require_timestamp(self.origin, name="decision batch origin")
        requested = tuple(self.requested)
        for key in requested:
            _validate_decision_key(key)
        if len(set(requested)) != len(requested):
            raise EngineError("decision batch requested keys must be unique")
        requested = tuple(sorted(requested, key=lambda key: (key[0].encode(), key[1].encode())))

        supplied = tuple(self.proposals)
        if len(supplied) > len(requested):
            raise EngineError("orderer returned more proposals than requested")
        if any(not isinstance(proposal, OrderProposal) for proposal in supplied):
            raise EngineError("orderer must return only OrderProposal values")
        proposals = tuple(
            OrderProposal(
                series_key=proposal.series_key,
                model_name=proposal.model_name,
                quantity=proposal.quantity,
            )
            for proposal in supplied
        )
        proposal_keys = tuple(proposal.key for proposal in proposals)
        if len(set(proposal_keys)) != len(proposal_keys):
            raise EngineError("orderer returned a duplicate decision proposal")
        foreign = set(proposal_keys) - set(requested)
        if foreign:
            raise EngineError(
                f"orderer proposed decision groups that were not requested: {sorted(foreign)!r}"
            )
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "proposals", proposals)


def materialize_decisions(
    decisions: DecisionBatch | None,
    *,
    configuration: OrderingConfiguration | None,
    calendar: Calendar,
) -> tuple[OrderRow, ...]:
    """Apply the one Commit-time ceiling and zero-fill projection."""
    if decisions is None:
        return ()
    if not isinstance(decisions, DecisionBatch):
        raise TypeError("order materialization requires a DecisionBatch or None")
    if not isinstance(configuration, OrderingConfiguration):
        raise EngineError("order materialization requires ordering configuration")
    if not isinstance(calendar, Calendar):
        raise TypeError("order materialization calendar must be a Calendar")
    foreign_series = {series_key for series_key, _model_name in decisions.requested} - set(
        configuration.series_keys
    )
    if foreign_series:
        raise EngineError(
            f"decision batch contains series outside ordering configuration: "
            f"{sorted(foreign_series, key=str.encode)!r}"
        )

    arrival = calendar.advance(decisions.origin, configuration.decision_timing.lead_time)
    proposed = {proposal.key: proposal.quantity for proposal in decisions.proposals}
    return tuple(
        OrderRow(
            session=decisions.session,
            series_key=series_key,
            origin=decisions.origin,
            model_name=model_name,
            quantity=float(max(math.ceil(proposed.get((series_key, model_name), 0.0)), 0)),
            arrival_period=arrival,
        )
        for series_key, model_name in decisions.requested
    )


def _validate_decision_key(key: object) -> None:
    if not isinstance(key, tuple) or len(key) != 2:
        raise EngineError("decision key must be an exact (series key, model name) pair")
    series_key, model_name = key
    _require_utf8_identifier(series_key, name="decision-key series key")
    _require_utf8_identifier(model_name, name="decision-key model name")


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise EngineError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise EngineError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise EngineError(f"{name} must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized


def _require_utf8_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise EngineError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise EngineError(f"{name} must be valid UTF-8") from error


def _require_timestamp(value: object, *, name: str) -> None:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise EngineError(f"{name} must be a non-missing pandas Timestamp")
    if value.tz is not None:
        raise EngineError(f"{name} must be timezone-naive")


__all__ = [
    "DecisionBatch",
    "DecisionKey",
    "OrderProposal",
    "materialize_decisions",
]
