"""Validate policy output and materialize durable whole-unit orders."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real

import pandas as pd

from newcalibre.domain import Calendar, DecisionEvidence, InventoryPosition, SessionIdentity
from newcalibre.engine.errors import EngineError
from newcalibre.ledger import BoundKey, ForecastIssuance, ForecastKey, OrderRow
from newcalibre.ordering import OrderingConfiguration, PolicyRequest, dispatch_policy

type DecisionKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class OrderProposal:
    """Propose one finite real-valued quantity for a decision group."""

    series_key: str
    model_name: str
    quantity: float
    evidence: DecisionEvidence | None = None

    def __post_init__(self) -> None:
        _require_utf8_identifier(self.series_key, name="proposal series key")
        _require_utf8_identifier(self.model_name, name="proposal model name")
        object.__setattr__(self, "quantity", _finite_real(self.quantity, name="proposal quantity"))
        if self.evidence is not None:
            if not isinstance(self.evidence, DecisionEvidence):
                raise EngineError("proposal evidence must be DecisionEvidence or omitted")
            object.__setattr__(self, "evidence", DecisionEvidence.snapshot(self.evidence))

    @property
    def key(self) -> DecisionKey:
        """Return the exact decision group addressed by this proposal."""
        return self.series_key, self.model_name


@dataclass(frozen=True, slots=True)
class ConfiguredPolicyOrderer:
    """Adapt the engine request facts to the built-in pure policy dispatcher."""

    def propose(
        self,
        *,
        frame: pd.DataFrame,
        issuances: Mapping[ForecastKey, Mapping[BoundKey, ForecastIssuance]],
        inventory_positions: Mapping[str, InventoryPosition],
        configuration: OrderingConfiguration,
    ) -> tuple[OrderProposal, ...]:
        """Return evidence-complete proposals without mutating engine inputs."""
        decisions = dispatch_policy(
            PolicyRequest(
                frame=frame,
                issuances=issuances,
                inventory_positions=inventory_positions,
                configuration=configuration,
            )
        )
        return tuple(
            OrderProposal(
                series_key=decision.series_key,
                model_name=decision.model_name,
                quantity=decision.quantity,
                evidence=decision.evidence,
            )
            for decision in decisions
        )


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
            (
                proposal
                if type(proposal) is OrderProposal
                else OrderProposal(
                    series_key=proposal.series_key,
                    model_name=proposal.model_name,
                    quantity=proposal.quantity,
                    evidence=proposal.evidence,
                )
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
    proposed = {proposal.key: proposal for proposal in decisions.proposals}
    return tuple(
        OrderRow(
            session=decisions.session,
            series_key=series_key,
            origin=decisions.origin,
            model_name=model_name,
            quantity=float(
                max(
                    math.ceil(
                        proposed[(series_key, model_name)].quantity
                        if (series_key, model_name) in proposed
                        else 0.0
                    ),
                    0,
                )
            ),
            arrival_period=arrival,
            evidence=(
                proposed[(series_key, model_name)].evidence
                if (series_key, model_name) in proposed
                else None
            ),
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
    "ConfiguredPolicyOrderer",
    "DecisionEvidence",
    "DecisionBatch",
    "DecisionKey",
    "OrderProposal",
    "materialize_decisions",
]
