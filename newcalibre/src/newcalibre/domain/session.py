"""Derive durable session identity from canonical defining inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral

from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.domain.calendar import Calendar
from newcalibre.domain.cost import CostStructure
from newcalibre.domain.decision import DecisionTiming, StockoutRule

_IDENTITY_SCHEMA = "newcalibre.session-identity"
_IDENTITY_VERSION = 2


class SessionIdentityError(ValueError):
    """Report an invalid session-defining input."""


@dataclass(frozen=True, slots=True, init=False)
class SessionIdentity:
    """Content-address one logical forecasting and decision lifecycle."""

    _value: str
    _payload: bytes = field(repr=False, compare=False, hash=False)
    _series_keys: tuple[str, ...] = field(repr=False, compare=False, hash=False)

    def __init__(self) -> None:
        raise TypeError("SessionIdentity must be created with derive()")

    @classmethod
    def derive(
        cls,
        *,
        tenant: str,
        series_keys: Iterable[str],
        calendar: Calendar,
        horizon: int,
        model_config: Mapping[str, object],
        conformal_config: Mapping[str, object] | None = None,
        ordering_policy: Mapping[str, object] | None = None,
        decision_series_keys: Iterable[str] | None = None,
        cost_structure: CostStructure | Mapping[str, CostStructure] | None = None,
        decision_timing: DecisionTiming | None = None,
        stockout_rule: StockoutRule | None = None,
    ) -> SessionIdentity:
        """Return the same identity for the same canonical defining inputs."""
        normalized_tenant = _require_text(tenant, name="tenant")
        normalized_series = _canonical_series_set(series_keys)
        if not isinstance(calendar, Calendar):
            raise SessionIdentityError("calendar must be a Calendar")
        if not isinstance(horizon, Integral) or isinstance(horizon, bool) or horizon < 1:
            raise SessionIdentityError("horizon must be a positive integer")

        normalized_model = _canonical_config(model_config, name="model_config")
        if "scope" in normalized_model:
            raise SessionIdentityError(
                "scope is engine configuration and cannot appear in model_config"
            )
        normalized_conformal = (
            None
            if conformal_config is None
            else _canonical_config(conformal_config, name="conformal_config")
        )
        decision_values = (ordering_policy, cost_structure, decision_timing, stockout_rule)
        has_decision_input = any(value is not None for value in decision_values)
        if has_decision_input and decision_series_keys is None:
            raise SessionIdentityError(
                "decision_series_keys must be supplied with decision configuration"
            )
        if not has_decision_input and decision_series_keys is not None:
            raise SessionIdentityError(
                "decision_series_keys require a complete decision configuration"
            )
        normalized_decision_series = (
            None
            if decision_series_keys is None
            else _canonical_decision_series(
                decision_series_keys,
                session_series=normalized_series,
            )
        )
        normalized_decision = _canonical_decision_config(
            ordering_policy=ordering_policy,
            cost_structure=cost_structure,
            decision_timing=decision_timing,
            stockout_rule=stockout_rule,
            series_keys=(
                normalized_series
                if normalized_decision_series is None
                else normalized_decision_series
            ),
        )
        if normalized_decision is not None and normalized_decision_series != normalized_series:
            assert normalized_decision_series is not None
            normalized_decision["series_set"] = list(normalized_decision_series)

        payload = {
            "calendar_frequency": calendar.frequency,
            "conformal_config": normalized_conformal,
            "decision": normalized_decision,
            "horizon": int(horizon),
            "model_config": normalized_model,
            "schema": _IDENTITY_SCHEMA,
            "series_set": list(normalized_series),
            "tenant": normalized_tenant,
            "version": _IDENTITY_VERSION,
        }
        try:
            encoded = canonical_json_bytes(payload, path="session identity")
        except CanonicalJsonError as error:
            raise SessionIdentityError(str(error)) from error

        instance = object.__new__(cls)
        object.__setattr__(instance, "_value", hashlib.sha256(encoded).hexdigest())
        object.__setattr__(instance, "_payload", encoded)
        object.__setattr__(instance, "_series_keys", normalized_series)
        return instance

    @property
    def value(self) -> str:
        """Return the complete lowercase SHA-256 hex digest."""
        return self._value

    @property
    def series_keys(self) -> tuple[str, ...]:
        """Return the immutable canonical session series set."""
        return self._series_keys

    def to_bytes(self) -> bytes:
        """Return the immutable canonical identity preimage."""
        return self._payload

    def __str__(self) -> str:
        return self._value


def _canonical_config(value: Mapping[str, object], *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise SessionIdentityError(f"{name} must be a mapping")
    candidate = dict(value)
    try:
        encoded = canonical_json_bytes(candidate, path=name)
    except CanonicalJsonError as error:
        raise SessionIdentityError(str(error)) from error
    return json.loads(encoded)


def _canonical_decision_config(
    *,
    ordering_policy: Mapping[str, object] | None,
    cost_structure: CostStructure | Mapping[str, CostStructure] | None,
    decision_timing: DecisionTiming | None,
    stockout_rule: StockoutRule | None,
    series_keys: tuple[str, ...],
) -> dict[str, object] | None:
    values = (ordering_policy, cost_structure, decision_timing, stockout_rule)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise SessionIdentityError(
            "ordering_policy, cost_structure, decision_timing, and stockout_rule "
            "must all be supplied or all be absent"
        )
    assert ordering_policy is not None
    if not isinstance(decision_timing, DecisionTiming):
        raise SessionIdentityError("decision_timing must be a DecisionTiming")
    if not isinstance(stockout_rule, StockoutRule):
        raise SessionIdentityError("stockout_rule must be a StockoutRule")

    return {
        "cost_structure": _canonical_cost_structure(
            cost_structure,
            series_keys=series_keys,
        ),
        "ordering_policy": _canonical_config(
            ordering_policy,
            name="ordering_policy",
        ),
        "stockout_rule": stockout_rule.value,
        "timing": {
            "lead_time": decision_timing.lead_time,
            "review_period": decision_timing.review_period,
        },
    }


def _canonical_cost_structure(
    value: object,
    *,
    series_keys: tuple[str, ...],
) -> dict[str, object]:
    if isinstance(value, CostStructure):
        return _cost_payload(value)
    if not isinstance(value, Mapping):
        raise SessionIdentityError(
            "cost_structure must be a CostStructure or an exact per-series mapping"
        )
    snapshot = dict(value)
    if set(snapshot) != set(series_keys):
        raise SessionIdentityError("per-series cost_structure keys must exactly match series_keys")
    if any(not isinstance(cost, CostStructure) for cost in snapshot.values()):
        raise SessionIdentityError("every per-series cost_structure value must be a CostStructure")
    return {
        "per_series": {
            series_key: _cost_payload(snapshot[series_key]) for series_key in series_keys
        }
    }


def _cost_payload(cost: CostStructure) -> dict[str, object]:
    return {
        "holding_cost": cost.holding,
        "overage_cost": cost.overage,
        "shortage_cost": cost.shortage,
        "underage_cost": cost.underage,
    }


def _canonical_series_set(series_keys: Iterable[str]) -> tuple[str, ...]:
    if isinstance(series_keys, (str, bytes)):
        raise SessionIdentityError("series_keys must be an iterable of series keys")
    try:
        values = tuple(series_keys)
    except TypeError as error:
        raise SessionIdentityError("series_keys must be an iterable of series keys") from error
    if not values:
        raise SessionIdentityError("series_keys must not be empty")
    normalized = tuple(_require_text(value, name="series key") for value in values)
    if len(set(normalized)) != len(normalized):
        raise SessionIdentityError("series_keys must not contain duplicates")
    return tuple(sorted(normalized, key=str.encode))


def _canonical_decision_series(
    series_keys: Iterable[str],
    *,
    session_series: tuple[str, ...],
) -> tuple[str, ...]:
    if isinstance(series_keys, (str, bytes)):
        raise SessionIdentityError("decision_series_keys must be an iterable of series keys")
    try:
        values = tuple(series_keys)
    except TypeError as error:
        raise SessionIdentityError(
            "decision_series_keys must be an iterable of series keys"
        ) from error
    if not values:
        raise SessionIdentityError("decision_series_keys must not be empty")
    normalized = tuple(_require_text(value, name="decision series key") for value in values)
    if len(set(normalized)) != len(normalized):
        raise SessionIdentityError("decision_series_keys must not contain duplicates")
    foreign = sorted(set(normalized) - set(session_series), key=str.encode)
    if foreign:
        raise SessionIdentityError(
            f"decision_series_keys must be a subset of series_keys; foreign={foreign!r}"
        )
    return tuple(sorted(normalized, key=str.encode))


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SessionIdentityError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise SessionIdentityError(f"{name} must be valid UTF-8") from error
    return value
