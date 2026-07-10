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

_IDENTITY_SCHEMA = "newcalibre.session-identity"
_IDENTITY_VERSION = 1


class SessionIdentityError(ValueError):
    """Report an invalid session-defining input."""


@dataclass(frozen=True, slots=True, init=False)
class SessionIdentity:
    """Content-address one logical forecasting and decision lifecycle."""

    _value: str
    _payload: bytes = field(repr=False, compare=False, hash=False)

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
        cost_structure: CostStructure | None = None,
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
        normalized_decision = _canonical_decision_config(
            ordering_policy=ordering_policy,
            cost_structure=cost_structure,
        )

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
        return instance

    @property
    def value(self) -> str:
        """Return the complete lowercase SHA-256 hex digest."""
        return self._value

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
    cost_structure: CostStructure | None,
) -> dict[str, object] | None:
    if ordering_policy is None and cost_structure is None:
        return None
    if ordering_policy is None or cost_structure is None:
        raise SessionIdentityError(
            "ordering_policy and cost_structure must both be supplied or both be absent"
        )
    if not isinstance(cost_structure, CostStructure):
        raise SessionIdentityError("cost_structure must be a CostStructure")

    return {
        "cost_structure": {
            "holding_cost": cost_structure.holding,
            "overage_cost": cost_structure.overage,
            "shortage_cost": cost_structure.shortage,
            "underage_cost": cost_structure.underage,
        },
        "ordering_policy": _canonical_config(
            ordering_policy,
            name="ordering_policy",
        ),
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


def _require_text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SessionIdentityError(f"{name} must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise SessionIdentityError(f"{name} must be valid UTF-8") from error
    return value
