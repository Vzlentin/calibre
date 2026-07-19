"""Encode opaque conformal state in strict canonical versioned envelopes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral
from typing import Final

from newcalibre.conformal.types import (
    METHOD_SCOPE_LABEL,
    RuntimeContractError,
    _decode_label,
)
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes

_STATE_SCHEMA: Final = "newcalibre.conformal-state"
_STATE_FIELDS: Final = frozenset(
    {"schema", "schema_version", "method", "scope", "label", "payload"}
)


class StateCodecError(ValueError):
    """Report invalid, mismatched, or non-canonical conformal state bytes."""


class StateScope(StrEnum):
    """Name the two independently persisted conformal state scopes."""

    PARTITION = "partition"
    METHOD = "method"


@dataclass(frozen=True, slots=True)
class StateAddress:
    """Expose only the validated address metadata of one opaque state row."""

    method_name: str
    schema_version: int
    scope: StateScope
    label: str


@dataclass(frozen=True, slots=True)
class JsonStateCodec:
    """Own one method's finite canonical-JSON state envelope boundary."""

    method_name: str
    schema_version: int

    def __post_init__(self) -> None:
        _require_method_name(self.method_name)
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, Integral)
            or self.schema_version < 1
        ):
            raise StateCodecError("state schema version must be a positive integer")
        object.__setattr__(self, "schema_version", int(self.schema_version))

    def scope_for(self, label: str) -> StateScope:
        """Return the structural scope encoded by one validated state label."""
        try:
            scope, _ = _decode_label(label)
        except RuntimeContractError as error:
            raise StateCodecError(str(error)) from error
        return StateScope(scope)

    def encode(self, label: str, payload: object) -> bytes:
        """Encode one independently addressable finite payload as opaque bytes."""
        scope = self.scope_for(label)
        envelope = {
            "label": label,
            "method": self.method_name,
            "payload": payload,
            "schema": _STATE_SCHEMA,
            "schema_version": self.schema_version,
            "scope": scope.value,
        }
        try:
            return canonical_json_bytes(envelope, path="conformal state")
        except CanonicalJsonError as error:
            raise StateCodecError(str(error)) from error

    def decode(self, state: bytes, *, expected_label: str | None = None) -> object:
        """Validate an opaque state row and return its conformal-owned payload."""
        envelope = _decode_envelope(state)
        self._validate_envelope(envelope, expected_label=expected_label)
        return envelope["payload"]

    def address(self, state: bytes) -> StateAddress:
        """Validate a state row and return only its persistence address."""
        envelope = _decode_envelope(state)
        scope, label = self._validate_envelope(envelope, expected_label=None)
        return StateAddress(
            method_name=self.method_name,
            schema_version=self.schema_version,
            scope=scope,
            label=label,
        )

    def _validate_envelope(
        self,
        envelope: dict[str, object],
        *,
        expected_label: str | None,
    ) -> tuple[StateScope, str]:
        if envelope["schema"] != _STATE_SCHEMA:
            raise StateCodecError("state envelope has an unsupported schema")
        if envelope["method"] != self.method_name:
            raise StateCodecError(
                f"state envelope has wrong method {envelope['method']!r}; "
                f"expected {self.method_name!r}"
            )
        version = envelope["schema_version"]
        if type(version) is not int or version != self.schema_version:
            raise StateCodecError(
                "state envelope has unsupported state schema version "
                f"{version!r}; expected {self.schema_version}"
            )
        label = envelope["label"]
        if not isinstance(label, str):
            raise StateCodecError("state envelope label must be a string")
        actual_scope = self.scope_for(label)
        try:
            declared_scope = StateScope(envelope["scope"])
        except (TypeError, ValueError) as error:
            raise StateCodecError("state envelope has an invalid scope") from error
        if actual_scope is not declared_scope:
            expected = "method scope" if declared_scope is StateScope.METHOD else "partition scope"
            raise StateCodecError(f"state envelope label does not belong to declared {expected}")
        if declared_scope is StateScope.METHOD and label != METHOD_SCOPE_LABEL:
            raise StateCodecError("method scope must use the reserved method label")
        if expected_label is not None and label != expected_label:
            raise StateCodecError(
                f"state envelope label {label!r} does not match expected label {expected_label!r}"
            )
        return declared_scope, label


def validate_state_blob(
    state: bytes,
    *,
    method_name: str,
    schema_version: int,
    expected_label: str,
) -> StateAddress:
    """Validate method, version, and address without exposing a state payload."""
    codec = JsonStateCodec(method_name, schema_version)
    address = codec.address(state)
    if address.label != expected_label:
        raise StateCodecError(
            f"state envelope label {address.label!r} does not match expected label "
            f"{expected_label!r}"
        )
    return address


def _decode_envelope(state: object) -> dict[str, object]:
    if not isinstance(state, bytes):
        raise StateCodecError("conformal state must be immutable bytes")
    try:
        text = state.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StateCodecError("conformal state must be valid UTF-8") from error
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except _DuplicateFieldError as error:
        raise StateCodecError(str(error)) from error
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise StateCodecError("conformal state must contain strict JSON") from error
    if not isinstance(value, dict) or set(value) != _STATE_FIELDS:
        raise StateCodecError("state envelope must contain exact fields")
    try:
        canonical = canonical_json_bytes(value, path="conformal state")
    except CanonicalJsonError as error:
        raise StateCodecError(str(error)) from error
    if state != canonical:
        raise StateCodecError("conformal state must use canonical JSON encoding")
    return value


class _DuplicateFieldError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateFieldError(f"conformal state contains duplicate field {key!r}")
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _require_method_name(value: object) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise StateCodecError("state method name must be a non-empty trimmed string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise StateCodecError("state method name must be valid UTF-8") from error
