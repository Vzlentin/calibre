"""Define immutable runtime values and collision-proof state addressing."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Final, cast

import pandas as pd

from newcalibre.conformal.manifest import EmissionForm
from newcalibre.domain import (
    FRAME_KEY_COLUMNS,
    AppliedBinding,
    CensoringAssertion,
    DecisionScope,
    EmissionScope,
    GuaranteeDescriptor,
    GuaranteeType,
    interval_columns,
)
from newcalibre.domain._canonical_json import CanonicalJsonError, canonical_json_bytes
from newcalibre.domain.forecast_frame import ForecastFrameError, forecast_bound_groups

_PARTITION_PREFIX: Final = "p1."
_METHOD_PREFIX: Final = "m1."
_METHOD_SCOPE_PAYLOAD: Final = {
    "namespace": "newcalibre.conformal",
    "scope": "method",
    "version": 1,
}


class RuntimeContractError(ValueError):
    """Report an invalid conformal runtime value or call contract."""


class StateLabel(str):
    """Address conformal state with a label that is canonical by construction.

    Only :func:`_encoded_label` mints one, so holding an instance is itself the
    proof of canonical encoding, and the decoded payload travels with it. A
    ``str`` subclass keeps every dict key, comparison, sort, and JSON write of a
    label unchanged. Use :func:`require_state_label` at every ingress where an
    untrusted string arrives.
    """

    __slots__ = ("payload", "scope")

    scope: str
    payload: object

    def __new__(cls, text: str, *, scope: str, payload: object) -> StateLabel:
        label = super().__new__(cls, text)
        label.scope = scope
        label.payload = payload
        return label


def _encoded_label(prefix: str, payload: object) -> StateLabel:
    try:
        raw = canonical_json_bytes(payload, path="conformal state label")
    except CanonicalJsonError as error:
        raise RuntimeContractError(str(error)) from error
    token = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    scope = "method" if prefix == _METHOD_PREFIX else "partition"
    return StateLabel(f"{prefix}{token}", scope=scope, payload=payload)


METHOD_SCOPE_LABEL: Final = _encoded_label(_METHOD_PREFIX, _METHOD_SCOPE_PAYLOAD)


def derive_partition_label(
    model_name: str,
    partition_value: str | bool | int | float,
    horizon_scope: EmissionScope,
    *,
    horizon_step: int | None = None,
) -> StateLabel:
    """Derive an injective label from model, typed partition, scope, and step."""
    model = _require_text(model_name, name="model name", trimmed=True)
    if not isinstance(horizon_scope, EmissionScope):
        raise RuntimeContractError("horizon scope must be an EmissionScope")
    tag, value = _partition_value(partition_value)
    payload: dict[str, object] = {
        "horizon_scope": horizon_scope.value,
        "model_name": model,
        "partition_value": {"tag": tag, "value": value},
    }
    if horizon_step is not None:
        if isinstance(horizon_step, bool) or not isinstance(horizon_step, Integral):
            raise RuntimeContractError("partition horizon step must be a positive integer")
        if horizon_step < 1:
            raise RuntimeContractError("partition horizon step must be a positive integer")
        payload["horizon_step"] = int(horizon_step)
    return _encoded_label(_PARTITION_PREFIX, payload)


def _partition_value(value: object) -> tuple[str, object]:
    if isinstance(value, str):
        return "string", _require_text(value, name="partition value")
    if isinstance(value, bool):
        return "boolean", value
    if isinstance(value, Integral):
        return "integer", int(value)
    if isinstance(value, Real):
        normalized = _finite_real(value, name="partition value")
        if normalized == 0.0:
            normalized = 0.0
        return "float", float(normalized).hex()
    raise RuntimeContractError(
        "partition value must be a string, boolean, integer, or finite float"
    )


def require_state_label(label: object) -> StateLabel:
    """Prove one label's canonicity, unless it is already a :class:`StateLabel`."""
    if isinstance(label, StateLabel):
        return label
    text = _require_text(label, name="state label")
    if text.startswith(_PARTITION_PREFIX):
        prefix = _PARTITION_PREFIX
        scope = "partition"
    elif text.startswith(_METHOD_PREFIX):
        prefix = _METHOD_PREFIX
        scope = "method"
    else:
        raise RuntimeContractError("state label has an unknown structural namespace")
    token = text.removeprefix(prefix)
    if not token:
        raise RuntimeContractError("state label payload must not be empty")
    try:
        padding = "=" * (-len(token) % 4)
        raw = base64.b64decode(
            token + padding,
            altchars=b"-_",
            validate=True,
        )
        decoded = raw.decode("utf-8")
    except (UnicodeError, ValueError) as error:
        raise RuntimeContractError(
            "state label payload must be canonical base64url UTF-8"
        ) from error

    try:
        payload = json.loads(decoded)
        canonical = canonical_json_bytes(payload, path="conformal state label")
    except (CanonicalJsonError, json.JSONDecodeError) as error:
        raise RuntimeContractError("state label payload must be canonical JSON") from error
    # Comparing the re-encoded token rejects the non-zero trailing bits that
    # base64 validation admits, without a second canonical-JSON encode.
    canonical_token = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    if raw != canonical or canonical_token != token:
        raise RuntimeContractError("state label payload must use canonical encoding")
    if scope == "method":
        if payload != _METHOD_SCOPE_PAYLOAD or text != METHOD_SCOPE_LABEL:
            raise RuntimeContractError("method-scope state label is not the reserved label")
    else:
        _validate_partition_payload(payload)
    return StateLabel(text, scope=scope, payload=payload)


def _validate_partition_payload(payload: object) -> None:
    base_fields = {"horizon_scope", "model_name", "partition_value"}
    if not isinstance(payload, dict) or set(payload) not in (
        base_fields,
        {*base_fields, "horizon_step"},
    ):
        raise RuntimeContractError("partition state label has malformed fields")
    fields = cast(dict[str, object], payload)
    _require_text(fields["model_name"], name="partition label model name", trimmed=True)
    try:
        EmissionScope(fields["horizon_scope"])
    except (TypeError, ValueError) as error:
        raise RuntimeContractError("partition state label has an invalid horizon scope") from error
    horizon_step = fields.get("horizon_step")
    if horizon_step is not None and (
        isinstance(horizon_step, bool) or not isinstance(horizon_step, int) or horizon_step < 1
    ):
        raise RuntimeContractError("partition state label has an invalid horizon step")
    value = fields["partition_value"]
    if not isinstance(value, dict) or set(value) != {"tag", "value"}:
        raise RuntimeContractError("partition state label has a malformed typed value")
    typed_value = cast(dict[str, object], value)
    tag = typed_value["tag"]
    scalar = typed_value["value"]
    valid = (
        (tag == "string" and isinstance(scalar, str) and bool(scalar))
        or (tag == "boolean" and isinstance(scalar, bool))
        or (tag == "integer" and isinstance(scalar, int) and not isinstance(scalar, bool))
        or (tag == "float" and isinstance(scalar, str) and _is_canonical_float_hex(scalar))
    )
    if not valid:
        raise RuntimeContractError("partition state label has a malformed typed value")


def _is_canonical_float_hex(value: str) -> bool:
    try:
        parsed = float.fromhex(value)
    except ValueError:
        return False
    if not math.isfinite(parsed):
        return False
    if parsed == 0.0:
        parsed = 0.0
    return parsed.hex() == value


@dataclass(frozen=True, slots=True)
class ForecastKey:
    """Identify one forecast row completely."""

    series_key: str
    origin: pd.Timestamp
    horizon_step: int
    model_name: str

    def __post_init__(self) -> None:
        _require_text(self.series_key, name="series key")
        _require_timestamp(self.origin, name="origin")
        if (
            isinstance(self.horizon_step, bool)
            or not isinstance(self.horizon_step, Integral)
            or self.horizon_step < 1
        ):
            raise RuntimeContractError("horizon step must be a positive integer")
        _require_text(self.model_name, name="model name")
        object.__setattr__(self, "horizon_step", int(self.horizon_step))


@dataclass(frozen=True, slots=True)
class IssuedBoundFacts:
    """Carry the immutable calibration facts recorded when one row was issued."""

    method_name: str
    emission_form: EmissionForm
    emission_scope: EmissionScope
    partition_label: StateLabel
    working_level: float
    state_reference: str
    lower_bound: float
    upper_bound: float
    calibration_ready: bool
    bounds_null_reason: str | None
    effective_descriptor: GuaranteeDescriptor
    bindings: tuple[AppliedBinding, ...] = ()

    @classmethod
    def snapshot(cls, facts: IssuedBoundFacts) -> IssuedBoundFacts:
        """Return an exact immutable snapshot of issued bound facts."""
        # Every field is normalized and deeply immutable once __post_init__ has
        # run, so an instance of exactly this class already is its own snapshot.
        if type(facts) is cls:
            return facts
        if not isinstance(facts, cls):
            raise RuntimeContractError("issuance metadata must contain IssuedBoundFacts")
        return cls(
            method_name=facts.method_name,
            emission_form=facts.emission_form,
            emission_scope=facts.emission_scope,
            partition_label=facts.partition_label,
            working_level=facts.working_level,
            state_reference=facts.state_reference,
            lower_bound=facts.lower_bound,
            upper_bound=facts.upper_bound,
            calibration_ready=facts.calibration_ready,
            bounds_null_reason=facts.bounds_null_reason,
            effective_descriptor=facts.effective_descriptor,
            bindings=facts.bindings,
        )

    def __post_init__(self) -> None:
        _require_text(self.method_name, name="issued method name", trimmed=True)
        if not isinstance(self.emission_form, EmissionForm):
            raise RuntimeContractError("issued emission form must be an EmissionForm")
        if not isinstance(self.emission_scope, EmissionScope):
            raise RuntimeContractError("issued emission scope must be an EmissionScope")
        label = self.partition_label
        if not isinstance(label, StateLabel) or label.scope != "partition":
            raise RuntimeContractError("issued partition label must be a data-derived label")
        level = _finite_real(self.working_level, name="working level")
        _require_text(self.state_reference, name="state reference")
        lower = _finite_or_nan_real(self.lower_bound, name="lower bound")
        upper = _finite_or_nan_real(self.upper_bound, name="upper bound")
        lower_finite = math.isfinite(lower)
        upper_finite = math.isfinite(upper)
        if lower_finite != upper_finite:
            raise RuntimeContractError("issued lower and upper bounds must be finite together")
        if not isinstance(self.calibration_ready, bool):
            raise RuntimeContractError("calibration readiness must be a boolean")
        if lower_finite:
            if lower > upper:
                raise RuntimeContractError("issued lower bound cannot exceed upper bound")
            if not self.calibration_ready:
                raise RuntimeContractError("finite bounds require calibration readiness")
            if self.bounds_null_reason is not None:
                raise RuntimeContractError("finite bounds cannot carry a bounds null reason")
        else:
            _require_text(self.bounds_null_reason, name="bounds null reason")
        descriptor = _snapshot_descriptor(self.effective_descriptor)
        if descriptor.window is not self.emission_scope:
            raise RuntimeContractError(
                "effective descriptor window must equal the issued emission scope"
            )
        bindings = _snapshot_bindings(self.bindings)
        if len({binding.name for binding in bindings}) != len(bindings):
            raise RuntimeContractError("issued binding names must be unique")
        object.__setattr__(self, "working_level", level)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "effective_descriptor", descriptor)
        object.__setattr__(self, "bindings", bindings)


@dataclass(frozen=True, slots=True)
class ResolvedObservation:
    """Carry one resolved forecast row into a conformal observe call."""

    forecast_key: ForecastKey
    target_timestamp: pd.Timestamp
    actual: float
    point_forecast: float
    censoring_assertion: CensoringAssertion | None
    availability_bound: float | None
    issued: IssuedBoundFacts

    def __post_init__(self) -> None:
        if not isinstance(self.forecast_key, ForecastKey):
            raise RuntimeContractError("forecast key must be a ForecastKey")
        _require_timestamp(self.target_timestamp, name="target timestamp")
        actual = _finite_real(self.actual, name="actual")
        point = _finite_real(self.point_forecast, name="point forecast")
        if self.censoring_assertion is not None and not isinstance(
            self.censoring_assertion,
            CensoringAssertion,
        ):
            raise RuntimeContractError(
                "censoring assertion must be a CensoringAssertion or undeclared"
            )
        availability = self.availability_bound
        if availability is not None:
            availability = _finite_real(availability, name="availability bound")
        if not isinstance(self.issued, IssuedBoundFacts):
            raise RuntimeContractError("issued facts must be IssuedBoundFacts")
        object.__setattr__(self, "actual", actual)
        object.__setattr__(self, "point_forecast", point)
        object.__setattr__(self, "availability_bound", availability)


@dataclass(frozen=True, slots=True)
class CalibrationContext:
    """Expose only row-aligned immutable facts from the declared hierarchy."""

    series_keys: tuple[str, ...]
    lattice_levels: tuple[str, ...]
    aggregate_memberships: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        keys = _snapshot_iterable(self.series_keys, name="context series keys")
        levels = _snapshot_iterable(self.lattice_levels, name="context lattice levels")
        memberships_raw = _snapshot_iterable(
            self.aggregate_memberships,
            name="context aggregate memberships",
        )
        if len(keys) != len(levels) or len(keys) != len(memberships_raw):
            raise RuntimeContractError("calibration context facts must be row-aligned")
        for value in keys:
            _require_text(value, name="context series key")
        for value in levels:
            _require_text(value, name="context lattice level")
        memberships: list[tuple[str, ...]] = []
        for row in memberships_raw:
            values = _snapshot_iterable(row, name="aggregate membership row")
            for value in values:
                _require_text(value, name="aggregate membership")
            if len(set(values)) != len(values):
                raise RuntimeContractError("aggregate memberships must be unique per row")
            memberships.append(values)
        object.__setattr__(self, "series_keys", keys)
        object.__setattr__(self, "lattice_levels", levels)
        object.__setattr__(self, "aggregate_memberships", tuple(memberships))

    @property
    def class_assignments(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Derive one class assignment per row solely from hierarchy facts."""
        return tuple(zip(self.lattice_levels, self.aggregate_memberships, strict=True))


@dataclass(frozen=True, slots=True)
class ObserveAnnotation:
    """Record one scored or attributable excluded observation."""

    forecast_key: ForecastKey
    score: float | None
    exclusion_cause: str | None
    advanced_delivered_score: bool

    def __post_init__(self) -> None:
        if not isinstance(self.forecast_key, ForecastKey):
            raise RuntimeContractError("annotation forecast key must be a ForecastKey")
        has_score = self.score is not None
        has_cause = self.exclusion_cause is not None
        if has_score == has_cause:
            raise RuntimeContractError(
                "an observe annotation requires exactly one of score or exclusion cause"
            )
        if has_score:
            object.__setattr__(self, "score", _finite_real(self.score, name="annotation score"))
        else:
            _require_text(self.exclusion_cause, name="exclusion cause")
        if not isinstance(self.advanced_delivered_score, bool):
            raise RuntimeContractError("delivered-score advancement must be a boolean")
        if self.advanced_delivered_score and not has_score:
            raise RuntimeContractError("only a scored observation may advance delivered score")


def _snapshot_issuances(
    forecasts: pd.DataFrame,
    issuances: Mapping[ForecastKey, IssuedBoundFacts] | None,
) -> dict[ForecastKey, IssuedBoundFacts]:
    try:
        interval_groups = tuple(
            group for group in forecast_bound_groups(forecasts.columns) if len(group) == 2
        )
    except ForecastFrameError as error:
        raise RuntimeContractError(str(error)) from error
    supplied = {} if issuances is None else issuances
    if not isinstance(supplied, Mapping):
        raise RuntimeContractError("calibration issuances must be a mapping")
    if not interval_groups:
        if supplied:
            raise RuntimeContractError("issuance metadata requires calibrated interval bounds")
        return {}
    if len(interval_groups) != 1:
        raise RuntimeContractError("calibration results must own exactly one interval pair")
    missing_columns = [column for column in FRAME_KEY_COLUMNS if column not in forecasts]
    if missing_columns:
        raise RuntimeContractError(
            "calibrated interval bounds require complete forecast key columns"
        )
    expected_keys = tuple(
        ForecastKey(
            series_key=row["series_key"],
            origin=pd.Timestamp(row["origin"]),
            horizon_step=row["horizon_step"],
            model_name=row["model_name"],
        )
        for row in forecasts.loc[:, list(FRAME_KEY_COLUMNS)].to_dict("records")
    )
    if len(set(expected_keys)) != len(expected_keys):
        raise RuntimeContractError("calibrated forecasts contain duplicate forecast keys")
    snapshot = dict(supplied)
    if set(snapshot) != set(expected_keys):
        raise RuntimeContractError(
            "calibration issuances must exactly cover calibrated forecast keys"
        )
    lower_column, upper_column = interval_groups[0]
    lower_values = forecasts[lower_column].to_numpy(dtype=float)
    upper_values = forecasts[upper_column].to_numpy(dtype=float)
    matching_levels: set[float] = set()
    frozen: dict[ForecastKey, IssuedBoundFacts] = {}
    for position, key in enumerate(expected_keys):
        facts = IssuedBoundFacts.snapshot(snapshot[key])
        level = facts.effective_descriptor.level
        if level not in matching_levels:
            if interval_columns(level) != (lower_column, upper_column):
                raise RuntimeContractError(
                    "issuance descriptor level must identify the calibrated interval columns"
                )
            matching_levels.add(level)
        if not _same_bound(float(lower_values[position]), facts.lower_bound) or not _same_bound(
            float(upper_values[position]),
            facts.upper_bound,
        ):
            raise RuntimeContractError("issuance bounds must equal the calibrated frame row")
        frozen[key] = facts
    return frozen


def _same_bound(left: float, right: float) -> bool:
    return left == right or (math.isnan(left) and math.isnan(right))


def _snapshot_descriptor(value: object) -> GuaranteeDescriptor:
    if not isinstance(value, GuaranteeDescriptor):
        raise RuntimeContractError("effective descriptor must be a GuaranteeDescriptor")
    descriptor_type = GuaranteeType(
        claim=value.type.claim,
        currency=value.type.currency,
        declared_slack=value.type.declared_slack,
    )
    scope = DecisionScope(
        kind=value.scope.kind,
        class_system_name=value.scope.class_system_name,
    )
    return GuaranteeDescriptor(
        type=descriptor_type,
        level=value.level,
        scored_series=value.scored_series,
        window=value.window,
        scope=scope,
    )


def _snapshot_bindings(values: object) -> tuple[AppliedBinding, ...]:
    bindings = _snapshot_iterable(values, name="issued bindings")
    if any(not isinstance(value, AppliedBinding) for value in bindings):
        raise RuntimeContractError("issued bindings must contain AppliedBinding values")
    return tuple(
        AppliedBinding(name=value.name, value=value.value, bound=value.bound) for value in bindings
    )


def _snapshot_iterable(values: object, *, name: str) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise RuntimeContractError(f"{name} must be an iterable of values")
    return tuple(values)


def _require_timestamp(value: object, *, name: str) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp) or pd.isna(value):
        raise RuntimeContractError(f"{name} must be a pandas Timestamp")
    if value.tz is not None:
        raise RuntimeContractError(f"{name} must be timezone-naive")
    return value


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeContractError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise RuntimeContractError(f"{name} must be a finite real number") from error
    if not math.isfinite(normalized):
        raise RuntimeContractError(f"{name} must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized


def _finite_or_nan_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeContractError(f"{name} must be a real number or NaN")
    normalized = float(value)
    if math.isinf(normalized):
        raise RuntimeContractError(f"{name} cannot be infinite")
    return 0.0 if normalized == 0.0 else normalized


def _require_text(value: object, *, name: str, trimmed: bool = False) -> str:
    if not isinstance(value, str) or not value or (trimmed and value != value.strip()):
        qualifier = " non-empty trimmed" if trimmed else " non-empty"
        raise RuntimeContractError(f"{name} must be a{qualifier} string")
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise RuntimeContractError(f"{name} must be valid UTF-8") from error
    return value
