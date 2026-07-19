"""Implement recency-weighted one-sided upper conformal bounds per step."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from newcalibre.conformal.manifest import (
    AssumptionClass,
    CensoringPolicy,
    ConservativeRankRequirement,
    EmissionForm,
    GuaranteeDeclaration,
    JointClaim,
    MethodManifest,
    PostWarmupNonFinite,
)
from newcalibre.conformal.runtime import require_calibration_context
from newcalibre.conformal.state import JsonStateCodec, StateCodecError, StateScope
from newcalibre.conformal.types import (
    METHOD_SCOPE_LABEL,
    CalibrationContext,
    CalibrationResult,
    Delivery,
    ForecastKey,
    IssuedBoundFacts,
    ObserveAnnotation,
    ObserveEffect,
    RuntimeContractError,
    _decode_label,
    derive_partition_label,
)
from newcalibre.domain import (
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    CensoringAssertion,
    DecisionScope,
    DecisionScopeKind,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
    GuaranteeDescriptor,
    GuaranteeType,
    ScoredSeries,
    interval_columns,
)

WEIGHTED_PER_STEP = "weighted-per-step"
_MAX_CALIBRATION_WINDOW = 5000
_STATE_SCHEMA_VERSION = 1
_WARMUP_CAUSE = "warm-up"
_HELD_OUT_WEIGHT_CAUSE = "held-out-weight-mass"
_CENSORED_CAUSE = "declared-censored"


class WeightedPerStepConfig(BaseModel):
    """Configure recency-weighted one-sided conformal bounds per step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: float = Field(default=0.9, gt=0.0, lt=1.0, allow_inf_nan=False)
    calibration_window: int = Field(
        default=_MAX_CALIBRATION_WINDOW,
        ge=1,
        le=_MAX_CALIBRATION_WINDOW,
    )
    partition_by: Literal["global", "series"] = "global"
    weight_decay: float = Field(default=0.99, gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _valid_rank(self) -> WeightedPerStepConfig:
        minimum = ConservativeRankRequirement().minimum_scores(self)
        if minimum > self.calibration_window:
            raise ValueError("calibration_window cannot satisfy the configured conservative rank")
        return self


_WEIGHTED_GUARANTEE = GuaranteeDeclaration(
    claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
    currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
)

WEIGHTED_PER_STEP_MANIFEST = MethodManifest(
    name=WEIGHTED_PER_STEP,
    emission_form=EmissionForm.ONE_SIDED_UPPER,
    emission_scope=EmissionScope.PER_STEP,
    guarantees=(_WEIGHTED_GUARANTEE,),
    assumption_class=AssumptionClass.WEIGHTED,
    calibration_requirement=ConservativeRankRequirement(),
    order_sensitive=True,
    censoring_policy=CensoringPolicy.CONSUMES_CENSORING_FACTS,
    imputation_policy=None,
    state_bound=_MAX_CALIBRATION_WINDOW,
    state_schema_version=_STATE_SCHEMA_VERSION,
    consumes_calibration_context=False,
    hosted_submodels=(),
    requires_fitted_values=False,
    post_warmup_non_finite=PostWarmupNonFinite.ALLOWED_WITH_ATTRIBUTION,
    clamps=(),
    joint_claim=JointClaim.NONE,
)


@dataclass(frozen=True, slots=True)
class _PartitionState:
    scores: tuple[float, ...]
    delivered_score_count: int
    scored_series: ScoredSeries


@dataclass(frozen=True, slots=True)
class _MethodState:
    issue_counter: int


class _WeightedStateCodec:
    def __init__(self) -> None:
        self._codec = JsonStateCodec(
            WEIGHTED_PER_STEP_MANIFEST.name,
            WEIGHTED_PER_STEP_MANIFEST.state_schema_version,
        )

    def encode_partition(self, label: str, state: _PartitionState) -> bytes:
        return self._codec.encode(
            label,
            {
                "delivered_score_count": state.delivered_score_count,
                "scored_series": state.scored_series.value,
                "scores": list(state.scores),
            },
        )

    def decode_partition(self, state: bytes, *, label: str) -> _PartitionState:
        payload = self._codec.decode(state, expected_label=label)
        if not isinstance(payload, dict) or set(payload) != {
            "delivered_score_count",
            "scored_series",
            "scores",
        }:
            raise StateCodecError("weighted partition state must contain exact fields")
        fields = cast(dict[str, object], payload)
        scores_raw = fields["scores"]
        if not isinstance(scores_raw, list):
            raise StateCodecError("weighted partition scores must be a list")
        if len(scores_raw) > WEIGHTED_PER_STEP_MANIFEST.state_bound:
            raise StateCodecError("weighted partition scores exceed the declared state bound")
        scores = tuple(_state_score(value) for value in scores_raw)
        delivered = fields["delivered_score_count"]
        if isinstance(delivered, bool) or not isinstance(delivered, int) or delivered < len(scores):
            raise StateCodecError(
                "weighted delivered-score count must be an integer no smaller than retained scores"
            )
        try:
            scored_series = ScoredSeries(fields["scored_series"])
        except (TypeError, ValueError) as error:
            raise StateCodecError("weighted scored-series label is invalid") from error
        return _PartitionState(scores, delivered, scored_series)

    def encode_method(self, state: _MethodState) -> bytes:
        return self._codec.encode(
            METHOD_SCOPE_LABEL,
            {"issue_counter": state.issue_counter},
        )

    def decode_method(self, state: bytes) -> _MethodState:
        payload = self._codec.decode(state, expected_label=METHOD_SCOPE_LABEL)
        if not isinstance(payload, dict) or set(payload) != {"issue_counter"}:
            raise StateCodecError("weighted method state must contain exact fields")
        fields = cast(dict[str, object], payload)
        counter = fields["issue_counter"]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise StateCodecError("weighted issue counter must be a nonnegative integer")
        return _MethodState(counter)

    def scope_for(self, label: str) -> StateScope:
        scope = self._codec.scope_for(label)
        if scope is StateScope.PARTITION:
            try:
                _namespace, payload = _decode_label(label)
            except RuntimeContractError as error:
                raise StateCodecError(str(error)) from error
            fields = cast(dict[str, object], payload)
            if fields["horizon_scope"] != EmissionScope.PER_STEP.value:
                raise StateCodecError(
                    "weighted partition label has the wrong method emission scope"
                )
        return scope


class WeightedConformalRuntime:
    """Apply recency-weighted conformal bounds through the stable runtime seam."""

    def __init__(
        self,
        config: WeightedPerStepConfig,
        states: Mapping[str, bytes],
    ) -> None:
        if type(config) is not WeightedPerStepConfig:
            raise RuntimeContractError("weighted runtime configuration does not match its manifest")
        self._config = config
        self._codec = _WeightedStateCodec()
        self._validate_states(states, allow_missing=False)

    @property
    def manifest(self) -> MethodManifest:
        """Return the registered immutable weighted-method declaration."""
        return WEIGHTED_PER_STEP_MANIFEST

    @property
    def config(self) -> BaseModel:
        """Return the frozen weighted runtime configuration."""
        return self._config

    def calibrate(self, scores: Mapping[str, Sequence[float]]) -> Mapping[str, bytes]:
        """Seed bounded chronological partition states from finite scores."""
        if not isinstance(scores, Mapping):
            raise RuntimeContractError("calibration scores must be a mapping")
        updates: dict[str, bytes] = {}
        for label, values in scores.items():
            try:
                scope = self._codec.scope_for(label)
            except StateCodecError as error:
                raise RuntimeContractError(str(error)) from error
            if scope is not StateScope.PARTITION:
                raise RuntimeContractError("calibration scores must use partition labels")
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise RuntimeContractError("partition calibration scores must be a sequence")
            normalized = tuple(_finite_score(value) for value in values)
            retained = normalized[-self._config.calibration_window :]
            updates[label] = self._codec.encode_partition(
                label,
                _PartitionState(
                    scores=retained,
                    delivered_score_count=len(normalized),
                    scored_series=ScoredSeries.DEMAND_HONEST,
                ),
            )
        updates[METHOD_SCOPE_LABEL] = self._codec.encode_method(_MethodState(0))
        return MappingProxyType(updates)

    def apply(
        self,
        forecasts: pd.DataFrame,
        states: Mapping[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        """Emit weighted upper bounds without reading actual-value columns."""
        if not isinstance(forecasts, pd.DataFrame):
            raise RuntimeContractError("weighted apply forecasts must be a pandas DataFrame")
        if forecasts.columns.has_duplicates:
            raise RuntimeContractError("weighted apply forecasts cannot have duplicate columns")
        required = (SERIES_KEY, POINT_FORECAST, HORIZON_STEP, ORIGIN, MODEL_NAME)
        missing = [column for column in required if column not in forecasts]
        if missing:
            raise RuntimeContractError(
                "weighted apply forecasts are missing required columns: " + ", ".join(missing)
            )
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(forecasts[SERIES_KEY]),
        )
        decoded = self._validate_states(states, allow_missing=True)
        rows = _apply_rows(forecasts)
        lower_column, upper_column = interval_columns(self._config.coverage)
        lower_values: list[float] = []
        upper_values: list[float] = []
        issuances: dict[ForecastKey, IssuedBoundFacts] = {}
        minimum = self.manifest.minimum_calibration_scores(self._config)
        quantiles: dict[str, float | None] = {}
        state_references: dict[str, str] = {}
        descriptors: dict[ScoredSeries, GuaranteeDescriptor] = {}

        for row in rows:
            label = self._partition_label(row.key.model_name, row.key.series_key)
            partition = decoded.get(label)
            if partition is None:
                partition = _empty_partition()
            ready = len(partition.scores) >= minimum
            if not ready:
                lower = math.nan
                upper = math.nan
                null_reason = _WARMUP_CAUSE
            else:
                if label not in quantiles:
                    quantiles[label] = _weighted_quantile(
                        partition.scores,
                        coverage=self._config.coverage,
                        decay=self._config.weight_decay,
                    )
                radius = quantiles[label]
                if radius is None:
                    lower = math.nan
                    upper = math.nan
                    null_reason = _HELD_OUT_WEIGHT_CAUSE
                else:
                    lower = 0.0
                    upper = row.point_forecast + radius
                    if not math.isfinite(upper):
                        raise RuntimeContractError(
                            "ready weighted arithmetic must produce a finite bound"
                        )
                    if upper < lower:
                        raise RuntimeContractError(
                            "weighted upper bound cannot fall below the zero support bound"
                        )
                    null_reason = None
            if label not in state_references:
                state_references[label] = _state_reference(
                    self.manifest.name,
                    self._codec.encode_partition(label, partition),
                )
            if partition.scored_series not in descriptors:
                descriptors[partition.scored_series] = self._descriptor(partition.scored_series)
            facts = IssuedBoundFacts(
                method_name=self.manifest.name,
                emission_form=self.manifest.emission_form,
                emission_scope=self.manifest.emission_scope,
                partition_label=label,
                working_level=self._config.coverage,
                state_reference=state_references[label],
                lower_bound=lower,
                upper_bound=upper,
                calibration_ready=ready,
                bounds_null_reason=null_reason,
                effective_descriptor=descriptors[partition.scored_series],
                bindings=(),
            )
            lower_values.append(lower)
            upper_values.append(upper)
            issuances[row.key] = facts

        calibrated = forecasts.copy(deep=True)
        calibrated[lower_column] = pd.Series(lower_values, index=calibrated.index, dtype="float64")
        calibrated[upper_column] = pd.Series(upper_values, index=calibrated.index, dtype="float64")
        method_state = self._method_state(states)
        updates = {
            METHOD_SCOPE_LABEL: self._codec.encode_method(
                _MethodState(method_state.issue_counter + len(rows))
            )
        }
        return CalibrationResult(calibrated, updates, issuances)

    def observe(
        self,
        delivery: Delivery,
        states: Mapping[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        """Append accepted scores in canonical delivery order."""
        if not isinstance(delivery, Delivery):
            raise RuntimeContractError("weighted observe delivery must be a Delivery")
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(
                observation.forecast_key.series_key for observation in delivery.observations
            ),
        )
        self._validate_delivery_issuance(delivery)
        decoded = self._validate_states(states, allow_missing=True)
        partition = decoded.get(delivery.partition_label)
        if partition is None:
            partition = _empty_partition()
        scores = list(partition.scores)
        count = partition.delivered_score_count
        scored_series = partition.scored_series
        annotations: list[ObserveAnnotation] = []
        for observation in delivery.observations:
            if observation.censoring_assertion is CensoringAssertion.CENSORED:
                annotations.append(
                    ObserveAnnotation(observation.forecast_key, None, _CENSORED_CAUSE, False)
                )
                continue
            score = abs(observation.actual - observation.point_forecast)
            scores.append(score)
            count += 1
            if observation.censoring_assertion is None:
                scored_series = ScoredSeries.RECORDED_SALES
            annotations.append(ObserveAnnotation(observation.forecast_key, score, None, True))
        bounded_scores = scores[-self._config.calibration_window :]
        updated = _PartitionState(tuple(bounded_scores), count, scored_series)
        state = self._codec.encode_partition(delivery.partition_label, updated)
        return ObserveEffect({delivery.partition_label: state}, tuple(annotations))

    def _partition_label(self, model_name: str, series_key: str) -> str:
        value = "global" if self._config.partition_by == "global" else series_key
        return derive_partition_label(model_name, value, self.manifest.emission_scope)

    def _validate_states(
        self,
        states: Mapping[str, bytes | None],
        *,
        allow_missing: bool,
    ) -> dict[str, _PartitionState]:
        if not isinstance(states, Mapping):
            raise RuntimeContractError("weighted states must be a mapping")
        partitions: dict[str, _PartitionState] = {}
        for label, value in dict(states).items():
            if value is None:
                if allow_missing:
                    continue
                raise RuntimeContractError("restored weighted states must contain bytes")
            if not isinstance(value, bytes):
                raise RuntimeContractError("weighted state values must be immutable bytes")
            try:
                scope = self._codec.scope_for(label)
                if scope is StateScope.METHOD:
                    self._codec.decode_method(value)
                    continue
                partition = self._codec.decode_partition(value, label=label)
            except StateCodecError as error:
                raise RuntimeContractError(str(error)) from error
            if len(partition.scores) > self._config.calibration_window:
                raise RuntimeContractError(
                    "restored weighted scores exceed the configured calibration window"
                )
            partitions[label] = partition
        return partitions

    def _method_state(self, states: Mapping[str, bytes | None]) -> _MethodState:
        value = states.get(METHOD_SCOPE_LABEL)
        if value is None:
            return _MethodState(0)
        if not isinstance(value, bytes):
            raise RuntimeContractError("weighted method state must be immutable bytes")
        try:
            return self._codec.decode_method(value)
        except StateCodecError as error:
            raise RuntimeContractError(str(error)) from error

    def _descriptor(self, scored_series: ScoredSeries) -> GuaranteeDescriptor:
        return GuaranteeDescriptor(
            type=GuaranteeType(
                claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
                currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                declared_slack=None,
            ),
            level=self._config.coverage,
            scored_series=scored_series,
            window=self.manifest.emission_scope,
            scope=DecisionScope(DecisionScopeKind.PER_DECISION_NODE, None),
        )

    def _validate_delivery_issuance(self, delivery: Delivery) -> None:
        for observation in delivery.observations:
            issued = observation.issued
            if issued.method_name != self.manifest.name:
                raise RuntimeContractError("delivered issuance has the wrong weighted method")
            if issued.emission_form is not self.manifest.emission_form:
                raise RuntimeContractError("delivered issuance has the wrong emission form")
            if issued.emission_scope is not self.manifest.emission_scope:
                raise RuntimeContractError("delivered issuance has the wrong emission scope")
            if issued.working_level != self._config.coverage:
                raise RuntimeContractError("delivered issuance has the wrong working level")
            expected = self._partition_label(
                observation.forecast_key.model_name,
                observation.forecast_key.series_key,
            )
            if expected != delivery.partition_label:
                raise RuntimeContractError(
                    "delivered forecast does not belong to the declared partition"
                )


@dataclass(frozen=True, slots=True)
class _ApplyRow:
    key: ForecastKey
    point_forecast: float


def build_weighted_per_step(
    config: BaseModel,
    states: Mapping[str, bytes],
) -> WeightedConformalRuntime:
    """Construct a fresh weighted runtime through its registered factory."""
    if not isinstance(config, WeightedPerStepConfig):
        raise RuntimeContractError("weighted factory requires WeightedPerStepConfig")
    return WeightedConformalRuntime(config, states)


def _weighted_quantile(
    scores: tuple[float, ...],
    *,
    coverage: float,
    decay: float,
) -> float | None:
    count = len(scores)
    weighted = [(score, decay ** (count - 1 - index)) for index, score in enumerate(scores)]
    weighted.sort(key=lambda item: item[0])
    threshold = coverage * (1.0 + sum(weight for _, weight in weighted))
    cumulative = 0.0
    for score, weight in weighted:
        cumulative += weight
        if cumulative >= threshold:
            return score
    return None


def _state_score(value: object) -> float:
    try:
        return _finite_score(value)
    except RuntimeContractError as error:
        raise StateCodecError(str(error)) from error


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeContractError("weighted scores must be finite real numbers")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise RuntimeContractError("weighted scores must be finite nonnegative real numbers")
    return 0.0 if normalized == 0.0 else normalized


def _empty_partition() -> _PartitionState:
    return _PartitionState((), 0, ScoredSeries.DEMAND_HONEST)


def _apply_rows(forecasts: pd.DataFrame) -> tuple[_ApplyRow, ...]:
    rows: list[_ApplyRow] = []
    seen: set[ForecastKey] = set()
    for values in forecasts.loc[
        :, [SERIES_KEY, ORIGIN, HORIZON_STEP, MODEL_NAME, POINT_FORECAST]
    ].to_dict("records"):
        key = ForecastKey(
            series_key=_text(values[SERIES_KEY], name="series key"),
            origin=_timestamp(values[ORIGIN]),
            horizon_step=_horizon(values[HORIZON_STEP]),
            model_name=_text(values[MODEL_NAME], name="model name"),
        )
        if key in seen:
            raise RuntimeContractError("weighted apply forecasts contain a duplicate row key")
        seen.add(key)
        point = values[POINT_FORECAST]
        if isinstance(point, bool) or not isinstance(point, Real):
            raise RuntimeContractError("weighted point forecasts must be finite real numbers")
        normalized = float(point)
        if not math.isfinite(normalized):
            raise RuntimeContractError("weighted point forecasts must be finite real numbers")
        rows.append(_ApplyRow(key, 0.0 if normalized == 0.0 else normalized))
    return tuple(rows)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeContractError(f"weighted {name} must be a non-empty string")
    return value


def _timestamp(value: object) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp):
        raise RuntimeContractError("weighted origins must be pandas Timestamps")
    if pd.isna(value) or value.tz is not None:
        raise RuntimeContractError("weighted origins must be timezone-naive timestamps")
    return value


def _horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise RuntimeContractError("weighted horizon steps must be positive integers")
    return int(value)


def _state_reference(method_name: str, state: bytes) -> str:
    return f"{method_name}:sha256:{hashlib.sha256(state).hexdigest()}"
