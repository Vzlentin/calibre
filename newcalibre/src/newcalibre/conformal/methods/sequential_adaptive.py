"""Implement sequential-adaptive one-sided upper conformal bounds per step."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, cast

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from newcalibre.conformal.batch import (
    CalibrationResult,
    CalibrationSeedBatch,
    ConformalStateBatch,
    DeliveryBatch,
    ObserveEffect,
)
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
    ForecastKey,
    IssuedBoundFacts,
    ObserveAnnotation,
    ResolvedObservation,
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

SEQUENTIAL_ADAPTIVE_PER_STEP = "sequential-adaptive-per-step"
_MAX_CALIBRATION_WINDOW = 5000
_STATE_SCHEMA_VERSION = 1
_WARMUP_CAUSE = "warm-up"
_UNRESOLVABLE_CAUSE = "unresolvable-working-level"
_CENSORED_CAUSE = "declared-censored"


class SequentialAdaptivePerStepConfig(BaseModel):
    """Configure sequential-adaptive one-sided conformal bounds per step."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: float = Field(default=0.9, gt=0.0, lt=1.0, allow_inf_nan=False)
    calibration_window: int = Field(
        default=_MAX_CALIBRATION_WINDOW,
        ge=1,
        le=_MAX_CALIBRATION_WINDOW,
    )
    partition_by: Literal["global", "series"] = "global"
    learning_rate: float = Field(default=0.05, ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _valid_rank(self) -> SequentialAdaptivePerStepConfig:
        minimum = ConservativeRankRequirement().minimum_scores(self)
        if minimum > self.calibration_window:
            raise ValueError("calibration_window cannot satisfy the configured conservative rank")
        return self


_SEQUENTIAL_ADAPTIVE_GUARANTEE = GuaranteeDeclaration(
    claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
    currency=GuaranteeCurrency.LONG_RUN_PATHWISE,
)

SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST = MethodManifest(
    name=SEQUENTIAL_ADAPTIVE_PER_STEP,
    emission_form=EmissionForm.ONE_SIDED_UPPER,
    emission_scope=EmissionScope.PER_STEP,
    guarantees=(_SEQUENTIAL_ADAPTIVE_GUARANTEE,),
    assumption_class=AssumptionClass.SEQUENTIAL_ADAPTIVE,
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
    raw_alpha: float
    feedback_count: int


@dataclass(frozen=True, slots=True)
class _MethodState:
    issue_counter: int


@dataclass(frozen=True, slots=True)
class _SequentialStateColumns:
    labels: tuple[str, ...]
    scores: tuple[tuple[float, ...], ...]
    delivered_score_counts: tuple[int, ...]
    scored_series: tuple[ScoredSeries, ...]
    raw_alphas: tuple[float, ...]
    feedback_counts: tuple[int, ...]
    route_by_label: Mapping[str, int]
    method_state: _MethodState

    @classmethod
    def from_rows(
        cls,
        rows: Mapping[str, _PartitionState],
        method_state: _MethodState,
    ) -> _SequentialStateColumns:
        labels = tuple(sorted(rows, key=str.encode))
        states = tuple(rows[label] for label in labels)
        return cls(
            labels=labels,
            scores=tuple(state.scores for state in states),
            delivered_score_counts=tuple(state.delivered_score_count for state in states),
            scored_series=tuple(state.scored_series for state in states),
            raw_alphas=tuple(state.raw_alpha for state in states),
            feedback_counts=tuple(state.feedback_count for state in states),
            route_by_label=MappingProxyType({label: route for route, label in enumerate(labels)}),
            method_state=method_state,
        )

    def get(self, label: str, default: _PartitionState) -> _PartitionState:
        route = self.route_by_label.get(label)
        if route is None:
            return default
        return _PartitionState(
            self.scores[route],
            self.delivered_score_counts[route],
            self.scored_series[route],
            self.raw_alphas[route],
            self.feedback_counts[route],
        )


class _SequentialAdaptiveStateCodec:
    def __init__(self) -> None:
        self._codec = JsonStateCodec(
            SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.name,
            SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.state_schema_version,
        )

    def encode_partition(self, label: str, state: _PartitionState) -> bytes:
        return self._codec.encode(
            label,
            {
                "delivered_score_count": state.delivered_score_count,
                "feedback_count": state.feedback_count,
                "raw_alpha": state.raw_alpha,
                "scored_series": state.scored_series.value,
                "scores": list(state.scores),
            },
        )

    def decode_partition(self, state: bytes, *, label: str) -> _PartitionState:
        payload = self._codec.decode(state, expected_label=label)
        if not isinstance(payload, dict) or set(payload) != {
            "delivered_score_count",
            "feedback_count",
            "raw_alpha",
            "scored_series",
            "scores",
        }:
            raise StateCodecError("sequential-adaptive partition state must contain exact fields")
        fields = cast(dict[str, object], payload)
        scores_raw = fields["scores"]
        if not isinstance(scores_raw, list):
            raise StateCodecError("sequential-adaptive partition scores must be a list")
        if len(scores_raw) > SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST.state_bound:
            raise StateCodecError(
                "sequential-adaptive partition scores exceed the declared state bound"
            )
        scores = tuple(_state_score(value) for value in scores_raw)
        delivered = fields["delivered_score_count"]
        if isinstance(delivered, bool) or not isinstance(delivered, int) or delivered < len(scores):
            raise StateCodecError(
                "sequential-adaptive delivered-score count must be an integer no smaller "
                "than retained scores"
            )
        feedback = fields["feedback_count"]
        if (
            isinstance(feedback, bool)
            or not isinstance(feedback, int)
            or feedback < 0
            or feedback > delivered
        ):
            raise StateCodecError(
                "sequential-adaptive feedback count must be a nonnegative integer no larger "
                "than delivered scores"
            )
        raw_alpha = _state_alpha(fields["raw_alpha"])
        try:
            scored_series = ScoredSeries(fields["scored_series"])
        except (TypeError, ValueError) as error:
            raise StateCodecError("sequential-adaptive scored-series label is invalid") from error
        return _PartitionState(scores, delivered, scored_series, raw_alpha, feedback)

    def encode_method(self, state: _MethodState) -> bytes:
        return self._codec.encode(
            METHOD_SCOPE_LABEL,
            {"issue_counter": state.issue_counter},
        )

    def decode_method(self, state: bytes) -> _MethodState:
        payload = self._codec.decode(state, expected_label=METHOD_SCOPE_LABEL)
        if not isinstance(payload, dict) or set(payload) != {"issue_counter"}:
            raise StateCodecError("sequential-adaptive method state must contain exact fields")
        fields = cast(dict[str, object], payload)
        counter = fields["issue_counter"]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise StateCodecError("sequential-adaptive issue counter must be a nonnegative integer")
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
                    "sequential-adaptive partition label has the wrong method emission scope"
                )
        return scope


class SequentialAdaptiveConformalRuntime:
    """Apply sequential-adaptive bounds through the stable runtime seam."""

    def __init__(
        self,
        config: SequentialAdaptivePerStepConfig,
        states: ConformalStateBatch,
    ) -> None:
        if type(config) is not SequentialAdaptivePerStepConfig:
            raise RuntimeContractError(
                "sequential-adaptive runtime configuration does not match its manifest"
            )
        self._config = config
        self._codec = _SequentialAdaptiveStateCodec()
        if not isinstance(states, ConformalStateBatch):
            raise RuntimeContractError(
                "sequential-adaptive runtime state must be a ConformalStateBatch"
            )
        self._validate_states(states)

    @property
    def manifest(self) -> MethodManifest:
        """Return the registered immutable sequential-adaptive declaration."""
        return SEQUENTIAL_ADAPTIVE_PER_STEP_MANIFEST

    @property
    def config(self) -> BaseModel:
        """Return the frozen sequential-adaptive runtime configuration."""
        return self._config

    def calibrate(self, seeds: CalibrationSeedBatch) -> ConformalStateBatch:
        """Seed bounded chronological scores while holding alpha at its target."""
        if not isinstance(seeds, CalibrationSeedBatch):
            raise RuntimeContractError(
                "sequential-adaptive calibration seeds must be a CalibrationSeedBatch"
            )
        updates: dict[str, bytes] = {}
        for label, values in seeds.items():
            try:
                scope = self._codec.scope_for(label)
            except StateCodecError as error:
                raise RuntimeContractError(str(error)) from error
            if scope is not StateScope.PARTITION:
                raise RuntimeContractError("calibration scores must use partition labels")
            normalized = values
            retained = normalized[-self._config.calibration_window :]
            updates[label] = self._codec.encode_partition(
                label,
                _PartitionState(
                    scores=retained,
                    delivered_score_count=len(normalized),
                    scored_series=ScoredSeries.DEMAND_HONEST,
                    raw_alpha=self._target_alpha,
                    feedback_count=0,
                ),
            )
        updates[METHOD_SCOPE_LABEL] = self._codec.encode_method(_MethodState(0))
        return ConformalStateBatch(updates)

    def apply(
        self,
        forecasts: pd.DataFrame,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        """Emit adaptive upper bounds without reading actual-value columns."""
        if not isinstance(forecasts, pd.DataFrame):
            raise RuntimeContractError(
                "sequential-adaptive apply forecasts must be a pandas DataFrame"
            )
        if forecasts.columns.has_duplicates:
            raise RuntimeContractError(
                "sequential-adaptive apply forecasts cannot have duplicate columns"
            )
        required = (SERIES_KEY, POINT_FORECAST, HORIZON_STEP, ORIGIN, MODEL_NAME)
        missing = [column for column in required if column not in forecasts]
        if missing:
            raise RuntimeContractError(
                "sequential-adaptive apply forecasts are missing required columns: "
                + ", ".join(missing)
            )
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(forecasts[SERIES_KEY]),
        )
        decoded = self._validate_states(state)
        rows = _apply_rows(forecasts)
        lower_column, upper_column = interval_columns(self._config.coverage)
        lower_values: list[float] = []
        upper_values: list[float] = []
        issuances: dict[ForecastKey, IssuedBoundFacts] = {}
        minimum = self.manifest.minimum_calibration_scores(self._config)
        quantiles: dict[str, float] = {}
        state_references: dict[str, str] = {}
        descriptors: dict[ScoredSeries, GuaranteeDescriptor] = {}
        method_state = decoded.method_state
        partitions: dict[str, _PartitionState] = {}
        empty_partition = self._empty_partition()

        for row in rows:
            label = self._partition_label(row.key.model_name, row.key.series_key)
            partition = partitions.get(label)
            if partition is None:
                partition = decoded.get(label, empty_partition)
                partitions[label] = partition
            ready = len(partition.scores) >= minimum
            if not ready:
                lower = math.nan
                upper = math.nan
                null_reason = _WARMUP_CAUSE
            elif partition.raw_alpha <= 1.0 / (len(partition.scores) + 1):
                lower = math.nan
                upper = math.nan
                null_reason = _UNRESOLVABLE_CAUSE
            else:
                if label not in quantiles:
                    quantile_level = 1.0 - min(1.0, max(0.0, partition.raw_alpha))
                    quantiles[label] = _higher_quantile(partition.scores, quantile_level)
                lower = 0.0
                upper = row.point_forecast + quantiles[label]
                if not math.isfinite(upper):
                    raise RuntimeContractError(
                        "ready sequential-adaptive arithmetic must produce a finite bound"
                    )
                if upper < lower:
                    raise RuntimeContractError(
                        "sequential-adaptive upper bound cannot fall below the zero support bound"
                    )
                null_reason = None
            if label not in state_references:
                state_references[label] = _state_reference(
                    self.manifest.name,
                    method_state.issue_counter,
                    label,
                    self._codec.encode_partition(label, partition),
                )
            if partition.scored_series not in descriptors:
                descriptors[partition.scored_series] = self._descriptor(partition.scored_series)
            facts = IssuedBoundFacts(
                method_name=self.manifest.name,
                emission_form=self.manifest.emission_form,
                emission_scope=self.manifest.emission_scope,
                partition_label=label,
                working_level=partition.raw_alpha,
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
        method_blob = self._codec.encode_method(
            _MethodState(method_state.issue_counter + len(rows))
        )
        post_state = state.with_rows({METHOD_SCOPE_LABEL: method_blob})
        dirty = () if state.get(METHOD_SCOPE_LABEL) == method_blob else (METHOD_SCOPE_LABEL,)
        return CalibrationResult(calibrated, post_state, dirty, issuances)

    def observe(
        self,
        deliveries: DeliveryBatch,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        """Append scores and update alpha across every canonical partition."""
        if not isinstance(deliveries, DeliveryBatch):
            raise RuntimeContractError(
                "sequential-adaptive observe deliveries must be a DeliveryBatch"
            )
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(
                observation.forecast_key.series_key for observation in deliveries.observations
            ),
        )
        decoded = self._validate_states(state)
        updated_rows: dict[str, bytes] = {}
        dirty: list[str] = []
        annotations: list[ObserveAnnotation] = []
        empty_partition = self._empty_partition()
        for label, observations in deliveries.items():
            self._validate_delivery_issuance(label, observations)
            partition = decoded.get(label, empty_partition)
            scores = list(partition.scores)
            delivered_count = partition.delivered_score_count
            scored_series = partition.scored_series
            raw_alpha = partition.raw_alpha
            feedback_count = partition.feedback_count

            for observation in observations:
                if observation.censoring_assertion is CensoringAssertion.CENSORED:
                    annotations.append(
                        ObserveAnnotation(observation.forecast_key, None, _CENSORED_CAUSE, False)
                    )
                    continue
                score = abs(observation.actual - observation.point_forecast)
                if not math.isfinite(score):
                    raise RuntimeContractError("sequential-adaptive residual scores must be finite")
                scores.append(score)
                delivered_count += 1
                if observation.censoring_assertion is None:
                    scored_series = ScoredSeries.RECORDED_SALES
                issued = observation.issued
                if issued.calibration_ready:
                    error = 0
                    if math.isfinite(issued.upper_bound):
                        threshold = issued.upper_bound - observation.point_forecast
                        error = int(score > threshold)
                    raw_alpha += self._config.learning_rate * (self._target_alpha - error)
                    if not math.isfinite(raw_alpha):
                        raise RuntimeContractError(
                            "sequential-adaptive alpha update must remain finite"
                        )
                    feedback_count += 1
                annotations.append(ObserveAnnotation(observation.forecast_key, score, None, True))

            bounded_scores = scores[-self._config.calibration_window :]
            updated = _PartitionState(
                scores=tuple(bounded_scores),
                delivered_score_count=delivered_count,
                scored_series=scored_series,
                raw_alpha=raw_alpha,
                feedback_count=feedback_count,
            )
            blob = self._codec.encode_partition(label, updated)
            updated_rows[label] = blob
            if state.get(label) != blob:
                dirty.append(label)
        post_state = state.with_rows(updated_rows)
        return ObserveEffect(post_state, dirty, annotations)

    @property
    def _target_alpha(self) -> float:
        return 1.0 - self._config.coverage

    def _empty_partition(self) -> _PartitionState:
        return _PartitionState((), 0, ScoredSeries.DEMAND_HONEST, self._target_alpha, 0)

    def _partition_label(self, model_name: str, series_key: str) -> str:
        value = "global" if self._config.partition_by == "global" else series_key
        return derive_partition_label(model_name, value, self.manifest.emission_scope)

    def _validate_states(
        self,
        state: ConformalStateBatch,
    ) -> _SequentialStateColumns:
        if not isinstance(state, ConformalStateBatch):
            raise RuntimeContractError("sequential-adaptive state must be a ConformalStateBatch")
        partitions: dict[str, _PartitionState] = {}
        method_state = _MethodState(0)
        for label, value in state.items():
            try:
                scope = self._codec.scope_for(label)
                if scope is StateScope.METHOD:
                    method_state = self._codec.decode_method(value)
                    continue
                partition = self._codec.decode_partition(value, label=label)
            except StateCodecError as error:
                raise RuntimeContractError(str(error)) from error
            if len(partition.scores) > self._config.calibration_window:
                raise RuntimeContractError(
                    "restored sequential-adaptive scores exceed the configured calibration window"
                )
            partitions[label] = partition
        return _SequentialStateColumns.from_rows(partitions, method_state)

    def _descriptor(self, scored_series: ScoredSeries) -> GuaranteeDescriptor:
        return GuaranteeDescriptor(
            type=GuaranteeType(
                claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
                currency=GuaranteeCurrency.LONG_RUN_PATHWISE,
                declared_slack=None,
            ),
            level=self._config.coverage,
            scored_series=scored_series,
            window=self.manifest.emission_scope,
            scope=DecisionScope(DecisionScopeKind.PER_DECISION_NODE, None),
        )

    def _validate_delivery_issuance(
        self,
        partition_label: str,
        observations: tuple[ResolvedObservation, ...],
    ) -> None:
        for observation in observations:
            issued = observation.issued
            if issued.method_name != self.manifest.name:
                raise RuntimeContractError(
                    "delivered issuance has the wrong sequential-adaptive method"
                )
            if issued.emission_form is not self.manifest.emission_form:
                raise RuntimeContractError("delivered issuance has the wrong emission form")
            if issued.emission_scope is not self.manifest.emission_scope:
                raise RuntimeContractError("delivered issuance has the wrong emission scope")
            descriptor = issued.effective_descriptor
            if (
                descriptor.type.claim is not GuaranteeClaim.ONE_SIDED_COVERAGE
                or descriptor.type.currency is not GuaranteeCurrency.LONG_RUN_PATHWISE
                or descriptor.level != self._config.coverage
                or descriptor.window is not self.manifest.emission_scope
                or descriptor.scope.kind is not DecisionScopeKind.PER_DECISION_NODE
            ):
                raise RuntimeContractError(
                    "delivered issuance has the wrong sequential-adaptive descriptor"
                )
            expected = self._partition_label(
                observation.forecast_key.model_name,
                observation.forecast_key.series_key,
            )
            if expected != partition_label:
                raise RuntimeContractError(
                    "delivered forecast does not belong to the declared partition"
                )
            if issued.calibration_ready:
                if math.isfinite(issued.upper_bound):
                    if issued.lower_bound != 0.0:
                        raise RuntimeContractError(
                            "delivered sequential-adaptive finite issuance has the wrong support"
                        )
                elif issued.bounds_null_reason != _UNRESOLVABLE_CAUSE:
                    raise RuntimeContractError(
                        "delivered sequential-adaptive non-finite issuance has the wrong cause"
                    )
            elif issued.bounds_null_reason != _WARMUP_CAUSE:
                raise RuntimeContractError(
                    "delivered sequential-adaptive warm-up issuance has the wrong cause"
                )


@dataclass(frozen=True, slots=True)
class _ApplyRow:
    key: ForecastKey
    point_forecast: float


def build_sequential_adaptive_per_step(
    config: BaseModel,
    states: ConformalStateBatch,
) -> SequentialAdaptiveConformalRuntime:
    """Construct a fresh sequential-adaptive runtime through its factory."""
    if not isinstance(config, SequentialAdaptivePerStepConfig):
        raise RuntimeContractError(
            "sequential-adaptive factory requires SequentialAdaptivePerStepConfig"
        )
    return SequentialAdaptiveConformalRuntime(config, states)


def _higher_quantile(scores: tuple[float, ...], level: float) -> float:
    return float(np.quantile(np.asarray(scores, dtype="float64"), level, method="higher"))


def _state_score(value: object) -> float:
    try:
        return _finite_score(value)
    except RuntimeContractError as error:
        raise StateCodecError(str(error)) from error


def _state_alpha(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise StateCodecError("sequential-adaptive raw alpha must be a finite real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise StateCodecError("sequential-adaptive raw alpha must be a finite real number")
    return 0.0 if normalized == 0.0 else normalized


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeContractError("sequential-adaptive scores must be finite real numbers")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise RuntimeContractError(
            "sequential-adaptive scores must be finite nonnegative real numbers"
        )
    return 0.0 if normalized == 0.0 else normalized


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
            raise RuntimeContractError(
                "sequential-adaptive apply forecasts contain a duplicate row key"
            )
        seen.add(key)
        point = values[POINT_FORECAST]
        if isinstance(point, bool) or not isinstance(point, Real):
            raise RuntimeContractError(
                "sequential-adaptive point forecasts must be finite real numbers"
            )
        normalized = float(point)
        if not math.isfinite(normalized):
            raise RuntimeContractError(
                "sequential-adaptive point forecasts must be finite real numbers"
            )
        rows.append(_ApplyRow(key, 0.0 if normalized == 0.0 else normalized))
    return tuple(rows)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeContractError(f"sequential-adaptive {name} must be a non-empty string")
    return value


def _timestamp(value: object) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp):
        raise RuntimeContractError("sequential-adaptive origins must be pandas Timestamps")
    if pd.isna(value) or value.tz is not None:
        raise RuntimeContractError("sequential-adaptive origins must be timezone-naive timestamps")
    return value


def _horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise RuntimeContractError("sequential-adaptive horizon steps must be positive integers")
    return int(value)


def _state_reference(
    method_name: str,
    issue_counter: int,
    partition_label: str,
    state: bytes,
) -> str:
    partition_identity = partition_label.encode("utf-8") + b"\x00" + state
    digest = hashlib.sha256(partition_identity).hexdigest()
    return f"{method_name}:{issue_counter}:sha256:{digest}"
