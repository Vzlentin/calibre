"""Implement one-sided upper split conformal for both supported emission scopes."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from newcalibre.conformal.manifest import (
    AssumptionClass,
    CensoringPolicy,
    ClampDeclaration,
    ClampGuaranteeImpact,
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
    AppliedBinding,
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

SPLIT_PER_STEP = "split-per-step"
SPLIT_WINDOW_SUM = "split-window-sum"
_MAX_CALIBRATION_WINDOW = 5000
_STATE_SCHEMA_VERSION = 1
_WARMUP_CAUSE = "warm-up"
_EMISSION_SCOPE_CAUSE = "emission-scope"
_CENSORED_CAUSE = "declared-censored"
_CENSORED_WINDOW_CAUSE = "declared-censored-window"
_COMPOSITE_MEMBER_CAUSE = "window-composite-member"


class _SplitConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: float = Field(default=0.9, gt=0.0, lt=1.0, allow_inf_nan=False)
    calibration_window: int = Field(
        default=_MAX_CALIBRATION_WINDOW,
        ge=1,
        le=_MAX_CALIBRATION_WINDOW,
    )
    partition_by: Literal["global", "series"] = "global"
    upper_floor: float | None = None
    upper_cap: float | None = None

    @field_validator("upper_floor", "upper_cap")
    @classmethod
    def _finite_clamp(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("clamp values must be finite")
        return value

    @model_validator(mode="after")
    def _valid_rank_and_clamps(self) -> _SplitConfig:
        minimum = ConservativeRankRequirement().minimum_scores(self)
        if minimum > self.calibration_window:
            raise ValueError("calibration_window cannot satisfy the configured conservative rank")
        if (
            self.upper_floor is not None
            and self.upper_cap is not None
            and self.upper_floor > self.upper_cap
        ):
            raise ValueError("upper_floor cannot exceed upper_cap")
        return self


class SplitPerStepConfig(_SplitConfig):
    """Configure one-sided split conformal for per-step bounds."""


class SplitWindowSumConfig(_SplitConfig):
    """Configure one-sided split conformal for protection-window sums."""

    protection_period: int = Field(default=1, ge=1)


_SPLIT_GUARANTEE = GuaranteeDeclaration(
    claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
    currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
)
_SPLIT_CLAMPS = (
    ClampDeclaration("upper_floor", ClampGuaranteeImpact.VOIDS_CLAIM),
    ClampDeclaration("upper_cap", ClampGuaranteeImpact.VOIDS_CLAIM),
)

SPLIT_PER_STEP_MANIFEST = MethodManifest(
    name=SPLIT_PER_STEP,
    emission_form=EmissionForm.ONE_SIDED_UPPER,
    emission_scope=EmissionScope.PER_STEP,
    guarantees=(_SPLIT_GUARANTEE,),
    assumption_class=AssumptionClass.EXCHANGEABLE,
    calibration_requirement=ConservativeRankRequirement(),
    order_sensitive=True,
    censoring_policy=CensoringPolicy.CONSUMES_CENSORING_FACTS,
    imputation_policy=None,
    state_bound=_MAX_CALIBRATION_WINDOW,
    state_schema_version=_STATE_SCHEMA_VERSION,
    consumes_calibration_context=False,
    hosted_submodels=(),
    requires_fitted_values=False,
    post_warmup_non_finite=PostWarmupNonFinite.FORBIDDEN,
    clamps=_SPLIT_CLAMPS,
    joint_claim=JointClaim.NONE,
)

SPLIT_WINDOW_SUM_MANIFEST = MethodManifest(
    name=SPLIT_WINDOW_SUM,
    emission_form=EmissionForm.ONE_SIDED_UPPER,
    emission_scope=EmissionScope.WINDOW_SUM,
    guarantees=(_SPLIT_GUARANTEE,),
    assumption_class=AssumptionClass.EXCHANGEABLE,
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
    clamps=_SPLIT_CLAMPS,
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


class _SplitStateCodec:
    def __init__(self, manifest: MethodManifest) -> None:
        self._codec = JsonStateCodec(manifest.name, manifest.state_schema_version)
        self._emission_scope = manifest.emission_scope
        self._state_bound = manifest.state_bound

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
            raise StateCodecError("split partition state must contain exact fields")
        fields = cast(dict[str, object], payload)
        scores_raw = fields["scores"]
        if not isinstance(scores_raw, list):
            raise StateCodecError("split partition scores must be a list")
        if len(scores_raw) > self._state_bound:
            raise StateCodecError("split partition scores exceed the declared state bound")
        scores = tuple(_state_score(value) for value in scores_raw)
        delivered = fields["delivered_score_count"]
        if isinstance(delivered, bool) or not isinstance(delivered, int) or delivered < len(scores):
            raise StateCodecError(
                "split delivered-score count must be an integer no smaller than retained scores"
            )
        try:
            scored_series = ScoredSeries(fields["scored_series"])
        except (TypeError, ValueError) as error:
            raise StateCodecError("split scored-series label is invalid") from error
        return _PartitionState(scores, delivered, scored_series)

    def encode_method(self, state: _MethodState) -> bytes:
        return self._codec.encode(
            METHOD_SCOPE_LABEL,
            {"issue_counter": state.issue_counter},
        )

    def decode_method(self, state: bytes) -> _MethodState:
        payload = self._codec.decode(state, expected_label=METHOD_SCOPE_LABEL)
        if not isinstance(payload, dict) or set(payload) != {"issue_counter"}:
            raise StateCodecError("split method state must contain exact fields")
        fields = cast(dict[str, object], payload)
        counter = fields["issue_counter"]
        if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
            raise StateCodecError("split issue counter must be a nonnegative integer")
        return _MethodState(counter)

    def scope_for(self, label: str) -> StateScope:
        scope = self._codec.scope_for(label)
        if scope is StateScope.PARTITION:
            try:
                _namespace, payload = _decode_label(label)
            except RuntimeContractError as error:
                raise StateCodecError(str(error)) from error
            fields = cast(dict[str, object], payload)
            if fields["horizon_scope"] != self._emission_scope.value:
                raise StateCodecError("split partition label has the wrong method emission scope")
        return scope


class SplitConformalRuntime:
    """Apply the split-conformal algorithm declared by one registered manifest."""

    def __init__(
        self,
        config: SplitPerStepConfig | SplitWindowSumConfig,
        states: Mapping[str, bytes],
        *,
        manifest: MethodManifest,
    ) -> None:
        if manifest not in (SPLIT_PER_STEP_MANIFEST, SPLIT_WINDOW_SUM_MANIFEST):
            raise RuntimeContractError("split runtime requires a registered split manifest")
        expected_schema = (
            SplitPerStepConfig
            if manifest.emission_scope is EmissionScope.PER_STEP
            else SplitWindowSumConfig
        )
        if type(config) is not expected_schema:
            raise RuntimeContractError("split runtime configuration does not match its manifest")
        self._config = config
        self._manifest = manifest
        self._codec = _SplitStateCodec(manifest)
        self._validate_states(states, allow_missing=False)

    @property
    def manifest(self) -> MethodManifest:
        """Return the registered immutable split-method declaration."""
        return self._manifest

    @property
    def config(self) -> BaseModel:
        """Return the frozen split runtime configuration."""
        return self._config

    def calibrate(self, scores: Mapping[str, Sequence[float]]) -> Mapping[str, bytes]:
        """Deterministically seed bounded partition states from finite scores."""
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
            state = _PartitionState(
                scores=retained,
                delivered_score_count=len(normalized),
                scored_series=ScoredSeries.DEMAND_HONEST,
            )
            updates[label] = self._codec.encode_partition(label, state)
        updates[METHOD_SCOPE_LABEL] = self._codec.encode_method(_MethodState(0))
        return MappingProxyType(updates)

    def apply(
        self,
        forecasts: pd.DataFrame,
        states: Mapping[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        """Emit one-sided split bounds without reading actual-value columns."""
        if not isinstance(forecasts, pd.DataFrame):
            raise RuntimeContractError("split apply forecasts must be a pandas DataFrame")
        if forecasts.columns.has_duplicates:
            raise RuntimeContractError("split apply forecasts cannot have duplicate columns")
        required = (SERIES_KEY, POINT_FORECAST, HORIZON_STEP, ORIGIN, MODEL_NAME)
        missing = [column for column in required if column not in forecasts]
        if missing:
            raise RuntimeContractError(
                "split apply forecasts are missing required columns: " + ", ".join(missing)
            )
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(forecasts[SERIES_KEY]),
        )
        decoded = self._validate_states(states, allow_missing=True)
        rows = _apply_rows(forecasts)
        centers = (
            self._window_centers(rows)
            if self.manifest.emission_scope is EmissionScope.WINDOW_SUM
            else {}
        )
        lower_column, upper_column = interval_columns(self._config.coverage)
        lower_values: list[float] = []
        upper_values: list[float] = []
        issuances: dict[ForecastKey, IssuedBoundFacts] = {}
        minimum = self.manifest.minimum_calibration_scores(self._config)
        radii: dict[str, float] = {}
        state_references: dict[str, str] = {}

        for row in rows:
            raw_upper = math.nan
            bindings: tuple[AppliedBinding, ...] = ()
            label = self._partition_label(row.key.model_name, row.key.series_key)
            partition = decoded.get(label, _empty_partition())
            ready = len(partition.scores) >= minimum
            emitted = self.manifest.emission_scope is EmissionScope.PER_STEP or (
                row.key.horizon_step == self._protection_period
            )
            if not emitted:
                lower = math.nan
                upper = math.nan
                null_reason = _EMISSION_SCOPE_CAUSE
            elif not ready:
                lower = math.nan
                upper = math.nan
                null_reason = _WARMUP_CAUSE
            else:
                if label not in radii:
                    rank = math.ceil((len(partition.scores) + 1) * self._config.coverage)
                    if rank > len(partition.scores):
                        raise RuntimeContractError(
                            "split conservative rank exceeds available ready scores"
                        )
                    radii[label] = sorted(partition.scores)[rank - 1]
                radius = radii[label]
                center = (
                    row.point_forecast
                    if self.manifest.emission_scope is EmissionScope.PER_STEP
                    else centers[_window_key(row.key)]
                )
                raw_upper = center + radius
                lower = 0.0
                upper, bindings = self._apply_clamps(raw_upper)
                if upper < lower:
                    raise RuntimeContractError(
                        "split upper bound cannot fall below the zero support bound"
                    )
                null_reason = None
            bindings = () if not math.isfinite(upper) else bindings
            descriptor = self._descriptor(
                partition.scored_series,
                voided=math.isfinite(upper) and upper != raw_upper,
            )
            if label not in state_references:
                state_references[label] = _state_reference(
                    self.manifest.name,
                    self._codec.encode_partition(label, partition),
                )
            state_reference = state_references[label]
            facts = IssuedBoundFacts(
                method_name=self.manifest.name,
                emission_form=self.manifest.emission_form,
                emission_scope=self.manifest.emission_scope,
                partition_label=label,
                working_level=self._config.coverage,
                state_reference=state_reference,
                lower_bound=lower,
                upper_bound=upper,
                calibration_ready=ready,
                bounds_null_reason=null_reason,
                effective_descriptor=descriptor,
                bindings=bindings,
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
        """Score one canonical delivery and return its bounded partition update."""
        if not isinstance(delivery, Delivery):
            raise RuntimeContractError("split observe delivery must be a Delivery")
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(
                observation.forecast_key.series_key for observation in delivery.observations
            ),
        )
        decoded = self._validate_states(states, allow_missing=True)
        self._validate_delivery_issuance(delivery)
        partition = decoded.get(delivery.partition_label, _empty_partition())
        if self.manifest.emission_scope is EmissionScope.PER_STEP:
            updated, annotations = self._observe_per_step(delivery, partition)
        else:
            updated, annotations = self._observe_window(delivery, partition)
        state = self._codec.encode_partition(delivery.partition_label, updated)
        return ObserveEffect({delivery.partition_label: state}, annotations)

    @property
    def _protection_period(self) -> int:
        if not isinstance(self._config, SplitWindowSumConfig):
            raise RuntimeContractError("per-step split has no protection period")
        return self._config.protection_period

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
            raise RuntimeContractError("split states must be a mapping")
        partitions: dict[str, _PartitionState] = {}
        for label, value in dict(states).items():
            if value is None:
                if allow_missing:
                    continue
                raise RuntimeContractError("restored split states must contain bytes")
            if not isinstance(value, bytes):
                raise RuntimeContractError("split state values must be immutable bytes")
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
                    "restored split scores exceed the configured calibration window"
                )
            partitions[label] = partition
        return partitions

    def _method_state(self, states: Mapping[str, bytes | None]) -> _MethodState:
        value = states.get(METHOD_SCOPE_LABEL)
        if value is None:
            return _MethodState(0)
        if not isinstance(value, bytes):
            raise RuntimeContractError("split method state must be immutable bytes")
        try:
            return self._codec.decode_method(value)
        except StateCodecError as error:
            raise RuntimeContractError(str(error)) from error

    def _window_centers(
        self, rows: tuple[_ApplyRow, ...]
    ) -> dict[tuple[str, pd.Timestamp, str], float]:
        members: dict[tuple[str, pd.Timestamp, str], dict[int, float]] = {}
        for row in rows:
            members.setdefault(_window_key(row.key), {})[row.key.horizon_step] = row.point_forecast
        centers: dict[tuple[str, pd.Timestamp, str], float] = {}
        expected = set(range(1, self._protection_period + 1))
        for row in rows:
            if row.key.horizon_step != self._protection_period:
                continue
            group = members[_window_key(row.key)]
            if not expected.issubset(group):
                raise RuntimeContractError(
                    "window-sum apply requires every leading protection-window member"
                )
            centers[_window_key(row.key)] = sum(
                group[step] for step in range(1, self._protection_period + 1)
            )
        return centers

    def _apply_clamps(self, raw_upper: float) -> tuple[float, tuple[AppliedBinding, ...]]:
        if not math.isfinite(raw_upper):
            raise RuntimeContractError("ready split arithmetic must produce a finite bound")
        upper = raw_upper
        bindings: list[AppliedBinding] = []
        if self._config.upper_floor is not None:
            bound = upper < self._config.upper_floor
            bindings.append(AppliedBinding("upper_floor", self._config.upper_floor, bound))
            upper = max(upper, self._config.upper_floor)
        if self._config.upper_cap is not None:
            bound = upper > self._config.upper_cap
            bindings.append(AppliedBinding("upper_cap", self._config.upper_cap, bound))
            upper = min(upper, self._config.upper_cap)
        return upper, tuple(bindings)

    def _descriptor(
        self,
        scored_series: ScoredSeries,
        *,
        voided: bool,
    ) -> GuaranteeDescriptor:
        guarantee_type = GuaranteeType(
            claim=GuaranteeClaim.NONE if voided else GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=None if voided else GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        )
        return GuaranteeDescriptor(
            type=guarantee_type,
            level=self._config.coverage,
            scored_series=scored_series,
            window=self.manifest.emission_scope,
            scope=DecisionScope(DecisionScopeKind.PER_DECISION_NODE, None),
        )

    def _validate_delivery_issuance(self, delivery: Delivery) -> None:
        for observation in delivery.observations:
            issued = observation.issued
            if issued.method_name != self.manifest.name:
                raise RuntimeContractError("delivered issuance has the wrong split method")
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

    def _observe_per_step(
        self,
        delivery: Delivery,
        partition: _PartitionState,
    ) -> tuple[_PartitionState, tuple[ObserveAnnotation, ...]]:
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
        return _PartitionState(tuple(bounded_scores), count, scored_series), tuple(annotations)

    def _observe_window(
        self,
        delivery: Delivery,
        partition: _PartitionState,
    ) -> tuple[_PartitionState, tuple[ObserveAnnotation, ...]]:
        observations = delivery.observations
        if len(observations) != self._protection_period:
            raise RuntimeContractError("window-sum observe requires one complete protection window")
        first = observations[0].forecast_key
        if any(
            observation.forecast_key.series_key != first.series_key
            or observation.forecast_key.model_name != first.model_name
            or observation.forecast_key.origin != first.origin
            for observation in observations
        ):
            raise RuntimeContractError(
                "window-sum observe members must share series, model, and origin"
            )
        steps = tuple(observation.forecast_key.horizon_step for observation in observations)
        if steps != tuple(range(1, self._protection_period + 1)):
            raise RuntimeContractError(
                "window-sum observe requires canonical horizon steps 1..protection_period"
            )
        if any(
            observation.censoring_assertion is CensoringAssertion.CENSORED
            for observation in observations
        ):
            annotations = tuple(
                ObserveAnnotation(
                    observation.forecast_key,
                    None,
                    _CENSORED_WINDOW_CAUSE,
                    False,
                )
                for observation in observations
            )
            return partition, annotations

        score = abs(
            sum(observation.actual for observation in observations)
            - sum(observation.point_forecast for observation in observations)
        )
        scores = (*partition.scores, score)[-self._config.calibration_window :]
        scored_series = (
            ScoredSeries.RECORDED_SALES
            if partition.scored_series is ScoredSeries.RECORDED_SALES
            or any(observation.censoring_assertion is None for observation in observations)
            else ScoredSeries.DEMAND_HONEST
        )
        annotations = tuple(
            ObserveAnnotation(
                observation.forecast_key,
                score if index == len(observations) - 1 else None,
                None if index == len(observations) - 1 else _COMPOSITE_MEMBER_CAUSE,
                index == len(observations) - 1,
            )
            for index, observation in enumerate(observations)
        )
        return (
            _PartitionState(
                scores=tuple(scores),
                delivered_score_count=partition.delivered_score_count + 1,
                scored_series=scored_series,
            ),
            annotations,
        )


@dataclass(frozen=True, slots=True)
class _ApplyRow:
    key: ForecastKey
    point_forecast: float


def build_split_per_step(
    config: BaseModel,
    states: Mapping[str, bytes],
) -> SplitConformalRuntime:
    """Construct a fresh per-step split runtime through its registered factory."""
    if not isinstance(config, SplitPerStepConfig):
        raise RuntimeContractError("per-step split factory requires SplitPerStepConfig")
    return SplitConformalRuntime(config, states, manifest=SPLIT_PER_STEP_MANIFEST)


def build_split_window_sum(
    config: BaseModel,
    states: Mapping[str, bytes],
) -> SplitConformalRuntime:
    """Construct a fresh window-sum split runtime through its registered factory."""
    if not isinstance(config, SplitWindowSumConfig):
        raise RuntimeContractError("window-sum split factory requires SplitWindowSumConfig")
    return SplitConformalRuntime(config, states, manifest=SPLIT_WINDOW_SUM_MANIFEST)


def _state_score(value: object) -> float:
    try:
        return _finite_score(value)
    except RuntimeContractError as error:
        raise StateCodecError(str(error)) from error


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RuntimeContractError("split scores must be finite real numbers")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        raise RuntimeContractError("split scores must be finite nonnegative real numbers")
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
            raise RuntimeContractError("split apply forecasts contain a duplicate row key")
        seen.add(key)
        point = values[POINT_FORECAST]
        if isinstance(point, bool) or not isinstance(point, Real):
            raise RuntimeContractError("split point forecasts must be finite real numbers")
        normalized = float(point)
        if not math.isfinite(normalized):
            raise RuntimeContractError("split point forecasts must be finite real numbers")
        rows.append(_ApplyRow(key, 0.0 if normalized == 0.0 else normalized))
    return tuple(rows)


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RuntimeContractError(f"split {name} must be a non-empty string")
    return value


def _timestamp(value: object) -> pd.Timestamp:
    if not isinstance(value, pd.Timestamp):
        raise RuntimeContractError("split origins must be pandas Timestamps")
    timestamp = value
    if pd.isna(timestamp) or timestamp.tz is not None:
        raise RuntimeContractError("split origins must be timezone-naive timestamps")
    return timestamp


def _horizon(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise RuntimeContractError("split horizon steps must be positive integers")
    return int(value)


def _window_key(key: ForecastKey) -> tuple[str, pd.Timestamp, str]:
    return key.series_key, key.origin, key.model_name


def _state_reference(method_name: str, state: bytes) -> str:
    return f"{method_name}:sha256:{hashlib.sha256(state).hexdigest()}"
