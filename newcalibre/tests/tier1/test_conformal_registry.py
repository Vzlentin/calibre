"""Exercise conformal registration and a test-owned three-verb fixture method."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    AssumptionClass,
    CalibrationContext,
    CalibrationResult,
    CalibrationSeedBatch,
    CensoringPolicy,
    ConformalRegistry,
    ConformalRegistryError,
    ConformalRuntime,
    ConformalStateBatch,
    DeliveryBatch,
    EmissionForm,
    FixedCountRequirement,
    ForecastKey,
    GuaranteeDeclaration,
    IssuedBoundFacts,
    JointClaim,
    MethodManifest,
    ObserveAnnotation,
    ObserveEffect,
    PostWarmupNonFinite,
    ResolvedObservation,
    RuntimeContractError,
    derive_partition_label,
    require_calibration_context,
)
from newcalibre.conformal.state import JsonStateCodec
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
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
)

pytestmark = pytest.mark.tier1


def Delivery(label: str, observations: tuple[ResolvedObservation, ...]) -> DeliveryBatch:
    """Build one partition row inside the batch API."""
    return DeliveryBatch({label: observations})


class _FixtureConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    offset: float = 1.0


class _SeededMismatchConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    offset: float = 1.0
    hidden_runtime_default: int = 7


FIXTURE_MANIFEST = MethodManifest(
    name="fixture",
    emission_form=EmissionForm.ONE_SIDED_UPPER,
    emission_scope=EmissionScope.PER_STEP,
    guarantees=(
        GuaranteeDeclaration(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
        ),
    ),
    assumption_class=AssumptionClass.EXCHANGEABLE,
    calibration_requirement=FixedCountRequirement(1),
    order_sensitive=True,
    censoring_policy=CensoringPolicy.REQUIRES_UNCENSORED,
    imputation_policy=None,
    state_bound=16,
    state_schema_version=1,
    consumes_calibration_context=False,
    hosted_submodels=(),
    requires_fitted_values=False,
    post_warmup_non_finite=PostWarmupNonFinite.FORBIDDEN,
    clamps=(),
    joint_claim=JointClaim.NONE,
)


class _FixtureCodec:
    def __init__(self) -> None:
        self._codec = JsonStateCodec("fixture", 1)

    def encode_partition(self, label: str, *, count: int, total: float) -> bytes:
        return self._codec.encode(label, {"count": count, "total": total})

    def encode_method(self, *, issue_counter: int) -> bytes:
        return self._codec.encode(METHOD_SCOPE_LABEL, {"issue_counter": issue_counter})

    def decode_partition(self, state: bytes, *, label: str) -> tuple[int, float]:
        payload = self._codec.decode(state, expected_label=label)
        if not isinstance(payload, dict) or set(payload) != {"count", "total"}:
            raise ValueError("fixture partition payload has invalid fields")
        return int(payload["count"]), float(payload["total"])


class _FixtureRuntime:
    instances: ClassVar[int] = 0
    constructed: ClassVar[list[_FixtureRuntime]] = []

    def __init__(
        self,
        config: _FixtureConfig,
        states: Mapping[str, bytes],
    ) -> None:
        _FixtureRuntime.instances += 1
        self.instance_number = _FixtureRuntime.instances
        _FixtureRuntime.constructed.append(self)
        self._config = config
        self._states = MappingProxyType(dict(states))
        self._codec = _FixtureCodec()

    @property
    def manifest(self) -> MethodManifest:
        return FIXTURE_MANIFEST

    @property
    def config(self) -> BaseModel:
        return self._config

    @property
    def restored_states(self) -> Mapping[str, bytes]:
        return self._states

    def calibrate(
        self,
        seeds: CalibrationSeedBatch,
    ) -> ConformalStateBatch:
        states = {
            label: self._codec.encode_partition(
                label,
                count=len(values),
                total=sum(values),
            )
            for label, values in seeds.items()
        }
        states[METHOD_SCOPE_LABEL] = self._codec.encode_method(issue_counter=0)
        return ConformalStateBatch(states)

    def apply(
        self,
        forecasts: pd.DataFrame,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(forecasts["series_key"]),
        )
        calibrated = forecasts.copy(deep=True)
        calibrated["lower_0.9"] = 0.0
        calibrated["upper_0.9"] = calibrated["point_forecast"] + self._config.offset
        partition = next(
            (label for label in state if label != METHOD_SCOPE_LABEL),
            derive_partition_label("fixture-model", "global", EmissionScope.PER_STEP),
        )
        issuances = {
            ForecastKey(
                series_key=row[SERIES_KEY],
                origin=pd.Timestamp(row[ORIGIN]),
                horizon_step=row[HORIZON_STEP],
                model_name=row[MODEL_NAME],
            ): IssuedBoundFacts(
                method_name="fixture",
                emission_form=EmissionForm.ONE_SIDED_UPPER,
                emission_scope=EmissionScope.PER_STEP,
                partition_label=partition,
                working_level=0.9,
                state_reference="fixture:0",
                lower_bound=0.0,
                upper_bound=float(row[POINT_FORECAST]) + self._config.offset,
                calibration_ready=True,
                bounds_null_reason=None,
                effective_descriptor=_descriptor(),
            )
            for row in calibrated.to_dict("records")
        }
        return CalibrationResult(calibrated, state, issuances=issuances)

    def observe(
        self,
        deliveries: DeliveryBatch,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(
                observation.forecast_key.series_key for observation in deliveries.observations
            ),
        )
        updates: dict[str, bytes] = {}
        annotations: list[ObserveAnnotation] = []
        for label, observations in deliveries.items():
            current = state.get(label)
            count, total = (
                (0, 0.0) if current is None else self._codec.decode_partition(current, label=label)
            )
            partition_annotations = tuple(
                ObserveAnnotation(
                    forecast_key=observation.forecast_key,
                    score=abs(observation.actual - observation.point_forecast),
                    exclusion_cause=None,
                    advanced_delivered_score=True,
                )
                for observation in observations
            )
            score_total = sum(annotation.score or 0.0 for annotation in partition_annotations)
            updates[label] = self._codec.encode_partition(
                label,
                count=count + len(partition_annotations),
                total=total + score_total,
            )
            annotations.extend(partition_annotations)
        post_state = state.with_rows(updates)
        return ObserveEffect(post_state, updates, annotations)


class _InvalidStateOutputRuntime(_FixtureRuntime):
    def __init__(
        self,
        config: _FixtureConfig,
        states: Mapping[str, bytes],
        invalid_state: bytes,
    ) -> None:
        super().__init__(config, states)
        self._invalid_state = invalid_state

    def calibrate(
        self,
        seeds: CalibrationSeedBatch,
    ) -> ConformalStateBatch:
        label = next(iter(seeds.labels))
        return ConformalStateBatch({label: self._invalid_state})

    def apply(
        self,
        forecasts: pd.DataFrame,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        result = super().apply(forecasts, state, context=context)
        label = next(iter(state))
        return CalibrationResult(
            result.forecasts,
            state.with_rows({label: self._invalid_state}),
            (label,),
            result.issuances,
        )

    def observe(
        self,
        deliveries: DeliveryBatch,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        effect = super().observe(deliveries, state, context=context)
        label = deliveries.labels[0]
        return ObserveEffect(
            state.with_rows({label: self._invalid_state}),
            (label,),
            effect.annotations,
        )


def _descriptor() -> GuaranteeDescriptor:
    return GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.ONE_SIDED_COVERAGE,
            currency=GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
            declared_slack=None,
        ),
        level=0.9,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=EmissionScope.PER_STEP,
        scope=DecisionScope(DecisionScopeKind.PER_DECISION_NODE, None),
    )


def _invalid_issuance(facts: IssuedBoundFacts, invalid_kind: str) -> IssuedBoundFacts:
    if invalid_kind == "method":
        return replace(facts, method_name="other")
    if invalid_kind == "form":
        return replace(facts, emission_form=EmissionForm.ONE_SIDED_LOWER)
    if invalid_kind == "scope":
        return replace(
            facts,
            emission_scope=EmissionScope.WINDOW_SUM,
            effective_descriptor=replace(
                facts.effective_descriptor,
                window=EmissionScope.WINDOW_SUM,
            ),
        )
    if invalid_kind == "claim":
        return replace(
            facts,
            effective_descriptor=replace(
                facts.effective_descriptor,
                type=GuaranteeType(
                    GuaranteeClaim.TWO_SIDED_COVERAGE,
                    GuaranteeCurrency.FINITE_SAMPLE_MARGINAL,
                    None,
                ),
            ),
        )
    if invalid_kind == "clamp":
        return replace(
            facts,
            bindings=(AppliedBinding("undeclared-cap", 5.0, False),),
        )
    if invalid_kind == "claim-binding":
        return replace(
            facts,
            effective_descriptor=replace(
                facts.effective_descriptor,
                type=GuaranteeType(GuaranteeClaim.NONE, None, None),
            ),
        )
    raise AssertionError(f"unknown invalid issuance kind: {invalid_kind}")


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series(["sku"], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime(["2025-01-06"]),
            ACTUAL_VALUE: pd.Series([float("nan")], dtype="float64"),
            POINT_FORECAST: pd.Series([4.0], dtype="float64"),
            HORIZON_STEP: pd.Series([1], dtype="int64"),
            ORIGIN: pd.to_datetime(["2025-01-06"]),
            MODEL_NAME: pd.Series(["fixture-model"], dtype="string"),
        }
    )


def _observation(
    series_key: str,
    *,
    partition_label: str,
    actual: float,
) -> ResolvedObservation:
    key = ForecastKey(
        series_key=series_key,
        origin=pd.Timestamp("2025-01-06"),
        horizon_step=1,
        model_name="fixture-model",
    )
    return ResolvedObservation(
        forecast_key=key,
        target_timestamp=pd.Timestamp("2025-01-06"),
        actual=actual,
        point_forecast=4.0,
        censoring_assertion=CensoringAssertion.UNCENSORED,
        availability_bound=None,
        issued=IssuedBoundFacts(
            method_name="fixture",
            emission_form=EmissionForm.ONE_SIDED_UPPER,
            emission_scope=EmissionScope.PER_STEP,
            partition_label=partition_label,
            working_level=0.9,
            state_reference="fixture:0",
            lower_bound=0.0,
            upper_bound=5.0,
            calibration_ready=True,
            bounds_null_reason=None,
            effective_descriptor=_descriptor(),
        ),
    )


def _factory(
    config: BaseModel,
    states: Mapping[str, bytes],
) -> ConformalRuntime:
    assert isinstance(config, _FixtureConfig)
    return _FixtureRuntime(config, states)


def _registry() -> ConformalRegistry:
    registry = ConformalRegistry()
    registry.register("fixture", FIXTURE_MANIFEST, _FixtureConfig, _factory)
    return registry


def test_registry_registers_and_constructs_a_fixture_with_schema_parity() -> None:
    registry = _registry()

    runtime = registry.resolve({"method": "fixture", "offset": 2.5})

    assert registry.available_methods == ("fixture",)
    assert registry.config_schema("fixture") is _FixtureConfig
    assert isinstance(runtime, ConformalRuntime)
    assert type(runtime.config) is _FixtureConfig
    assert runtime.config.model_dump() == {"offset": 2.5}
    assert runtime.manifest is FIXTURE_MANIFEST


def test_fixture_executes_calibrate_apply_observe_and_factory_restoration() -> None:
    registry = _registry()
    original = registry.resolve({"method": "fixture"})
    original_implementation = _FixtureRuntime.constructed[-1]
    partition = derive_partition_label("fixture-model", "global", EmissionScope.PER_STEP)

    calibrated_states = original.calibrate(CalibrationSeedBatch({partition: [1.0, 2.0]}))
    restored = registry.resolve({"method": "fixture"}, states=calibrated_states)
    restored_implementation = _FixtureRuntime.constructed[-1]
    frame = _frame()
    applied = restored.apply(frame, calibrated_states)
    observed = restored.observe(
        Delivery(partition, (_observation("sku", partition_label=partition, actual=7.0),)),
        calibrated_states,
    )

    assert restored is not original
    assert restored_implementation is not original_implementation
    assert restored_implementation.instance_number != original_implementation.instance_number
    assert dict(restored_implementation.restored_states) == dict(calibrated_states)
    assert applied.forecasts.loc[0, "upper_0.9"] == 5.0
    assert observed.annotations[0].score == 3.0
    assert _FixtureCodec().decode_partition(
        observed.dirty_state[partition],
        label=partition,
    ) == (3, 6.0)


def test_registration_rejects_runtime_and_exposed_schema_mismatch_atomically() -> None:
    registry = ConformalRegistry()

    def mismatch_factory(
        config: BaseModel,
        states: Mapping[str, bytes],
    ) -> ConformalRuntime:
        del config
        return _FixtureRuntime(
            _SeededMismatchConfig(),  # type: ignore[arg-type]
            states,
        )

    with pytest.raises(ConformalRegistryError, match="runtime configuration schema"):
        registry.register(
            "fixture",
            FIXTURE_MANIFEST,
            _FixtureConfig,
            mismatch_factory,
        )
    assert registry.available_methods == ()


def test_registration_rejects_default_requirement_beyond_the_state_bound() -> None:
    registry = ConformalRegistry()
    impossible = replace(
        FIXTURE_MANIFEST,
        calibration_requirement=FixedCountRequirement(FIXTURE_MANIFEST.state_bound + 1),
    )

    with pytest.raises(ConformalRegistryError, match="requirement exceeds"):
        registry.register("fixture", impossible, _FixtureConfig, _factory)
    assert registry.available_methods == ()


def test_registration_requires_frozen_extra_forbidding_pydantic_schema() -> None:
    class MutableConfig(BaseModel):
        value: int = 1

    class ExtraConfig(BaseModel):
        model_config = ConfigDict(frozen=True, extra="allow")

        value: int = 1

    for schema, message in (
        (MutableConfig, "frozen"),
        (ExtraConfig, "extra.*forbid"),
    ):
        registry = ConformalRegistry()
        with pytest.raises(ConformalRegistryError, match=message):
            registry.register("fixture", FIXTURE_MANIFEST, schema, _factory)
        assert registry.available_methods == ()


def test_registration_rejects_name_manifest_factory_and_duplicate_failures() -> None:
    registry = ConformalRegistry()
    with pytest.raises(ConformalRegistryError, match="manifest name"):
        registry.register("other", FIXTURE_MANIFEST, _FixtureConfig, _factory)
    with pytest.raises(ConformalRegistryError, match="factory.*callable"):
        registry.register("fixture", FIXTURE_MANIFEST, _FixtureConfig, None)  # type: ignore[arg-type]

    registry.register("fixture", FIXTURE_MANIFEST, _FixtureConfig, _factory)
    with pytest.raises(ConformalRegistryError, match="already registered"):
        registry.register("fixture", FIXTURE_MANIFEST, _FixtureConfig, _factory)


def test_registry_has_no_default_and_lists_methods_in_deterministic_diagnostics() -> None:
    registry = _registry()
    with pytest.raises(
        ConformalRegistryError,
        match=r"explicit 'method'.*available methods: fixture",
    ):
        registry.resolve({"offset": 1.0})
    with pytest.raises(
        ConformalRegistryError,
        match="unknown method 'missing'.*available methods: fixture",
    ):
        registry.resolve({"method": "missing"})


def test_registry_validates_configuration_strictly_and_forbids_extras() -> None:
    registry = _registry()
    with pytest.raises(ConformalRegistryError, match="invalid configuration"):
        registry.resolve({"method": "fixture", "offset": "2.0"})
    with pytest.raises(ConformalRegistryError, match="invalid configuration"):
        registry.resolve({"method": "fixture", "offset": 2.0, "unknown": True})


def test_restoration_validates_every_blob_before_calling_factory() -> None:
    registry = _registry()
    partition = derive_partition_label("fixture-model", "global", EmissionScope.PER_STEP)
    before = _FixtureRuntime.instances
    wrong_method = JsonStateCodec("other", 1).encode(partition, {"count": 0})
    wrong_version = JsonStateCodec("fixture", 2).encode(partition, {"count": 0})
    wrong_label = JsonStateCodec("fixture", 1).encode(
        derive_partition_label("fixture-model", "other", EmissionScope.PER_STEP),
        {"count": 0},
    )

    for state in (wrong_method, wrong_version, wrong_label):
        with pytest.raises(ConformalRegistryError, match="state"):
            registry.resolve(
                {"method": "fixture"},
                states={partition: state},
            )
    assert _FixtureRuntime.instances == before


def test_factory_must_return_a_fresh_runtime_instance() -> None:
    registry = ConformalRegistry()
    singleton = _FixtureRuntime(_FixtureConfig(), {})

    def singleton_factory(
        config: BaseModel,
        states: Mapping[str, bytes],
    ) -> ConformalRuntime:
        del config, states
        return singleton

    with pytest.raises(ConformalRegistryError, match="fresh runtime"):
        registry.register(
            "fixture",
            FIXTURE_MANIFEST,
            _FixtureConfig,
            singleton_factory,
        )


def test_factory_cannot_alternate_between_previously_issued_instances() -> None:
    registry = ConformalRegistry()
    cached = (
        _FixtureRuntime(_FixtureConfig(), {}),
        _FixtureRuntime(_FixtureConfig(), {}),
    )
    call_count = 0

    def alternating_factory(
        config: BaseModel,
        states: Mapping[str, bytes],
    ) -> ConformalRuntime:
        nonlocal call_count
        del config, states
        runtime = cached[call_count % len(cached)]
        call_count += 1
        return runtime

    registry.register(
        "fixture",
        FIXTURE_MANIFEST,
        _FixtureConfig,
        alternating_factory,
    )
    with pytest.raises(ConformalRegistryError, match="fresh runtime"):
        registry.resolve({"method": "fixture"})


@pytest.mark.parametrize("verb", ["calibrate", "apply", "observe"])
@pytest.mark.parametrize(
    "invalid_kind",
    ["corrupt", "wrong-method", "wrong-version", "wrong-label"],
)
def test_registry_rejects_invalid_state_emitted_by_every_runtime_verb(
    verb: str,
    invalid_kind: str,
) -> None:
    partition = derive_partition_label("fixture-model", "global", EmissionScope.PER_STEP)
    other_partition = derive_partition_label("fixture-model", "other", EmissionScope.PER_STEP)
    invalid_states = {
        "corrupt": b"corrupt",
        "wrong-method": JsonStateCodec("other", 1).encode(partition, {"count": 0}),
        "wrong-version": JsonStateCodec("fixture", 2).encode(partition, {"count": 0}),
        "wrong-label": JsonStateCodec("fixture", 1).encode(other_partition, {"count": 0}),
    }
    invalid_state = invalid_states[invalid_kind]

    def invalid_output_factory(
        config: BaseModel,
        states: Mapping[str, bytes],
    ) -> ConformalRuntime:
        assert isinstance(config, _FixtureConfig)
        return _InvalidStateOutputRuntime(config, states, invalid_state)

    registry = ConformalRegistry()
    registry.register(
        "fixture",
        FIXTURE_MANIFEST,
        _FixtureConfig,
        invalid_output_factory,
    )
    runtime = registry.resolve({"method": "fixture"})
    frame = _frame()
    delivery = Delivery(
        partition,
        (_observation("sku", partition_label=partition, actual=7.0),),
    )
    valid_state = ConformalStateBatch(
        {partition: _FixtureCodec().encode_partition(partition, count=0, total=0.0)}
    )

    with pytest.raises(RuntimeContractError, match=f"{verb} emitted invalid state"):
        if verb == "calibrate":
            runtime.calibrate(CalibrationSeedBatch({partition: [1.0]}))
        elif verb == "apply":
            runtime.apply(frame, valid_state)
        else:
            runtime.observe(delivery, ConformalStateBatch())


@pytest.mark.parametrize("mismatch", ["missing", "extra"])
def test_registry_requires_observe_annotations_for_exactly_the_delivered_rows(
    mismatch: str,
) -> None:
    class InvalidAnnotationRuntime(_FixtureRuntime):
        def observe(
            self,
            deliveries: DeliveryBatch,
            state: ConformalStateBatch,
            *,
            context: CalibrationContext | None = None,
        ) -> ObserveEffect:
            effect = super().observe(deliveries, state, context=context)
            if mismatch == "missing":
                annotations = effect.annotations[:-1]
            else:
                extra_key = _observation(
                    "unexpected",
                    partition_label=deliveries.labels[0],
                    actual=7.0,
                ).forecast_key
                annotations = (
                    *effect.annotations,
                    ObserveAnnotation(extra_key, 3.0, None, True),
                )
            return ObserveEffect(effect.state, effect.dirty_labels, annotations)

    def invalid_annotation_factory(
        config: BaseModel,
        states: Mapping[str, bytes],
    ) -> ConformalRuntime:
        assert isinstance(config, _FixtureConfig)
        return InvalidAnnotationRuntime(config, states)

    registry = ConformalRegistry()
    registry.register(
        "fixture",
        FIXTURE_MANIFEST,
        _FixtureConfig,
        invalid_annotation_factory,
    )
    runtime = registry.resolve({"method": "fixture"})
    partition = derive_partition_label("fixture-model", "global", EmissionScope.PER_STEP)
    delivery = Delivery(
        partition,
        (
            _observation("sku-a", partition_label=partition, actual=7.0),
            _observation("sku-b", partition_label=partition, actual=8.0),
        ),
    )

    with pytest.raises(RuntimeContractError, match="exactly cover"):
        runtime.observe(delivery, ConformalStateBatch())


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("method", "method must equal"),
        ("form", "form must equal"),
        ("scope", "scope must equal"),
        ("claim", "descriptor is not declared"),
        ("clamp", "binding is not declared"),
        ("claim-binding", "claim must be voided exactly"),
    ],
)
def test_registry_rejects_issuance_shapes_that_disagree_with_the_manifest(
    invalid_kind: str,
    message: str,
) -> None:
    class InvalidIssuanceRuntime(_FixtureRuntime):
        def apply(
            self,
            forecasts: pd.DataFrame,
            state: ConformalStateBatch,
            *,
            context: CalibrationContext | None = None,
        ) -> CalibrationResult:
            result = super().apply(forecasts, state, context=context)
            issuances = {
                key: _invalid_issuance(facts, invalid_kind)
                for key, facts in result.issuances.items()
            }
            return CalibrationResult(
                result.forecasts,
                result.state,
                result.dirty_labels,
                issuances,
            )

    def factory(config: BaseModel, states: Mapping[str, bytes]) -> ConformalRuntime:
        assert isinstance(config, _FixtureConfig)
        return InvalidIssuanceRuntime(config, states)

    registry = ConformalRegistry()
    registry.register("fixture", FIXTURE_MANIFEST, _FixtureConfig, factory)
    runtime = registry.resolve({"method": "fixture"})
    with pytest.raises(RuntimeContractError, match=message):
        runtime.apply(_frame(), ConformalStateBatch())


def test_registry_rejects_undeclared_post_readiness_nonfinite_bounds() -> None:
    class NonFiniteRuntime(_FixtureRuntime):
        def apply(
            self,
            forecasts: pd.DataFrame,
            state: ConformalStateBatch,
            *,
            context: CalibrationContext | None = None,
        ) -> CalibrationResult:
            result = super().apply(forecasts, state, context=context)
            calibrated = result.forecasts
            calibrated.loc[:, ["lower_0.9", "upper_0.9"]] = float("nan")
            issuances = {
                key: replace(
                    facts,
                    lower_bound=float("nan"),
                    upper_bound=float("nan"),
                    bounds_null_reason="unattributed fallback",
                )
                for key, facts in result.issuances.items()
            }
            return CalibrationResult(
                calibrated,
                result.state,
                result.dirty_labels,
                issuances,
            )

    def factory(config: BaseModel, states: Mapping[str, bytes]) -> ConformalRuntime:
        assert isinstance(config, _FixtureConfig)
        return NonFiniteRuntime(config, states)

    registry = ConformalRegistry()
    registry.register("fixture", FIXTURE_MANIFEST, _FixtureConfig, factory)
    runtime = registry.resolve({"method": "fixture"})
    with pytest.raises(RuntimeContractError, match="post-readiness non-finite"):
        runtime.apply(_frame(), ConformalStateBatch())


def test_fixture_extension_is_test_owned_and_runs_without_engine_changes() -> None:
    fixture_path = Path(inspect.getsourcefile(_FixtureRuntime) or "")

    assert fixture_path.name == "test_conformal_registry.py"
    assert "tests/tier1" in fixture_path.as_posix()
    assert _registry().resolve({"method": "fixture"}).manifest.name == "fixture"
