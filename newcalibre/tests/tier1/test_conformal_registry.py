"""Exercise conformal registration and a test-owned three-verb fixture method."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict

import newcalibre.conformal as conformal_package
from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    AssumptionClass,
    CalibrationContext,
    CalibrationResult,
    CensoringPolicy,
    ConformalRegistry,
    ConformalRegistryError,
    ConformalRuntime,
    Delivery,
    EmissionForm,
    ForecastKey,
    GuaranteeDeclaration,
    IssuedBoundFacts,
    JointClaim,
    MethodManifest,
    ObserveAnnotation,
    ObserveEffect,
    PostWarmupNonFinite,
    ResolvedObservation,
    derive_partition_label,
    require_calibration_context,
)
from newcalibre.conformal.state import JsonStateCodec
from newcalibre.domain import (
    CensoringAssertion,
    EmissionScope,
    GuaranteeClaim,
    GuaranteeCurrency,
)

pytestmark = pytest.mark.tier1


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
    minimum_calibration_scores=1,
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
    instances = 0

    def __init__(
        self,
        config: _FixtureConfig,
        states: Mapping[str, bytes],
    ) -> None:
        type(self).instances += 1
        self.instance_number = type(self).instances
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
        scores: Mapping[str, Sequence[float]],
    ) -> Mapping[str, bytes]:
        states = {
            label: self._codec.encode_partition(
                label,
                count=len(values),
                total=sum(values),
            )
            for label, values in scores.items()
        }
        states[METHOD_SCOPE_LABEL] = self._codec.encode_method(issue_counter=0)
        return states

    def apply(
        self,
        forecasts: pd.DataFrame,
        states: Mapping[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        del states
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(forecasts["series_key"]),
        )
        calibrated = forecasts.copy(deep=True)
        calibrated["lower_0.9"] = 0.0
        calibrated["upper_0.9"] = calibrated["point_forecast"] + self._config.offset
        return CalibrationResult(calibrated, {})

    def observe(
        self,
        delivery: Delivery,
        states: Mapping[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(
                observation.forecast_key.series_key for observation in delivery.observations
            ),
        )
        current = states.get(delivery.partition_label)
        count, total = (
            (0, 0.0)
            if current is None
            else self._codec.decode_partition(current, label=delivery.partition_label)
        )
        annotations = tuple(
            ObserveAnnotation(
                forecast_key=observation.forecast_key,
                score=abs(observation.actual - observation.point_forecast),
                exclusion_cause=None,
                advanced_delivered_score=True,
            )
            for observation in delivery.observations
        )
        score_total = sum(annotation.score or 0.0 for annotation in annotations)
        update = self._codec.encode_partition(
            delivery.partition_label,
            count=count + len(annotations),
            total=total + score_total,
        )
        return ObserveEffect({delivery.partition_label: update}, annotations)


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
            partition_label=partition_label,
            working_level=0.9,
            state_reference="fixture:0",
            lower_bound=0.0,
            upper_bound=5.0,
            calibration_ready=True,
            bounds_null_reason=None,
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
    partition = derive_partition_label("fixture-model", "global", EmissionScope.PER_STEP)

    calibrated_states = original.calibrate({partition: [1.0, 2.0]})
    restored = registry.resolve({"method": "fixture"}, states=calibrated_states)
    frame = pd.DataFrame({"series_key": ["sku"], "point_forecast": [4.0]})
    applied = restored.apply(frame, calibrated_states)
    observed = restored.observe(
        Delivery(partition, (_observation("sku", partition_label=partition, actual=7.0),)),
        calibrated_states,
    )

    assert restored is not original
    assert isinstance(restored, _FixtureRuntime)
    assert restored.instance_number != original.instance_number
    assert restored.restored_states == calibrated_states
    assert applied.forecasts.loc[0, "upper_0.9"] == 5.0
    assert observed.annotations[0].score == 3.0
    assert _FixtureCodec().decode_partition(
        observed.state_updates[partition],
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


def test_fixture_extension_and_production_registry_have_no_engine_coupling() -> None:
    fixture_path = Path(inspect.getsourcefile(_FixtureRuntime) or "")
    conformal_root = Path(inspect.getsourcefile(conformal_package) or "").parent
    production_sources = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(conformal_root.glob("*.py"))
    )

    assert fixture_path.name == "test_conformal_registry.py"
    assert "tests/tier1" in fixture_path.as_posix()
    assert "newcalibre.engine" not in production_sources
    assert "_FixtureRuntime" not in production_sources
    assert _registry().resolve({"method": "fixture"}).manifest.name == "fixture"
