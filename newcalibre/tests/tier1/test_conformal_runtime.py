"""Exercise conformal runtime values, labels, codecs, and protocol shape."""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import FrozenInstanceError
from typing import Any, cast

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
from newcalibre.conformal.state import JsonStateCodec, StateCodecError, StateScope
from newcalibre.domain import (
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


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    offset: float = 1.0


def _manifest(*, consumes_context: bool = False) -> MethodManifest:
    return MethodManifest(
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
        consumes_calibration_context=consumes_context,
        hosted_submodels=(),
        requires_fitted_values=False,
        post_warmup_non_finite=PostWarmupNonFinite.FORBIDDEN,
        clamps=(),
        joint_claim=JointClaim.NONE,
    )


def _partition(
    value: str | bool | int | float = "global",
    *,
    model_name: str = "fixture-model",
) -> str:
    return derive_partition_label(model_name, value, EmissionScope.PER_STEP)


def _key(series_key: str, *, horizon_step: int = 1) -> ForecastKey:
    return ForecastKey(
        series_key=series_key,
        origin=pd.Timestamp("2025-01-06"),
        horizon_step=horizon_step,
        model_name="fixture-model",
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


def _issued(partition_label: str, *, ready: bool = True) -> IssuedBoundFacts:
    return IssuedBoundFacts(
        method_name="fixture",
        emission_form=EmissionForm.ONE_SIDED_UPPER,
        emission_scope=EmissionScope.PER_STEP,
        partition_label=partition_label,
        working_level=0.9,
        state_reference="fixture:7",
        lower_bound=0.0 if ready else math.nan,
        upper_bound=8.0 if ready else math.nan,
        calibration_ready=ready,
        bounds_null_reason=None if ready else "warm-up",
        effective_descriptor=_descriptor(),
    )


def _observation(
    series_key: str,
    *,
    partition_label: str | None = None,
    horizon_step: int = 1,
    actual: float = 5.0,
) -> ResolvedObservation:
    label = partition_label or _partition()
    return ResolvedObservation(
        forecast_key=_key(series_key, horizon_step=horizon_step),
        target_timestamp=pd.Timestamp("2025-01-06") + pd.Timedelta(weeks=horizon_step - 1),
        actual=actual,
        point_forecast=4.0,
        censoring_assertion=CensoringAssertion.UNCENSORED,
        availability_bound=None,
        issued=_issued(label),
    )


class _Runtime:
    def __init__(self, *, consumes_context: bool = False) -> None:
        self._manifest = _manifest(consumes_context=consumes_context)
        self._config = _Config()
        self._codec = JsonStateCodec("fixture", 1)

    @property
    def manifest(self) -> MethodManifest:
        return self._manifest

    @property
    def config(self) -> BaseModel:
        return self._config

    def calibrate(self, seeds: CalibrationSeedBatch) -> ConformalStateBatch:
        return ConformalStateBatch(
            {
                label: self._codec.encode(label, {"scores": list(values)})
                for label, values in seeds.items()
            }
        )

    def apply(
        self,
        forecasts: pd.DataFrame,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        del state
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(forecasts["series_key"]),
        )
        return CalibrationResult(forecasts, ConformalStateBatch())

    def observe(
        self,
        deliveries: DeliveryBatch,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        del state
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(
                observation.forecast_key.series_key for observation in deliveries.observations
            ),
        )
        annotations = tuple(
            ObserveAnnotation(
                forecast_key=observation.forecast_key,
                score=abs(observation.actual - observation.point_forecast),
                exclusion_cause=None,
                advanced_delivered_score=True,
            )
            for observation in deliveries.observations
        )
        return ObserveEffect(ConformalStateBatch(), annotations=annotations)


def test_partition_labels_are_injective_across_delimiters_unicode_and_types() -> None:
    witnesses = {
        derive_partition_label("a:b", "c", EmissionScope.PER_STEP),
        derive_partition_label("a", "b:c", EmissionScope.PER_STEP),
        derive_partition_label("模型", "é", EmissionScope.PER_STEP),
        derive_partition_label("模型", "e\u0301", EmissionScope.PER_STEP),
        derive_partition_label("model", "1", EmissionScope.PER_STEP),
        derive_partition_label("model", 1, EmissionScope.PER_STEP),
        derive_partition_label("model", 1.0, EmissionScope.PER_STEP),
        derive_partition_label("model", True, EmissionScope.PER_STEP),
        derive_partition_label("model", "global", EmissionScope.WINDOW_SUM),
    }

    assert len(witnesses) == 9
    assert all(label.startswith("p1.") for label in witnesses)
    assert not METHOD_SCOPE_LABEL.startswith("p1.")


def test_partition_label_reserved_namespace_cannot_be_forged_by_data() -> None:
    forged_as_model = derive_partition_label(
        METHOD_SCOPE_LABEL,
        "global",
        EmissionScope.PER_STEP,
    )
    forged_as_value = derive_partition_label(
        "model",
        METHOD_SCOPE_LABEL,
        EmissionScope.PER_STEP,
    )

    assert forged_as_model != METHOD_SCOPE_LABEL
    assert forged_as_value != METHOD_SCOPE_LABEL
    assert JsonStateCodec("fixture", 1).scope_for(METHOD_SCOPE_LABEL) is StateScope.METHOD
    assert JsonStateCodec("fixture", 1).scope_for(forged_as_value) is StateScope.PARTITION


@pytest.mark.parametrize(
    ("model_name", "partition_value", "horizon_scope"),
    [
        ("", "value", EmissionScope.PER_STEP),
        (" model", "value", EmissionScope.PER_STEP),
        ("model", ["unsupported"], EmissionScope.PER_STEP),
        ("model", math.inf, EmissionScope.PER_STEP),
        ("model", "\ud800", EmissionScope.PER_STEP),
        ("model", "value", "per-step"),
    ],
)
def test_partition_label_rejects_malformed_inputs(
    model_name: object,
    partition_value: object,
    horizon_scope: object,
) -> None:
    with pytest.raises(RuntimeContractError):
        derive_partition_label(
            cast(Any, model_name),
            cast(Any, partition_value),
            cast(Any, horizon_scope),
        )


def test_delivery_preserves_partition_row_order_and_snapshots_observations() -> None:
    label = _partition()
    first = _observation("z-series", partition_label=label, horizon_step=2)
    second = _observation("a-series", partition_label=label, horizon_step=1)
    supplied = [first, second]

    delivery = DeliveryBatch({label: cast(Any, supplied)})
    supplied.reverse()

    assert delivery.observations_for(label) == (first, second)
    assert delivery.observations == (second, first)
    assert [item.forecast_key.series_key for item in delivery.observations] == [
        "a-series",
        "z-series",
    ]
    with pytest.raises(FrozenInstanceError):
        cast(Any, delivery)._labels = ("changed",)


def test_delivery_validates_complete_keys_values_censoring_and_issued_facts() -> None:
    label = _partition()
    observation = _observation("series", partition_label=label)
    assert DeliveryBatch({label: (observation,)}).observations == (observation,)

    with pytest.raises(RuntimeContractError, match="partition"):
        DeliveryBatch({_partition("other"): (observation,)})
    with pytest.raises(RuntimeContractError, match="duplicate forecast key"):
        DeliveryBatch({label: (observation, observation)})

    other_label = _partition("other")
    duplicated_key = _observation("series", partition_label=other_label)
    with pytest.raises(RuntimeContractError, match="duplicate forecast key"):
        DeliveryBatch(
            {
                label: (observation,),
                other_label: (duplicated_key,),
            }
        )

    for field, value, message in (
        ("series_key", "", "series key"),
        ("origin", pd.Timestamp("NaT"), "origin"),
        ("horizon_step", 0, "horizon step"),
        ("model_name", "", "model name"),
    ):
        values = {
            "series_key": "series",
            "origin": pd.Timestamp("2025-01-06"),
            "horizon_step": 1,
            "model_name": "model",
        }
        values[field] = value
        with pytest.raises(RuntimeContractError, match=message):
            ForecastKey(**values)  # type: ignore[arg-type]

    for field, value in (
        ("actual", math.nan),
        ("actual", math.inf),
        ("point_forecast", math.nan),
        ("availability_bound", math.inf),
    ):
        values = {
            "forecast_key": observation.forecast_key,
            "target_timestamp": observation.target_timestamp,
            "actual": observation.actual,
            "point_forecast": observation.point_forecast,
            "censoring_assertion": observation.censoring_assertion,
            "availability_bound": observation.availability_bound,
            "issued": observation.issued,
        }
        values[field] = value
        with pytest.raises(RuntimeContractError, match=field.replace("_", " ")):
            ResolvedObservation(**values)  # type: ignore[arg-type]

    with pytest.raises(RuntimeContractError, match="censoring assertion"):
        ResolvedObservation(
            observation.forecast_key,
            observation.target_timestamp,
            observation.actual,
            observation.point_forecast,
            cast(Any, "uncensored"),
            None,
            observation.issued,
        )
    with pytest.raises(RuntimeContractError, match="bounds null reason"):
        _issued(label, ready=True).__class__(
            method_name="fixture",
            emission_form=EmissionForm.ONE_SIDED_UPPER,
            emission_scope=EmissionScope.PER_STEP,
            partition_label=label,
            working_level=0.9,
            state_reference="fixture:7",
            lower_bound=0.0,
            upper_bound=8.0,
            calibration_ready=True,
            bounds_null_reason="warm-up",
            effective_descriptor=_descriptor(),
        )


def test_issued_bound_facts_accept_unclipped_finite_working_levels() -> None:
    label = _partition()

    for working_level in (-2.0, 3.0):
        facts = IssuedBoundFacts(
            method_name="fixture",
            emission_form=EmissionForm.ONE_SIDED_UPPER,
            emission_scope=EmissionScope.PER_STEP,
            partition_label=label,
            working_level=working_level,
            state_reference="fixture:7",
            lower_bound=0.0,
            upper_bound=8.0,
            calibration_ready=True,
            bounds_null_reason=None,
            effective_descriptor=_descriptor(),
        )
        assert facts.working_level == working_level
        assert facts.effective_descriptor.level == 0.9

    for working_level in (math.nan, math.inf, -math.inf):
        with pytest.raises(RuntimeContractError, match="working level"):
            IssuedBoundFacts(
                method_name="fixture",
                emission_form=EmissionForm.ONE_SIDED_UPPER,
                emission_scope=EmissionScope.PER_STEP,
                partition_label=label,
                working_level=working_level,
                state_reference="fixture:7",
                lower_bound=0.0,
                upper_bound=8.0,
                calibration_ready=True,
                bounds_null_reason=None,
                effective_descriptor=_descriptor(),
            )


def test_issued_bound_facts_reject_impossible_finite_bound_states() -> None:
    label = _partition()
    common = {
        "method_name": "fixture",
        "emission_form": EmissionForm.ONE_SIDED_UPPER,
        "emission_scope": EmissionScope.PER_STEP,
        "partition_label": label,
        "working_level": 0.9,
        "state_reference": "fixture:7",
        "bounds_null_reason": None,
        "effective_descriptor": _descriptor(),
    }

    with pytest.raises(RuntimeContractError, match="lower bound cannot exceed"):
        IssuedBoundFacts(
            **common,
            lower_bound=9.0,
            upper_bound=8.0,
            calibration_ready=True,
        )
    with pytest.raises(RuntimeContractError, match="require calibration readiness"):
        IssuedBoundFacts(
            **common,
            lower_bound=0.0,
            upper_bound=8.0,
            calibration_ready=False,
        )

    declared_non_finite = IssuedBoundFacts(
        **(common | {"bounds_null_reason": "method-declared non-finite"}),
        lower_bound=math.nan,
        upper_bound=math.nan,
        calibration_ready=True,
    )
    assert declared_non_finite.calibration_ready
    assert math.isnan(declared_non_finite.lower_bound)


def test_calibration_context_exposes_only_row_aligned_immutable_hierarchy_facts() -> None:
    context = CalibrationContext(
        series_keys=cast(Any, ["sku-b", "sku-a"]),
        lattice_levels=cast(Any, ["bottom", "aggregate"]),
        aggregate_memberships=cast(Any, [["dept:b", "total"], ["total"]]),
    )

    assert context.series_keys == ("sku-b", "sku-a")
    assert context.lattice_levels == ("bottom", "aggregate")
    assert context.aggregate_memberships == (("dept:b", "total"), ("total",))
    assert context.class_assignments == (
        ("bottom", ("dept:b", "total")),
        ("aggregate", ("total",)),
    )
    assert not hasattr(context, "hierarchy_index")
    assert not hasattr(context, "engine")
    with pytest.raises(FrozenInstanceError):
        cast(Any, context).series_keys = ()


def test_context_permutation_changes_only_corresponding_assignment_order() -> None:
    original = CalibrationContext(
        series_keys=("a", "b"),
        lattice_levels=("bottom", "aggregate"),
        aggregate_memberships=(("total",), ("region:x", "total")),
    )
    permuted = CalibrationContext(
        series_keys=("b", "a"),
        lattice_levels=("aggregate", "bottom"),
        aggregate_memberships=(("region:x", "total"), ("total",)),
    )

    assert permuted.class_assignments == tuple(reversed(original.class_assignments))
    assert permuted.series_keys == tuple(reversed(original.series_keys))


@pytest.mark.parametrize(
    "arguments",
    [
        {
            "series_keys": ("a",),
            "lattice_levels": (),
            "aggregate_memberships": (("total",),),
        },
        {
            "series_keys": ("a",),
            "lattice_levels": ("bottom",),
            "aggregate_memberships": (("total", "total"),),
        },
        {
            "series_keys": ("",),
            "lattice_levels": ("bottom",),
            "aggregate_memberships": (("total",),),
        },
    ],
)
def test_calibration_context_rejects_misaligned_or_malformed_facts(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(RuntimeContractError):
        CalibrationContext(**arguments)  # type: ignore[arg-type]


def test_observe_annotations_enforce_score_exclusion_and_advancement_consistency() -> None:
    key = _key("series")
    scored = ObserveAnnotation(key, 1.5, None, True)
    excluded = ObserveAnnotation(key, None, "censored", False)
    assert scored.score == 1.5
    assert excluded.exclusion_cause == "censored"

    for score, cause, advanced in (
        (None, None, False),
        (1.0, "censored", True),
        (None, "censored", True),
        (math.nan, None, True),
    ):
        with pytest.raises(RuntimeContractError):
            ObserveAnnotation(key, score, cause, advanced)


def test_effect_and_calibration_result_snapshot_values_and_require_bytes() -> None:
    key = _key("series")
    annotations = [ObserveAnnotation(key, 1.0, None, True)]
    updates = {_partition(): b"state"}
    effect = ObserveEffect(ConformalStateBatch(updates), updates, cast(Any, annotations))
    annotations.clear()
    updates.clear()

    assert len(effect.annotations) == 1
    assert list(effect.dirty_state.values()) == [b"state"]
    with pytest.raises(TypeError):
        cast(Any, effect.dirty_state)["new"] = b"state"
    with pytest.raises(RuntimeContractError, match="bytes"):
        ObserveEffect(  # type: ignore[dict-item]
            ConformalStateBatch({_partition(): bytearray(b"state")}), ()
        )

    frame = pd.DataFrame({"series_key": ["a"], "point_forecast": [1.0]})
    result = CalibrationResult(frame, ConformalStateBatch({_partition(): b"state"}))
    frame.loc[0, "point_forecast"] = 99.0
    returned = result.forecasts
    returned.loc[0, "point_forecast"] = 88.0
    assert result.forecasts.loc[0, "point_forecast"] == 1.0


def test_calibration_result_requires_exact_row_keyed_issuance_for_owned_bounds() -> None:
    label = _partition()
    frame = pd.DataFrame(
        {
            "series_key": pd.Series(["series"], dtype="string"),
            "origin": pd.to_datetime(["2025-01-06"]),
            "horizon_step": pd.Series([1], dtype="int64"),
            "model_name": pd.Series(["fixture-model"], dtype="string"),
            "lower_0.9": pd.Series([0.0], dtype="float64"),
            "upper_0.9": pd.Series([8.0], dtype="float64"),
        }
    )
    key = _key("series")

    raw_alpha_facts = IssuedBoundFacts(
        method_name="fixture",
        emission_form=EmissionForm.ONE_SIDED_UPPER,
        emission_scope=EmissionScope.PER_STEP,
        partition_label=label,
        working_level=-0.25,
        state_reference="fixture:7",
        lower_bound=0.0,
        upper_bound=8.0,
        calibration_ready=True,
        bounds_null_reason=None,
        effective_descriptor=_descriptor(),
    )
    result = CalibrationResult(frame, ConformalStateBatch(), issuances={key: raw_alpha_facts})
    assert result.issuances[key].upper_bound == 8.0
    assert result.issuances[key].working_level == -0.25
    with pytest.raises(RuntimeContractError, match="exactly cover"):
        CalibrationResult(frame, ConformalStateBatch())
    with pytest.raises(RuntimeContractError, match="exactly cover"):
        CalibrationResult(frame, ConformalStateBatch(), issuances={})
    with pytest.raises(RuntimeContractError, match="bounds must equal"):
        CalibrationResult(
            frame,
            ConformalStateBatch(),
            issuances={
                key: _issued(label).__class__(
                    method_name="fixture",
                    emission_form=EmissionForm.ONE_SIDED_UPPER,
                    emission_scope=EmissionScope.PER_STEP,
                    partition_label=label,
                    working_level=0.9,
                    state_reference="fixture:7",
                    lower_bound=0.0,
                    upper_bound=9.0,
                    calibration_ready=True,
                    bounds_null_reason=None,
                    effective_descriptor=_descriptor(),
                )
            },
        )


def test_context_presence_must_match_manifest_and_row_alignment() -> None:
    context = CalibrationContext(("a",), ("bottom",), (("total",),))
    require_calibration_context(_manifest(consumes_context=True), context, series_keys=("a",))

    with pytest.raises(RuntimeContractError, match="required"):
        require_calibration_context(
            _manifest(consumes_context=True),
            None,
            series_keys=("a",),
        )
    with pytest.raises(RuntimeContractError, match="must be absent"):
        require_calibration_context(
            _manifest(consumes_context=False),
            context,
            series_keys=("a",),
        )
    with pytest.raises(RuntimeContractError, match="row alignment"):
        require_calibration_context(
            _manifest(consumes_context=True),
            context,
            series_keys=("b",),
        )


def test_state_codec_round_trip_is_canonical_stable_and_scope_independent() -> None:
    codec = JsonStateCodec("fixture", 1)
    partition = _partition("a:b")
    payload = {"count": 2, "scores": [1.0, 2.5], "label": "é"}

    first = codec.encode(partition, payload)
    second = codec.encode(partition, payload)
    method = codec.encode(METHOD_SCOPE_LABEL, {"issue_counter": 7})

    assert first == second
    assert codec.decode(first, expected_label=partition) == payload
    assert codec.decode(method, expected_label=METHOD_SCOPE_LABEL) == {"issue_counter": 7}
    assert codec.address(first).scope is StateScope.PARTITION
    assert codec.address(method).scope is StateScope.METHOD
    assert codec.address(first).label != codec.address(method).label


def test_state_codec_strictly_refuses_wrong_method_version_and_malformed_bytes() -> None:
    codec = JsonStateCodec("fixture", 1)
    label = _partition()
    state = codec.encode(label, {"count": 1})

    with pytest.raises(StateCodecError, match="wrong method"):
        JsonStateCodec("other", 1).decode(state)
    with pytest.raises(StateCodecError, match="unsupported state schema version"):
        JsonStateCodec("fixture", 2).decode(state)
    with pytest.raises(StateCodecError, match="expected label"):
        codec.decode(state, expected_label=_partition("other"))
    with pytest.raises(StateCodecError, match="canonical"):
        codec.decode(state.replace(b"{", b"{ ", 1))
    with pytest.raises(StateCodecError, match="UTF-8"):
        codec.decode(b"\xff")

    raw = json.loads(state)
    raw["unknown"] = True
    with pytest.raises(StateCodecError, match="exact fields"):
        codec.decode(json.dumps(raw, separators=(",", ":"), sort_keys=True).encode())

    duplicate = state[:-1] + b',"schema":"newcalibre.conformal-state"}'
    with pytest.raises(StateCodecError, match="duplicate"):
        codec.decode(duplicate)


def test_state_codec_rejects_nonfinite_payloads_and_malformed_partition_scope() -> None:
    codec = JsonStateCodec("fixture", 1)
    with pytest.raises(StateCodecError, match="non-finite"):
        codec.encode(_partition(), {"score": math.nan})
    with pytest.raises(StateCodecError, match="state label"):
        codec.encode("p1.not-base64!", {"score": 1.0})

    state = codec.encode(_partition(), {"score": 1.0})
    raw = json.loads(state)
    raw["scope"] = "method"
    malformed = json.dumps(
        raw,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    with pytest.raises(StateCodecError, match="method scope"):
        codec.decode(malformed)


def test_runtime_protocol_is_conforming_and_has_no_mutating_load_path() -> None:
    runtime = _Runtime()
    assert isinstance(runtime, ConformalRuntime)
    assert not hasattr(runtime, "load_state")
    assert "load_state" not in inspect.getsource(ConformalRuntime)

    label = _partition()
    states = runtime.calibrate(CalibrationSeedBatch({label: [1.0, 2.0]}))
    assert JsonStateCodec("fixture", 1).decode(states[label])["scores"] == [1.0, 2.0]  # type: ignore[index]

    frame = pd.DataFrame({"series_key": ["series"], "point_forecast": [4.0]})
    result = runtime.apply(frame, ConformalStateBatch({label: states[label]}))
    effect = runtime.observe(
        DeliveryBatch({label: (_observation("series", partition_label=label),)}),
        ConformalStateBatch({label: states[label]}),
    )
    assert result.forecasts.equals(frame)
    assert effect.annotations[0].advanced_delivered_score
