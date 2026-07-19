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
    CensoringPolicy,
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
    RuntimeContractError,
    derive_partition_label,
    require_calibration_context,
)
from newcalibre.conformal.state import JsonStateCodec, StateCodecError, StateScope
from newcalibre.domain import CensoringAssertion, EmissionScope, GuaranteeClaim, GuaranteeCurrency

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
        minimum_calibration_scores=1,
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


def _issued(partition_label: str, *, ready: bool = True) -> IssuedBoundFacts:
    return IssuedBoundFacts(
        method_name="fixture",
        partition_label=partition_label,
        working_level=0.9,
        state_reference="fixture:7",
        lower_bound=0.0 if ready else math.nan,
        upper_bound=8.0 if ready else math.nan,
        calibration_ready=ready,
        bounds_null_reason=None if ready else "warm-up",
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

    def calibrate(self, scores: dict[str, list[float]]) -> dict[str, bytes]:
        return {
            label: self._codec.encode(label, {"scores": list(values)})
            for label, values in scores.items()
        }

    def apply(
        self,
        forecasts: pd.DataFrame,
        states: dict[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        del states
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(forecasts["series_key"]),
        )
        return CalibrationResult(forecasts, {})

    def observe(
        self,
        delivery: Delivery,
        states: dict[str, bytes | None],
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        del states
        require_calibration_context(
            self.manifest,
            context,
            series_keys=tuple(
                observation.forecast_key.series_key for observation in delivery.observations
            ),
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
        return ObserveEffect(state_updates={}, annotations=annotations)


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


def test_delivery_preserves_caller_order_and_defensively_snapshots_observations() -> None:
    label = _partition()
    first = _observation("z-series", partition_label=label, horizon_step=2)
    second = _observation("a-series", partition_label=label, horizon_step=1)
    supplied = [first, second]

    delivery = Delivery(partition_label=label, observations=cast(Any, supplied))
    supplied.reverse()

    assert delivery.observations == (first, second)
    assert [item.forecast_key.series_key for item in delivery.observations] == [
        "z-series",
        "a-series",
    ]
    with pytest.raises(FrozenInstanceError):
        cast(Any, delivery).partition_label = "changed"


def test_delivery_validates_complete_keys_values_censoring_and_issued_facts() -> None:
    label = _partition()
    observation = _observation("series", partition_label=label)
    assert Delivery(label, (observation,)).observations == (observation,)

    with pytest.raises(RuntimeContractError, match="partition"):
        Delivery(_partition("other"), (observation,))
    with pytest.raises(RuntimeContractError, match="duplicate forecast key"):
        Delivery(label, (observation, observation))

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
            partition_label=label,
            working_level=0.9,
            state_reference="fixture:7",
            lower_bound=0.0,
            upper_bound=8.0,
            calibration_ready=True,
            bounds_null_reason="warm-up",
        )


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
    effect = ObserveEffect(updates, cast(Any, annotations))
    annotations.clear()
    updates.clear()

    assert len(effect.annotations) == 1
    assert list(effect.state_updates.values()) == [b"state"]
    with pytest.raises(TypeError):
        cast(Any, effect.state_updates)["new"] = b"state"
    with pytest.raises(RuntimeContractError, match="bytes"):
        ObserveEffect({_partition(): bytearray(b"state")}, ())  # type: ignore[dict-item]

    frame = pd.DataFrame({"series_key": ["a"], "point_forecast": [1.0]})
    result = CalibrationResult(frame, {_partition(): b"state"})
    frame.loc[0, "point_forecast"] = 99.0
    returned = result.forecasts
    returned.loc[0, "point_forecast"] = 88.0
    assert result.forecasts.loc[0, "point_forecast"] == 1.0


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
    states = runtime.calibrate({label: [1.0, 2.0]})
    assert JsonStateCodec("fixture", 1).decode(states[label])["scores"] == [1.0, 2.0]  # type: ignore[index]

    frame = pd.DataFrame({"series_key": ["series"], "point_forecast": [4.0]})
    result = runtime.apply(frame, {label: states[label]})
    effect = runtime.observe(
        Delivery(label, (_observation("series", partition_label=label),)),
        {label: states[label]},
    )
    assert result.forecasts.equals(frame)
    assert effect.annotations[0].advanced_delivered_score
