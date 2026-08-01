"""Exercise complete-only resolution and canonical observe-loop delivery."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    CalibrationContext,
    CalibrationResult,
    CalibrationSeedBatch,
    ConformalRuntime,
    ConformalStateBatch,
    DeliveryBatch,
    ForecastKey,
    ObserveEffect,
    derive_partition_label,
    resolve_method,
)
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    CensoringAssertion,
    EmissionScope,
    HierarchyIndex,
    interval_columns,
)
from newcalibre.observe import (
    ActualRecord,
    ActualsSubmission,
    ObserveError,
    ObserveLoop,
    PendingObservation,
)

pytestmark = pytest.mark.tier1
_MODEL = "fixture-model"
_ISSUE_ORIGIN = pd.Timestamp("2026-01-01")
_TARGET = pd.Timestamp("2026-01-02")
_CYCLE_ORIGIN = pd.Timestamp("2026-01-03")


class _CountingRuntime:
    def __init__(
        self,
        delegate: ConformalRuntime,
        *,
        update_method_state: bool = False,
    ) -> None:
        self.delegate = delegate
        self.update_method_state = update_method_state
        self.calls: list[tuple[str, ...]] = []
        self.states: list[dict[str, bytes]] = []

    @property
    def manifest(self):  # type: ignore[no-untyped-def]
        return self.delegate.manifest

    @property
    def config(self) -> BaseModel:
        return self.delegate.config

    def calibrate(self, seeds: CalibrationSeedBatch) -> ConformalStateBatch:
        return self.delegate.calibrate(seeds)

    def apply(
        self,
        forecasts: pd.DataFrame,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> CalibrationResult:
        return self.delegate.apply(forecasts, state, context=context)

    def observe(
        self,
        deliveries: DeliveryBatch,
        state: ConformalStateBatch,
        *,
        context: CalibrationContext | None = None,
    ) -> ObserveEffect:
        self.calls.append(deliveries.labels)
        self.states.append(dict(state))
        delegate_state = state
        if self.update_method_state:
            delegate_state = ConformalStateBatch(
                {label: value for label, value in state.items() if label != METHOD_SCOPE_LABEL}
            )
        effect = self.delegate.observe(deliveries, delegate_state, context=context)
        if not self.update_method_state:
            return effect
        method_state = b"method-update-1"
        post_state = effect.state.with_rows({METHOD_SCOPE_LABEL: method_state})
        return ObserveEffect(
            post_state,
            (*effect.dirty_labels, METHOD_SCOPE_LABEL),
            effect.annotations,
        )


def _hierarchy() -> HierarchyIndex:
    return HierarchyIndex.from_facts(
        pd.DataFrame(
            {
                SERIES_KEY: ["sku-a", "sku-b", "sku-c"],
                "category": ["tops", "tops", "bottoms"],
            }
        ),
        bottom_series=("sku-c", "sku-b", "sku-a"),
    )


def _frame(
    rows: tuple[tuple[str, pd.Timestamp, int, pd.Timestamp, float, str], ...],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            SERIES_KEY: pd.Series([row[0] for row in rows], dtype="string"),
            TARGET_TIMESTAMP: pd.to_datetime([row[3] for row in rows]),
            ACTUAL_VALUE: pd.Series([math.nan] * len(rows), dtype="float64"),
            POINT_FORECAST: pd.Series([row[4] for row in rows], dtype="float64"),
            HORIZON_STEP: pd.Series([row[2] for row in rows], dtype="int64"),
            ORIGIN: pd.to_datetime([row[1] for row in rows]),
            MODEL_NAME: pd.Series([row[5] for row in rows], dtype="string"),
        }
    )


def _issued_pending(
    runtime: ConformalRuntime,
    rows: tuple[tuple[str, pd.Timestamp, int, pd.Timestamp, float, str], ...],
    states: ConformalStateBatch,
) -> tuple[PendingObservation, ...]:
    result = runtime.apply(_frame(rows), states)
    pending: list[PendingObservation] = []
    for row in result.forecasts.to_dict("records"):
        key = ForecastKey(
            row[SERIES_KEY],
            pd.Timestamp(row[ORIGIN]),
            row[HORIZON_STEP],
            row[MODEL_NAME],
        )
        pending.append(
            PendingObservation(
                forecast_key=key,
                target_timestamp=pd.Timestamp(row[TARGET_TIMESTAMP]),
                point_forecast=row[POINT_FORECAST],
                issued=result.issuances[key],
            )
        )
    return tuple(pending)


def _label(
    series_key: str,
    scope: EmissionScope,
    *,
    model: str = _MODEL,
) -> str:
    return derive_partition_label(model, series_key, scope)


def _actual(
    series_key: str,
    value: int | float,
    *,
    assertion: CensoringAssertion | None = CensoringAssertion.UNCENSORED,
    bound: float | None = None,
    timestamp: pd.Timestamp = _TARGET,
) -> ActualRecord:
    return ActualRecord(series_key, timestamp, value, assertion, bound)


def test_runtime_free_cycle_resolves_due_bottom_and_complete_aggregate_rows() -> None:
    hierarchy = _hierarchy()
    aggregate = next(
        node.label
        for node in hierarchy.nodes
        if node.label.startswith("__aggregate__") and node.members == ("sku-a", "sku-b")
    )
    pending = (
        PendingObservation(ForecastKey(aggregate, _ISSUE_ORIGIN, 1, _MODEL), _TARGET, 8.0),
        PendingObservation(ForecastKey("sku-a", _ISSUE_ORIGIN, 1, _MODEL), _TARGET, 3.0),
    )
    loop = ObserveLoop(hierarchy=hierarchy, pending_observations=pending)
    loop.accept(
        ActualsSubmission(
            (
                _actual("sku-a", 2, assertion=CensoringAssertion.UNCENSORED, bound=5.0),
                _actual("sku-c", 100),
            )
        )
    )

    at_target = loop.cycle(_TARGET)
    assert at_target.resolutions == ()
    assert at_target.pending_retentions == pending

    incomplete = loop.cycle(_CYCLE_ORIGIN)
    assert [value.forecast_key.series_key for value in incomplete.resolutions] == ["sku-a"]
    assert incomplete.resolutions[0].availability_bound == 5.0
    assert [value.forecast_key.series_key for value in incomplete.pending_retentions] == [aggregate]

    resumed = ObserveLoop(
        hierarchy=hierarchy,
        observed_history=incomplete.history_appends,
        pending_observations=incomplete.pending_retentions,
    )
    resumed.accept(ActualsSubmission((_actual("sku-b", 7, assertion=CensoringAssertion.CENSORED),)))
    complete = resumed.cycle(_CYCLE_ORIGIN)

    assert len(complete.resolutions) == 1
    resolution = complete.resolutions[0]
    assert resolution.forecast_key.series_key == aggregate
    assert resolution.actual == 9
    assert type(resolution.actual) is int
    assert resolution.censoring_assertion is CensoringAssertion.CENSORED
    assert resolution.availability_bound is None
    assert complete.pending_retentions == ()
    assert len(complete.deliveries) == 0
    assert complete.state_updates == {}


def test_canonical_delivery_calls_each_partition_once_without_cross_partition_state() -> None:
    hierarchy = _hierarchy()
    delegate = resolve_method(
        {"method": "split-per-step", "coverage": 0.5, "partition_by": "series"}
    )
    runtime = _CountingRuntime(delegate, update_method_state=True)
    labels = {
        series: _label(series, EmissionScope.PER_STEP) for series in ("sku-a", "sku-b", "sku-c")
    }
    states = runtime.calibrate(
        CalibrationSeedBatch({label: [1.0, 2.0] for label in labels.values()})
    )
    rows = (
        ("sku-b", _ISSUE_ORIGIN, 2, _TARGET, 5.0, _MODEL),
        ("sku-a", _ISSUE_ORIGIN, 2, _TARGET, 4.0, _MODEL),
        ("sku-a", _ISSUE_ORIGIN, 1, _TARGET, 3.0, _MODEL),
        ("sku-b", _ISSUE_ORIGIN, 1, _TARGET, 2.0, _MODEL),
    )
    pending = _issued_pending(runtime, rows, states)
    loop = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=pending,
        conformal_states=states,
        runtime=runtime,
    )
    loop.accept(ActualsSubmission((_actual("sku-b", 9), _actual("sku-a", 6))))

    cycle = loop.cycle(_CYCLE_ORIGIN)

    expected_labels = tuple(sorted((labels["sku-a"], labels["sku-b"]), key=str.encode))
    assert runtime.calls == [expected_labels]
    assert runtime.states == [dict(states)]
    assert cycle.deliveries.labels == expected_labels
    assert [
        (observation.forecast_key.series_key, observation.forecast_key.horizon_step)
        for observation in cycle.deliveries.observations
    ] == [("sku-a", 1), ("sku-a", 2), ("sku-b", 1), ("sku-b", 2)]
    assert set(cycle.state_updates) == {
        METHOD_SCOPE_LABEL,
        labels["sku-a"],
        labels["sku-b"],
    }
    assert cycle.state_updates[METHOD_SCOPE_LABEL] == b"method-update-1"
    assert labels["sku-c"] not in cycle.state_updates
    assert states[labels["sku-c"]] == loop.conformal_states[labels["sku-c"]]
    assert len(cycle.annotations) == 4


def test_partial_window_retains_resolved_members_then_delivers_exactly_once() -> None:
    hierarchy = _hierarchy()
    runtime = resolve_method(
        {
            "method": "split-window-sum",
            "coverage": 0.5,
            "partition_by": "series",
            "protection_period": 2,
        }
    )
    label = _label("sku-a", EmissionScope.WINDOW_SUM)
    states = runtime.calibrate(CalibrationSeedBatch({label: [1.0, 2.0]}))
    rows = (
        ("sku-a", _ISSUE_ORIGIN, 2, pd.Timestamp("2026-01-03"), 5.0, _MODEL),
        ("sku-a", _ISSUE_ORIGIN, 1, _TARGET, 4.0, _MODEL),
    )
    pending = _issued_pending(runtime, rows, states)
    first = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=pending,
        conformal_states=states,
        runtime=runtime,
    )
    first.accept(ActualsSubmission((_actual("sku-a", 6),)))

    partial = first.cycle(pd.Timestamp("2026-01-04"))

    assert len(partial.deliveries) == 0
    assert partial.resolutions == ()
    by_step = {value.forecast_key.horizon_step: value for value in partial.pending_retentions}
    assert by_step[1].resolution is not None
    assert by_step[2].resolution is None

    second = ObserveLoop(
        hierarchy=hierarchy,
        observed_history=partial.history_appends,
        pending_observations=partial.pending_retentions,
        conformal_states=states,
        runtime=runtime,
    )
    second.accept(ActualsSubmission((_actual("sku-a", 8, timestamp=pd.Timestamp("2026-01-03")),)))
    completed = second.cycle(pd.Timestamp("2026-01-04"))

    assert len(completed.deliveries) == 1
    assert [
        value.forecast_key.horizon_step for value in completed.deliveries.observations_for(label)
    ] == [
        1,
        2,
    ]
    assert len(completed.resolutions) == 2
    assert completed.pending_retentions == ()

    drained = ObserveLoop(
        hierarchy=hierarchy,
        observed_history=(*partial.history_appends, *completed.history_appends),
        pending_observations=completed.pending_retentions,
        conformal_states=states.with_rows(completed.state_updates),
        runtime=runtime,
    ).cycle(pd.Timestamp("2026-01-05"))
    assert len(drained.deliveries) == 0
    assert drained.resolutions == ()


def test_global_window_delivery_groups_interleaved_canonical_rows_by_window() -> None:
    hierarchy = _hierarchy()
    delegate = resolve_method(
        {
            "method": "split-window-sum",
            "coverage": 0.5,
            "partition_by": "global",
            "protection_period": 2,
        }
    )
    runtime = _CountingRuntime(delegate)
    label = _label("global", EmissionScope.WINDOW_SUM)
    states = runtime.calibrate(CalibrationSeedBatch({label: [1.0, 2.0]}))
    second_target = pd.Timestamp("2026-01-03")
    rows = (
        ("sku-b", _ISSUE_ORIGIN, 2, second_target, 20.0, _MODEL),
        ("sku-a", _ISSUE_ORIGIN, 1, _TARGET, 2.0, _MODEL),
        ("sku-b", _ISSUE_ORIGIN, 1, _TARGET, 10.0, _MODEL),
        ("sku-a", _ISSUE_ORIGIN, 2, second_target, 3.0, _MODEL),
    )
    loop = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=_issued_pending(runtime, rows, states),
        conformal_states=states,
        runtime=runtime,
    )
    loop.accept(
        ActualsSubmission(
            (
                _actual("sku-b", 24, timestamp=second_target),
                _actual("sku-a", 4),
                _actual("sku-b", 13),
                _actual("sku-a", 7, timestamp=second_target),
            )
        )
    )

    cycle = loop.cycle(pd.Timestamp("2026-01-04"))

    assert runtime.calls == [(label,)]
    delivery = cycle.deliveries.observations_for(label)
    assert [
        (value.forecast_key.series_key, value.forecast_key.horizon_step) for value in delivery
    ] == [("sku-a", 1), ("sku-b", 1), ("sku-a", 2), ("sku-b", 2)]
    assert [annotation.score for annotation in cycle.annotations] == [None, None, 6.0, 7.0]
    assert [annotation.advanced_delivered_score for annotation in cycle.annotations] == [
        False,
        False,
        True,
        True,
    ]


def test_one_partition_call_can_carry_multiple_complete_windows() -> None:
    hierarchy = _hierarchy()
    delegate = resolve_method(
        {
            "method": "split-window-sum",
            "coverage": 0.5,
            "partition_by": "series",
            "protection_period": 2,
        }
    )
    runtime = _CountingRuntime(delegate)
    label = _label("sku-a", EmissionScope.WINDOW_SUM)
    states = runtime.calibrate(CalibrationSeedBatch({label: [1.0, 2.0]}))
    second_origin = pd.Timestamp("2026-01-02")
    rows = (
        ("sku-a", second_origin, 2, pd.Timestamp("2026-01-04"), 6.0, _MODEL),
        ("sku-a", _ISSUE_ORIGIN, 1, _TARGET, 3.0, _MODEL),
        ("sku-a", second_origin, 1, pd.Timestamp("2026-01-03"), 5.0, _MODEL),
        ("sku-a", _ISSUE_ORIGIN, 2, pd.Timestamp("2026-01-03"), 4.0, _MODEL),
    )
    loop = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=_issued_pending(runtime, rows, states),
        conformal_states=states,
        runtime=runtime,
    )
    loop.accept(
        ActualsSubmission(
            (
                _actual("sku-a", 4),
                _actual("sku-a", 6, timestamp=pd.Timestamp("2026-01-03")),
                _actual("sku-a", 8, timestamp=pd.Timestamp("2026-01-04")),
            )
        )
    )

    cycle = loop.cycle(pd.Timestamp("2026-01-05"))

    assert runtime.calls == [(label,)]
    assert len(cycle.deliveries) == 1
    assert [
        (value.forecast_key.origin, value.forecast_key.horizon_step)
        for value in cycle.deliveries.observations_for(label)
    ] == [
        (_ISSUE_ORIGIN, 1),
        (_ISSUE_ORIGIN, 2),
        (second_origin, 1),
        (second_origin, 2),
    ]
    assert len(cycle.annotations) == 4


def test_sequence_preserving_cycle_chunking_has_identical_partition_state() -> None:
    hierarchy = _hierarchy()
    configuration = {
        "method": "split-per-step",
        "coverage": 0.5,
        "partition_by": "series",
    }
    runtime = resolve_method(configuration)
    label = _label("sku-a", EmissionScope.PER_STEP)
    states = runtime.calibrate(CalibrationSeedBatch({label: [1.0, 2.0]}))
    rows = (
        ("sku-a", _ISSUE_ORIGIN, 1, _TARGET, 3.0, _MODEL),
        ("sku-a", _ISSUE_ORIGIN, 2, pd.Timestamp("2026-01-03"), 5.0, _MODEL),
    )
    pending = _issued_pending(runtime, rows, states)
    one = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=pending,
        conformal_states=states,
        runtime=runtime,
    )
    one.accept(
        ActualsSubmission(
            (
                _actual("sku-a", 7),
                _actual("sku-a", 9, timestamp=pd.Timestamp("2026-01-03")),
            )
        )
    )
    batched = one.cycle(pd.Timestamp("2026-01-04"))

    first = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=pending,
        conformal_states=states,
        runtime=resolve_method(configuration, states=states),
    )
    first.accept(ActualsSubmission((_actual("sku-a", 7),)))
    first_cycle = first.cycle(pd.Timestamp("2026-01-03"))
    second_states = states.with_rows(first_cycle.state_updates)
    second = ObserveLoop(
        hierarchy=hierarchy,
        observed_history=first_cycle.history_appends,
        pending_observations=first_cycle.pending_retentions,
        conformal_states=second_states,
        runtime=resolve_method(configuration, states=second_states),
    )
    second.accept(ActualsSubmission((_actual("sku-a", 9, timestamp=pd.Timestamp("2026-01-03")),)))
    split = second.cycle(pd.Timestamp("2026-01-04"))

    assert batched.state_updates[label] == split.state_updates[label]


def test_cold_start_nan_issuance_still_delivers_and_advances_state() -> None:
    hierarchy = _hierarchy()
    runtime = resolve_method({"method": "split-per-step", "partition_by": "series"})
    row = (("sku-a", _ISSUE_ORIGIN, 1, _TARGET, 3.0, _MODEL),)
    pending = _issued_pending(runtime, row, ConformalStateBatch())
    assert pending[0].issued is not None
    assert math.isnan(pending[0].issued.upper_bound)
    loop = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=pending,
        runtime=runtime,
    )
    loop.accept(ActualsSubmission((_actual("sku-a", 6),)))

    cycle = loop.cycle(_CYCLE_ORIGIN)

    label = _label("sku-a", EmissionScope.PER_STEP)
    assert len(cycle.deliveries) == 1
    assert cycle.annotations[0].advanced_delivered_score
    assert label in cycle.state_updates
    assert cycle.pending_retentions == ()


def test_runtime_backed_cycle_refuses_missing_or_conflicting_issuance_facts() -> None:
    hierarchy = _hierarchy()
    runtime = resolve_method(
        {"method": "split-per-step", "coverage": 0.5, "partition_by": "series"}
    )
    row = PendingObservation(
        ForecastKey("sku-a", _ISSUE_ORIGIN, 1, _MODEL),
        _TARGET,
        3.0,
    )
    loop = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=(row,),
        runtime=runtime,
    )
    loop.accept(ActualsSubmission((_actual("sku-a", 6),)))

    with pytest.raises(ObserveError, match="missing issued"):
        loop.cycle(_CYCLE_ORIGIN)

    other = resolve_method(
        {
            "method": "split-window-sum",
            "coverage": 0.5,
            "partition_by": "series",
            "protection_period": 1,
        }
    )
    conflicting = _issued_pending(
        other,
        (("sku-a", _ISSUE_ORIGIN, 1, _TARGET, 3.0, _MODEL),),
        ConformalStateBatch(),
    )
    wrong = ObserveLoop(
        hierarchy=hierarchy,
        pending_observations=conflicting,
        runtime=runtime,
    )
    wrong.accept(ActualsSubmission((_actual("sku-a", 6),)))
    with pytest.raises(ObserveError, match="conflicting runtime issuance"):
        wrong.cycle(_CYCLE_ORIGIN)


def test_recent_score_perturbation_changes_only_the_later_perturbed_bound() -> None:
    hierarchy = _hierarchy()
    configuration = {
        "method": "split-per-step",
        "coverage": 0.5,
        "partition_by": "series",
    }
    base_runtime = resolve_method(configuration)
    label = _label("sku-a", EmissionScope.PER_STEP)
    seed = base_runtime.calibrate(CalibrationSeedBatch({label: [1.0]}))
    issued_row = (("sku-a", _ISSUE_ORIGIN, 1, _TARGET, 5.0, _MODEL),)
    pending = _issued_pending(base_runtime, issued_row, seed)

    def later_bound(actual: float) -> float:
        runtime = resolve_method(configuration, states=seed)
        loop = ObserveLoop(
            hierarchy=hierarchy,
            pending_observations=pending,
            conformal_states=seed,
            runtime=runtime,
        )
        loop.accept(ActualsSubmission((_actual("sku-a", actual),)))
        observed = loop.cycle(_CYCLE_ORIGIN)
        updated = seed.with_rows(observed.state_updates)
        later = runtime.apply(
            _frame(
                (("sku-a", pd.Timestamp("2026-01-03"), 1, pd.Timestamp("2026-01-04"), 5.0, _MODEL),)
            ),
            updated,
        )
        return float(later.forecasts.loc[0, interval_columns(0.5)[1]])

    control = later_bound(7.0)
    repeated_control = later_bound(7.0)
    perturbed = later_bound(10.0)

    assert control == repeated_control
    assert perturbed != control


def test_observe_package_owns_no_transport_store_or_readiness_counter() -> None:
    root = Path(__file__).resolve().parents[2] / "src" / "newcalibre" / "observe"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py")))

    assert "class ObserveLoop" in sources
    for forbidden in (
        "class Queue",
        "class Store",
        "class Transport",
        "readiness_counter",
        "delivered_score_count",
    ):
        assert forbidden not in sources
