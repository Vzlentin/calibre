"""Prove the conformal runtime owns canonical immutable batch values."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

import newcalibre.conformal as conformal
from newcalibre.conformal import (
    METHOD_SCOPE_LABEL,
    CalibrationSeedBatch,
    ConformalStateBatch,
    DeliveryBatch,
    ForecastKey,
    ResolvedObservation,
    RuntimeContractError,
    derive_partition_label,
    resolve_method,
)
from newcalibre.domain import CensoringAssertion, EmissionScope


def _label(value: str) -> str:
    return derive_partition_label("model", value, EmissionScope.PER_STEP)


def test_seed_batch_canonicalizes_routes_and_snapshots_scores() -> None:
    first = _label("z")
    second = _label("a")
    raw = [1.0, 2.0]

    batch = CalibrationSeedBatch({first: raw, second: [3.0]})
    raw.append(99.0)

    assert batch.labels == tuple(sorted((first, second), key=str.encode))
    assert batch.scores_for(first) == (1.0, 2.0)
    assert batch.scores_for(second) == (3.0,)
    assert dict(batch.items()) == {label: batch.scores_for(label) for label in batch.labels}


def test_seed_batch_rejects_bad_labels_duplicates_and_scores() -> None:
    label = _label("a")
    with pytest.raises(RuntimeContractError, match="duplicate"):
        CalibrationSeedBatch([(label, [1.0]), (label, [2.0])])
    with pytest.raises(RuntimeContractError, match="state label"):
        CalibrationSeedBatch({"bad": [1.0]})
    with pytest.raises(RuntimeContractError, match="finite nonnegative"):
        CalibrationSeedBatch({label: [-1.0]})


def test_state_batch_is_canonical_immutable_and_supports_exact_transitions() -> None:
    first = _label("z")
    second = _label("a")
    states = {first: b"one", second: b"two"}

    batch = ConformalStateBatch(states)
    states[first] = b"mutated"
    transitioned = batch.with_rows({first: b"changed"})
    unchanged = batch.with_rows({first: b"one"})

    assert batch.labels == tuple(sorted((first, second), key=str.encode))
    assert unchanged is batch
    assert batch[first] == b"one"
    assert transitioned[first] == b"changed"
    assert transitioned[second] == b"two"
    assert transitioned.labels is batch.labels
    assert transitioned.project((first,)) == {first: b"changed"}
    with pytest.raises(TypeError):
        transitioned.project((first,))[first] = b"again"  # type: ignore[index]
    with pytest.raises(RuntimeContractError, match="dirty labels"):
        transitioned.project((_label("missing"),))


def test_empty_batches_are_valid() -> None:
    assert CalibrationSeedBatch().labels == ()
    assert ConformalStateBatch().labels == ()


@pytest.mark.parametrize(
    "configuration",
    [
        {"method": "split-per-step", "coverage": 0.5, "partition_by": "series"},
        {
            "method": "split-window-sum",
            "coverage": 0.5,
            "partition_by": "series",
            "protection_period": 1,
        },
        {"method": "weighted-per-step", "coverage": 0.5, "partition_by": "series"},
        {
            "method": "sequential-adaptive-per-step",
            "coverage": 0.5,
            "partition_by": "series",
        },
    ],
)
def test_every_registration_is_invariant_to_batch_row_and_label_placement(
    configuration: dict[str, object],
) -> None:
    runtime = resolve_method(configuration)
    scope = runtime.manifest.emission_scope
    labels = {
        series: derive_partition_label("model", series, scope) for series in ("sku-b", "sku-a")
    }
    seeds = CalibrationSeedBatch([(labels["sku-b"], [2.0, 3.0]), (labels["sku-a"], [1.0, 2.0])])
    state = runtime.calibrate(seeds)
    origin = pd.Timestamp("2026-01-05")
    frame = pd.DataFrame(
        {
            "series_key": pd.Series(["sku-b", "sku-a"], dtype="string"),
            "target_timestamp": pd.to_datetime([origin, origin]),
            "actual_value": pd.Series([float("nan"), float("nan")], dtype="float64"),
            "point_forecast": pd.Series([5.0, 4.0], dtype="float64"),
            "horizon_step": pd.Series([1, 1], dtype="int64"),
            "origin": pd.to_datetime([origin, origin]),
            "model_name": pd.Series(["model", "model"], dtype="string"),
        }
    )

    issued = runtime.apply(frame, state)
    permuted = runtime.apply(frame.iloc[::-1].reset_index(drop=True), state)

    chunk_state = state
    chunked_frames: list[pd.DataFrame] = []
    for index in range(len(frame)):
        chunked = runtime.apply(frame.iloc[[index]].reset_index(drop=True), chunk_state)
        chunk_state = chunked.state
        chunked_frames.append(chunked.forecasts)

    left = issued.forecasts.sort_values("series_key").reset_index(drop=True)
    right = permuted.forecasts.sort_values("series_key").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert issued.state == permuted.state
    assert issued.dirty_labels == permuted.dirty_labels
    assert issued.issuances == permuted.issuances
    pd.testing.assert_frame_equal(
        issued.forecasts.reset_index(drop=True),
        pd.concat(chunked_frames, ignore_index=True),
        check_exact=True,
    )
    assert issued.state == chunk_state

    observations: dict[str, tuple[ResolvedObservation, ...]] = {}
    for row in issued.forecasts.to_dict("records"):
        key = ForecastKey(
            row["series_key"],
            pd.Timestamp(row["origin"]),
            row["horizon_step"],
            row["model_name"],
        )
        label = labels[key.series_key]
        observations[label] = (
            ResolvedObservation(
                key,
                pd.Timestamp(row["target_timestamp"]),
                8.0 if key.series_key == "sku-b" else 6.0,
                row["point_forecast"],
                CensoringAssertion.UNCENSORED,
                None,
                issued.issuances[key],
            ),
        )
    deliveries = DeliveryBatch(list(reversed(tuple(observations.items()))))
    assert deliveries.observations is deliveries.observations
    first = runtime.observe(deliveries, issued.state)
    second = runtime.observe(DeliveryBatch(observations), issued.state)

    assert first == second
    assert first.state.project(first.dirty_labels) == second.state.project(second.dirty_labels)
    assert issued.state[labels["sku-a"]] != issued.state[labels["sku-b"]]
    for label, partition_observations in deliveries.items():
        isolated_state = ConformalStateBatch(
            {
                label: issued.state[label],
                METHOD_SCOPE_LABEL: issued.state[METHOD_SCOPE_LABEL],
            }
        )
        isolated = runtime.observe(
            DeliveryBatch({label: partition_observations}),
            isolated_state,
        )
        assert isolated.state[label] == first.state[label]
        assert isolated.annotations == tuple(
            annotation
            for annotation in first.annotations
            if annotation.forecast_key
            in {observation.forecast_key for observation in partition_observations}
        )


def test_delivery_flattening_uses_total_key_order_not_partition_order() -> None:
    first_label = _label("a")
    second_label = _label("z")
    origin = pd.Timestamp("2026-01-05")

    def observation(series: str, label: str, horizon: int) -> ResolvedObservation:
        key = ForecastKey(series, origin, horizon, "model")
        runtime = resolve_method(
            {"method": "split-per-step", "coverage": 0.5, "partition_by": "series"}
        )
        frame = pd.DataFrame(
            {
                "series_key": pd.Series([series], dtype="string"),
                "target_timestamp": pd.to_datetime([origin]),
                "actual_value": pd.Series([float("nan")], dtype="float64"),
                "point_forecast": pd.Series([4.0], dtype="float64"),
                "horizon_step": pd.Series([horizon], dtype="int64"),
                "origin": pd.to_datetime([origin]),
                "model_name": pd.Series(["model"], dtype="string"),
            }
        )
        state = runtime.calibrate(CalibrationSeedBatch({label: [1.0, 2.0]}))
        issued = runtime.apply(frame, state)
        return ResolvedObservation(
            key,
            origin,
            6.0,
            4.0,
            CensoringAssertion.UNCENSORED,
            None,
            issued.issuances[key],
        )

    later_key = observation("a", first_label, 2)
    earlier_key = observation("z", second_label, 1)
    batch = DeliveryBatch(
        {
            first_label: (later_key,),
            second_label: (earlier_key,),
        }
    )

    assert batch.labels == tuple(sorted((first_label, second_label), key=str.encode))
    assert batch.observations == (earlier_key, later_key)


def test_runtime_protocol_has_only_batch_state_signatures() -> None:
    root = Path(__file__).parents[2] / "src" / "newcalibre" / "conformal"
    modules = tuple(ast.parse(path.read_text()) for path in root.rglob("*.py"))
    runtime = ast.parse((root / "runtime.py").read_text())
    functions = {
        node.name: node
        for node in ast.walk(runtime)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert not any(
        isinstance(node, ast.Name) and node.id == "Delivery"
        for module in modules
        for node in ast.walk(module)
    )
    assert "Mapping[str, bytes | None]" not in ast.unparse(functions["apply"])
    assert "Mapping[str, bytes | None]" not in ast.unparse(functions["observe"])
    state_entry_points = {
        "apply",
        "observe",
        "build_split_per_step",
        "build_split_window_sum",
        "build_weighted_per_step",
        "build_sequential_adaptive_per_step",
    }
    for module in modules:
        for node in ast.walk(module):
            if isinstance(node, ast.FunctionDef) and node.name in state_entry_points:
                assert "Mapping[str, bytes]" not in ast.unparse(node.args)
            if isinstance(node, ast.ClassDef) and node.name in {
                "SequentialAdaptiveConformalRuntime",
                "SplitConformalRuntime",
                "WeightedConformalRuntime",
            }:
                constructor = next(
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef) and child.name == "__init__"
                )
                assert "Mapping[str, bytes]" not in ast.unparse(constructor.args)
    assert "Delivery" not in conformal.__all__
    assert {
        "CalibrationSeedBatch",
        "ConformalStateBatch",
        "DeliveryBatch",
    }.issubset(conformal.__all__)
    observe_source = (
        Path(__file__).parents[2] / "src" / "newcalibre" / "observe" / "loop.py"
    ).read_text()
    assert "_observe_deliveries" not in observe_source
    assert observe_source.count("self._runtime.observe(") == 1
