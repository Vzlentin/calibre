"""Prove the conformal runtime owns canonical immutable batch values."""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from newcalibre.conformal import (
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

    assert batch.labels == tuple(sorted((first, second), key=str.encode))
    assert batch[first] == b"one"
    assert transitioned[first] == b"changed"
    assert transitioned[second] == b"two"
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

    left = issued.forecasts.sort_values("series_key").reset_index(drop=True)
    right = permuted.forecasts.sort_values("series_key").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_exact=True)
    assert issued.state == permuted.state
    assert issued.dirty_labels == permuted.dirty_labels
    assert issued.issuances == permuted.issuances

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
    first = runtime.observe(
        DeliveryBatch(list(reversed(tuple(observations.items())))),
        issued.state,
    )
    second = runtime.observe(DeliveryBatch(observations), issued.state)

    assert first == second
    assert first.state.project(first.dirty_labels) == second.state.project(second.dirty_labels)


def test_runtime_protocol_has_only_batch_state_signatures() -> None:
    root = Path(__file__).parents[2] / "src" / "newcalibre" / "conformal"
    runtime = ast.parse((root / "runtime.py").read_text())
    source = (root / "types.py").read_text()
    functions = {
        node.name: node
        for node in ast.walk(runtime)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert "Delivery" not in source
    assert "Mapping[str, bytes | None]" not in ast.unparse(functions["apply"])
    assert "Mapping[str, bytes | None]" not in ast.unparse(functions["observe"])
    observe_source = (
        Path(__file__).parents[2] / "src" / "newcalibre" / "observe" / "loop.py"
    ).read_text()
    assert "_observe_deliveries" not in observe_source
    assert observe_source.count("self._runtime.observe(") == 1
