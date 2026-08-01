"""Exercise typed deterministic forecast dispatch."""

from __future__ import annotations

import base64
import inspect
import json
from dataclasses import replace

import pandas as pd
import pytest

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    Calendar,
    CycleToken,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import ForecastLifecycle, IndexedPanel, InProcessDispatch
from newcalibre.engine.dispatch import (
    DispatchBackend,
    ForecastDispatchError,
    build_forecast_work,
    canonical_shard_ranges,
    validate_forecast_envelopes,
)
from newcalibre.forecasting import AdapterExecutionMode, resolve_adapter

pytestmark = pytest.mark.tier1


def test_canonical_shards_are_contiguous_near_equal_and_stable() -> None:
    """Partition canonical series order by quotient and remainder."""
    assert canonical_shard_ranges(series_count=18, shard_count=16) == (
        (0, 2),
        (2, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (15, 16),
        (16, 17),
        (17, 18),
    )


@pytest.mark.parametrize("series_count, shard_count", [(0, 1), (1, 0)])
def test_canonical_shards_reject_empty_work_or_budget(
    series_count: int,
    shard_count: int,
) -> None:
    """Reject budgets that cannot produce non-empty logical work."""
    with pytest.raises(ValueError):
        canonical_shard_ranges(series_count=series_count, shard_count=shard_count)


def test_canonical_shards_keep_fixed_ordinals_for_small_populations() -> None:
    """Represent unused fixed ordinals as trailing empty contiguous ranges."""
    assert canonical_shard_ranges(series_count=3, shard_count=5) == (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 3),
        (3, 3),
    )


def _work_and_results():
    calendar = Calendar("D", phase=pd.Timestamp("2026-01-01"))
    series = tuple(f"s-{ordinal:02d}" for ordinal in range(18))
    frame = pd.DataFrame.from_records(
        [
            {SERIES_KEY: key, TIMESTAMP: timestamp, OBSERVED_VALUE: float(ordinal + day)}
            for ordinal, key in enumerate(series)
            for day, timestamp in enumerate(pd.date_range("2026-01-01", periods=4, freq="D"))
        ]
    ).astype({SERIES_KEY: "string", OBSERVED_VALUE: "float64"})
    panel = IndexedPanel.from_panel(
        Panel.from_frame(frame, calendar=calendar, target_support=TargetSupport.REAL)
    )
    config = {"backend": "seasonal-naive", "m": 2}
    session = SessionIdentity.derive(
        tenant="dispatch-tier1",
        series_keys=series,
        calendar=calendar,
        horizon=2,
        model_config=config,
    )
    task = panel.tasks(
        origin=pd.Timestamp("2026-01-04"),
        horizon=2,
        scope=Scope.GLOBAL,
        model_config=config,
    )[0]
    token = CycleToken(session, task.origin, 1, 1, "0" * 32)
    dispatch = InProcessDispatch(logical_shards=16)
    lifecycle = ForecastLifecycle(adapter_resolver=resolve_adapter)
    work = lifecycle.prepare_work(
        session=session,
        task=task,
        token=token,
        checkpoints={},
        checkpoint_indexes={},
        backend=dispatch.backend,
        budget=dispatch.budget,
    )
    return lifecycle, work, dispatch.dispatch(work, lifecycle)


def test_series_separable_work_has_exact_fixed_membership() -> None:
    """Bind 18 canonical series to the required 16 stable ordinals."""
    _lifecycle, work, _results = _work_and_results()

    assert len(work.shards) == 16
    assert work.shards[0].series_keys == ("s-00", "s-01")
    assert work.shards[1].series_keys == ("s-02", "s-03")
    assert tuple(key for shard in work.shards for key in shard.series_keys) == work.task.series_keys
    assert len({shard.key for shard in work.shards}) == 16


def test_monolithic_work_remains_one_item_under_a_sixteen_shard_budget() -> None:
    """Keep truly cross-series adapter work unsplit."""
    _lifecycle, separable, _results = _work_and_results()
    monolithic = build_forecast_work(
        backend=separable.backend,
        budget=separable.budget,
        session=separable.session,
        token=separable.token,
        task=separable.task,
        execution_mode=AdapterExecutionMode.MONOLITHIC,
    )

    assert len(monolithic.shards) == 1
    assert monolithic.shards[0].series_keys == separable.task.series_keys


def test_result_merge_is_completion_order_invariant() -> None:
    """Merge complete outcomes only by stable logical ordinal."""
    _lifecycle, work, results = _work_and_results()
    canonical, _ = validate_forecast_envelopes(work, results)
    reversed_completion, _ = validate_forecast_envelopes(work, tuple(reversed(results)))

    pd.testing.assert_frame_equal(reversed_completion, canonical)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda work, results: results[:-1],
        lambda work, results: (results[0], results[0], *results[2:]),
        lambda work, results: (replace(results[0], backend="foreign"), *results[1:]),
        lambda work, results: (replace(results[0], ordinal=99), *results[1:]),
        lambda work, results: (
            replace(results[0], frame=results[0].frame.iloc[::-1].reset_index(drop=True)),
            *results[1:],
        ),
    ],
    ids=["missing", "duplicate", "foreign", "ordinal", "non-canonical-rows"],
)
def test_result_envelopes_fail_closed(mutate) -> None:
    """Reject incomplete, duplicate, foreign, and non-canonical outcomes."""
    _lifecycle, work, results = _work_and_results()

    with pytest.raises(ForecastDispatchError):
        validate_forecast_envelopes(work, tuple(mutate(work, results)))


def test_dispatch_port_has_no_arbitrary_callable_map_surface() -> None:
    """Expose typed work and executor values without arbitrary callables."""
    source = inspect.getsource(DispatchBackend)
    assert "Callable" not in source
    assert "TypeVar" not in source
    assert "def map" not in source
    assert "ForecastWork" in source


def test_combined_checkpoint_contains_no_dispatch_identity() -> None:
    """Keep backend, work keys, shard keys, and ordinals out of durable bytes."""
    lifecycle, work, results = _work_and_results()
    completed = lifecycle.complete_work(work, results)

    assert completed.checkpoint is not None
    checkpoint = json.loads(completed.checkpoint.value)
    combined = json.loads(base64.b64decode(checkpoint["native_state"], validate=True))
    encoded = completed.checkpoint.value
    assert set(combined) == {"schema", "states"}
    assert all(set(state) == {"fit_time_bound", "native_state"} for state in combined["states"])
    assert work.backend.encode() not in encoded
    assert work.key.encode() not in encoded
    assert all(shard.key.encode() not in encoded for shard in work.shards)
    assert b"ordinal" not in encoded
