from __future__ import annotations

import pandas as pd
import pytest

from calibre.cli.config import load_config_from_mapping
from calibre.core.forecast_frame import DS, FORECAST_ORIGIN, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.order_types import CostStruct
from calibre.execution.actuals import HierarchyActualsSource
from calibre.execution.dataset import DatasetBundle
from calibre.execution.hierarchy_preparation import prepare_run
from calibre.reconciliation import BottomUpReconciler
from calibre.reconciliation.summing import build_summing_matrix


def _history() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    return pd.DataFrame(
        [
            {UNIQUE_ID: uid, DS: ds, Y: float(index + step + 1)}
            for index, uid in enumerate(["a", "b"])
            for step, ds in enumerate(dates)
        ]
    )


def _hierarchy() -> pd.DataFrame:
    return pd.DataFrame({UNIQUE_ID: ["a", "b"], "dept": ["D", "D"]})


def _bundle(*, hierarchy: pd.DataFrame | None = None) -> DatasetBundle:
    return DatasetBundle(
        history=_history(),
        future_x=None,
        costs=CostStruct(),
        hierarchy=hierarchy,
        censoring=None,
    )


def _config(**overrides):
    data = {
        "config_schema": "1.0",
        "dataset": {"adapter": "unit", "path": "ignored"},
        "tasks": [
            {
                "model": "SeasonalNaive",
                "horizon": 2,
                "config": {"backend": "statsforecast", "season_length": 2},
            }
        ],
        "origins": {"start": "2024-01-03", "end": "2024-01-03", "freq": "D"},
        "output": {"ledger_path": "ignored.parquet", "streaming": False},
        "execution": {"backend": "local", "seed": 42},
    }
    data.update(overrides)
    return load_config_from_mapping(data)


def test_flat_preparation_keeps_bottom_actuals_and_skips_memory_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_guard(*args, **kwargs):
        raise AssertionError("flat preparation should not run the hierarchy memory guard")

    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.enforce_hierarchical_expansion_memory_limit",
        fail_guard,
    )
    bundle = _bundle(hierarchy=_hierarchy())

    preparation = prepare_run(_config(), bundle)

    pd.testing.assert_frame_equal(preparation.actuals, bundle.history)
    assert preparation.reconciliation_hierarchy is None


def test_reconciliation_preparation_materializes_node_actuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_calls = []

    def record_guard(*args, **kwargs):
        guard_calls.append((args, kwargs))

    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.enforce_hierarchical_expansion_memory_limit",
        record_guard,
    )
    hierarchy = _hierarchy()

    preparation = prepare_run(
        _config(reconciliation={"strategy": "ols"}),
        _bundle(hierarchy=hierarchy),
    )

    summing = build_summing_matrix(hierarchy)
    assert preparation.reconciliation_hierarchy is not None
    pd.testing.assert_frame_equal(
        preparation.reconciliation_hierarchy.reset_index(drop=True),
        hierarchy.reset_index(drop=True),
    )
    assert preparation.reconciler is not None
    assert isinstance(preparation.actuals, pd.DataFrame)
    assert set(preparation.actuals[UNIQUE_ID]) == set(summing.node_labels)
    assert len(guard_calls) == 1


def test_bottom_up_preparation_builds_bottom_only_tasks_and_lazy_actuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_guard(*args, **kwargs):
        raise AssertionError("bottom-only bottom_up must not run the expansion memory guard")

    def fail_build_node_history(*args, **kwargs):
        raise AssertionError("bottom-only bottom_up must not materialize node history")

    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.enforce_hierarchical_expansion_memory_limit",
        fail_guard,
    )
    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.build_node_history",
        fail_build_node_history,
    )

    preparation = prepare_run(
        _config(reconciliation={"strategy": "bottom_up"}),
        _bundle(hierarchy=_hierarchy()),
    )

    assert isinstance(preparation.actuals, HierarchyActualsSource)
    assert isinstance(preparation.reconciler, BottomUpReconciler)
    task_uids = {str(task.history[UNIQUE_ID].iloc[0]) for task in preparation.tasks.local}
    assert task_uids == {"a", "b"}
    assert preparation.tasks.global_ == []


def test_prepare_run_threads_one_shared_hierarchy_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The lazy actuals source consumes the SAME index object run preparation
    built and exposes (mirrors #147's shared-facts identity assertion)."""
    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.enforce_hierarchical_expansion_memory_limit",
        lambda *args, **kwargs: None,
    )

    preparation = prepare_run(
        _config(reconciliation={"strategy": "bottom_up"}),
        _bundle(hierarchy=_hierarchy()),
    )

    assert preparation.hierarchy_index is not None
    assert isinstance(preparation.actuals, HierarchyActualsSource)
    # Identity, not equality: no consumer rebuilds the index.
    assert preparation.actuals._index is preparation.hierarchy_index


def test_flat_preparation_has_no_hierarchy_index() -> None:
    preparation = prepare_run(_config(), _bundle(hierarchy=_hierarchy()))

    assert preparation.hierarchy_index is None


def test_empty_origins_raise_clear_error() -> None:
    config = _config(origins={"start": "2024-01-01", "end": "2024-01-02", "freq": "W-SUN"})

    with pytest.raises(ValueError, match="origins resolved to an empty list"):
        prepare_run(config, _bundle())


@pytest.mark.parametrize(
    "config_override",
    [
        {"reconciliation": {"strategy": "ols"}},
        {"hierarchical_intervals": {"method": "nixtla_conformal"}},
    ],
)
def test_hierarchy_guard_runs_before_node_history(
    monkeypatch: pytest.MonkeyPatch,
    config_override,
) -> None:
    monkeypatch.setattr(
        "calibre.execution.hierarchy_memory.read_effective_available_memory_bytes", lambda: 1
    )

    def fail_build_node_history(*args, **kwargs):
        raise AssertionError("build_node_history should not run after the preflight guard fails")

    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.build_node_history",
        fail_build_node_history,
    )

    with pytest.raises(ValueError, match="projected node-history rows"):
        prepare_run(_config(**config_override), _bundle(hierarchy=_hierarchy()))


def test_hierarchical_interval_preparation_uses_fused_phase_without_point_reconciler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.enforce_hierarchical_expansion_memory_limit",
        lambda *args, **kwargs: None,
    )

    preparation = prepare_run(
        _config(hierarchical_intervals={"method": "nixtla_conformal"}),
        _bundle(hierarchy=_hierarchy()),
    )

    assert preparation.reconciliation_hierarchy is not None
    assert preparation.reconciler is None
    assert preparation.hierarchical_interval_phase is not None
    assert preparation.conformal_config is None


def test_series_partition_limit_uses_prepared_task_estimate() -> None:
    config = _config(conformal={"method": "mscp", "partition": "series", "max_partitions": 3})

    with pytest.raises(ValueError, match="approximately 4"):
        prepare_run(config, _bundle())


def test_multiple_model_configs_multiply_series_partition_estimate() -> None:
    config = _config(
        tasks=[
            {
                "model": "SeasonalNaive",
                "horizon": 2,
                "config": {"backend": "statsforecast", "season_length": 2},
            },
            {
                "model": "Naive",
                "horizon": 2,
                "config": {"backend": "statsforecast"},
            },
        ],
        conformal={"method": "mscp", "partition": "series", "max_partitions": 7},
    )

    with pytest.raises(ValueError, match="approximately 8"):
        prepare_run(config, _bundle())


def test_series_partition_limit_counts_hierarchy_expanded_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.enforce_hierarchical_expansion_memory_limit",
        lambda *args, **kwargs: None,
    )
    config = _config(
        reconciliation={"strategy": "bottom_up"},
        conformal={"method": "mscp", "partition": "series", "max_partitions": 7},
    )

    with pytest.raises(ValueError, match="approximately 8"):
        prepare_run(config, _bundle(hierarchy=_hierarchy()))


def test_eager_guard_and_partition_limit_consume_shared_expansion_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One source of truth: the memory guard receives the estimate computed by
    run preparation, and the series-partition limit uses that same estimate's
    forecast_partitions instead of recounting from built tasks."""
    from calibre.execution.hierarchy_memory import HierarchicalExpansionEstimate

    sentinel = HierarchicalExpansionEstimate(
        bottom_unique_ids=2,
        aggregate_nodes=2,
        node_count=4,
        bottom_rows=8,
        projected_node_count=4,
        projected_node_history_rows=16,
        forecast_partitions=8,
        model_count=1,
    )
    captured: dict[str, HierarchicalExpansionEstimate] = {}
    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.estimate_hierarchical_expansion",
        lambda *args, **kwargs: sentinel,
    )
    monkeypatch.setattr(
        "calibre.execution.hierarchy_preparation.enforce_hierarchical_expansion_memory_limit",
        lambda estimate: captured.setdefault("estimate", estimate),
    )
    config = _config(
        reconciliation={"strategy": "ols"},
        conformal={"method": "mscp", "partition": "series", "max_partitions": 7},
    )

    with pytest.raises(ValueError, match="approximately 8"):
        prepare_run(config, _bundle(hierarchy=_hierarchy()))

    assert captured["estimate"] is sentinel


def test_eager_hierarchy_partition_estimate_matches_materialized_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real (non-mocked) eager estimate reproduces the count the old
    task-based recount produced: materialized node count x horizon x models."""
    monkeypatch.setattr(
        "calibre.execution.hierarchy_memory.read_effective_available_memory_bytes",
        lambda: 2**60,
    )
    config = _config(
        tasks=[
            {
                "model": "SeasonalNaive",
                "horizon": 2,
                "config": {"backend": "statsforecast", "season_length": 2},
            },
            {
                "model": "Naive",
                "horizon": 2,
                "config": {"backend": "statsforecast"},
            },
        ],
        reconciliation={"strategy": "ols"},
        conformal={"method": "mscp", "partition": "series", "max_partitions": 100},
    )

    preparation = prepare_run(config, _bundle(hierarchy=_hierarchy()))

    assert isinstance(preparation.actuals, pd.DataFrame)
    materialized_nodes = preparation.actuals[UNIQUE_ID].nunique()
    assert preparation.conformal_partition_estimate == materialized_nodes * 2 * 2


def test_bottom_up_preparation_runs_end_to_end_through_engine() -> None:
    """The shipped bottom-only wiring works as a whole: bottom tasks +
    BottomUpReconciler + HierarchyActualsSource through BackendEngine, with
    synthesized aggregate forecasts staying coherent with the bottom rows and
    aggregate actuals resolving to the full member sums."""
    from calibre.execution.backend import (
        BackendEngine,
        ExecutionOptions,
        ReconciliationOptions,
    )
    from calibre.reconciliation.summing import TOTAL_LABEL

    hierarchy = _hierarchy()
    config = _config(reconciliation={"strategy": "bottom_up"})
    preparation = prepare_run(config, _bundle(hierarchy=hierarchy))

    engine = BackendEngine(
        execution=ExecutionOptions(freq="D", backend="local", seed=42),
        reconciliation=ReconciliationOptions(
            reconciler=preparation.reconciler,
            hierarchy_index=preparation.hierarchy_index,
        ),
    )
    try:
        result = engine.execute(preparation.tasks, preparation.actuals, preparation.origins)
    finally:
        engine.close()
    ledger = result.ledger.to_df()

    # Aggregate rows exist without any aggregate task having run.
    assert set(ledger[UNIQUE_ID]) == {"a", "b", "dept=D", TOTAL_LABEL}

    # Synthesized aggregate forecasts equal the bottom sums per cross-section.
    for _, cross_section in ledger.groupby([FORECAST_ORIGIN, H]):
        values = cross_section.set_index(UNIQUE_ID)[Y_HAT]
        bottom_sum = values["a"] + values["b"]
        assert values["dept=D"] == pytest.approx(bottom_sum)
        assert values[TOTAL_LABEL] == pytest.approx(bottom_sum)

    # Aggregate actuals resolved lazily equal the full member sums: on
    # 2024-01-03 the bottom actuals are a=3.0 and b=4.0.
    resolved = ledger[ledger[DS] == pd.Timestamp("2024-01-03")].set_index(UNIQUE_ID)[Y]
    assert resolved["a"] == pytest.approx(3.0)
    assert resolved["b"] == pytest.approx(4.0)
    assert resolved["dept=D"] == pytest.approx(7.0)
    assert resolved[TOTAL_LABEL] == pytest.approx(7.0)


def test_global_scope_configs_still_deduplicate_through_build_tasks() -> None:
    config = _config(
        tasks=[
            {
                "model": "SeasonalNaive",
                "horizon": 2,
                "config": {"backend": "statsforecast", "season_length": 2, "scope": "global"},
            }
        ],
        conformal={"method": "mscp", "partition": "series", "max_partitions": 10},
    )

    preparation = prepare_run(config, _bundle())

    assert preparation.tasks.local == []
    assert len(preparation.tasks.global_) == 1
    assert preparation.conformal_partition_estimate == 4
