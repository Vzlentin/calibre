"""Prepare engine inputs for flat and hierarchical CLI runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import pandas as pd

from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.core.forecast_task import TaskGroups
from calibre.execution.actuals import ActualsSource, HierarchyActualsSource
from calibre.execution.dataset import DatasetBundle
from calibre.execution.hierarchy_memory import (
    enforce_hierarchical_expansion_memory_limit,
    estimate_hierarchical_expansion,
)
from calibre.execution.task_builder import build_node_history, build_tasks
from calibre.reconciliation import HierarchicalIntervalPhase, Reconciler, resolve_reconciler
from calibre.reconciliation.summing import HierarchyIndex, build_hierarchy_index


class _TaskConfig(Protocol):
    @property
    def horizon(self) -> int: ...

    def resolved_model_config(self) -> dict[str, Any]: ...


class _OriginsConfig(Protocol):
    def to_list(self) -> list[pd.Timestamp]: ...


class _ConformalConfig(Protocol):
    @property
    def partition(self) -> str: ...

    @property
    def max_partitions(self) -> int | None: ...

    def to_runtime_config(self) -> SymmetricIntervalConfig: ...


class _ReconciliationConfig(Protocol):
    @property
    def strategy(self) -> str: ...

    def to_reconciler(self) -> Reconciler: ...


class _HierarchicalIntervalConfig(Protocol):
    def to_phase(self) -> HierarchicalIntervalPhase: ...


class RunPreparationConfig(Protocol):
    @property
    def tasks(self) -> Sequence[_TaskConfig]: ...

    @property
    def origins(self) -> _OriginsConfig: ...

    @property
    def conformal(self) -> _ConformalConfig | None: ...

    @property
    def reconciliation(self) -> _ReconciliationConfig | None: ...

    @property
    def hierarchical_intervals(self) -> _HierarchicalIntervalConfig | None: ...


@dataclass(frozen=True)
class RunPreparation:
    tasks: TaskGroups
    actuals: pd.DataFrame | ActualsSource
    origins: list[pd.Timestamp]
    conformal_config: SymmetricIntervalConfig | None
    reconciliation_hierarchy: pd.DataFrame | None
    hierarchy_index: HierarchyIndex | None
    reconciler: Reconciler | None
    hierarchical_interval_phase: HierarchicalIntervalPhase | None
    conformal_partition_estimate: int | None


def prepare_run(config: RunPreparationConfig, bundle: DatasetBundle) -> RunPreparation:
    """Resolve hierarchy-specific run inputs for ``BackendEngine`` construction."""
    model_configs = [task.resolved_model_config() for task in config.tasks]
    horizon = config.tasks[0].horizon
    reconciliation_hierarchy = _hierarchy_for_run(config, bundle)
    hierarchy_index = (
        build_hierarchy_index(reconciliation_hierarchy)
        if reconciliation_hierarchy is not None
        else None
    )
    bottom_only = hierarchy_index is not None and _is_point_bottom_up(config)

    actuals: pd.DataFrame | ActualsSource
    hierarchy_partitions: int | None = None
    if bottom_only:
        # Native point bottom_up consumes bottom-only forecasts and synthesizes
        # aggregate node rows itself, so the run never materializes the eager
        # node-history frame: tasks come from bottom history and actuals
        # resolve lazily. The #136 expansion guard targets exactly that
        # materialization, so it does not apply here. Ledger partitions still
        # span the full hierarchy node set once aggregates are synthesized.
        assert reconciliation_hierarchy is not None and hierarchy_index is not None
        actuals = HierarchyActualsSource(bundle.history, hierarchy_index)
        tasks = build_tasks(bundle.history, model_configs, horizon)
        hierarchy_partitions = len(hierarchy_index.node_labels) * horizon * len(config.tasks)
    else:
        if hierarchy_index is not None:
            expansion = estimate_hierarchical_expansion(
                bundle.history,
                hierarchy_index,
                horizon=horizon,
                model_count=len(config.tasks),
            )
            # This branch is the densifying path (ols/erm/MinT point reconcilers
            # and the fused interval phase all build the dense S per origin);
            # native bottom_up takes the bottom_only branch and never reaches
            # here, so accounting the dense-S term unconditionally here is the
            # implicit gate. node_count x n_bottom x 8 bytes (float64).
            dense_s_bytes = len(hierarchy_index.node_labels) * len(hierarchy_index.bottom_ids) * 8
            expansion = replace(expansion, dense_s_bytes=dense_s_bytes)
            enforce_hierarchical_expansion_memory_limit(expansion)
            hierarchy_partitions = expansion.forecast_partitions
        actuals = build_node_history(bundle.history, hierarchy_index)
        tasks = build_tasks(actuals, model_configs, horizon)
    conformal_partition_estimate = _enforce_conformal_partition_limit(
        config,
        tasks,
        horizon,
        hierarchy_partitions=hierarchy_partitions,
    )
    origins = config.origins.to_list()
    if not origins:
        raise ValueError("origins resolved to an empty list")

    conformal_config: SymmetricIntervalConfig | None = (
        config.conformal.to_runtime_config() if config.conformal is not None else None
    )
    return RunPreparation(
        tasks=tasks,
        actuals=actuals,
        origins=origins,
        conformal_config=conformal_config,
        reconciliation_hierarchy=reconciliation_hierarchy,
        hierarchy_index=hierarchy_index,
        reconciler=(
            None
            if config.hierarchical_intervals is not None
            else config.reconciliation.to_reconciler()
            if config.reconciliation is not None
            else resolve_reconciler("none")
        ),
        hierarchical_interval_phase=(
            config.hierarchical_intervals.to_phase()
            if config.hierarchical_intervals is not None
            else None
        ),
        conformal_partition_estimate=conformal_partition_estimate,
    )


def _is_point_bottom_up(config: RunPreparationConfig) -> bool:
    return (
        config.hierarchical_intervals is None
        and config.reconciliation is not None
        and config.reconciliation.strategy == "bottom_up"
    )


def _hierarchy_for_run(config: RunPreparationConfig, bundle: DatasetBundle) -> pd.DataFrame | None:
    if config.hierarchical_intervals is not None:
        if bundle.hierarchy is None:
            raise ValueError("hierarchical_intervals requires a dataset hierarchy")
        return bundle.hierarchy
    if config.reconciliation is None or config.reconciliation.strategy == "none":
        return None
    return bundle.hierarchy


def _enforce_conformal_partition_limit(
    config: RunPreparationConfig,
    tasks: TaskGroups,
    horizon: int,
    *,
    hierarchy_partitions: int | None = None,
) -> int | None:
    if (
        config.conformal is None
        or config.conformal.partition != "series"
        or config.conformal.max_partitions is None
    ):
        return None

    if hierarchy_partitions is not None:
        # Hierarchy runs consume the partition count derived from the shared
        # preparation facts: the full node set for bottom-only synthesis, the
        # projected node set from the expansion estimate for eager runs.
        estimated_partitions = hierarchy_partitions
    else:
        # Memoize the per-history uid count on a content-derived key — the sorted
        # stringified uid tuple, the load-bearing component of task_builder's
        # _global_dedup_key — instead of id(task.history), which silently
        # recounts a cloned-but-equal history frame. The cached fact is a
        # uid-count, which depends only on the uid set, so config/horizon are not
        # part of the key (a tuple, not a joined string, so comma-bearing
        # unique_ids cannot alias). This memo is reachable only on the flat-panel
        # partition path (hierarchy runs take the branch above), and the key is
        # itself a uid scan, so the change is correctness-motivated (clone-safe),
        # not a performance optimization.
        unique_ids_by_history: dict[tuple[str, ...], int] = {}
        estimated_partitions = 0
        for task_group in (tasks.local, tasks.global_):
            for task in task_group:
                history_key = tuple(sorted(task.history[UNIQUE_ID].astype(str).unique()))
                if history_key not in unique_ids_by_history:
                    unique_ids_by_history[history_key] = len(history_key)
                estimated_partitions += unique_ids_by_history[history_key] * horizon

    if estimated_partitions > config.conformal.max_partitions:
        raise ValueError(
            "conformal.partition='series' would create approximately "
            f"{estimated_partitions} model/node/horizon partitions; configured maximum is "
            f"{config.conformal.max_partitions}. Increase conformal.max_partitions only after "
            "confirming the run has enough memory for the resulting calibration state."
        )
    return estimated_partitions
