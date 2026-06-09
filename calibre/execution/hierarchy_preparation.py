"""Prepare engine inputs for flat and hierarchical CLI runs."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from calibre.cli.config import BackendConfig, ReconciliationConfig
from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.core.forecast_task import TaskGroups
from calibre.execution.dataset import DatasetBundle
from calibre.execution.hierarchy_memory import enforce_hierarchical_expansion_memory_limit
from calibre.execution.task_builder import build_node_history, build_tasks
from calibre.reconciliation import HierarchicalIntervalPhase, Reconciler


@dataclass(frozen=True)
class RunPreparationDiagnostics:
    conformal_partition_estimate: int | None = None
    hierarchy_memory_guard_applied: bool = False


@dataclass(frozen=True)
class RunPreparation:
    tasks: TaskGroups
    actuals: pd.DataFrame
    origins: list[pd.Timestamp]
    conformal_config: SymmetricIntervalConfig | None
    reconciliation_hierarchy: pd.DataFrame | None
    reconciler: Reconciler | None
    hierarchical_interval_phase: HierarchicalIntervalPhase | None
    diagnostics: RunPreparationDiagnostics


def prepare_run(config: BackendConfig, bundle: DatasetBundle) -> RunPreparation:
    """Resolve hierarchy-specific run inputs for ``BackendEngine`` construction."""
    model_configs = [task.resolved_model_config() for task in config.tasks]
    horizon = config.tasks[0].horizon
    reconciliation_hierarchy = _hierarchy_for_run(config, bundle)
    guard_applied = reconciliation_hierarchy is not None
    if guard_applied:
        enforce_hierarchical_expansion_memory_limit(
            bundle.history,
            reconciliation_hierarchy,
            horizon=horizon,
            model_count=len(config.tasks),
        )

    actuals = build_node_history(bundle.history, reconciliation_hierarchy)
    tasks = build_tasks(actuals, model_configs, horizon)
    conformal_partition_estimate = _enforce_conformal_partition_limit(config, tasks, horizon)
    origins = config.origins.to_list()
    if not origins:
        raise ValueError("origins resolved to an empty list")

    conformal_config: SymmetricIntervalConfig | None = (
        config.conformal.to_runtime_config() if config.conformal is not None else None
    )
    reconciliation_config = config.reconciliation or ReconciliationConfig()
    return RunPreparation(
        tasks=tasks,
        actuals=actuals,
        origins=origins,
        conformal_config=conformal_config,
        reconciliation_hierarchy=reconciliation_hierarchy,
        reconciler=(
            None
            if config.hierarchical_intervals is not None
            else reconciliation_config.to_reconciler()
        ),
        hierarchical_interval_phase=(
            config.hierarchical_intervals.to_phase()
            if config.hierarchical_intervals is not None
            else None
        ),
        diagnostics=RunPreparationDiagnostics(
            conformal_partition_estimate=conformal_partition_estimate,
            hierarchy_memory_guard_applied=guard_applied,
        ),
    )


def _hierarchy_for_run(config: BackendConfig, bundle: DatasetBundle) -> pd.DataFrame | None:
    if config.hierarchical_intervals is not None:
        if bundle.hierarchy is None:
            raise ValueError("hierarchical_intervals requires a dataset hierarchy")
        return bundle.hierarchy
    if config.reconciliation is None or config.reconciliation.strategy == "none":
        return None
    return bundle.hierarchy


def _enforce_conformal_partition_limit(
    config: BackendConfig,
    tasks: TaskGroups,
    horizon: int,
) -> int | None:
    if config.conformal is None or config.conformal.partition != "series":
        return None

    estimated_partitions = sum(
        int(task.history[UNIQUE_ID].astype(str).nunique()) * horizon for task in tasks.tasks
    )
    if (
        config.conformal.max_partitions is not None
        and estimated_partitions > config.conformal.max_partitions
    ):
        raise ValueError(
            "conformal.partition='series' would create approximately "
            f"{estimated_partitions} model/node/horizon partitions; configured maximum is "
            f"{config.conformal.max_partitions}. Increase conformal.max_partitions only after "
            "confirming the run has enough memory for the resulting calibration state."
        )
    return estimated_partitions
