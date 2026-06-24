"""Prepare engine inputs for flat and hierarchical CLI runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import pandas as pd

from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig
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
from calibre.reconciliation import Reconciler, resolve_reconciler
from calibre.reconciliation.nixtla_adapter import NIXTLA_SPARSE_STRATEGIES
from calibre.reconciliation.summing import HierarchyIndex, build_hierarchy_index


class _TaskConfig(Protocol):
    @property
    def horizon(self) -> int: ...

    def resolved_model_config(self) -> dict[str, Any]: ...


class _OriginsConfig(Protocol):
    @property
    def freq(self) -> str: ...

    def to_list(self) -> list[pd.Timestamp]: ...


class _ConformalConfig(Protocol):
    @property
    def partition(self) -> str: ...

    @property
    def max_partitions(self) -> int | None: ...

    @property
    def spread(self) -> str: ...

    def to_runtime_config(self) -> SymmetricIntervalConfig: ...


class _OrderConformalConfig(Protocol):
    def to_runtime_config(self) -> CumulativeConformalRiskConfig: ...


class _ReconciliationConfig(Protocol):
    @property
    def strategy(self) -> str: ...

    def to_reconciler(self) -> Reconciler: ...


class RunPreparationConfig(Protocol):
    """Structural view of the run config :func:`prepare_run` reads."""

    @property
    def tasks(self) -> Sequence[_TaskConfig]: ...

    @property
    def origins(self) -> _OriginsConfig: ...

    @property
    def conformal(self) -> _ConformalConfig | None: ...

    @property
    def order_conformal(self) -> _OrderConformalConfig | None: ...

    @property
    def reconciliation(self) -> _ReconciliationConfig | None: ...


@dataclass(frozen=True)
class RunPreparation:
    """Resolved engine inputs for one run.

    Carries tasks, actuals, origins, the reconciler, and conformal configs. The
    diagnostic ``conformal_config`` (two-sided band) and the decision
    ``order_conformal_config`` (one-sided cumulative bound) are resolved
    independently, but only one becomes the engine's runtime — the CLI rejects
    configuring both.
    """

    tasks: TaskGroups
    actuals: pd.DataFrame | ActualsSource
    origins: list[pd.Timestamp]
    conformal_config: SymmetricIntervalConfig | None
    order_conformal_config: CumulativeConformalRiskConfig | None
    hierarchy_index: HierarchyIndex | None
    reconciler: Reconciler | None
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

    # Resolved before the bottom_only branch so its actuals precompute can derive
    # the evaluation window from the origins; to_list() depends only on
    # config.origins, so the hoist is independent of the tasks/actuals built below.
    origins = config.origins.to_list()
    if not origins:
        raise ValueError("origins resolved to an empty list")

    actuals: pd.DataFrame | ActualsSource
    hierarchy_partitions: int | None = None
    if bottom_only:
        # Native point bottom_up consumes bottom-only forecasts and synthesizes
        # aggregate node rows itself, so the run never materializes the eager
        # node-history frame: tasks come from bottom history and actuals
        # resolve lazily. The expansion guard targets exactly that
        # materialization, so it does not apply here. Ledger partitions still
        # span the full hierarchy node set once aggregates are synthesized.
        assert reconciliation_hierarchy is not None and hierarchy_index is not None
        actuals = HierarchyActualsSource(bundle.history, hierarchy_index)
        # Warm the aggregate-actuals cache for the whole window up front: the
        # window-sliding group-bys move out of the per-origin resolve hot path,
        # making the actuals lookup O(due). The ds set is a best-effort superset
        # of the dates the ledger requests (the model adapter's future-frame freq
        # is configured independently of OriginsConfig.freq), so a missed pair
        # falls through to the unchanged lazy compute byte-identically.
        ds_values = _evaluation_window_ds(origins, config.origins.freq, horizon)
        actuals.precompute(hierarchy_index.node_labels, ds_values)
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
            # This branch is the matrix-requiring path (the Nixtla point
            # reconcilers build S per origin); native bottom_up takes the
            # bottom_only branch and never reaches here. The S term is
            # strategy-conditional: the sparse-capable
            # roster builds a csr (nnz*8 data + nnz*4 indices + (rows+1)*4
            # indptr, nnz analytic from index facts), while erm/mint_shrink
            # have no upstream sparse implementation and keep the dense
            # node_count x n_bottom x 8 ceiling.
            expansion = replace(
                expansion,
                summing_matrix_bytes=_summing_matrix_bytes(config, hierarchy_index),
            )
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

    conformal_config: SymmetricIntervalConfig | None = (
        config.conformal.to_runtime_config() if config.conformal is not None else None
    )
    order_conformal_config: CumulativeConformalRiskConfig | None = (
        config.order_conformal.to_runtime_config() if config.order_conformal is not None else None
    )
    return RunPreparation(
        tasks=tasks,
        actuals=actuals,
        origins=origins,
        conformal_config=conformal_config,
        order_conformal_config=order_conformal_config,
        hierarchy_index=hierarchy_index,
        reconciler=(
            config.reconciliation.to_reconciler()
            if config.reconciliation is not None
            else resolve_reconciler("none")
        ),
        conformal_partition_estimate=conformal_partition_estimate,
    )


def _evaluation_window_ds(
    origins: Sequence[pd.Timestamp], freq: str, horizon: int
) -> list[pd.Timestamp]:
    """Derive the best-effort ``ds`` superset for the bottom_up actuals precompute.

    Unions, per origin, the prediction-date span ``date_range(origin, periods=
    horizon + 1, freq=freq)`` and dedups. The run ``freq`` is the same one
    ``OriginsConfig.to_list`` steps origins by; the model adapter's future-frame
    freq and its h=1 anchor (origin vs origin+1) are configured independently, so
    the span starts at the origin and runs one step past ``horizon`` to bracket
    both anchoring conventions — a *superset* of the dates the ledger requests,
    not an engine-canonical reconstruction. A pair the ledger requests but this
    misses falls through to the lazy compute, so seed coverage — never
    correctness — is what depends on ``freq`` matching; extra seeded dates are
    harmless dead cache entries.
    """
    dates: pd.DatetimeIndex = pd.DatetimeIndex([])
    for origin in origins:
        dates = dates.union(pd.date_range(pd.Timestamp(origin), periods=horizon + 1, freq=freq))
    return list(dates)


def _summing_matrix_bytes(config: RunPreparationConfig, hierarchy_index: HierarchyIndex) -> int:
    """Per-origin summing-matrix bytes for the eager (matrix-requiring) branch.

    The strategy decides the representation, so the preflight charges what the
    run will actually allocate: the csr estimate for the sparse-capable roster
    (computable from index facts without building anything), the dense float64
    product only for the strategies with no upstream sparse implementation.
    """
    n_bottom = len(hierarchy_index.bottom_ids)
    n_nodes = len(hierarchy_index.node_labels)
    if config.reconciliation is not None and config.reconciliation.strategy != "none":
        strategy = config.reconciliation.strategy
    elif _coherent_spread_active(config):
        # The coherent-draws spread builds the sparse csr S at runtime
        # construction; charge the same csr estimate the sparse roster does.
        nnz = n_bottom * (2 + len(hierarchy_index.attr_cols))
        return nnz * 8 + nnz * 4 + (n_nodes + 1) * 4
    else:
        # _hierarchy_for_run returns no hierarchy for these configs, so the
        # eager branch never calls this function; a silent dense estimate here
        # would mask a wiring regression.
        raise ValueError("_summing_matrix_bytes requires an active reconciliation strategy")
    if strategy in NIXTLA_SPARSE_STRATEGIES:
        # Identity block + one membership per attribute column + total row.
        nnz = n_bottom * (2 + len(hierarchy_index.attr_cols))
        return nnz * 8 + nnz * 4 + (n_nodes + 1) * 4
    return n_nodes * n_bottom * 8


def _is_point_bottom_up(config: RunPreparationConfig) -> bool:
    return config.reconciliation is not None and config.reconciliation.strategy == "bottom_up"


def _coherent_spread_active(config: RunPreparationConfig) -> bool:
    """Whether the conformal config selects the coherent-draws spread."""
    return config.conformal is not None and config.conformal.spread == "coherent_draws"


def _hierarchy_for_run(config: RunPreparationConfig, bundle: DatasetBundle) -> pd.DataFrame | None:
    if _coherent_spread_active(config):
        # The coherent-draws spread needs S even though point reconciliation is
        # none; supply the hierarchy so the runtime builder can fold S in.
        if bundle.hierarchy is None:
            raise ValueError("conformal spread='coherent_draws' requires a dataset hierarchy")
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
