"""Memory preflight for hierarchical node-history expansion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID
from calibre.reconciliation.summing import TOTAL_LABEL, HierarchyIndex

_LOADED_HISTORY_BYTES_PER_ROW = 32
_NODE_HISTORY_BYTES_PER_ROW = 128
_NODE_HISTORY_TEMPORARY_MULTIPLIER = 3
_TASK_HISTORY_BYTES_PER_ROW = 96
_FORECAST_PARTITION_BYTES = 256
_RUNTIME_OVERHEAD_BYTES = 512 * 1024**2
_CGROUP_UNLIMITED_BYTES = 1 << 60


@dataclass(frozen=True, slots=True)
class HierarchicalExpansionEstimate:
    bottom_unique_ids: int
    aggregate_nodes: int
    node_count: int
    bottom_rows: int
    projected_node_count: int
    projected_node_history_rows: int
    forecast_partitions: int
    model_count: int
    # Summing-matrix bytes a matrix-requiring strategy must allocate per
    # origin: the csr estimate for the sparse-capable roster (bottom_up fused,
    # ols, wls_struct, wls_var), the dense node_count x n_bottom x 8 product
    # only for erm/mint_shrink, which have no upstream sparse implementation.
    # Zero unless run preparation sets it in the eager branch — native
    # bottom_up and strategy="none" never build S, so the term never applies
    # to them. Defaults to 0 so direct-call estimates are unaffected.
    summing_matrix_bytes: int = 0


def estimate_hierarchical_expansion(
    history: pd.DataFrame,
    hierarchy_index: HierarchyIndex,
    *,
    horizon: int,
    model_count: int = 1,
) -> HierarchicalExpansionEstimate:
    """Project node-history rows using the same completeness rules as expansion.

    The preflight must not materialize the expanded frame, but it must mirror
    ``build_node_history``: aggregate and total rows only exist for dates where
    all bottom members of that node are present. ``hierarchy_index`` is the
    index run preparation already built, so cardinality facts have one source.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if model_count < 1:
        raise ValueError("model_count must be at least 1")
    missing = [col for col in (UNIQUE_ID, DS) if col not in history.columns]
    if missing:
        raise ValueError(f"history missing required column(s): {missing}")
    if history.empty:
        raise ValueError("history has no rows")

    data = history[[UNIQUE_ID, DS]].copy()
    data[UNIQUE_ID] = data[UNIQUE_ID].astype(str)
    data[DS] = pd.to_datetime(data[DS]).astype("datetime64[ns]")

    bottom_ids = set(hierarchy_index.bottom_ids)
    observed_ids = set(data[UNIQUE_ID].unique())
    unknown = observed_ids - bottom_ids
    if unknown:
        raise ValueError(
            f"history contains unique_id values not present in hierarchy: {sorted(unknown)}"
        )

    projected_node_labels = set(observed_ids)
    projected_node_history_rows = len(data)
    hierarchy_frame = hierarchy_index.frame
    joined = data.merge(
        hierarchy_frame[[UNIQUE_ID, *hierarchy_index.attr_cols]],
        on=UNIQUE_ID,
        how="inner",
    )

    expected_members = hierarchy_index.expected_members()
    for col in hierarchy_index.attr_cols:
        # Group on the stringified attribute column so the projection reads the
        # single stringified counting authority (#148), matching expansion.
        col_str = joined[col].astype(str)
        grouped = (
            joined.assign(**{col: col_str})
            .groupby([col, DS], sort=True)
            .agg(_member_count=(UNIQUE_ID, "nunique"))
            .reset_index()
        )
        complete = grouped[grouped["_member_count"] == grouped[col].map(expected_members[col])]
        projected_node_history_rows += len(complete)
        projected_node_labels.update(
            f"{col}={value}" for value in complete[col].astype(str).unique()
        )

    total_counts = data.groupby(DS, sort=True)[UNIQUE_ID].nunique()
    complete_total_rows = int((total_counts == len(hierarchy_index.bottom_ids)).sum())
    projected_node_history_rows += complete_total_rows
    if complete_total_rows:
        projected_node_labels.add(TOTAL_LABEL)

    projected_node_count = len(projected_node_labels)
    node_count = len(hierarchy_index.node_labels)
    return HierarchicalExpansionEstimate(
        bottom_unique_ids=len(observed_ids),
        aggregate_nodes=node_count - len(hierarchy_index.bottom_ids),
        node_count=node_count,
        bottom_rows=len(data),
        projected_node_count=projected_node_count,
        projected_node_history_rows=projected_node_history_rows,
        forecast_partitions=projected_node_count * horizon * model_count,
        model_count=model_count,
    )


def read_linux_available_memory_bytes() -> int | None:
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        name, _, rest = line.partition(":")
        if name != "MemAvailable":
            continue
        value, unit, *_ = rest.strip().split()
        if unit != "kB":
            return None
        return int(value) * 1024
    return None


def _read_memory_limit_value(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value >= _CGROUP_UNLIMITED_BYTES:
        return None
    return max(value, 0)


def _read_memory_usage_value(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return max(value, 0)


def _cgroup_candidate_dirs(cgroup_root: Path, self_cgroup: Path) -> list[Path]:
    candidates: list[Path] = []
    try:
        lines = self_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        relative_path = relative.lstrip("/")
        if controllers == "":
            candidates.append(cgroup_root / relative_path)
        elif "memory" in controllers.split(","):
            candidates.append(cgroup_root / "memory" / relative_path)
            candidates.append(cgroup_root / relative_path)

    candidates.extend([cgroup_root, cgroup_root / "memory"])
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def read_cgroup_available_memory_bytes(
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    self_cgroup: Path = Path("/proc/self/cgroup"),
) -> int | None:
    remaining_values: list[int] = []
    for candidate in _cgroup_candidate_dirs(cgroup_root, self_cgroup):
        for limit_name, usage_name in (
            ("memory.max", "memory.current"),
            ("memory.limit_in_bytes", "memory.usage_in_bytes"),
        ):
            limit = _read_memory_limit_value(candidate / limit_name)
            usage = _read_memory_usage_value(candidate / usage_name)
            if limit is None or usage is None:
                continue
            remaining_values.append(max(limit - usage, 0))
    if not remaining_values:
        return None
    return min(remaining_values)


def effective_available_memory_bytes(
    host_available: int | None,
    cgroup_available: int | None,
) -> int | None:
    values = [value for value in (host_available, cgroup_available) if value is not None]
    if not values:
        return None
    return min(values)


def read_effective_available_memory_bytes() -> int | None:
    return effective_available_memory_bytes(
        read_linux_available_memory_bytes(),
        read_cgroup_available_memory_bytes(),
    )


def estimated_node_history_peak_bytes(estimate: HierarchicalExpansionEstimate) -> int:
    """Conservative upper bound for expansion plus immediately downstream copies.

    The row-count term is separately locked against actual M5 fixture expansion
    in tests; the constants deliberately over-reserve pandas materialization,
    per-task history copies, forecast state, and runtime headroom because this
    guard blocks only when the projected run cannot fit in detected memory.
    """
    loaded_history = estimate.bottom_rows * _LOADED_HISTORY_BYTES_PER_ROW
    node_history_materialization = estimate.projected_node_history_rows * (
        _NODE_HISTORY_BYTES_PER_ROW * (1 + _NODE_HISTORY_TEMPORARY_MULTIPLIER)
    )
    task_histories = (
        estimate.projected_node_history_rows * estimate.model_count * _TASK_HISTORY_BYTES_PER_ROW
    )
    forecast_partition_state = estimate.forecast_partitions * _FORECAST_PARTITION_BYTES
    return (
        loaded_history
        + node_history_materialization
        + task_histories
        + forecast_partition_state
        + estimate.summing_matrix_bytes
        + _RUNTIME_OVERHEAD_BYTES
    )


def format_bytes(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def enforce_hierarchical_expansion_memory_limit(
    estimate: HierarchicalExpansionEstimate,
) -> None:
    """Block eager node-history expansion that cannot fit in detected memory.

    Consumes the shared :class:`HierarchicalExpansionEstimate` facts computed
    once by run preparation rather than recomputing a parallel model of the
    hierarchy run.
    """
    available_memory = read_effective_available_memory_bytes()
    if available_memory is None:
        return

    estimated_peak = estimated_node_history_peak_bytes(estimate)
    if estimated_peak <= available_memory:
        return

    raise ValueError(
        "hierarchical node-history expansion is estimated to need "
        f"{format_bytes(estimated_peak)} before forecasting, which exceeds the "
        f"detected effective memory guard of {format_bytes(available_memory)}. "
        f"Estimated bottom rows: {estimate.bottom_rows}; "
        f"node count: {estimate.node_count}; "
        f"projected node-history rows: {estimate.projected_node_history_rows}; "
        f"projected forecast nodes: {estimate.projected_node_count}; "
        f"forecast partitions: {estimate.forecast_partitions}; "
        f"summing-matrix bytes: {format_bytes(estimate.summing_matrix_bytes)}. "
        "The estimate includes pandas materialization, task-history copies, the "
        "summing-matrix representation the strategy allocates per origin (csr "
        "for the sparse-capable strategies; the dense matrix only for erm and "
        "mint_shrink, which have no upstream sparse implementation and keep a "
        "dense memory ceiling), and runtime overhead. "
        "Streaming output does not avoid this input-side materialization; use a "
        "smaller hierarchy/input or run on a host with more memory."
    )
