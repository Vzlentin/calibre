"""Compose strict M5 inputs through the generic successor engine."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pandas as pd

from newcalibre.domain import (
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    ActualsSemantics,
    HierarchyIndex,
    Panel,
    Scope,
    SessionIdentity,
)
from newcalibre.engine import (
    Engine,
    InMemoryIndexedRunStore,
    InMemoryLedgerReader,
    InMemoryPanelSource,
    InProcessDispatch,
    TimeLoop,
    TimeLoopRequest,
)
from newcalibre.forecasting import resolve_adapter
from newcalibre.protocols.m5.compiler import compile_m5_protocol
from newcalibre.protocols.m5.config import load_m5_config
from newcalibre.protocols.m5.loader import load_m5_dataset
from newcalibre.protocols.m5.scorer import M5Diagnostics, score_m5

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class M5RunResult:
    """Return compact completion facts and closed-ledger M5 diagnostics."""

    session: SessionIdentity
    input_inventory_sha256: str
    forecast_origin_count: int
    commit_count: int
    node_count: int
    expected_row_count: int
    resolved_row_count: int
    eligible_row_count: int
    scored_row_count: int
    pending_row_count: int
    diagnostics: M5Diagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("M5 run session must be a SessionIdentity")
        if (
            not isinstance(self.input_inventory_sha256, str)
            or _SHA256.fullmatch(self.input_inventory_sha256) is None
        ):
            raise TypeError("M5 run input inventory identity must be a SHA-256 digest")
        count_names = (
            "forecast_origin_count",
            "commit_count",
            "node_count",
            "expected_row_count",
            "resolved_row_count",
            "eligible_row_count",
            "scored_row_count",
            "pending_row_count",
        )
        for name in count_names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TypeError(f"M5 run {name} must be a non-negative integer")
        if not isinstance(self.diagnostics, M5Diagnostics):
            raise TypeError("M5 run diagnostics must be M5Diagnostics")
        context = self.diagnostics.context
        counts = self.diagnostics.population.counts
        expected_rows = self.forecast_origin_count * self.node_count * context.horizon
        if self.commit_count != self.forecast_origin_count + 1:
            raise ValueError("M5 run must contain one close commit after its forecast origins")
        if self.session.value != context.session_id:
            raise ValueError("M5 run session does not match its diagnostics")
        if context.origin_count != self.forecast_origin_count:
            raise ValueError("M5 run origin count does not match its diagnostics")
        if context.node_count != self.node_count:
            raise ValueError("M5 run node count does not match its diagnostics")
        if context.expected_row_count != self.expected_row_count:
            raise ValueError("M5 run expected row count does not match its diagnostics")
        if self.expected_row_count != expected_rows or counts.total != expected_rows:
            raise ValueError("M5 run row universe does not match its configured dimensions")
        if self.resolved_row_count != counts.resolved:
            raise ValueError("M5 run resolved count does not match its diagnostics")
        if self.eligible_row_count != counts.eligible:
            raise ValueError("M5 run eligible count does not match its diagnostics")
        if self.scored_row_count != counts.scored:
            raise ValueError("M5 run scored count does not match its diagnostics")
        if self.resolved_row_count + self.pending_row_count != self.expected_row_count:
            raise ValueError("M5 run resolved and pending counts do not cover its row universe")


def run_m5(config_path: Path) -> M5RunResult:
    """Run one strict M5 configuration through the generic time-loop engine."""
    config = load_m5_config(config_path)
    dataset = load_m5_dataset(_PROJECT_ROOT, config)
    compiled = compile_m5_protocol(dataset, config)
    forecast_panel = _all_node_panel(compiled.panel, hierarchy=compiled.hierarchy)
    session = SessionIdentity.derive(
        tenant=config.dataset,
        series_keys=compiled.hierarchy.node_labels,
        calendar=forecast_panel.calendar,
        horizon=config.horizon,
        model_config=compiled.model_config,
        conformal_config=compiled.conformal_config,
    )
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=forecast_panel.calendar,
        actuals=compiled.panel,
        actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
        hierarchy=compiled.hierarchy,
    )
    engine = Engine(
        session=session,
        panel_source=InMemoryPanelSource(forecast_panel),
        run_store=store,
        dispatch_backend=InProcessDispatch(),
        hierarchy=compiled.hierarchy,
        adapter_resolver=resolve_adapter,
        reconciliation_strategy=compiled.reconciliation_strategy,
        orderer=None,
    )
    time_loop = TimeLoop(
        engine=engine,
        run_store=store,
        request=TimeLoopRequest(
            session=session,
            origins=compiled.origins,
            settlement_end=compiled.origins[-1],
            scope=Scope(config.model_scope),
            initial_inventory_positions={},
            actuals_semantics=ActualsSemantics.CENSORED_SALES_SURROGATE,
        ),
    ).run()
    reader = InMemoryLedgerReader(store)
    diagnostics = score_m5(
        config,
        reader,
        output_dir=_PROJECT_ROOT / compiled.output_dir,
    )
    pending_count = store.pending_observation_count
    return M5RunResult(
        session=session,
        input_inventory_sha256=dataset.input_inventory_sha256,
        forecast_origin_count=store.forecast_origin_count,
        commit_count=len(time_loop.receipts),
        node_count=len(compiled.hierarchy.node_labels),
        expected_row_count=diagnostics.context.expected_row_count,
        resolved_row_count=diagnostics.population.counts.resolved,
        eligible_row_count=diagnostics.population.counts.eligible,
        scored_row_count=diagnostics.population.counts.scored,
        pending_row_count=pending_count,
        diagnostics=diagnostics,
    )


def _all_node_panel(panel: Panel, *, hierarchy: HierarchyIndex) -> Panel:
    """Aggregate complete bottom histories into one canonical all-node panel."""
    frame = panel.frame
    node_labels = hierarchy.node_labels
    node_values: dict[str, list[int | float | None]] = {label: [] for label in node_labels}
    timestamps: list[pd.Timestamp] = []
    for timestamp, section in frame.groupby(TIMESTAMP, sort=True, observed=True):
        values = dict(zip(section[SERIES_KEY], section[OBSERVED_VALUE], strict=True))
        aggregated = hierarchy.aggregate(values)
        timestamps.append(cast(pd.Timestamp, timestamp))
        for label in node_labels:
            node_values[label].append(aggregated[label])
    row_count = len(timestamps)
    all_nodes = pd.DataFrame(
        {
            SERIES_KEY: pd.Series(
                [label for label in node_labels for _ in range(row_count)],
                dtype="string",
            ),
            TIMESTAMP: pd.to_datetime(timestamps * len(node_labels)),
            OBSERVED_VALUE: [value for label in node_labels for value in node_values[label]],
        }
    )
    return Panel.from_frame(
        all_nodes,
        calendar=panel.calendar,
        target_support=panel.target_support,
    )


__all__ = ["M5RunResult", "run_m5"]
