"""Compile validated M5 data into canonical domain inputs and execution intent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

from newcalibre.domain import (
    AGGREGATE_NODE_PREFIX,
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    TOTAL_NODE_LABEL,
    Calendar,
    HierarchyIndex,
    Panel,
)
from newcalibre.protocols.m5.config import M5ExecutionConfig, M5ProtocolConfig
from newcalibre.protocols.m5.loader import M5DataError, M5Dataset

_SOURCE_TO_CANONICAL_FACT = {
    "item_id": "item",
    "dept_id": "department",
    "cat_id": "category",
    "store_id": "store",
    "state_id": "state",
}
_CANONICAL_FACTS = frozenset(_SOURCE_TO_CANONICAL_FACT.values())


@dataclass(frozen=True, slots=True)
class _CompiledM5Protocol:
    """Retain canonical M5 inputs and immutable validated pipeline intent."""

    panel: Panel
    hierarchy: HierarchyIndex
    origins: tuple[pd.Timestamp, ...]
    config: M5ProtocolConfig

    @property
    def model_config(self) -> dict[str, object]:
        """Return an isolated validated forecast intent snapshot."""
        return self.config.model_config

    @property
    def reconciliation_strategy(self) -> str:
        """Return the validated point-reconciliation intent."""
        return self.config.reconciliation_strategy

    @property
    def conformal_config(self) -> dict[str, object]:
        """Return an isolated validated conformal intent snapshot."""
        return self.config.conformal_config

    @property
    def conformal_partition(self) -> str:
        """Return the semantic M5 calibration partition declaration."""
        return self.config.conformal_partition

    @property
    def execution(self) -> M5ExecutionConfig:
        """Return fixed logical execution intent without dispatch behavior."""
        return self.config.execution

    @property
    def output_dir(self) -> Path:
        """Return the validated output-root intent without emitting files."""
        return self.config.output_dir


def compile_m5_protocol(
    dataset: M5Dataset,
    config: M5ProtocolConfig,
) -> _CompiledM5Protocol:
    """Compile the selected release without constructing tasks or an engine."""
    if not isinstance(dataset, M5Dataset):
        raise M5DataError("dataset must be an M5Dataset")
    if not isinstance(config, M5ProtocolConfig):
        raise M5DataError("config must be an M5ProtocolConfig")
    if dataset.config is not config:
        raise M5DataError("dataset and compiler must use the same M5 configuration")

    sales = dataset.sales
    panel = _compile_panel(sales, dates=dataset.dates)
    facts = sales.loc[:, ["series_key", *_SOURCE_TO_CANONICAL_FACT]].rename(
        columns=_SOURCE_TO_CANONICAL_FACT
    )
    hierarchy = _compile_hierarchy(facts, dataset.bottom_series)
    first_origin = panel.calendar.retreat(dataset.history_end, config.origin_count - 1)
    origins = tuple(
        panel.calendar.advance(first_origin, offset) for offset in range(config.origin_count)
    )
    return _CompiledM5Protocol(panel, hierarchy, origins, config)


def _compile_panel(
    sales: pd.DataFrame,
    *,
    dates: tuple[pd.Timestamp, ...],
) -> Panel:
    day_columns = tuple(f"d_{index}" for index in range(1, len(dates) + 1))
    expected = {"series_key", *_SOURCE_TO_CANONICAL_FACT, *day_columns}
    if set(sales.columns) != expected:
        raise M5DataError("selected sales must contain exact identity, hierarchy, and day facts")
    long = sales.melt(
        id_vars=["series_key"],
        value_vars=list(day_columns),
        var_name="day",
        value_name=OBSERVED_VALUE,
    )
    timestamp_blocks = np.repeat(np.asarray(dates, dtype="datetime64[ns]"), len(sales))
    frame = pd.DataFrame(
        {
            SERIES_KEY: long["series_key"].astype("string"),
            TIMESTAMP: pd.to_datetime(timestamp_blocks),
            OBSERVED_VALUE: long[OBSERVED_VALUE].astype("int64"),
        }
    )
    return Panel.from_frame(frame, calendar=Calendar("D"))


def _compile_hierarchy(
    facts: pd.DataFrame,
    bottom_series: tuple[str, ...],
) -> HierarchyIndex:
    expected = {SERIES_KEY, *_CANONICAL_FACTS}
    if not isinstance(facts, pd.DataFrame) or set(facts.columns) != expected:
        raise M5DataError(
            "M5 hierarchy facts must contain exact series, item, department, category, "
            "store, and state columns"
        )
    return HierarchyIndex.from_facts(facts, bottom_series=bottom_series)


def _level_from_node_label(label: str) -> str:
    if not isinstance(label, str) or not label:
        raise M5DataError("M5 hierarchy node label must be a non-empty string")
    if label == TOTAL_NODE_LABEL:
        return "total"
    aggregate_prefix = f"{AGGREGATE_NODE_PREFIX}:"
    if label.startswith(aggregate_prefix):
        tokens = label.split(":", maxsplit=3)
        if len(tokens) != 4:
            raise M5DataError(f"malformed M5 aggregate node label: {label!r}")
        attribute = unquote(tokens[1])
        if attribute not in _CANONICAL_FACTS:
            raise M5DataError(f"unknown M5 aggregate level in node label: {label!r}")
        return attribute
    if label.startswith(AGGREGATE_NODE_PREFIX):
        raise M5DataError(f"malformed M5 aggregate node label: {label!r}")
    return "bottom"
