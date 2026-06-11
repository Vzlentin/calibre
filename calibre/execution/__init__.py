"""Execution layer: backend engine, ledgers, pipeline runners, decision loops."""

from calibre.core.forecast_task import TaskGroups
from calibre.core.io import exists, join_uri, open_fs, resolve_path
from calibre.execution.backend import (
    BackendEngine,
    BackendResult,
    ConformalOptions,
    ExecutionOptions,
    LedgerOutputOptions,
)
from calibre.execution.data_loading import (
    load_master,
    load_period,
    melt_wide_instock,
    melt_wide_sales,
)
from calibre.execution.dataset import DatasetAdapter, DatasetBundle
from calibre.execution.dataset_registry import (
    available_dataset_adapters,
    get_dataset_adapter_cls,
    register_dataset_adapter,
    resolve_dataset_adapter,
)
from calibre.execution.decision_loop import (
    DecisionLoop,
    DecisionLoopConfig,
    RoundResult,
    observe_cumulative,
    observe_pending,
    observe_per_horizon,
)
from calibre.execution.ledger import (
    InMemoryLedger,
    InMemoryOrderLedger,
    Ledger,
    OrderLedger,
    StreamingLedger,
    StreamingOrderLedger,
)
from calibre.execution.runner import PipelineResult, run_backtest, run_forecast
from calibre.execution.task_builder import build_tasks
from calibre.execution.validation import load_costs, validate_dataset_bundle

__all__ = [
    "BackendEngine",
    "BackendResult",
    "ConformalOptions",
    "ExecutionOptions",
    "LedgerOutputOptions",
    "load_master",
    "load_period",
    "melt_wide_instock",
    "melt_wide_sales",
    "DatasetAdapter",
    "DatasetBundle",
    "available_dataset_adapters",
    "get_dataset_adapter_cls",
    "register_dataset_adapter",
    "resolve_dataset_adapter",
    "DecisionLoop",
    "DecisionLoopConfig",
    "RoundResult",
    "observe_cumulative",
    "observe_pending",
    "observe_per_horizon",
    "Ledger",
    "InMemoryLedger",
    "StreamingLedger",
    "OrderLedger",
    "InMemoryOrderLedger",
    "StreamingOrderLedger",
    "exists",
    "join_uri",
    "open_fs",
    "resolve_path",
    "PipelineResult",
    "run_backtest",
    "run_forecast",
    "build_tasks",
    "TaskGroups",
    "load_costs",
    "validate_dataset_bundle",
]
