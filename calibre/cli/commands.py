from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from uuid import UUID

import pandas as pd

from calibre.cli.config import (
    BackendConfig,
    ReconciliationConfig,
    load_config,
    load_config_from_mapping,
)
from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.core.io import is_local_fs, open_fs
from calibre.core.metrics import set_order_cost
from calibre.execution.backend import (
    BackendEngine,
    BackendResult,
    ConformalOptions,
    HierarchicalIntervalEngineOptions,
    LedgerOutputOptions,
    ReconciliationOptions,
)
from calibre.execution.dataset import DatasetBundle
from calibre.execution.dataset_registry import resolve_dataset_adapter
from calibre.execution.task_builder import build_node_history, build_tasks
from calibre.execution.validation import validate_dataset_bundle
from calibre.ordering.policy_config import (
    NewsvendorConfig,
    OrderPolicy,
    RsConfig,
    RssConfig,
)
from calibre.storage.state import ConformalStateStore

logger = logging.getLogger(__name__)


def _fs_result_uri(fs, path: str) -> str:
    if is_local_fs(fs):
        return path
    return str(fs.unstrip_protocol(path))


_HEALTH_CONFIG: dict[str, Any] = {
    "config_schema": "1.0",
    "dataset": {"adapter": "vn2", "path": "benchmarks/vn2/fixture", "period": 0},
    "tasks": [
        {
            "model": "SeasonalNaive",
            "horizon": 2,
            "config": {"backend": "statsforecast", "season_length": 2},
        }
    ],
    "origins": {"start": "2024-01-29", "end": "2024-01-29", "freq": "W-MON"},
    "output": {"ledger_path": "results/vn2/smoke-ledger.parquet", "streaming": False},
    "execution": {"backend": "local", "seed": 42},
}


def _load_dataset(config: BackendConfig):
    adapter = resolve_dataset_adapter(config.dataset.adapter)
    kwargs = dict(config.dataset.options)
    if config.dataset.period is not None:
        kwargs["period"] = config.dataset.period
    bundle = adapter.load(config.dataset.path, **kwargs)
    validate_dataset_bundle(bundle)
    return bundle


def _enforce_unique_id_limit(bundle: DatasetBundle, max_unique_ids: int | None) -> None:
    if max_unique_ids is None:
        return
    if max_unique_ids < 1:
        raise ValueError("max_unique_ids must be at least 1")
    unique_ids = int(bundle.history[UNIQUE_ID].astype(str).nunique())
    if unique_ids > max_unique_ids:
        raise ValueError(
            f"dataset contains {unique_ids} unique_id values; maximum allowed is {max_unique_ids}"
        )


def _build_order_config(config: BackendConfig) -> OrderPolicy | None:
    if config.ordering is None:
        return None
    if config.ordering.params is None:
        raise ValueError("ordering.params is required for generic CLI ordering runs")
    params = config.ordering.params
    if isinstance(params, dict):
        params = [params]
    params_frame = pd.DataFrame(params)
    ordering = config.ordering
    if ordering.policy == "rs":
        return RsConfig(
            params=params_frame,
            coverage=ordering.coverage,
            quantile=ordering.quantile,
        )
    if ordering.policy == "rss":
        if ordering.quantile is not None:
            raise ValueError("ordering.quantile is not a valid knob for the rss policy")
        return RssConfig(params=params_frame, coverage=ordering.coverage)
    if ordering.policy == "newsvendor":
        if ordering.quantile is not None:
            raise ValueError("ordering.quantile is not a valid knob for the newsvendor policy")
        return NewsvendorConfig(params=params_frame, coverage=ordering.coverage)
    raise ValueError(f"unknown order policy: {ordering.policy!r}")


def _hierarchy_for_run(config: BackendConfig, bundle: DatasetBundle) -> pd.DataFrame | None:
    if config.hierarchical_intervals is not None:
        if bundle.hierarchy is None:
            raise ValueError("hierarchical_intervals requires a dataset hierarchy")
        return bundle.hierarchy
    if config.reconciliation is None or config.reconciliation.strategy == "none":
        return None
    return bundle.hierarchy


def _metric_currency(config: BackendConfig) -> str:
    currency = config.dataset.options.get("currency")
    return str(currency) if currency is not None else "EUR"


def _record_order_cost_metric(frame: pd.DataFrame, *, dataset: str, currency: str) -> None:
    if frame.empty:
        return
    if "total_cost" in frame.columns:
        total_cost = float(frame["total_cost"].sum())
    else:
        cost_columns = [
            column
            for column in frame.columns
            if column.endswith("_cost") and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not cost_columns:
            return
        total_cost = float(frame[cost_columns].sum(numeric_only=True).sum())
    set_order_cost(currency, dataset, total_cost)


def run(config_path: str | Path, *, metrics_port: int | None = None) -> BackendResult:
    if metrics_port is not None:
        from calibre.core.metrics import serve

        serve(metrics_port)
    config = load_config(config_path)
    return run_config(config)


def run_config(
    config: BackendConfig,
    *,
    run_id: UUID | None = None,
    conformal_state_store: ConformalStateStore | None = None,
    initial_ledger: pd.DataFrame | None = None,
    max_unique_ids: int | None = None,
) -> BackendResult:
    bundle = _load_dataset(config)
    _enforce_unique_id_limit(bundle, max_unique_ids)
    model_configs = [task.resolved_model_config() for task in config.tasks]
    horizon = config.tasks[0].horizon
    reconciliation_hierarchy = _hierarchy_for_run(config, bundle)
    actuals = build_node_history(bundle.history, reconciliation_hierarchy)
    tasks = build_tasks(actuals, model_configs, horizon)
    origins = config.origins.to_list()
    if not origins:
        raise ValueError("origins resolved to an empty list")

    conformal_config: SymmetricIntervalConfig | None = (
        config.conformal.to_runtime_config() if config.conformal is not None else None
    )
    reconciliation_config = config.reconciliation or ReconciliationConfig()
    streaming_output = config.output.ledger_path if config.output.streaming else None
    streaming_order_output = config.output.order_ledger_path if config.output.streaming else None

    engine = BackendEngine(
        execution=config.execution.to_execution_options(freq=config.origins.freq),
        output=LedgerOutputOptions(
            forecast_path=streaming_output,
            order_path=streaming_order_output,
            streaming=config.output.streaming,
        ),
        conformal=ConformalOptions(
            config=conformal_config,
            run_id=run_id,
            state_store=conformal_state_store,
            initial_ledger=initial_ledger,
        ),
        reconciliation=ReconciliationOptions(
            reconciler=(
                None
                if config.hierarchical_intervals is not None
                else reconciliation_config.to_reconciler()
            ),
            hierarchy=reconciliation_hierarchy,
        ),
        hierarchical_intervals=HierarchicalIntervalEngineOptions(
            phase=(
                config.hierarchical_intervals.to_phase()
                if config.hierarchical_intervals is not None
                else None
            )
        ),
        order=_build_order_config(config),
    )
    try:
        result = engine.execute(tasks, actuals, origins)
    finally:
        engine.close()

    if not config.output.streaming and config.output.ledger_path is not None:
        result.ledger.to_parquet(config.output.ledger_path)
    if (
        not config.output.streaming
        and result.order_ledger is not None
        and config.output.order_ledger_path is not None
    ):
        result.order_ledger.to_parquet(config.output.order_ledger_path)
    if result.order_ledger is not None:
        _record_order_cost_metric(
            result.order_ledger.to_df(),
            dataset=config.dataset.adapter,
            currency=_metric_currency(config),
        )

    ledger_rows = len(result.ledger.to_df())
    logger.info("run complete", extra={"rows": ledger_rows})
    if config.output.ledger_path is not None:
        logger.info("ledger written", extra={"ledger_path": config.output.ledger_path})
    return result


def validate(config_path: str | Path) -> BackendConfig:
    config = load_config(config_path)
    logger.info(
        "config valid",
        extra={"config_schema": config.config_schema, "tasks": len(config.tasks)},
    )
    return config


def score_m5_coverage(
    ledger_path: str | Path,
    *,
    coverage: float = 0.9,
    output_dir: str | Path | None = None,
):
    from calibre.evaluation.m5_coverage import score_resolved_ledger

    artifacts = score_resolved_ledger(
        ledger_path,
        coverage=coverage,
        output_dir=output_dir,
    )
    logger.info(
        "m5 coverage artifacts written",
        extra={
            "coverage_by_node_path": str(artifacts.coverage_by_node_path),
            "report_path": str(artifacts.report_path),
        },
    )
    return artifacts


def health() -> dict[str, Any]:
    import importlib.metadata

    try:
        version = importlib.metadata.version("calibre")
    except importlib.metadata.PackageNotFoundError:
        version = "0.1.0"
    config = load_config_from_mapping(_HEALTH_CONFIG)
    bundle = _load_dataset(config)
    payload = {
        "status": "ok",
        "version": version,
        "config_schema": config.config_schema,
        "fixture_adapter": config.dataset.adapter,
        "fixture_rows": len(bundle.history),
        "fixture_series": int(bundle.history[UNIQUE_ID].astype(str).nunique()),
    }
    return payload


def run_sweep(configs_dir: str | Path) -> list[BackendResult]:
    fs, root = open_fs(configs_dir)
    if not fs.exists(root):
        raise FileNotFoundError(f"Config directory not found: {configs_dir}")
    normalized = root.rstrip("/\\")
    configs = sorted(
        {
            *fs.glob(f"{normalized}/*.yaml"),
            *fs.glob(f"{normalized}/*.yml"),
        }
    )
    if not configs:
        raise ValueError(f"No YAML configs found under {configs_dir}")
    return [run(_fs_result_uri(fs, path)) for path in configs]
