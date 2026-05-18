from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

import fsspec  # type: ignore[import-untyped]
import pandas as pd

from calibre.cli.config import BackendConfig, load_config, load_config_from_mapping
from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.core.forecast_frame import UNIQUE_ID
from calibre.core.metrics import set_order_cost
from calibre.execution.backend import BackendEngine, BackendResult
from calibre.execution.dataset import DatasetBundle
from calibre.execution.dataset_registry import resolve_dataset_adapter
from calibre.execution.io import ensure_parent_dir, open_fs
from calibre.execution.task_builder import build_tasks
from calibre.execution.validation import validate_dataset_bundle
from calibre.ordering.policy_config import OrderPolicyConfig
from calibre.storage.state import ConformalStateStore


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _fs_result_uri(fs, path: str) -> str:
    protocol = fs.protocol
    protocols = {protocol} if isinstance(protocol, str) else set(protocol)
    if not protocols.isdisjoint({"file", "local"}):
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
    "execution": {"engine": None, "seed": 42},
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


def _build_order_config(config: BackendConfig) -> OrderPolicyConfig | None:
    if config.ordering is None:
        return None
    if config.ordering.params is None:
        raise ValueError("ordering.params is required for generic CLI ordering runs")
    params = config.ordering.params
    if isinstance(params, dict):
        params = [params]
    return OrderPolicyConfig(
        policy=config.ordering.policy,  # type: ignore[arg-type]
        params=pd.DataFrame(params),
        coverage=config.ordering.coverage,
        quantile=config.ordering.quantile,
    )


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


def _write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    ensure_parent_dir(path)
    with fsspec.open(str(path), "wb") as handle:
        frame.to_parquet(handle, index=False)


def _resolve_execution_engine(config: BackendConfig) -> Any:
    if config.execution.engine is None:
        return None
    if config.execution.engine == "dask":
        from distributed import Client, LocalCluster
        from fugue_dask import DaskExecutionEngine

        if config.execution.dask_address is not None:
            client = Client(config.execution.dask_address)
            cluster = None
        else:
            cluster = LocalCluster(processes=False, dashboard_address=None)
            client = Client(cluster)
        engine = DaskExecutionEngine(client)
        engine._calibre_dask_client = client
        engine._calibre_dask_cluster = cluster
        return engine
    if config.execution.engine == "spark":
        from fugue_spark import SparkExecutionEngine
        from pyspark.sql import SparkSession

        session_config = config.execution.spark_session
        builder = SparkSession.builder
        if master := session_config.get("master"):
            builder = builder.master(str(master))
        if app_name := session_config.get("app_name"):
            builder = builder.appName(str(app_name))
        for key, value in session_config.get("config", {}).items():
            builder = builder.config(str(key), str(value))
        return SparkExecutionEngine(builder.getOrCreate())
    raise ValueError(f"Unsupported execution engine: {config.execution.engine!r}")


def _close_execution_engine(execution_engine: Any) -> None:
    if execution_engine is None:
        return
    if hasattr(execution_engine, "_calibre_dask_client"):
        execution_engine._calibre_dask_client.close()
    if (
        hasattr(execution_engine, "_calibre_dask_cluster")
        and execution_engine._calibre_dask_cluster is not None
    ):
        execution_engine._calibre_dask_cluster.close()


def _run_builtin_benchmark(config: BackendConfig) -> pd.DataFrame:
    if config.benchmark not in {"vn2_winning", "vn2_tuned"}:
        raise ValueError(f"Unknown benchmark runner: {config.benchmark!r}")

    from benchmarks.vn2.run_benchmark import run_benchmark

    execution_engine = _resolve_execution_engine(config)
    try:
        summary = run_benchmark(
            data_dir=config.dataset.path,
            horizon=config.tasks[0].horizon,
            tune=False,
            results_dir=None,
            verbose=True,
            execution_engine=execution_engine,
        )
    finally:
        _close_execution_engine(execution_engine)
    if config.output.ledger_path is not None:
        _write_parquet(summary, config.output.ledger_path)
    return summary


def run(
    config_path: str | Path, *, metrics_port: int | None = None
) -> BackendResult | pd.DataFrame:
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
) -> BackendResult | pd.DataFrame:
    if config.benchmark is not None:
        summary = _run_builtin_benchmark(config)
        total_cost = float(summary["total_cost"].sum()) if "total_cost" in summary else float("nan")
        _record_order_cost_metric(
            summary,
            dataset=config.benchmark,
            currency=_metric_currency(config),
        )
        _emit(f"benchmark={config.benchmark} rows={len(summary)} total_cost={total_cost:.2f}")
        return summary

    bundle = _load_dataset(config)
    _enforce_unique_id_limit(bundle, max_unique_ids)
    model_configs = [task.model_config() for task in config.tasks]
    horizon = config.tasks[0].horizon
    tasks = build_tasks(bundle.history, model_configs, horizon)
    origins = config.origins.to_list()
    if not origins:
        raise ValueError("origins resolved to an empty list")

    conformal_config: SymmetricIntervalConfig | None = (
        config.conformal.to_runtime_config() if config.conformal is not None else None
    )
    streaming_output = config.output.ledger_path if config.output.streaming else None
    streaming_order_output = config.output.order_ledger_path if config.output.streaming else None

    execution_engine = _resolve_execution_engine(config)
    try:
        result = BackendEngine(
            freq=config.origins.freq,
            engine=execution_engine,
            conformal_config=conformal_config,
            order_config=_build_order_config(config),
            streaming_output=streaming_output,
            streaming_order_output=streaming_order_output,
            seed=config.execution.seed,
            run_id=run_id,
            conformal_state_store=conformal_state_store,
            initial_ledger=initial_ledger,
        ).execute(tasks, bundle.history, origins)
    finally:
        # Clean up distributed execution engines to avoid thread/connection leaks
        # in long-running processes (e.g. FastAPI server). Runs even on failure.
        _close_execution_engine(execution_engine)

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
    _emit(f"run complete rows={ledger_rows}")
    if config.output.ledger_path is not None:
        _emit(f"ledger={config.output.ledger_path}")
    return result


def validate(config_path: str | Path) -> BackendConfig:
    config = load_config(config_path)
    _emit(f"valid config_schema={config.config_schema} tasks={len(config.tasks)}")
    return config


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
    _emit(json.dumps(payload, sort_keys=True))
    return payload


def run_sweep(configs_dir: str | Path) -> list[BackendResult | pd.DataFrame]:
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
