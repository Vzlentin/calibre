from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from calibre.cli.config import BackendConfig, load_config
from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.execution.backend import BackendEngine, BackendResult
from calibre.execution.dataset_registry import resolve_dataset_adapter
from calibre.execution.task_builder import build_tasks
from calibre.execution.validation import validate_dataset_bundle
from calibre.ordering.policy_config import OrderPolicyConfig


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _load_dataset(config: BackendConfig):
    adapter = resolve_dataset_adapter(config.dataset.adapter)
    kwargs = dict(config.dataset.options)
    if config.dataset.period is not None:
        kwargs["period"] = config.dataset.period
    bundle = adapter.load(config.dataset.path, **kwargs)
    validate_dataset_bundle(bundle)
    return bundle


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


def _run_builtin_benchmark(config: BackendConfig) -> pd.DataFrame:
    if config.benchmark not in {"vn2_winning", "vn2_tuned"}:
        raise ValueError(f"Unknown benchmark runner: {config.benchmark!r}")

    from benchmarks.vn2.run_benchmark import run_benchmark

    summary = run_benchmark(
        data_dir=Path(config.dataset.path),
        horizon=config.tasks[0].horizon,
        tune=False,
        results_dir=None,
        verbose=True,
    )
    if config.output.ledger_path is not None:
        Path(config.output.ledger_path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_parquet(config.output.ledger_path, index=False)
    return summary


def run(
    config_path: str | Path, *, metrics_port: int | None = None
) -> BackendResult | pd.DataFrame:
    if metrics_port is not None:
        from calibre.core.metrics import serve

        serve(metrics_port)
    config = load_config(config_path)
    return run_config(config)


def run_config(config: BackendConfig) -> BackendResult | pd.DataFrame:
    if config.benchmark is not None:
        summary = _run_builtin_benchmark(config)
        total_cost = float(summary["total_cost"].sum()) if "total_cost" in summary else float("nan")
        _emit(f"benchmark={config.benchmark} rows={len(summary)} total_cost={total_cost:.2f}")
        return summary

    bundle = _load_dataset(config)
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
        ).execute(tasks, bundle.history, origins)
    finally:
        # Clean up distributed execution engines to avoid thread/connection leaks
        # in long-running processes (e.g. FastAPI server). Runs even on failure.
        if execution_engine is not None:
            if hasattr(execution_engine, "_calibre_dask_client"):
                execution_engine._calibre_dask_client.close()
            if (
                hasattr(execution_engine, "_calibre_dask_cluster")
                and execution_engine._calibre_dask_cluster is not None
            ):
                execution_engine._calibre_dask_cluster.close()

    if not config.output.streaming and config.output.ledger_path is not None:
        Path(config.output.ledger_path).parent.mkdir(parents=True, exist_ok=True)
        result.ledger.to_parquet(config.output.ledger_path)
    if (
        not config.output.streaming
        and result.order_ledger is not None
        and config.output.order_ledger_path is not None
    ):
        Path(config.output.order_ledger_path).parent.mkdir(parents=True, exist_ok=True)
        result.order_ledger.to_parquet(config.output.order_ledger_path)

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
    payload = {"status": "ok", "version": version}
    _emit(json.dumps(payload, sort_keys=True))
    return payload


def run_sweep(configs_dir: str | Path) -> list[BackendResult | pd.DataFrame]:
    root = Path(configs_dir)
    if not root.exists():
        raise FileNotFoundError(f"Config directory not found: {root}")
    configs = sorted([*root.glob("*.yaml"), *root.glob("*.yml")])
    if not configs:
        raise ValueError(f"No YAML configs found under {root}")
    return [run(path) for path in configs]
