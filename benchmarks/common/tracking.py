"""Shared MLflow tracking utilities for Calibre benchmarks."""

from __future__ import annotations  # keeps pd.DataFrame annotation lazy at runtime

import dataclasses
import logging
import os
import platform
import subprocess
import sys
import tempfile
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any


class _NoopMlflow:
    """Small runtime shim used when benchmark tracking extras are not installed."""

    def active_run(self) -> None:
        return None

    def __getattr__(self, name: str):
        del name

        def _noop(*args, **kwargs):
            del args, kwargs
            return None

        return _noop


try:
    import mlflow
except ModuleNotFoundError:
    mlflow = _NoopMlflow()  # type: ignore[assignment]
    _MLFLOW_AVAILABLE = False
else:
    _MLFLOW_AVAILABLE = True

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent.parent


def _mlflow_disabled() -> bool:
    """Return True when CALIBRE_NO_MLFLOW is set to a truthy value."""
    return os.environ.get("CALIBRE_NO_MLFLOW", "").lower() in {"1", "true", "yes"}


def _mlflow_unavailable() -> bool:
    if _MLFLOW_AVAILABLE:
        return False
    logger.info("MLflow is not installed; benchmark tracking is disabled.")
    return True


def _tracking_disabled() -> bool:
    return _mlflow_disabled() or _mlflow_unavailable()


def _load_dotenv() -> None:
    """Load key=value pairs from .env files into os.environ (if not already set).

    Searches: repo root .env, then the CWD .env. Values already set in the
    environment are NOT overridden. This avoids adding a python-dotenv dependency
    while still supporting ``MLFLOW_TRACKING_URI`` and other config from .env.
    """
    for dotenv_path in [_REPO_ROOT / ".env", Path.cwd() / ".env"]:
        if dotenv_path.is_file():
            with open(dotenv_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key, value = key.strip(), value.strip()
                    if key and key not in os.environ:
                        os.environ[key] = value


_load_dotenv()


def resolve_tracking_uri() -> str:
    """Return an MLflow-compatible tracking URI, honouring MLFLOW_TRACKING_URI if set.

    On Windows, bare paths like ``C:\\...\\mlruns`` are rejected by MLflow because
    the drive letter is parsed as a URI scheme. Any path-like value is converted to
    a ``file:///`` URI via Path.as_uri(); proper scheme URIs (http/databricks/…)
    are returned as-is.
    """
    raw = os.environ.get("MLFLOW_TRACKING_URI", str(_REPO_ROOT / "mlruns"))
    if "://" in raw or raw in {"databricks", "databricks-uc", "uc"}:
        return raw
    return Path(raw).as_uri()


def git_sha() -> str | None:
    """Return current HEAD SHA, or None if git is unavailable."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    sha = result.stdout.strip()
    return sha or None


def _flatten_to_str_params(key: str, value: Any) -> dict[str, str] | None:
    """Convert a config value to a flat {key: str_value} dict for mlflow.log_params.

    Returns None for complex types (list, plain dict) that cannot be meaningfully
    flattened — callers should log those as JSON artifacts instead.
    """
    if isinstance(value, Path):
        return {key[:250]: str(value)[:500]}
    if isinstance(value, (bool, int, float, str)):
        return {key[:250]: str(value)[:500]}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: dict[str, str] = {}
        for field_name, field_value in dataclasses.asdict(value).items():
            result[f"{key}.{field_name}"[:250]] = str(field_value)[:500]
        return result
    return None  # list, dict, etc. — log as artifact


@contextmanager
def start_benchmark_run(
    experiment: str,
    run_name: str,
    *,
    tags: dict[str, str] | None = None,
):
    """Open an MLflow run for a benchmark entry point.

    Sets the tracking URI (from resolve_tracking_uri), creates the experiment
    if needed, and attaches standard tags (git_sha, python, platform). Caller
    logs params/metrics inside the yielded context.
    """
    if _tracking_disabled():
        yield None
        return

    mlflow.set_tracking_uri(resolve_tracking_uri())
    mlflow.set_experiment(experiment)

    all_tags: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.system(),
    }
    sha = git_sha()
    if sha:
        all_tags["git_sha"] = sha
    if tags:
        all_tags.update(tags)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(all_tags)
        yield run


def log_config_module(mod: types.ModuleType) -> None:
    """Log uppercase constants from a config module as MLflow params/artifacts.

    Scalar values (int, float, bool, str, Path, dataclasses) are logged via
    mlflow.log_params. Non-scalar values (lists, plain dicts) are written to
    config_non_scalar.json as an artifact.
    """
    scalar_params: dict[str, str] = {}
    non_scalar: dict[str, Any] = {}

    for name in dir(mod):
        if not name.isupper():
            continue
        value = getattr(mod, name)
        flat = _flatten_to_str_params(name, value)
        if flat is not None:
            scalar_params.update(flat)
        else:
            non_scalar[name] = value

    if _tracking_disabled():
        return

    if scalar_params:
        mlflow.log_params(scalar_params)
    if non_scalar:
        try:
            mlflow.log_dict(
                {k: str(v) for k, v in non_scalar.items()},
                "config_non_scalar.json",
            )
        except Exception as exc:
            warnings.warn(f"Could not log non-scalar config: {exc}", stacklevel=2)


def safe_log_metric(key: str, value: float, step: int | None = None) -> None:
    """Log a metric via MLflow, silently skipping if CALIBRE_NO_MLFLOW is set."""
    if _tracking_disabled():
        return
    mlflow.log_metric(key, value, step=step)


def log_costs_dataframe(costs_df: pd.DataFrame, *, artifact_subdir: str = "costs") -> None:
    """Log aggregate costs as MLflow metrics and the full frame as a CSV artifact.

    Metrics are namespaced (cost/holding_total, cost/shortage_total, cost/total)
    so cross-dataset comparisons work in the MLflow UI.
    """
    if _tracking_disabled():
        return
    mlflow.log_metric("cost/holding_total", float(costs_df["holding_cost"].sum()))
    mlflow.log_metric("cost/shortage_total", float(costs_df["shortage_cost"].sum()))
    mlflow.log_metric("cost/total", float(costs_df["total_cost"].sum()))

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "per_product.csv"
        costs_df.to_csv(str(csv_path), index=False)
        mlflow.log_artifact(str(csv_path), artifact_path=artifact_subdir)


def optuna_mlflow_callback(experiment_name: str, metric_name: str = "objective") -> Any:
    """Return an optuna-integration MLflowCallback configured for nested runs.

    Passes nested=True when a parent MLflow run is already active (the benchmark
    run), so each Optuna trial appears as a child run in the UI. The Optuna
    integration creates one experiment per study by default; pin it to the
    requested experiment instead so benchmark trial runs remain discoverable.
    """
    if _tracking_disabled():
        return lambda study, trial: None

    from optuna_integration.mlflow import MLflowCallback

    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow_kwargs: dict[str, Any] = {}
    active_run = mlflow.active_run()
    if active_run is not None:
        mlflow_kwargs["nested"] = True
        mlflow_kwargs["experiment_id"] = active_run.info.experiment_id
    else:
        mlflow.set_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is not None:
            mlflow_kwargs["experiment_id"] = experiment.experiment_id

    return MLflowCallback(
        tracking_uri=tracking_uri,
        metric_name=metric_name,
        create_experiment=False,
        mlflow_kwargs=mlflow_kwargs,
    )
