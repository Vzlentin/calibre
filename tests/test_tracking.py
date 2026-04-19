"""Smoke tests for benchmarks/common/tracking.py."""
from __future__ import annotations

import mlflow
import pandas as pd
import pytest

from benchmarks.common.tracking import (
    log_costs_dataframe,
    resolve_tracking_uri,
    start_benchmark_run,
)


def _sample_costs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unique_id": ["A", "B"],
            "holding_cost": [100.0, 200.0],
            "shortage_cost": [50.0, 75.0],
            "total_cost": [150.0, 275.0],
        }
    )


def test_resolve_tracking_uri_uses_env(monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "custom"))
    uri = resolve_tracking_uri()
    # Bare Windows paths are converted to file:// URIs; verify the path is preserved.
    assert str(tmp_path / "custom").replace("\\", "/") in uri.replace("\\", "/")


def test_start_benchmark_run_creates_run():
    with start_benchmark_run("test_experiment", "test_run"):
        run = mlflow.active_run()
        assert run is not None
        assert run.info.run_name == "test_run"


def test_start_benchmark_run_sets_tags():
    with start_benchmark_run("test_experiment", "tagged_run", tags={"dataset": "vn_test"}):
        run = mlflow.active_run()
        assert run is not None
        # Fetch fresh from store; active_run().data is a snapshot from run start.
        tags = mlflow.tracking.MlflowClient().get_run(run.info.run_id).data.tags
        assert tags.get("dataset") == "vn_test"
        assert "python" in tags
        assert "platform" in tags


def test_log_costs_dataframe_metrics_and_artifact():
    costs = _sample_costs()
    with start_benchmark_run("test_experiment", "costs_run"):
        log_costs_dataframe(costs)
        run = mlflow.active_run()
        assert run is not None
        client = mlflow.tracking.MlflowClient()
        metrics = client.get_run(run.info.run_id).data.metrics
        assert metrics["cost/holding_total"] == pytest.approx(300.0)
        assert metrics["cost/shortage_total"] == pytest.approx(125.0)
        assert metrics["cost/total"] == pytest.approx(425.0)
        artifacts = client.list_artifacts(run.info.run_id, "costs")
        assert any(a.path == "costs/per_product.csv" for a in artifacts)
