"""Smoke tests for benchmarks/common/tracking.py."""

from __future__ import annotations

from uuid import uuid4

import mlflow
import optuna
import pandas as pd
import pytest

from benchmarks.common.tracking import (
    log_costs_dataframe,
    optuna_mlflow_callback,
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


def test_optuna_callback_logs_trials_in_requested_experiment():
    study_name = f"tracking_smoke_{uuid4().hex}"
    with start_benchmark_run("test_experiment", "optuna_parent") as parent:
        study = optuna.create_study(study_name=study_name, direction="minimize")
        study.optimize(
            lambda trial: 1.0,
            n_trials=1,
            callbacks=[optuna_mlflow_callback("test_experiment", metric_name="objective")],
        )
        parent_run_id = parent.info.run_id
        experiment_id = parent.info.experiment_id

    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        [experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )

    assert runs
    assert mlflow.get_experiment_by_name(study_name) is None
