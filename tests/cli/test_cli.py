from __future__ import annotations

import socket
import urllib.request
from pathlib import Path

import fsspec
import pandas as pd
import pytest
from prometheus_client import REGISTRY

from calibre.cli.commands import (
    _build_order_config,
    _record_order_cost_metric,
    health,
    run,
    run_sweep,
    validate,
)
from calibre.cli.config import load_config, load_config_from_mapping
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter
from calibre.forecasting.adapter_base import ModelAdapter
from calibre.ordering.policy_config import NewsvendorConfig, RsConfig, RssConfig


class _CliDatasetAdapter:
    def name(self) -> str:
        return "unit_cli"

    def load(self, path: str, **kwargs) -> DatasetBundle:
        del path, kwargs
        dates = pd.date_range("2024-01-07", periods=8, freq="W")
        return DatasetBundle(
            history=pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]}),
            future_x=None,
            costs=CostStruct(),
            hierarchy=None,
            censoring=None,
        )


class _StubAdapter(ModelAdapter):
    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        del collect_fitted_values
        self.task = task

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id],
                DS: [task.forecast_origin + pd.Timedelta(weeks=1)],
                Y_HAT: [10.0],
                H: [1],
            }
        )


register_dataset_adapter("unit_cli")(_CliDatasetAdapter)


def _config_text(output: str) -> str:
    return f"""
config_schema: "1.0"
dataset:
  adapter: unit_cli
  path: ignored
tasks:
  - model: stub_model
    horizon: 1
    config:
      backend: stub
origins:
  start: 2024-02-04
  end: 2024-02-04
  freq: W-SUN
output:
  ledger_path: {output}
  streaming: false
execution:
  backend: local
  seed: 123
"""


def _metrics_config_text(output: str) -> str:
    return f"""
config_schema: "1.0"
dataset:
  adapter: unit_cli
  path: ignored
tasks:
  - model: stub_model
    horizon: 1
    config:
      backend: stub
conformal:
  method: aci
  coverage: 0.9
  calibration_window: 4
  gamma: 0.05
ordering:
  policy: newsvendor
  coverage: 0.9
  params:
    - unique_id: A
      underage_cost: 3.0
      overage_cost: 1.0
      inventory_position: 0.0
origins:
  start: 2024-02-04
  end: 2024-02-11
  freq: W-SUN
output:
  ledger_path: {output}
  streaming: false
execution:
  backend: local
  seed: 123
"""


def _write_config(tmp_path, *, ledger_path: str | None = None) -> str:
    path = tmp_path / "config.yaml"
    output = ledger_path or (tmp_path / "ledger.parquet").as_posix()
    path.write_text(_config_text(output), encoding="utf-8")
    return str(path)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_load_config_and_validate_command(tmp_path) -> None:
    path = _write_config(tmp_path)
    config = load_config(path)
    assert config.config_schema == "1.0"
    assert config.tasks[0].horizon == 1
    assert validate(path).dataset.adapter == "unit_cli"


def test_health_validates_embedded_smoke_config() -> None:
    payload = health()

    assert payload["status"] == "ok"
    assert payload["config_schema"] == "1.0"
    assert payload["fixture_adapter"] == "vn2"
    assert payload["fixture_rows"] > 0
    assert payload["fixture_series"] > 0


def test_order_cost_metric_uses_total_cost_column() -> None:
    frame = pd.DataFrame({"unique_id": ["A", "B"], "total_cost": [1.25, 2.75]})

    _record_order_cost_metric(frame, dataset="unit", currency="EUR")

    assert (
        REGISTRY.get_sample_value("calibre_order_cost", {"currency": "EUR", "dataset": "unit"})
        == 4.0
    )


def _config_with_ordering(ordering: dict) -> dict:
    return {
        "config_schema": "1.0",
        "dataset": {"adapter": "unit_cli", "path": "ignored"},
        "tasks": [{"model": "stub_model", "horizon": 1, "config": {"backend": "stub"}}],
        "ordering": ordering,
        "origins": {"start": "2024-02-04", "end": "2024-02-04", "freq": "W-SUN"},
        "output": {"ledger_path": "ignored.parquet", "streaming": False},
        "execution": {"backend": "local", "seed": 123},
    }


def test_cli_yaml_rs_ordering_builds_rs_config() -> None:
    config = load_config_from_mapping(
        _config_with_ordering(
            {
                "policy": "rs",
                "coverage": 0.95,
                "params": [
                    {
                        "unique_id": "A",
                        "inventory_position": 5.0,
                        "lead_time": 1,
                        "review_period": 1,
                    }
                ],
            }
        )
    )
    order_config = _build_order_config(config)
    assert isinstance(order_config, RsConfig)
    assert order_config.coverage == 0.95
    assert order_config.quantile is None


def test_cli_yaml_rs_ordering_with_quantile_builds_rs_config() -> None:
    config = load_config_from_mapping(
        _config_with_ordering(
            {
                "policy": "rs",
                "quantile": 0.833,
                "params": [
                    {
                        "unique_id": "A",
                        "inventory_position": 5.0,
                        "lead_time": 1,
                        "review_period": 1,
                    }
                ],
            }
        )
    )
    order_config = _build_order_config(config)
    assert isinstance(order_config, RsConfig)
    assert order_config.quantile == pytest.approx(0.833)


def test_cli_yaml_rss_ordering_builds_rss_config() -> None:
    config = load_config_from_mapping(
        _config_with_ordering(
            {
                "policy": "rss",
                "params": [
                    {
                        "unique_id": "A",
                        "inventory_position": 5.0,
                        "reorder_point": 20.0,
                        "lead_time": 1,
                        "review_period": 1,
                    }
                ],
            }
        )
    )
    order_config = _build_order_config(config)
    assert isinstance(order_config, RssConfig)


def test_cli_yaml_newsvendor_ordering_builds_newsvendor_config() -> None:
    config = load_config_from_mapping(
        _config_with_ordering(
            {
                "policy": "newsvendor",
                "params": [
                    {
                        "unique_id": "A",
                        "underage_cost": 3.0,
                        "overage_cost": 1.0,
                        "inventory_position": 0.0,
                    }
                ],
            }
        )
    )
    order_config = _build_order_config(config)
    assert isinstance(order_config, NewsvendorConfig)


def test_cli_yaml_newsvendor_ordering_with_period_builds_newsvendor_config() -> None:
    config = load_config_from_mapping(
        _config_with_ordering(
            {
                "policy": "newsvendor",
                "period": 2,
                "params": [
                    {
                        "unique_id": "A",
                        "underage_cost": 3.0,
                        "overage_cost": 1.0,
                        "inventory_position": 0.0,
                    }
                ],
            }
        )
    )
    order_config = _build_order_config(config)
    assert isinstance(order_config, NewsvendorConfig)
    assert order_config.period == 2


def test_cli_yaml_newsvendor_with_quantile_is_rejected() -> None:
    config = load_config_from_mapping(
        _config_with_ordering(
            {
                "policy": "newsvendor",
                "quantile": 0.9,
                "params": [
                    {
                        "unique_id": "A",
                        "underage_cost": 3.0,
                        "overage_cost": 1.0,
                        "inventory_position": 0.0,
                    }
                ],
            }
        )
    )
    with pytest.raises(
        ValueError, match="ordering.quantile is not a valid knob for the newsvendor"
    ):
        _build_order_config(config)


def test_load_config_accepts_ray_execution_options(tmp_path) -> None:
    path = _write_config(tmp_path)
    text = Path(path).read_text(encoding="utf-8")
    Path(path).write_text(
        text.replace(
            "backend: local",
            "backend: ray\n  ray_address: ray://scheduler:10001\n"
            "  staging_uri: s3://bucket/calibre-staging\n  ray_threshold: 3\n"
            "  max_concurrency: 2\n  cpu_per_task: 1.5",
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.execution.backend == "ray"
    assert config.execution.ray_address == "ray://scheduler:10001"
    assert config.execution.staging_uri == "s3://bucket/calibre-staging"
    assert config.execution.ray_threshold == 3
    assert config.execution.max_concurrency == 2
    assert config.execution.cpu_per_task == 1.5


def test_load_config_rejects_unknown_execution_key(tmp_path) -> None:
    path = _write_config(tmp_path)
    text = Path(path).read_text(encoding="utf-8")
    Path(path).write_text(
        text.replace("backend: local", "backend: local\n  unknown_scheduler: true"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown execution key: unknown_scheduler"):
        load_config(path)


def test_load_config_requires_staging_uri_for_remote_ray(tmp_path) -> None:
    path = _write_config(tmp_path)
    text = Path(path).read_text(encoding="utf-8")
    Path(path).write_text(
        text.replace("backend: local", "backend: ray\n  ray_address: ray://scheduler:10001"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="execution.staging_uri is required"):
        load_config(path)


def test_load_config_rejects_hpo_section_until_cli_tuning_is_wired(tmp_path) -> None:
    path = _write_config(tmp_path)
    text = Path(path).read_text(encoding="utf-8")
    Path(path).write_text(
        text.replace(
            "execution:\n  backend: local",
            "hpo:\n  tune_experiment_dir: results/tune\nexecution:\n  backend: local",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config.hpo is not supported"):
        load_config(path)


def test_load_config_reads_fsspec_uri() -> None:
    uri = "memory://calibre-cli-config-load/config.yaml"
    ledger_uri = "memory://calibre-cli-config-load/output/ledger.parquet"
    fs = fsspec.filesystem("memory")
    if fs.exists("/calibre-cli-config-load"):
        fs.rm("/calibre-cli-config-load", recursive=True)
    with fsspec.open(uri, "wt", encoding="utf-8") as fh:
        fh.write(_config_text(ledger_uri))

    config = load_config(uri)

    assert config.source_path == uri
    assert config.output.ledger_path == ledger_uri


def test_winning_config_uses_auto_backend() -> None:
    config = load_config("benchmarks/vn2/config/winning.yaml")

    assert config.execution.backend == "auto"
    assert config.tasks[0].config["scope"] == "global"
    assert config.execution.ray_threshold == 10


def test_run_command_executes_config(monkeypatch, tmp_path) -> None:
    path = _write_config(tmp_path)
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

    result = run(path)

    frame = result.ledger.to_df()
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row[UNIQUE_ID] == "A"
    assert int(row[H]) == 1
    assert row[Y_HAT] == 10.0
    assert (tmp_path / "ledger.parquet").exists()


def test_run_command_metrics_port_exposes_required_series(monkeypatch, tmp_path) -> None:
    path = tmp_path / "metrics-config.yaml"
    path.write_text(
        _metrics_config_text((tmp_path / "metrics-ledger.parquet").as_posix()),
        encoding="utf-8",
    )
    port = _free_port()
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

    run(path, metrics_port=port)

    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=3) as response:
        payload = response.read().decode("utf-8")
    assert "calibre_forecast_duration_seconds" in payload
    assert "calibre_conformal_coverage_ratio" in payload
    assert "calibre_order_cost" in payload


def test_run_command_writes_non_streaming_output_to_fsspec_uri(monkeypatch, tmp_path) -> None:
    uri = "memory://calibre-cli-test/output/ledger.parquet"
    fs = fsspec.filesystem("memory")
    if fs.exists("calibre-cli-test"):
        fs.rm("calibre-cli-test", recursive=True)
    path = _write_config(tmp_path, ledger_path=uri)
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

    run(path)

    frame = pd.read_parquet(uri)
    assert len(frame) == 1
    assert frame.iloc[0][Y_HAT] == 10.0


def test_run_sweep_reads_fsspec_config_dir(monkeypatch) -> None:
    root_uri = "memory://calibre-cli-sweep/configs"
    config_uri = f"{root_uri}/config.yaml"
    ledger_uri = "memory://calibre-cli-sweep/output/ledger.parquet"
    fs = fsspec.filesystem("memory")
    if fs.exists("/calibre-cli-sweep"):
        fs.rm("/calibre-cli-sweep", recursive=True)
    fs.mkdirs("/calibre-cli-sweep/configs", exist_ok=True)
    with fsspec.open(config_uri, "wt", encoding="utf-8") as fh:
        fh.write(_config_text(ledger_uri))
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.prediction.resolve_adapter", lambda _: _StubAdapter())

    results = run_sweep(root_uri)

    assert len(results) == 1
    frame = pd.read_parquet(ledger_uri)
    assert len(frame) == 1
    assert frame.iloc[0][Y_HAT] == 10.0
