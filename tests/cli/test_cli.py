from __future__ import annotations

from pathlib import Path

import fsspec
import pandas as pd

from calibre.cli.commands import _record_order_cost_metric, health, run, validate
from calibre.cli.config import load_config
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.core.metrics import order_cost
from calibre.core.order_types import CostStruct
from calibre.execution.dataset import DatasetBundle
from calibre.execution.dataset_registry import register_dataset_adapter


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


class _StubAdapter:
    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}

    def fit(self, task: ForecastTask) -> None:
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


def _write_config(tmp_path, *, ledger_path: str | None = None) -> str:
    path = tmp_path / "config.yaml"
    output = ledger_path or (tmp_path / "ledger.parquet").as_posix()
    path.write_text(
        f"""
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
  engine: null
  seed: 123
""",
        encoding="utf-8",
    )
    return str(path)


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

    assert order_cost.labels(currency="EUR", dataset="unit")._value.get() == 4.0


def test_load_config_accepts_dask_execution_options(tmp_path) -> None:
    path = _write_config(tmp_path)
    text = Path(path).read_text(encoding="utf-8")
    Path(path).write_text(
        text.replace("engine: null", "engine: dask\n  dask_address: tcp://scheduler:8786"),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.execution.engine == "dask"
    assert config.execution.dask_address == "tcp://scheduler:8786"


def test_winning_dask_config_uses_dask_engine() -> None:
    config = load_config("benchmarks/vn2/config/winning_dask.yaml")

    assert config.execution.engine == "dask"
    assert config.tasks[0].config["scope"] == "global"
    assert "lag_transforms" in config.tasks[0].config


def test_run_command_executes_config(monkeypatch, tmp_path) -> None:
    path = _write_config(tmp_path)
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    result = run(path)

    frame = result.ledger.to_df()
    assert len(frame) == 1
    assert (tmp_path / "ledger.parquet").exists()


def test_run_command_writes_non_streaming_output_to_fsspec_uri(monkeypatch, tmp_path) -> None:
    uri = "memory://calibre-cli-test/output/ledger.parquet"
    fs = fsspec.filesystem("memory")
    if fs.exists("calibre-cli-test"):
        fs.rm("calibre-cli-test", recursive=True)
    path = _write_config(tmp_path, ledger_path=uri)
    monkeypatch.setattr("calibre.execution.task_builder.get_adapter_cls", lambda _: _StubAdapter)
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    run(path)

    frame = pd.read_parquet(uri)
    assert len(frame) == 1
