from __future__ import annotations

import pandas as pd

from benchmarks.vn2 import __main__ as vn2_main
from benchmarks.vn2.run_benchmark import run_from_config
from calibre.cli.config import load_config_from_mapping
from calibre.core.forecast_frame import UNIQUE_ID


def test_run_from_config_maps_execution_kwargs(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({UNIQUE_ID: ["A"], "total_cost": [1.0]})

    monkeypatch.setattr(
        "benchmarks.vn2.run_benchmark.run_benchmark",
        _fake_run_benchmark,
    )
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "vn2", "path": "data/vn2"},
            "tasks": [{"model": "global_lgbm", "horizon": 3, "config": {"backend": "mlforecast"}}],
            "origins": {"start": "2024-01-01", "end": "2024-01-01", "freq": "W-MON"},
            "output": {"streaming": False},
            "execution": {
                "backend": "auto",
                "ray_threshold": 10,
                "max_concurrency": 4,
                "cpu_per_task": 0.5,
            },
        }
    )

    run_from_config(config)

    assert captured["data_dir"] == "data/vn2"
    assert captured["horizon"] == 3
    assert captured["tune"] is False
    assert captured["execution_backend"] == "auto"
    assert captured["ray_threshold"] == 10
    assert captured["max_concurrency"] == 4
    assert captured["cpu_per_task"] == 0.5


def test_run_from_config_preserves_fsspec_dataset_uri(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_benchmark(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({UNIQUE_ID: ["A"], "total_cost": [0.0]})

    monkeypatch.setattr(
        "benchmarks.vn2.run_benchmark.run_benchmark",
        _fake_run_benchmark,
    )
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "vn2", "path": "memory://calibre-vn2-benchmark/data"},
            "tasks": [{"model": "global_lgbm", "horizon": 3, "config": {"backend": "mlforecast"}}],
            "origins": {"start": "2024-01-01", "end": "2024-01-01", "freq": "W-MON"},
            "output": {"streaming": False},
            "execution": {"backend": "local", "seed": 42},
        }
    )

    run_from_config(config)

    assert captured["data_dir"] == "memory://calibre-vn2-benchmark/data"


def test_run_from_config_writes_ledger_when_path_set(monkeypatch, tmp_path) -> None:
    ledger_path = tmp_path / "ledger.parquet"

    def _fake_run_benchmark(**_kwargs):
        return pd.DataFrame({UNIQUE_ID: ["A"], "total_cost": [2.0]})

    monkeypatch.setattr(
        "benchmarks.vn2.run_benchmark.run_benchmark",
        _fake_run_benchmark,
    )
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "vn2", "path": "data/vn2"},
            "tasks": [{"model": "global_lgbm", "horizon": 3, "config": {"backend": "mlforecast"}}],
            "origins": {"start": "2024-01-01", "end": "2024-01-01", "freq": "W-MON"},
            "output": {"ledger_path": str(ledger_path), "streaming": False},
            "execution": {"backend": "local", "seed": 42},
        }
    )

    run_from_config(config)

    assert ledger_path.exists()


def test_run_from_config_skips_ledger_when_path_unset(monkeypatch, tmp_path) -> None:
    def _fake_run_benchmark(**_kwargs):
        return pd.DataFrame({UNIQUE_ID: ["A"], "total_cost": [2.0]})

    monkeypatch.setattr(
        "benchmarks.vn2.run_benchmark.run_benchmark",
        _fake_run_benchmark,
    )
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "vn2", "path": "data/vn2"},
            "tasks": [{"model": "global_lgbm", "horizon": 3, "config": {"backend": "mlforecast"}}],
            "origins": {"start": "2024-01-01", "end": "2024-01-01", "freq": "W-MON"},
            "output": {"streaming": False},
            "execution": {"backend": "local", "seed": 42},
        }
    )

    summary = run_from_config(config)

    assert not (tmp_path / "ledger.parquet").exists()
    assert summary["total_cost"].sum() == 2.0


def test_module_main_invokes_run_from_config(monkeypatch, tmp_path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                'config_schema: "1.0"',
                "dataset:",
                '  adapter: vn2',
                f'  path: "{tmp_path / "data"}"',
                "tasks:",
                "  - model: global_lgbm",
                "    horizon: 3",
                '    config: {backend: mlforecast}',
                "origins:",
                '  start: "2024-01-01"',
                '  end: "2024-01-01"',
                '  freq: W-MON',
                "output:",
                "  streaming: false",
                "execution:",
                "  backend: local",
                "  seed: 42",
            ]
        ),
        encoding="utf-8",
    )
    seen: list[object] = []

    def _fake_run_from_config(config):
        seen.append(config)
        return pd.DataFrame({UNIQUE_ID: ["A"], "total_cost": [0.0]})

    monkeypatch.setattr("benchmarks.vn2.__main__.run_from_config", _fake_run_from_config)
    monkeypatch.setattr("benchmarks.vn2.__main__.load_dotenv", lambda: None)

    vn2_main.main(["--config", str(cfg_path)])

    assert len(seen) == 1
