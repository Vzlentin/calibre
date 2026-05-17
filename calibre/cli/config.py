from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml

from calibre.conformal.runtime import SymmetricIntervalConfig

CONFIG_SCHEMA = "1.0"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    adapter: str
    path: str
    period: int | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskConfig:
    model: str
    horizon: int
    config: dict[str, Any] = field(default_factory=dict)

    def model_config(self) -> dict[str, Any]:
        resolved = dict(self.config)
        resolved.setdefault("model", self.model)
        resolved.setdefault("name", self.model)
        return resolved


@dataclass(frozen=True, slots=True)
class ConformalConfig:
    method: Literal["mscp", "aci"]
    coverage: float = 0.9
    calibration_window: int = 100
    gamma: float = 0.05
    mode: Literal["perhorizon", "cumulative"] = "perhorizon"
    protection_period: int | None = None

    def to_runtime_config(self) -> SymmetricIntervalConfig:
        return SymmetricIntervalConfig(
            method=self.method,
            coverage=self.coverage,
            calibration_window=self.calibration_window,
            gamma=self.gamma,
            mode=self.mode,
            protection_period=self.protection_period,
        )


@dataclass(frozen=True, slots=True)
class OrderingConfig:
    policy: str
    coverage: float = 0.9
    quantile: float | None = None
    params: list[dict[str, Any]] | dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class OriginsConfig:
    start: pd.Timestamp
    end: pd.Timestamp
    freq: str

    def to_list(self) -> list[pd.Timestamp]:
        return [
            pd.Timestamp(value) for value in pd.date_range(self.start, self.end, freq=self.freq)
        ]


@dataclass(frozen=True, slots=True)
class OutputConfig:
    ledger_path: str | None = None
    order_ledger_path: str | None = None
    streaming: bool = False


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    engine: str | None = None
    seed: int | None = None
    dask_address: str | None = None
    spark_session: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BackendConfig:
    config_schema: str
    dataset: DatasetConfig
    tasks: list[TaskConfig]
    origins: OriginsConfig
    output: OutputConfig
    conformal: ConformalConfig | None = None
    ordering: OrderingConfig | None = None
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    benchmark: str | None = None
    source_path: Path | None = None


def _require_mapping(data: Any, section: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{section} must be a mapping")
    return data


def _optional_mapping(data: Any, section: str) -> dict[str, Any] | None:
    if data is None:
        return None
    return _require_mapping(data, section)


def _require_key(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{section} missing required key: {key}")
    return mapping[key]


def _parse_dataset(data: Any) -> DatasetConfig:
    raw = _require_mapping(data, "dataset")
    options = dict(raw)
    adapter = str(_require_key(options, "adapter", "dataset"))
    path = str(_require_key(options, "path", "dataset"))
    period = options.pop("period", None)
    options.pop("adapter", None)
    options.pop("path", None)
    return DatasetConfig(
        adapter=adapter,
        path=path,
        period=int(period) if period is not None else None,
        options=options,
    )


def _parse_tasks(data: Any) -> list[TaskConfig]:
    if not isinstance(data, list) or not data:
        raise ValueError("tasks must be a non-empty list")
    tasks: list[TaskConfig] = []
    for idx, item in enumerate(data):
        raw = _require_mapping(item, f"tasks[{idx}]")
        model = str(_require_key(raw, "model", f"tasks[{idx}]"))
        horizon = int(_require_key(raw, "horizon", f"tasks[{idx}]"))
        if horizon < 1:
            raise ValueError(f"tasks[{idx}].horizon must be at least 1")
        config = raw.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"tasks[{idx}].config must be a mapping")
        tasks.append(TaskConfig(model=model, horizon=horizon, config=dict(config)))
    return tasks


def _parse_conformal(data: Any) -> ConformalConfig | None:
    raw = _optional_mapping(data, "conformal")
    if raw is None:
        return None
    return ConformalConfig(
        method=str(_require_key(raw, "method", "conformal")),  # type: ignore[arg-type]
        coverage=float(raw.get("coverage", 0.9)),
        calibration_window=int(raw.get("calibration_window", 100)),
        gamma=float(raw.get("gamma", 0.05)),
        mode=str(raw.get("mode", "perhorizon")),  # type: ignore[arg-type]
        protection_period=int(raw["protection_period"])
        if raw.get("protection_period") is not None
        else None,
    )


def _parse_ordering(data: Any) -> OrderingConfig | None:
    raw = _optional_mapping(data, "ordering")
    if raw is None:
        return None
    return OrderingConfig(
        policy=str(_require_key(raw, "policy", "ordering")),
        coverage=float(raw.get("coverage", 0.9)),
        quantile=float(raw["quantile"]) if raw.get("quantile") is not None else None,
        params=raw.get("params"),
    )


def _parse_origins(data: Any) -> OriginsConfig:
    raw = _require_mapping(data, "origins")
    start = pd.Timestamp(_require_key(raw, "start", "origins"))
    end = pd.Timestamp(_require_key(raw, "end", "origins"))
    if end < start:
        raise ValueError("origins.end must be greater than or equal to origins.start")
    return OriginsConfig(start=start, end=end, freq=str(_require_key(raw, "freq", "origins")))


def _parse_output(data: Any) -> OutputConfig:
    raw = _require_mapping(data or {}, "output")
    return OutputConfig(
        ledger_path=str(raw["ledger_path"]) if raw.get("ledger_path") is not None else None,
        order_ledger_path=str(raw["order_ledger_path"])
        if raw.get("order_ledger_path") is not None
        else None,
        streaming=bool(raw.get("streaming", False)),
    )


def _parse_execution(data: Any) -> ExecutionConfig:
    raw = _require_mapping(data or {}, "execution")
    engine = raw.get("engine")
    if engine not in (None, "dask", "spark"):
        raise ValueError("execution.engine must be null, 'dask', or 'spark'")
    return ExecutionConfig(
        engine=engine,
        dask_address=str(raw["dask_address"]) if raw.get("dask_address") is not None else None,
        spark_session=dict(raw.get("spark_session", {})),
        seed=int(raw["seed"]) if raw.get("seed") is not None else None,
    )


def load_config_from_mapping(
    data: dict[str, Any], *, source_path: Path | None = None
) -> BackendConfig:
    raw = _require_mapping(data, "config")
    schema = str(_require_key(raw, "config_schema", "config"))
    if schema != CONFIG_SCHEMA:
        raise ValueError(f"config_schema must be {CONFIG_SCHEMA!r}, got {schema!r}")

    tasks = _parse_tasks(_require_key(raw, "tasks", "config"))
    horizons = {task.horizon for task in tasks}
    if len(horizons) != 1:
        raise ValueError("all tasks in a single CLI run must use the same horizon")

    config = BackendConfig(
        config_schema=schema,
        dataset=_parse_dataset(_require_key(raw, "dataset", "config")),
        tasks=tasks,
        conformal=_parse_conformal(raw.get("conformal")),
        ordering=_parse_ordering(raw.get("ordering")),
        origins=_parse_origins(_require_key(raw, "origins", "config")),
        output=_parse_output(raw.get("output")),
        execution=_parse_execution(raw.get("execution")),
        benchmark=str(raw["benchmark"]) if raw.get("benchmark") is not None else None,
        source_path=source_path,
    )
    if config.output.streaming and config.output.ledger_path is None:
        raise ValueError("output.ledger_path is required when output.streaming is true")
    return config


def load_config(path: str | Path) -> BackendConfig:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file is empty: {source_path}")
    return load_config_from_mapping(data, source_path=source_path)
