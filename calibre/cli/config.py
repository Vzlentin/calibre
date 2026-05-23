from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import fsspec  # type: ignore[import-untyped]
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from calibre.conformal.runtime import SymmetricIntervalConfig

CONFIG_SCHEMA = "1.0"

if TYPE_CHECKING:
    from calibre.execution.backend import ExecutionOptions


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
    backend: Literal["local", "ray", "auto"] = "auto"
    seed: int | None = None
    ray_address: str | None = None
    staging_uri: str | None = None
    ray_threshold: int = 10
    max_concurrency: int | None = None
    cpu_per_task: float | None = None

    def to_execution_options(self, *, freq: str) -> ExecutionOptions:
        from calibre.execution.backend import ExecutionOptions

        return ExecutionOptions(
            freq=freq,
            backend=self.backend,
            ray_address=self.ray_address,
            staging_uri=self.staging_uri,
            ray_threshold=self.ray_threshold,
            max_concurrency=self.max_concurrency,
            cpu_per_task=self.cpu_per_task,
            seed=self.seed,
        )


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
    source_path: str | None = None


# ── pydantic input schemas (parse-time validation only) ──────────────────────


class _RawConformal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: Literal["mscp", "aci"]
    coverage: float = 0.9
    calibration_window: int = 100
    gamma: float = 0.05
    mode: Literal["perhorizon", "cumulative"] = "perhorizon"
    protection_period: int | None = None


class _RawExecution(BaseModel):
    model_config = ConfigDict(extra="allow")  # extra checked manually for compat
    backend: Literal["local", "ray", "auto"] = "auto"
    seed: int | None = None
    ray_address: str | None = None
    staging_uri: str | None = None
    ray_threshold: int = 10
    max_concurrency: int | None = None
    cpu_per_task: float | None = None

    @model_validator(mode="after")
    def _cross_field_checks(self) -> _RawExecution:
        if self.ray_address is not None and self.staging_uri is None:
            raise ValueError("execution.staging_uri is required when execution.ray_address is set")
        if self.ray_threshold < 1:
            raise ValueError("execution.ray_threshold must be at least 1")
        if self.max_concurrency is not None and self.max_concurrency < 1:
            raise ValueError("execution.max_concurrency must be at least 1")
        if self.cpu_per_task is not None and self.cpu_per_task <= 0:
            raise ValueError("execution.cpu_per_task must be positive")
        return self


# ── section parsers ─────────────────────────────────────────────────────────


def _parse_dataset(data: object) -> DatasetConfig:
    if not isinstance(data, dict):
        raise ValueError("dataset must be a mapping")
    if "adapter" not in data:
        raise ValueError("dataset missing required key: adapter")
    if "path" not in data:
        raise ValueError("dataset missing required key: path")
    options = {k: v for k, v in data.items() if k not in ("adapter", "path", "period")}
    return DatasetConfig(
        adapter=str(data["adapter"]),
        path=str(data["path"]),
        period=int(data["period"]) if data.get("period") is not None else None,
        options=options,
    )


def _parse_tasks(data: object) -> list[TaskConfig]:
    if not isinstance(data, list) or not data:
        raise ValueError("tasks must be a non-empty list")
    tasks: list[TaskConfig] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"tasks[{idx}] must be a mapping")
        if "model" not in item:
            raise ValueError(f"tasks[{idx}] missing required key: model")
        if "horizon" not in item:
            raise ValueError(f"tasks[{idx}] missing required key: horizon")
        horizon = int(item["horizon"])
        if horizon < 1:
            raise ValueError(f"tasks[{idx}].horizon must be at least 1")
        config = item.get("config", {})
        if not isinstance(config, dict):
            raise ValueError(f"tasks[{idx}].config must be a mapping")
        tasks.append(TaskConfig(model=str(item["model"]), horizon=horizon, config=dict(config)))
    return tasks


def _parse_conformal(data: object) -> ConformalConfig | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("conformal must be a mapping")
    try:
        raw = _RawConformal.model_validate(data)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return ConformalConfig(
        method=raw.method,
        coverage=raw.coverage,
        calibration_window=raw.calibration_window,
        gamma=raw.gamma,
        mode=raw.mode,
        protection_period=raw.protection_period,
    )


def _parse_ordering(data: object) -> OrderingConfig | None:
    if data is None:
        return None
    if not isinstance(data, dict):
        raise ValueError("ordering must be a mapping")
    if "policy" not in data:
        raise ValueError("ordering missing required key: policy")
    raw: dict[str, Any] = data
    return OrderingConfig(
        policy=str(raw["policy"]),
        coverage=float(raw.get("coverage", 0.9)),
        quantile=float(raw["quantile"]) if raw.get("quantile") is not None else None,
        params=raw.get("params"),
    )


def _parse_origins(data: object) -> OriginsConfig:
    if not isinstance(data, dict):
        raise ValueError("origins must be a mapping")
    if "start" not in data:
        raise ValueError("origins missing required key: start")
    if "end" not in data:
        raise ValueError("origins missing required key: end")
    if "freq" not in data:
        raise ValueError("origins missing required key: freq")
    start = pd.Timestamp(data["start"])
    end = pd.Timestamp(data["end"])
    if end < start:
        raise ValueError("origins.end must be greater than or equal to origins.start")
    return OriginsConfig(start=start, end=end, freq=str(data["freq"]))


def _parse_output(data: object) -> OutputConfig:
    raw = data if isinstance(data, dict) else {}
    return OutputConfig(
        ledger_path=str(raw["ledger_path"]) if raw.get("ledger_path") is not None else None,
        order_ledger_path=str(raw["order_ledger_path"])
        if raw.get("order_ledger_path") is not None
        else None,
        streaming=bool(raw.get("streaming", False)),
    )


def _parse_execution(data: object) -> ExecutionConfig:
    raw: dict[str, object] = data if isinstance(data, dict) else {}
    _EXECUTION_KEYS = {
        "backend",
        "seed",
        "ray_address",
        "staging_uri",
        "ray_threshold",
        "max_concurrency",
        "cpu_per_task",
    }
    unknown = sorted(set(raw) - _EXECUTION_KEYS)
    if unknown:
        raise ValueError(f"unknown execution key: {unknown[0]}")
    try:
        parsed = _RawExecution.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return ExecutionConfig(
        backend=parsed.backend,
        ray_address=parsed.ray_address,
        staging_uri=parsed.staging_uri,
        ray_threshold=parsed.ray_threshold,
        max_concurrency=parsed.max_concurrency,
        cpu_per_task=parsed.cpu_per_task,
        seed=parsed.seed,
    )


def load_config_from_mapping(
    data: dict[str, Any], *, source_path: str | Path | None = None
) -> BackendConfig:
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    schema = str(data.get("config_schema", ""))
    if schema != CONFIG_SCHEMA:
        raise ValueError(f"config_schema must be {CONFIG_SCHEMA!r}, got {schema!r}")
    if "hpo" in data:
        raise ValueError("config.hpo is not supported until CLI tuning is wired")
    if "tasks" not in data:
        raise ValueError("config missing required key: tasks")
    if "dataset" not in data:
        raise ValueError("config missing required key: dataset")
    if "origins" not in data:
        raise ValueError("config missing required key: origins")

    tasks = _parse_tasks(data["tasks"])
    horizons = {task.horizon for task in tasks}
    if len(horizons) != 1:
        raise ValueError("all tasks in a single CLI run must use the same horizon")

    config = BackendConfig(
        config_schema=schema,
        dataset=_parse_dataset(data["dataset"]),
        tasks=tasks,
        conformal=_parse_conformal(data.get("conformal")),
        ordering=_parse_ordering(data.get("ordering")),
        origins=_parse_origins(data["origins"]),
        output=_parse_output(data.get("output")),
        execution=_parse_execution(data.get("execution")),
        benchmark=str(data["benchmark"]) if data.get("benchmark") is not None else None,
        source_path=str(source_path) if source_path is not None else None,
    )
    if config.output.streaming and config.output.ledger_path is None:
        raise ValueError("output.ledger_path is required when output.streaming is true")
    return config


def load_config(path: str | Path) -> BackendConfig:
    source_path = str(path)
    with fsspec.open(source_path, "rt", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file is empty: {source_path}")
    return load_config_from_mapping(data, source_path=source_path)
