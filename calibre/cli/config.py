from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, cast

import fsspec  # type: ignore[import-untyped]
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.ordering.policy_config import OrderPolicyType

CONFIG_SCHEMA = "1.0"
ConfigMap = dict[str, object]

if TYPE_CHECKING:
    from calibre.execution.backend import ExecutionOptions


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    adapter: str
    path: str
    period: int | None = None
    options: ConfigMap = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskConfig:
    model: str
    horizon: int
    config: ConfigMap = field(default_factory=dict)

    def model_config(self) -> ConfigMap:
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
    policy: OrderPolicyType
    coverage: float = 0.9
    quantile: float | None = None
    params: list[ConfigMap] | ConfigMap | None = None


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


class _StrictSection(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class _DatasetSection(BaseModel):
    model_config = ConfigDict(extra="allow")

    adapter: str
    path: str
    period: int | None = None

    def to_config(self) -> DatasetConfig:
        return DatasetConfig(
            adapter=self.adapter,
            path=self.path,
            period=self.period,
            options=cast(ConfigMap, self.__pydantic_extra__ or {}),
        )


class _TaskSection(_StrictSection):
    model: str
    horizon: int
    config: ConfigMap = Field(default_factory=dict)

    @field_validator("horizon")
    @classmethod
    def _validate_horizon(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1")
        return value

    def to_config(self) -> TaskConfig:
        return TaskConfig(model=self.model, horizon=self.horizon, config=dict(self.config))


class _ConformalSection(_StrictSection):
    method: Literal["mscp", "aci"]
    coverage: float = 0.9
    calibration_window: int = 100
    gamma: float = 0.05
    mode: Literal["perhorizon", "cumulative"] = "perhorizon"
    protection_period: int | None = None

    def to_config(self) -> ConformalConfig:
        return ConformalConfig(
            method=self.method,
            coverage=self.coverage,
            calibration_window=self.calibration_window,
            gamma=self.gamma,
            mode=self.mode,
            protection_period=self.protection_period,
        )


class _OrderingSection(_StrictSection):
    policy: OrderPolicyType
    coverage: float = 0.9
    quantile: float | None = None
    params: list[ConfigMap] | ConfigMap | None = None

    def to_config(self) -> OrderingConfig:
        return OrderingConfig(
            policy=self.policy,
            coverage=self.coverage,
            quantile=self.quantile,
            params=self.params,
        )


class _OriginsSection(_StrictSection):
    start: pd.Timestamp
    end: pd.Timestamp
    freq: str

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce_timestamp(cls, value: object) -> pd.Timestamp:
        if isinstance(value, pd.Timestamp | datetime | date | str | int | float):
            return pd.Timestamp(value)
        raise ValueError("must be timestamp-like")

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.end < self.start:
            raise ValueError("origins.end must be greater than or equal to origins.start")
        return self

    def to_config(self) -> OriginsConfig:
        return OriginsConfig(start=self.start, end=self.end, freq=self.freq)


class _OutputSection(_StrictSection):
    ledger_path: str | None = None
    order_ledger_path: str | None = None
    streaming: bool = False

    def to_config(self) -> OutputConfig:
        return OutputConfig(
            ledger_path=self.ledger_path,
            order_ledger_path=self.order_ledger_path,
            streaming=self.streaming,
        )


class _ExecutionSection(_StrictSection):
    backend: Literal["local", "ray", "auto"] = "auto"
    seed: int | None = None
    ray_address: str | None = None
    staging_uri: str | None = None
    ray_threshold: int = 10
    max_concurrency: int | None = None
    cpu_per_task: float | None = None

    @field_validator("ray_threshold")
    @classmethod
    def _validate_ray_threshold(cls, value: int) -> int:
        if value < 1:
            raise ValueError("execution.ray_threshold must be at least 1")
        return value

    @field_validator("max_concurrency")
    @classmethod
    def _validate_max_concurrency(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("execution.max_concurrency must be at least 1")
        return value

    @field_validator("cpu_per_task")
    @classmethod
    def _validate_cpu_per_task(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("execution.cpu_per_task must be positive")
        return value

    @model_validator(mode="after")
    def _validate_staging_uri(self) -> Self:
        if self.ray_address is not None and self.staging_uri is None:
            raise ValueError("execution.staging_uri is required when execution.ray_address is set")
        return self

    def to_config(self) -> ExecutionConfig:
        return ExecutionConfig(
            backend=self.backend,
            ray_address=self.ray_address,
            staging_uri=self.staging_uri,
            ray_threshold=self.ray_threshold,
            max_concurrency=self.max_concurrency,
            cpu_per_task=self.cpu_per_task,
            seed=self.seed,
        )


class _BackendSection(_StrictSection):
    config_schema: str
    dataset: _DatasetSection
    tasks: list[_TaskSection] = Field(min_length=1)
    origins: _OriginsSection
    output: _OutputSection = Field(default_factory=_OutputSection)
    conformal: _ConformalSection | None = None
    ordering: _OrderingSection | None = None
    execution: _ExecutionSection = Field(default_factory=_ExecutionSection)
    benchmark: str | None = None

    @field_validator("config_schema")
    @classmethod
    def _validate_schema(cls, value: str) -> str:
        if value != CONFIG_SCHEMA:
            raise ValueError(f"config_schema must be {CONFIG_SCHEMA!r}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_horizons(self) -> Self:
        horizons = {task.horizon for task in self.tasks}
        if len(horizons) != 1:
            raise ValueError("all tasks in a single CLI run must use the same horizon")
        return self

    def to_config(self, source_path: str | Path | None) -> BackendConfig:
        config = BackendConfig(
            config_schema=self.config_schema,
            dataset=self.dataset.to_config(),
            tasks=[task.to_config() for task in self.tasks],
            conformal=self.conformal.to_config() if self.conformal is not None else None,
            ordering=self.ordering.to_config() if self.ordering is not None else None,
            origins=self.origins.to_config(),
            output=self.output.to_config(),
            execution=self.execution.to_config(),
            benchmark=self.benchmark,
            source_path=str(source_path) if source_path is not None else None,
        )
        if config.output.streaming and config.output.ledger_path is None:
            raise ValueError("output.ledger_path is required when output.streaming is true")
        return config


def load_config_from_mapping(
    data: ConfigMap,
    *,
    source_path: str | Path | None = None,
) -> BackendConfig:
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    if "hpo" in data:
        raise ValueError("config.hpo is not supported until CLI tuning is wired")
    try:
        return _BackendSection.model_validate(data).to_config(source_path)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc


def load_config(path: str | Path) -> BackendConfig:
    source_path = str(path)
    with fsspec.open(source_path, "rt", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file is empty: {source_path}")
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    return load_config_from_mapping(cast(ConfigMap, data), source_path=source_path)
