from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import fsspec
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.reconciliation import (
    HierarchicalIntervalOptions,
    NixtlaHierarchicalIntervalPhase,
    Reconciler,
    resolve_reconciler,
)

CONFIG_SCHEMA = "1.0"

if TYPE_CHECKING:
    from calibre.execution.backend import ExecutionOptions


class _Section(BaseModel):
    """Base for frozen config sections that reject unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetConfig(_Section):
    adapter: str
    path: str
    period: int | None = None
    options: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_options(cls, data: Any) -> Any:
        # The dataset section is a flat mapping: ``adapter``, ``path`` and
        # ``period`` are reserved; every other key is adapter-specific and
        # collected into ``options``.
        if not isinstance(data, dict):
            return data
        reserved = {"adapter", "path", "period"}
        structured = {key: data[key] for key in reserved if key in data}
        structured["options"] = {key: value for key, value in data.items() if key not in reserved}
        return structured


class TaskConfig(_Section):
    model: str
    horizon: int = Field(ge=1)
    config: dict[str, Any] = Field(default_factory=dict)

    def resolved_model_config(self) -> dict[str, Any]:
        resolved = dict(self.config)
        resolved.setdefault("model", self.model)
        resolved.setdefault("name", self.model)
        return resolved


class ConformalConfig(_Section):
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


class ReconciliationConfig(_Section):
    """Point-forecast reconciliation strategy knob (defaults to a no-op).

    An absent ``reconciliation`` section is equivalent to ``strategy: none``, so
    existing flat-panel runs are unaffected.
    """

    strategy: Literal[
        "none",
        "bottom_up",
        "ols",
        "wls_struct",
        "mint_shrink",
        "wls_var",
        "erm",
    ] = "none"

    def to_reconciler(self) -> Reconciler:
        return resolve_reconciler(self.strategy)


class HierarchicalIntervalConfig(_Section):
    """Fused hierarchy + marginal interval path, off by default."""

    method: Literal["nixtla_conformal"]
    coverage: float = Field(default=0.9, gt=0.0, lt=1.0)
    strategy: Literal[
        "bottom_up",
        "ols",
        "wls_struct",
        "mint_shrink",
        "wls_var",
        "erm",
    ] = "bottom_up"
    seed: int = 0

    def to_phase(self) -> NixtlaHierarchicalIntervalPhase:
        return NixtlaHierarchicalIntervalPhase(
            HierarchicalIntervalOptions(
                method=self.method,
                coverage=self.coverage,
                strategy=self.strategy,
                seed=self.seed,
            )
        )


class OrderingConfig(_Section):
    policy: str
    coverage: float = 0.9
    quantile: float | None = None
    params: list[dict[str, Any]] | dict[str, Any] | None = None


class OriginsConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    start: pd.Timestamp
    end: pd.Timestamp
    freq: str

    @field_validator("start", "end", mode="before")
    @classmethod
    def _coerce_timestamp(cls, value: Any) -> pd.Timestamp:
        return pd.Timestamp(value)

    @model_validator(mode="after")
    def _check_order(self) -> OriginsConfig:
        if self.end < self.start:
            raise ValueError("origins.end must be greater than or equal to origins.start")
        return self

    def to_list(self) -> list[pd.Timestamp]:
        return [
            pd.Timestamp(value) for value in pd.date_range(self.start, self.end, freq=self.freq)
        ]


class OutputConfig(_Section):
    ledger_path: str | None = None
    order_ledger_path: str | None = None
    streaming: bool = False

    @model_validator(mode="after")
    def _check_streaming(self) -> OutputConfig:
        if self.streaming and self.ledger_path is None:
            raise ValueError("output.ledger_path is required when output.streaming is true")
        return self


class ExecutionConfig(_Section):
    backend: Literal["local", "ray", "auto"] = "auto"
    seed: int | None = None
    ray_address: str | None = None
    staging_uri: str | None = None
    ray_threshold: int = Field(default=10, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)
    cpu_per_task: float | None = Field(default=None, gt=0)

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        # Preserve the legacy single-key error message that callers rely on.
        if isinstance(data, dict):
            unknown = sorted(set(data) - set(cls.model_fields))
            if unknown:
                raise ValueError(f"unknown execution key: {unknown[0]}")
        return data

    @model_validator(mode="after")
    def _require_staging_uri(self) -> ExecutionConfig:
        if self.ray_address is not None and self.staging_uri is None:
            raise ValueError("execution.staging_uri is required when execution.ray_address is set")
        return self

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


class BackendConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    config_schema: str
    dataset: DatasetConfig
    tasks: list[TaskConfig] = Field(min_length=1)
    origins: OriginsConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    conformal: ConformalConfig | None = None
    reconciliation: ReconciliationConfig | None = None
    hierarchical_intervals: HierarchicalIntervalConfig | None = None
    ordering: OrderingConfig | None = None
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    source_path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_hpo(cls, data: Any) -> Any:
        if isinstance(data, dict) and "hpo" in data:
            raise ValueError("config.hpo is not supported until CLI tuning is wired")
        return data

    @field_validator("config_schema")
    @classmethod
    def _check_schema(cls, value: str) -> str:
        if value != CONFIG_SCHEMA:
            raise ValueError(f"config_schema must be {CONFIG_SCHEMA!r}, got {value!r}")
        return value

    @model_validator(mode="after")
    def _single_horizon(self) -> BackendConfig:
        if len({task.horizon for task in self.tasks}) != 1:
            raise ValueError("all tasks in a single CLI run must use the same horizon")
        return self

    @model_validator(mode="after")
    def _hierarchical_intervals_are_exclusive(self) -> BackendConfig:
        if self.hierarchical_intervals is None:
            return self
        if self.conformal is not None:
            raise ValueError("hierarchical_intervals cannot be combined with conformal")
        if self.reconciliation is not None and self.reconciliation.strategy != "none":
            raise ValueError(
                "hierarchical_intervals cannot be combined with non-none reconciliation"
            )
        return self


def load_config_from_mapping(
    data: dict[str, Any], *, source_path: str | Path | None = None
) -> BackendConfig:
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    payload = dict(data)
    if source_path is not None:
        payload["source_path"] = str(source_path)
    return BackendConfig.model_validate(payload)


def load_config(path: str | Path) -> BackendConfig:
    source_path = str(path)
    with fsspec.open(source_path, "rt", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file is empty: {source_path}")
    return load_config_from_mapping(data, source_path=source_path)
