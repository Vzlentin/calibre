"""Typed YAML config schema (:class:`BackendConfig`) and its loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import fsspec
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig
from calibre.conformal.partitions import global_partition, series_partition
from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.reconciliation import Reconciler, resolve_reconciler

CONFIG_SCHEMA = "1.0"

if TYPE_CHECKING:
    from calibre.execution.backend import ExecutionOptions


class _Section(BaseModel):
    """Base for frozen config sections that reject unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class DatasetConfig(_Section):
    """Dataset section: adapter, path, period, and adapter-specific options."""

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
    """A single forecast task: model name, horizon, and model config."""

    model: str
    horizon: int = Field(ge=1)
    config: dict[str, Any] = Field(default_factory=dict)

    def resolved_model_config(self) -> dict[str, Any]:
        resolved = dict(self.config)
        resolved.setdefault("model", self.model)
        resolved.setdefault("name", self.model)
        return resolved


_PARTITION_MAP = {"global": global_partition, "series": series_partition}


class ConformalConfig(_Section):
    """Conformal calibration knobs, convertible to the runtime config."""

    method: Literal["mscp", "aci"]
    coverage: float = 0.9
    calibration_window: int = 100
    gamma: float = 0.05
    mode: Literal["perhorizon", "cumulative"] = "perhorizon"
    partition: Literal["global", "series"] = "global"
    protection_period: int | None = None
    max_partitions: int | None = Field(default=None, ge=1)
    spread: Literal["analytic", "coherent_draws"] = "analytic"
    draw_count: int = Field(default=200, ge=1)
    draw_seed: int = 0

    def to_runtime_config(self) -> SymmetricIntervalConfig:
        partition_key = _PARTITION_MAP[self.partition]
        return SymmetricIntervalConfig(
            method=self.method,
            coverage=self.coverage,
            calibration_window=self.calibration_window,
            gamma=self.gamma,
            partition_key=partition_key,
            mode=self.mode,
            protection_period=self.protection_period,
            spread=self.spread,
            draw_count=self.draw_count,
            draw_seed=self.draw_seed,
        )


class OrderConformalConfig(_Section):
    """One-sided order-conformal decision knobs, convertible to the runtime config.

    Sibling to :class:`ConformalConfig` (the two-sided diagnostic): this block
    selects the cumulative-risk *decision* runtime the ordering policy consumes.
    ``to_runtime_config`` returns a :class:`CumulativeConformalRiskConfig`; the
    engine builds a ``CumulativeRiskRuntime`` from it and emits the upper bound
    into the ``hi_<coverage>`` column (e.g. ``coverage=0.74`` -> ``hi_0p74``).
    ``coverage`` is the CRC risk-control level (the residual quantile), not a
    cost fractile — cost-shaping lives upstream of this block.
    """

    coverage: float = Field(default=0.5, gt=0.0, lt=1.0)
    protection_period: int = Field(default=3, ge=1)
    calibration_window: int = Field(default=5000, ge=1)
    partition: Literal["global", "series"] = "global"
    base_column: str | None = None
    buffer_max: float | None = None

    def to_runtime_config(self) -> CumulativeConformalRiskConfig:
        partition_key = _PARTITION_MAP[self.partition]
        return CumulativeConformalRiskConfig(
            coverage=self.coverage,
            protection_period=self.protection_period,
            calibration_window=self.calibration_window,
            partition_key=partition_key,
            base_column=self.base_column,
            buffer_max=self.buffer_max,
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


class OrderingConfig(_Section):
    """Ordering section: policy choice, coverage, and policy parameters."""

    policy: str
    coverage: float = 0.9
    quantile: float | None = None
    # None means "unset": the domain default lives on NewsvendorConfig.period.
    period: int | None = None
    params: list[dict[str, Any]] | dict[str, Any] | None = None


class OriginsConfig(BaseModel):
    """Backtest origin range (``start``/``end``/``freq``) as a timestamp list."""

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
    """Output section: ledger paths and streaming-write toggle."""

    ledger_path: str | None = None
    order_ledger_path: str | None = None
    streaming: bool = False

    @model_validator(mode="after")
    def _check_streaming(self) -> OutputConfig:
        if self.streaming and self.ledger_path is None:
            raise ValueError("output.ledger_path is required when output.streaming is true")
        return self


class ExecutionConfig(_Section):
    """Execution backend section, convertible to :class:`ExecutionOptions`."""

    backend: Literal["local", "ray", "auto"] = "auto"
    seed: int | None = None
    ray_address: str | None = None
    staging_uri: str | None = None
    ray_threshold: int = Field(default=10, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)
    cpu_per_task: float | None = Field(default=None, gt=0)
    chunk_size: int = Field(default=256, ge=1)

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
            chunk_size=self.chunk_size,
            seed=self.seed,
        )


_HPO_SPEC_TYPES = frozenset({"categorical", "int", "float"})

# Search-space keys that name a derived or deployment-decision number rather than
# a sampled hyperparameter. The objective ``tau`` is the newsvendor cost fractile
# (derived from the dataset cost struct, overridable only via ``cost_fractile``),
# and ``coverage`` is the orthogonal CRC decision level on ``order_conformal`` —
# neither is a search dimension, so a key targeting them is a config defect.
_HPO_FORBIDDEN_SEARCH_KEYS = frozenset(
    {"coverage", "order_conformal.coverage", "cost_fractile", "tau"}
)


class HpoConfig(_Section):
    """Hyperparameter-search block consumed only by ``calibre run --tune``.

    Present-but-unused on a plain run (validated, never executed). ``budget``
    becomes the study's trial count and ``search_space`` is the declarative
    spec :func:`calibre.tuning.suggest_from_spec` samples each trial. The
    objective cost fractile is derived from the dataset cost struct, overridable
    only via ``cost_fractile`` — it is never a ``search_space`` dimension.
    """

    budget: int = Field(..., ge=1)
    seed: int | None = None
    search_space: dict[str, dict[str, Any]]
    cost_fractile: float | None = Field(default=None, gt=0.0, lt=1.0)
    asha_grace_period: int = Field(default=1, ge=1)

    @field_validator("search_space")
    @classmethod
    def _validate_search_space(cls, value: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        # Defence-in-depth: ``tau`` is un-leakable by construction (sourced from
        # cost_fractile/critical_ratio, never read out of search_space), but a
        # key named for the fractile or the orthogonal decision coverage signals
        # a misconception, so reject it loud. This is a flat name-set check — no
        # alias resolution; the dotted ``order_conformal.coverage`` entry only
        # bites if dotted-path keys are ever introduced.
        forbidden = sorted(set(value) & _HPO_FORBIDDEN_SEARCH_KEYS)
        if forbidden:
            raise ValueError(
                f"hpo.search_space may not target {forbidden[0]!r}: the objective "
                "cost fractile is derived from the dataset cost struct (override via "
                "hpo.cost_fractile) and order_conformal.coverage is the orthogonal "
                "decision level — neither is a search dimension"
            )
        for name, spec in value.items():
            kind = spec.get("type")
            if kind not in _HPO_SPEC_TYPES:
                raise ValueError(
                    f"hpo.search_space[{name!r}].type must be one of "
                    f"{sorted(_HPO_SPEC_TYPES)}, got {kind!r}"
                )
        return value


class BackendConfig(BaseModel):
    """Top-level backtest config validated from a YAML mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_schema: str
    dataset: DatasetConfig
    tasks: list[TaskConfig] = Field(min_length=1)
    origins: OriginsConfig
    output: OutputConfig = Field(default_factory=OutputConfig)
    conformal: ConformalConfig | None = None
    order_conformal: OrderConformalConfig | None = None
    reconciliation: ReconciliationConfig | None = None
    ordering: OrderingConfig | None = None
    hpo: HpoConfig | None = None
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    source_path: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _inherit_ordering_coverage(cls, data: Any) -> Any:
        # The R,S policy reads the hi_<coverage> decision bound the order-conformal
        # runtime writes, so the two coverages must be one value. Back-fill an
        # *omitted* ordering.coverage from order_conformal's *effective* coverage —
        # its explicit value, or its own default when it too is omitted — here, on
        # the raw mapping, because key presence ("unset" vs "set to its default") is
        # only visible before construction. The two block defaults differ (ordering
        # 0.9 vs order_conformal 0.5), so without inheriting order_conformal's
        # default an omit-both config would collide them. An explicitly-set,
        # mismatched value is left for the mode="after" backstop.
        if not isinstance(data, dict):
            return data
        order_conformal = data.get("order_conformal")
        ordering = data.get("ordering")
        if not isinstance(order_conformal, dict) or not isinstance(ordering, dict):
            return data
        if (
            ordering.get("policy") == "rs"
            and "coverage" not in ordering
            and ordering.get("quantile") is None
        ):
            ordering = dict(ordering)
            ordering["coverage"] = order_conformal.get(
                "coverage", OrderConformalConfig.model_fields["coverage"].default
            )
            data = dict(data)
            data["ordering"] = ordering
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
    def _decision_xor_diagnostic_conformal(self) -> BackendConfig:
        # The engine holds one conformal runtime slot (ConformalOptions), so the
        # decision runtime (order_conformal) and the diagnostic band (conformal)
        # cannot both reach it in one pass. Running both needs a second runtime
        # channel on the engine — out of scope; reject the combo at parse time.
        if self.conformal is not None and self.order_conformal is not None:
            raise ValueError(
                "conformal (diagnostic band) and order_conformal (decision bound) "
                "cannot both be configured: the engine has one runtime slot, so "
                "select one per run"
            )
        return self

    @model_validator(mode="after")
    def _ordering_coverage_matches_order_conformal(self) -> BackendConfig:
        # Backstop for the mode="before" back-fill: an *explicitly-set* ordering
        # coverage that disagrees with order_conformal would drift the column the
        # bound is written into away from the one the R,S rule reads back. Reject
        # it (don't silently override) so the config bug surfaces at parse time.
        # Quantile mode reads a q_<p> column, not the conformal bound, so it is
        # exempt.
        if (
            self.order_conformal is not None
            and self.ordering is not None
            and self.ordering.policy == "rs"
            and self.ordering.quantile is None
            and self.ordering.coverage != self.order_conformal.coverage
        ):
            raise ValueError(
                f"ordering.coverage ({self.ordering.coverage}) must equal "
                f"order_conformal.coverage ({self.order_conformal.coverage}): the R,S "
                "policy reads the hi_<coverage> decision bound the order-conformal "
                "runtime writes, so the two coverages must match (or omit "
                "ordering.coverage to inherit it)"
            )
        return self

    @model_validator(mode="after")
    def _coherent_spread_requires_no_reconciliation(self) -> BackendConfig:
        if self.conformal is None or self.conformal.spread != "coherent_draws":
            return self
        # The coherent-draws spread owns reconciliation-of-uncertainty; the point
        # Reconcile phase must stay a no-op while the hierarchy still supplies S.
        if self.reconciliation is not None and self.reconciliation.strategy != "none":
            raise ValueError(
                "conformal spread='coherent_draws' requires reconciliation strategy "
                "'none' (the coherent method owns reconciliation-of-uncertainty)"
            )
        return self


def load_config_from_mapping(
    data: dict[str, Any], *, source_path: str | Path | None = None
) -> BackendConfig:
    """Validate a config mapping into a :class:`BackendConfig`."""
    if not isinstance(data, dict):
        raise ValueError("config must be a mapping")
    payload = dict(data)
    if source_path is not None:
        payload["source_path"] = str(source_path)
    return BackendConfig.model_validate(payload)


def load_config(path: str | Path) -> BackendConfig:
    """Load and validate a :class:`BackendConfig` from a YAML file at ``path``."""
    source_path = str(path)
    with fsspec.open(source_path, "rt", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        raise ValueError(f"Config file is empty: {source_path}")
    return load_config_from_mapping(data, source_path=source_path)
