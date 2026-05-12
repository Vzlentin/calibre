from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from calibre.conformal.partitions import GLOBAL_PARTITION, global_partition
from calibre.conformal.runtime import serialize_calibration_state
from calibre.contracts.forecast_frame import (
    CALIBRATION_STATE,
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
    validate_forecast_frame,
)

WeightedQuantileMode = Literal["empirical", "nonexchangeable"]


@dataclass(frozen=True, slots=True)
class CumulativeConformalRiskConfig:
    """Config for one-sided cumulative conformal order calibration.

    The controller calibrates signed cumulative residuals
    ``sum(y) - sum(base_forecast)`` and emits a single upper bound at the
    protection-period horizon. This is the inventory-ordering analogue of a
    one-sided conformal upper set: the order policy consumes that ``hi_*``
    bound as its target stock level.
    """

    coverage: float = 0.5
    protection_period: int = 3
    calibration_window: int = 5000
    partition_key: Callable[[pd.Series], Hashable] = global_partition
    weight_decay: float | None = 0.85
    weighted_quantile_mode: WeightedQuantileMode = "empirical"
    base_column: str | None = None
    fallback_buffer: float = 0.0
    buffer_min: float | None = None
    buffer_max: float | None = None
    shrinkage_strength: float = 0.0
    method_name: str = "weighted_crc"

    def __post_init__(self) -> None:
        if not 0.0 < float(self.coverage) < 1.0:
            raise ValueError("coverage must satisfy 0 < coverage < 1")
        if int(self.protection_period) < 1:
            raise ValueError("protection_period must be at least 1")
        if int(self.calibration_window) < 1:
            raise ValueError("calibration_window must be at least 1")
        if not callable(self.partition_key):
            raise ValueError("partition_key must be callable")
        if self.weight_decay is not None and not 0.0 < float(self.weight_decay) <= 1.0:
            raise ValueError("weight_decay must satisfy 0 < weight_decay <= 1")
        if self.weighted_quantile_mode not in {"empirical", "nonexchangeable"}:
            raise ValueError("weighted_quantile_mode must be 'empirical' or 'nonexchangeable'")
        if not 0.0 <= float(self.shrinkage_strength) <= 1.0:
            raise ValueError("shrinkage_strength must satisfy 0 <= shrinkage_strength <= 1")
        if (
            self.buffer_min is not None
            and self.buffer_max is not None
            and float(self.buffer_min) > float(self.buffer_max)
        ):
            raise ValueError("buffer_min must be <= buffer_max")

    @property
    def alpha(self) -> float:
        return 1.0 - float(self.coverage)

    @property
    def interval_columns(self) -> tuple[str, str]:
        return interval_column_names(self.coverage)

    @property
    def resolved_base_column(self) -> str:
        return self.base_column or Y_HAT


@dataclass(frozen=True, slots=True)
class _ResidualRecord:
    sequence: int
    unique_id: str
    partition: Hashable
    residual: float


def _conformal_quantile(values: Iterable[float], level: float, fallback: float = 0.0) -> float:
    """Unweighted split-conformal quantile with clipped finite-sample rank."""
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return float(fallback)
    rank = math.ceil((len(sorted_values) + 1) * float(level))
    idx = min(max(rank, 1), len(sorted_values)) - 1
    return float(sorted_values[idx])


def _weighted_quantile(
    records: list[_ResidualRecord],
    level: float,
    decay: float,
    fallback: float,
    mode: WeightedQuantileMode,
) -> float:
    """Recency-weighted quantile for non-exchangeable calibration.

    ``mode="nonexchangeable"`` keeps the unit test-point mass from weighted
    conformal prediction/CRC, so the calibration weights are normalized by
    ``sum(weights) + 1``. If the requested level falls into that held-out mass,
    the finite-sample upper quantile is infinite unless the caller caps it.
    """
    if not records:
        return float(fallback)
    max_sequence = max(record.sequence for record in records)
    ordered = sorted(
        (
            record.residual,
            float(decay) ** float(max_sequence - record.sequence),
        )
        for record in records
    )
    total_weight = sum(weight for _, weight in ordered)
    if total_weight <= 0.0:
        return float(fallback)
    denominator = total_weight + (1.0 if mode == "nonexchangeable" else 0.0)
    threshold = float(level) * denominator
    cumulative = 0.0
    for residual, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return float(residual)
    if mode == "nonexchangeable":
        return float("inf")
    return float(ordered[-1][0])


class CumulativeConformalRiskRuntime:
    """One-sided cumulative residual conformal runtime for order decisions."""

    def __init__(self, config: CumulativeConformalRiskConfig) -> None:
        self.config = config
        self._records: list[_ResidualRecord] = []
        self._records_by_partition: dict[Hashable, list[_ResidualRecord]] = {}
        self._buffer_cache: dict[str, float] = {}
        self._sequence = 0

    def _partition_for_row(self, row: pd.Series) -> Hashable:
        partition = self.config.partition_key(row)
        try:
            hash(partition)
        except TypeError:
            return str(partition)
        return partition

    def _append_record(self, record: _ResidualRecord) -> None:
        self._records.append(record)
        self._records_by_partition.setdefault(record.partition, []).append(record)
        self._buffer_cache.clear()
        if len(self._records) > self.config.calibration_window:
            self._records = self._records[-self.config.calibration_window :]
            self._rebuild_indexes()

    def _rebuild_indexes(self) -> None:
        self._records_by_partition = {}
        for record in self._records:
            self._records_by_partition.setdefault(record.partition, []).append(record)
        self._buffer_cache.clear()

    def _records_for_partition(self, partition: Hashable) -> list[_ResidualRecord]:
        return self._partition_records_and_cache_key(partition)[0]

    def _partition_records_and_cache_key(
        self, partition: Hashable
    ) -> tuple[list[_ResidualRecord], str]:
        if partition == GLOBAL_PARTITION:
            return self._records, "global"
        return self._records_by_partition.get(partition, []), f"partition:{partition}"

    def _raw_buffer(self, records: list[_ResidualRecord]) -> float:
        if self.config.weight_decay is None:
            return _conformal_quantile(
                (record.residual for record in records),
                self.config.coverage,
                fallback=self.config.fallback_buffer,
            )
        return _weighted_quantile(
            records,
            self.config.coverage,
            decay=float(self.config.weight_decay),
            fallback=self.config.fallback_buffer,
            mode=self.config.weighted_quantile_mode,
        )

    def _raw_buffer_cached(self, cache_key: str, records: list[_ResidualRecord]) -> float:
        cached = self._buffer_cache.get(cache_key)
        if cached is not None:
            return cached
        buffer = self._raw_buffer(records)
        self._buffer_cache[cache_key] = buffer
        return buffer

    def _buffer_components_for(self, partition: Hashable) -> dict[str, float]:
        scoped_records, scoped_key = self._partition_records_and_cache_key(partition)
        scoped = self._raw_buffer_cached(scoped_key, scoped_records)
        shrinkage = float(self.config.shrinkage_strength)
        if partition == GLOBAL_PARTITION or shrinkage == 0.0:
            buffer = scoped
            global_buffer = scoped
        else:
            global_buffer = self._raw_buffer_cached("global", self._records)
            buffer = (1.0 - shrinkage) * scoped + shrinkage * global_buffer
        if self.config.buffer_min is not None:
            buffer = max(buffer, float(self.config.buffer_min))
        if self.config.buffer_max is not None:
            buffer = min(buffer, float(self.config.buffer_max))
        if not math.isfinite(buffer):
            raise ValueError(
                "Cumulative conformal buffer is non-finite; set buffer_max to make "
                "non-exchangeable weighted CRC consumable by order policies."
            )
        return {
            "scoped_buffer": float(scoped),
            "global_buffer": float(global_buffer),
            "buffer": float(buffer),
        }

    def _buffer_for(self, partition: Hashable) -> float:
        return self._buffer_components_for(partition)["buffer"]

    def _records_for(self, unique_id: str) -> list[_ResidualRecord]:
        """Return the residual pool used by a series partition.

        Kept as a compatibility shim for callers that inspect diagnostics in tests.
        """
        return self._records_by_partition.get(unique_id, [])

    def _snapshot(self, partition: Hashable) -> dict[str, Any]:
        records = self._records_for_partition(partition)
        buffer_components = self._buffer_components_for(partition)
        return {
            "method": self.config.method_name,
            "coverage": self.config.coverage,
            "protection_period": self.config.protection_period,
            "partition": str(partition),
            "partition_key": getattr(
                self.config.partition_key,
                "__name__",
                repr(self.config.partition_key),
            ),
            "weight_decay": self.config.weight_decay,
            "weighted_quantile_mode": self.config.weighted_quantile_mode,
            "base_column": self.config.resolved_base_column,
            "buffer_min": self.config.buffer_min,
            "buffer_max": self.config.buffer_max,
            "shrinkage_strength": self.config.shrinkage_strength,
            "n_scores": len(records),
            **buffer_components,
        }

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        # Interval columns are populated only on the terminal-H row of each
        # (uid, model, origin) group; earlier rows keep NaN. Aggregating
        # consumers must select the terminal row, not sum across H.
        if frame.empty:
            return frame.copy()

        validate_forecast_frame(frame)
        base_column = self.config.resolved_base_column
        if base_column not in frame.columns:
            raise ValueError(f"Missing base forecast column for conformal order: {base_column!r}")

        lower_col, upper_col = self.config.interval_columns
        result = frame.copy()
        result[lower_col] = np.nan
        result[upper_col] = np.nan
        result[CONFORMAL_METHOD] = self.config.method_name
        result[CONFORMAL_MODE] = "cumulative"
        result[CONFORMAL_ALPHA] = self.config.alpha
        if NONCONFORMITY_SCORE not in result.columns:
            result[NONCONFORMITY_SCORE] = np.nan

        group_cols = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]
        for _, group in result.groupby(group_cols, sort=False):
            ordered = group.sort_values(H)
            window = ordered[ordered[H] <= self.config.protection_period]
            if len(window) < self.config.protection_period:
                continue
            terminal = window[window[H] == self.config.protection_period]
            if terminal.empty:
                continue

            partition = self._partition_for_row(ordered.iloc[0])
            base_sum = float(window[base_column].sum())
            buffer = self._buffer_for(partition)
            upper = base_sum + buffer
            terminal_idx = terminal.index[-1]

            result.loc[terminal_idx, lower_col] = min(base_sum, upper)
            result.loc[terminal_idx, upper_col] = upper
            result.loc[terminal_idx, CALIBRATION_STATE] = serialize_calibration_state(
                self._snapshot(partition)
            )

        if CALIBRATION_STATE not in result.columns:
            result[CALIBRATION_STATE] = ""
        result[CALIBRATION_STATE] = result[CALIBRATION_STATE].fillna("")
        return result

    def observe(self, resolved: pd.DataFrame) -> pd.DataFrame:
        if resolved.empty:
            return resolved.copy()

        validate_forecast_frame(resolved)
        base_column = self.config.resolved_base_column
        if base_column not in resolved.columns:
            raise ValueError(f"Missing base forecast column for conformal order: {base_column!r}")

        observed = resolved.copy()
        if NONCONFORMITY_SCORE not in observed.columns:
            observed[NONCONFORMITY_SCORE] = np.nan

        group_cols = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]
        # sort=True is load-bearing: _sequence drives recency weights, so
        # origins must be ingested in chronological order.
        for _, origin_group in observed.groupby(FORECAST_ORIGIN, sort=True):
            self._sequence += 1
            for _, group in origin_group.groupby(group_cols, sort=False):
                ordered = group.sort_values(H)
                window = ordered[ordered[H] <= self.config.protection_period]
                if len(window) < self.config.protection_period:
                    continue
                if window[H].duplicated().any():
                    raise ValueError("Duplicate H values in cumulative conformal order window")
                if window[Y].isna().any():
                    continue

                unique_id = str(ordered[UNIQUE_ID].iloc[0])
                partition = self._partition_for_row(ordered.iloc[0])
                actual_sum = float(window[Y].sum())
                base_sum = float(window[base_column].sum())
                residual = actual_sum - base_sum
                self._append_record(
                    _ResidualRecord(
                        sequence=self._sequence,
                        unique_id=unique_id,
                        partition=partition,
                        residual=residual,
                    )
                )
                observed.at[window.index[-1], NONCONFORMITY_SCORE] = residual

        return observed

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "method": self.config.method_name,
            "coverage": self.config.coverage,
            "protection_period": self.config.protection_period,
            "partition_key": getattr(
                self.config.partition_key,
                "__name__",
                repr(self.config.partition_key),
            ),
            "weight_decay": self.config.weight_decay,
            "weighted_quantile_mode": self.config.weighted_quantile_mode,
            "base_column": self.config.resolved_base_column,
            "buffer_min": self.config.buffer_min,
            "buffer_max": self.config.buffer_max,
            "shrinkage_strength": self.config.shrinkage_strength,
            "n_scores": len(self._records),
            "n_partitions": len(self._records_by_partition),
            "residuals": np.asarray([record.residual for record in self._records], dtype=float),
        }
