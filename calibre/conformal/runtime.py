from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

from calibre.conformal.calibrators import RollingQuantileCalibrator
from calibre.conformal.controllers import AdaptiveAlphaController, FixedAlphaController
from calibre.conformal.partitions import global_partition
from calibre.conformal.protocols import Calibrator, Controller, Score
from calibre.conformal.scores import absolute_error_score
from calibre.conformal.types import IntervalPrediction
from calibre.core.forecast_frame import (
    CALIBRATION_STATE,
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    interval_column_names,
)

logger = logging.getLogger(__name__)

ConformalMethod = Literal["mscp", "aci"]
ConformalMode = Literal["perhorizon", "cumulative"]
QuantileRule = Literal["conformal", "higher"]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def serialize_calibration_state(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"), default=_json_default)


def deserialize_calibration_state(payload: str | None) -> dict[str, Any]:
    if not payload:
        return {}
    return json.loads(payload)  # type: ignore[arg-type]


def to_json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
    """Coerce numpy/ndarray/Timestamp values into JSON-safe Python objects."""
    return json.loads(serialize_calibration_state(state))


class ConformalRuntime(Protocol):
    @property
    def interval_columns(self) -> tuple[str, str]: ...

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame: ...

    def observe(self, resolved: pd.DataFrame) -> pd.DataFrame: ...

    def get_diagnostics(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SymmetricIntervalConfig:
    method: ConformalMethod
    coverage: float = 0.9
    calibration_window: int = 100
    gamma: float = 0.05
    partition_key: Callable[[pd.Series], Hashable] = global_partition
    quantile_rule: QuantileRule | None = None
    mode: ConformalMode = "perhorizon"
    protection_period: int | None = None

    def __post_init__(self) -> None:
        if self.method not in {"mscp", "aci"}:
            raise ValueError("method must be 'mscp' or 'aci'")
        if not 0.0 < float(self.coverage) < 1.0:
            raise ValueError("coverage must satisfy 0 < coverage < 1")
        if int(self.calibration_window) < 1:
            raise ValueError("calibration_window must be at least 1")
        if float(self.gamma) < 0.0:
            raise ValueError("gamma must be non-negative")
        if not callable(self.partition_key):
            raise ValueError("partition_key must be callable")
        if self.quantile_rule is not None and self.quantile_rule not in {"conformal", "higher"}:
            raise ValueError("quantile_rule must be 'conformal', 'higher', or None")
        if self.mode not in {"perhorizon", "cumulative"}:
            raise ValueError("mode must be 'perhorizon' or 'cumulative'")
        if self.mode == "cumulative":
            if self.method != "mscp":
                raise ValueError("cumulative mode currently supports only method='mscp'")
            if self.protection_period is None or int(self.protection_period) < 1:
                raise ValueError("cumulative mode requires protection_period >= 1")

    @property
    def alpha(self) -> float:
        return 1.0 - float(self.coverage)

    @property
    def interval_columns(self) -> tuple[str, str]:
        return interval_column_names(self.coverage)

    @property
    def resolved_quantile_rule(self) -> QuantileRule:
        if self.quantile_rule is not None:
            return self.quantile_rule
        return "higher" if self.method == "mscp" else "conformal"


def _as_scalar_score(score) -> float:
    arr = np.asarray(score, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError("Expected Score to return a scalar score")
    return float(arr[0])


def _validate_horizon_layout(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(H).copy()
    horizons = ordered[H].to_numpy()
    expected = np.arange(1, len(ordered) + 1, dtype=horizons.dtype)
    if not np.array_equal(horizons, expected):
        raise ValueError("Conformal runtime expects one row per horizon in ascending order")
    return ordered


def _hashable(value: Hashable) -> Hashable:
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


class SymmetricIntervalRuntime:
    def __init__(
        self,
        config: SymmetricIntervalConfig,
        score: Score | None = None,
        calibrator: Calibrator | None = None,
        controller: Controller | None = None,
        *,
        method_name: str | None = None,
    ) -> None:
        if score is None or calibrator is None or controller is None:
            score, calibrator, controller = _components_from_config(config)
        self.config = config
        self.score = score
        self.calibrator = calibrator
        self.controller = controller
        self.method_name = method_name or config.method
        self._issued_count = 0

    @classmethod
    def from_state(
        cls,
        config: SymmetricIntervalConfig,
        state_payload: str | dict[str, Any] | None,
    ) -> SymmetricIntervalRuntime:
        """Rehydrate a runtime from a serialized calibration-state snapshot."""
        if isinstance(state_payload, dict):
            state = state_payload
        else:
            state = deserialize_calibration_state(state_payload)
        runtime = cls(config, method_name=state.get("method", config.method))
        runtime._issued_count = int(state.get("issued_count", 0))
        if "calibrator" in state:
            runtime.calibrator.set_state(state["calibrator"])
        if "controller" in state:
            runtime.controller.set_state(state["controller"])
        return runtime

    @property
    def interval_columns(self) -> tuple[str, str]:
        return self.config.interval_columns

    def _base_partition(self, row: pd.Series) -> str:
        value = _hashable(self.config.partition_key(row))
        return str(value)

    def _partition_for_row(self, row: pd.Series, *, cumulative: bool = False) -> str:
        model_name = str(row[MODEL_NAME])
        base = self._base_partition(row)
        if cumulative:
            return f"{model_name}:cumulative:{base}"
        return f"{model_name}:h{int(row[H])}:{base}"

    def _calibrator_ready(self, partition: str, alpha: float) -> bool:
        ready = getattr(self.calibrator, "ready", None)
        if ready is None:
            return bool(np.isfinite(self.calibrator.predict(alpha, partition)))
        return bool(ready(partition, alpha))

    def _snapshot(self, partition: str) -> dict[str, Any]:
        calibrator_state: dict[str, Any] = getattr(self.calibrator, "get_state", lambda: {})()
        return {
            "method": self.method_name,
            "mode": self.config.mode,
            "coverage": self.config.coverage,
            "partition": partition,
            "issued_count": self._issued_count,
            "controller": self.controller.get_state(),
            "calibrator": calibrator_state,
        }

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        started = time.perf_counter()
        if frame.empty:
            result = frame.copy()
        elif self.config.mode == "cumulative":
            result = self._apply_cumulative(frame)
        else:
            result = self._apply_perhorizon(frame)
        logger.info(
            "applied conformal runtime",
            extra={
                "phase": "conformal",
                "operation": "apply",
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "rows": len(frame),
                "method": self.method_name,
                "mode": self.config.mode,
            },
        )
        return result

    def _apply_perhorizon(self, frame: pd.DataFrame) -> pd.DataFrame:
        lower_col, upper_col = self.config.interval_columns
        parts: list[pd.DataFrame] = []

        for _, group in frame.groupby([UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN], sort=False):
            ordered = _validate_horizon_layout(group)
            lower_values = np.full(len(ordered), np.nan, dtype=float)
            upper_values = np.full(len(ordered), np.nan, dtype=float)
            alpha_values = np.full(len(ordered), np.nan, dtype=float)
            state_values: list[str] = []

            for pos, (_, row) in enumerate(ordered.iterrows()):
                alpha = self.controller.get_alpha()
                partition = self._partition_for_row(row)
                radius = self.calibrator.predict(alpha, partition)
                alpha_values[pos] = alpha
                if self._calibrator_ready(partition, alpha) and np.isfinite(radius):
                    center = float(row[Y_HAT])
                    lower_values[pos] = center - float(radius)
                    upper_values[pos] = center + float(radius)
                state_values.append(serialize_calibration_state(self._snapshot(partition)))

            ordered[lower_col] = lower_values
            ordered[upper_col] = upper_values
            ordered[CONFORMAL_METHOD] = self.method_name
            ordered[CONFORMAL_MODE] = self.config.mode
            ordered[CONFORMAL_ALPHA] = alpha_values
            ordered[CALIBRATION_STATE] = state_values
            if NONCONFORMITY_SCORE not in ordered.columns:
                ordered[NONCONFORMITY_SCORE] = np.nan
            parts.append(ordered)
            self._issued_count += 1

        return pd.concat(parts).sort_index()

    def _apply_cumulative(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.config.protection_period is None:
            raise ValueError("cumulative mode requires config.protection_period")
        protection_period = int(self.config.protection_period)
        lower_col, upper_col = self.config.interval_columns
        result = frame.copy()
        result[lower_col] = np.nan
        result[upper_col] = np.nan
        result[CONFORMAL_METHOD] = self.method_name
        result[CONFORMAL_MODE] = self.config.mode
        result[CONFORMAL_ALPHA] = self.controller.get_alpha()
        if NONCONFORMITY_SCORE not in result.columns:
            result[NONCONFORMITY_SCORE] = np.nan

        for _, group in result.groupby([UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN], sort=False):
            ordered = group.sort_values(H)
            if int(ordered[H].max()) < protection_period:
                continue
            window = ordered[ordered[H] <= protection_period]
            if len(window) < protection_period:
                continue
            terminal = window[window[H] == protection_period]
            if terminal.empty:
                continue
            row = terminal.iloc[-1]
            terminal_idx = terminal.index[-1]
            alpha = self.controller.get_alpha()
            partition = self._partition_for_row(row, cumulative=True)
            radius = self.calibrator.predict(alpha, partition)
            if self._calibrator_ready(partition, alpha) and np.isfinite(radius):
                center = float(window[Y_HAT].sum())
                result.loc[terminal_idx, lower_col] = center - float(radius)
                result.loc[terminal_idx, upper_col] = center + float(radius)
            result.loc[terminal_idx, CALIBRATION_STATE] = serialize_calibration_state(
                self._snapshot(partition)
            )
            self._issued_count += 1

        if CALIBRATION_STATE not in result.columns:
            result[CALIBRATION_STATE] = ""
        result[CALIBRATION_STATE] = result[CALIBRATION_STATE].fillna("")
        return result

    def observe(self, resolved: pd.DataFrame) -> pd.DataFrame:
        started = time.perf_counter()
        if resolved.empty:
            observed = resolved.copy()
        else:
            lower_col, upper_col = self.config.interval_columns
            if lower_col not in resolved.columns or upper_col not in resolved.columns:
                raise ValueError("Resolved rows must include conformal interval columns")

            observed = resolved.copy()
            if NONCONFORMITY_SCORE not in observed.columns:
                observed[NONCONFORMITY_SCORE] = np.nan

            if self.config.mode == "cumulative":
                observed = self._observe_cumulative(observed)
            else:
                observed = self._observe_perhorizon(observed, lower_col, upper_col)
        logger.info(
            "observed conformal runtime",
            extra={
                "phase": "conformal",
                "operation": "observe",
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "rows": len(resolved),
                "method": self.method_name,
                "mode": self.config.mode,
            },
        )
        return observed

    def _observe_perhorizon(
        self, observed: pd.DataFrame, lower_col: str, upper_col: str
    ) -> pd.DataFrame:
        for _, group in observed.groupby([UNIQUE_ID, MODEL_NAME], sort=False):
            ordered = group.sort_values([DS, FORECAST_ORIGIN, H])
            for idx, row in ordered.iterrows():
                if pd.isna(row[Y]) or pd.isna(row[Y_HAT]):
                    continue
                partition = self._partition_for_row(row)
                score = _as_scalar_score(self.score(float(row[Y]), float(row[Y_HAT])))
                self.calibrator.update(score, partition)
                prediction = IntervalPrediction(
                    center=float(row[Y_HAT]),
                    lower=float(row[lower_col]) if pd.notna(row[lower_col]) else np.nan,
                    upper=float(row[upper_col]) if pd.notna(row[upper_col]) else np.nan,
                    radius=max(
                        abs(float(row[Y_HAT]) - float(row[lower_col]))
                        if pd.notna(row[lower_col])
                        else 0.0,
                        abs(float(row[upper_col]) - float(row[Y_HAT]))
                        if pd.notna(row[upper_col])
                        else 0.0,
                    ),
                    alpha=float(row.get(CONFORMAL_ALPHA, self.controller.get_alpha())),
                )
                self.controller.observe(float(row[Y]), prediction, int(row[H]))
                observed.at[idx, NONCONFORMITY_SCORE] = score
        return observed

    def _observe_cumulative(self, observed: pd.DataFrame) -> pd.DataFrame:
        if self.config.protection_period is None:
            raise ValueError("cumulative mode requires config.protection_period")
        protection_period = int(self.config.protection_period)
        group_cols = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]

        for _, group in observed.groupby(group_cols, sort=False):
            ordered = group.sort_values(H)
            window = ordered[ordered[H] <= protection_period]
            if window[H].duplicated().any():
                raise ValueError("Duplicate H values in cumulative observe window")
            if len(window) < protection_period or window[Y].isna().any():
                continue
            terminal = window[window[H] == protection_period]
            if terminal.empty:
                continue
            row = terminal.iloc[-1]
            partition = self._partition_for_row(row, cumulative=True)
            actual_sum = float(window[Y].sum())
            forecast_sum = float(window[Y_HAT].sum())
            score = _as_scalar_score(self.score(actual_sum, forecast_sum))
            self.calibrator.update(score, partition)
            self.controller.observe(actual_sum, forecast_sum, protection_period)
            observed.at[terminal.index[-1], NONCONFORMITY_SCORE] = score
        return observed

    def get_diagnostics(self) -> dict[str, Any]:
        calibrator_state: dict[str, Any] = getattr(self.calibrator, "get_state", lambda: {})()
        return {
            "method": self.method_name,
            "mode": self.config.mode,
            "coverage": self.config.coverage,
            "calibration_window": self.config.calibration_window,
            "gamma": self.config.gamma,
            "partition_key": getattr(
                self.config.partition_key,
                "__name__",
                repr(self.config.partition_key),
            ),
            "quantile_rule": self.config.resolved_quantile_rule,
            "protection_period": self.config.protection_period,
            "issued_count": self._issued_count,
            "controller": self.controller.get_state(),
            "calibrator": calibrator_state,
        }


def _components_from_config(
    config: SymmetricIntervalConfig,
) -> tuple[Score, Calibrator, Controller]:
    calibrator = RollingQuantileCalibrator(
        calibration_window=config.calibration_window,
        quantile_rule=config.resolved_quantile_rule,
        ready_on_empty=config.method == "aci",
    )
    if config.method == "aci":
        controller: Controller = AdaptiveAlphaController(alpha=config.alpha, gamma=config.gamma)
    else:
        controller = FixedAlphaController(config.alpha)
    return absolute_error_score, calibrator, controller


def build_symmetric_interval_runtime(config: SymmetricIntervalConfig) -> SymmetricIntervalRuntime:
    score, calibrator, controller = _components_from_config(config)
    return SymmetricIntervalRuntime(
        config=config,
        score=score,
        calibrator=calibrator,
        controller=controller,
        method_name=config.method,
    )
