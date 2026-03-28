from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from calibre.conformal.aci import AdaptiveConformalInference
from calibre.conformal.intervals import symmetric_intervals
from calibre.conformal.mscp import MultiStepSplitConformalInference
from calibre.conformal.scores import absolute_error
from calibre.conformal.types import IntervalPrediction, MultiStepIntervalPrediction
from calibre.contracts.forecast_frame import (
    CALIBRATION_STATE,
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
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

ConformalMethod = Literal["mscp", "aci"]
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


@dataclass(frozen=True, slots=True)
class ConformalPolicyConfig:
    method: ConformalMethod
    coverage: float = 0.9
    calibration_window: int = 100
    gamma: float = 0.05
    score_fn: Callable = absolute_error
    quantile_rule: QuantileRule | None = None

    def __post_init__(self) -> None:
        if self.method not in {"mscp", "aci"}:
            raise ValueError("method must be 'mscp' or 'aci'")
        if not 0.0 < float(self.coverage) < 1.0:
            raise ValueError("coverage must satisfy 0 < coverage < 1")
        if int(self.calibration_window) < 1:
            raise ValueError("calibration_window must be at least 1")
        if float(self.gamma) < 0.0:
            raise ValueError("gamma must be non-negative")
        if self.quantile_rule is not None and self.quantile_rule not in {"conformal", "higher"}:
            raise ValueError("quantile_rule must be 'conformal', 'higher', or None")

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


class _BasePolicyState:
    def __init__(self, config: ConformalPolicyConfig, horizon: int) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        self.config = config
        self.horizon = int(horizon)
        self._issued_count = 0

    def predict(self, point_forecasts: np.ndarray) -> MultiStepIntervalPrediction:
        raise NotImplementedError

    def observe_row(self, row: pd.Series, lower_col: str, upper_col: str) -> float:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        raise NotImplementedError

    def emittable_mask(self, prediction: MultiStepIntervalPrediction) -> np.ndarray:
        lower = np.asarray(prediction.lower, dtype=float)
        upper = np.asarray(prediction.upper, dtype=float)
        return np.isfinite(lower) & np.isfinite(upper)


class _MscpPolicyState(_BasePolicyState):
    def __init__(self, config: ConformalPolicyConfig, horizon: int) -> None:
        super().__init__(config, horizon)
        self._controller = MultiStepSplitConformalInference(
            horizon=horizon,
            alpha=config.alpha,
            calibration_window=config.calibration_window,
            score_fn=config.score_fn,
            quantile_rule=config.resolved_quantile_rule,
        )

    def predict(self, point_forecasts: np.ndarray) -> MultiStepIntervalPrediction:
        return self._controller.predict_interval(point_forecasts)

    def observe_row(self, row: pd.Series, lower_col: str, upper_col: str) -> float:
        result = self._controller.observe(
            horizon=int(row[H]),
            y_true=float(row[Y]),
            point_forecast=float(row[Y_HAT]),
        )
        return float(result["score"])

    def snapshot(self) -> dict[str, Any]:
        diagnostics = self._controller.get_diagnostics()
        diagnostics["method"] = "mscp"
        return diagnostics

    def emittable_mask(self, prediction: MultiStepIntervalPrediction) -> np.ndarray:
        base_mask = super().emittable_mask(prediction)
        return base_mask & self._controller.ready_mask()


class _AciPolicyState(_BasePolicyState):
    """Runtime adapter for horizon-wise ACI.

    The engine can issue forecasts on sparse origin grids, so the runtime uses
    one single-step controller per horizon instead of relying on a contiguous
    issued/observed sequence shared across all horizons.
    """

    def __init__(self, config: ConformalPolicyConfig, horizon: int) -> None:
        super().__init__(config, horizon)
        self._controllers = [
            AdaptiveConformalInference(
                alpha=config.alpha,
                gamma=config.gamma,
                score_fn=config.score_fn,
                quantile_rule=config.resolved_quantile_rule,
            )
            for _ in range(horizon)
        ]

    def predict(self, point_forecasts: np.ndarray) -> MultiStepIntervalPrediction:
        center = np.asarray(point_forecasts, dtype=float)
        radii = []
        alphas = []
        for idx, controller in enumerate(self._controllers):
            prediction = controller.predict_interval(center[idx])
            radii.append(prediction.radius)
            alphas.append(prediction.alpha)
        interval = symmetric_intervals(
            center=center,
            radius=np.asarray(radii, dtype=float),
            alpha=np.asarray(alphas, dtype=float),
            issued_at=self._issued_count,
        )
        self._issued_count += 1
        return interval

    def observe_row(self, row: pd.Series, lower_col: str, upper_col: str) -> float:
        horizon_idx = int(row[H]) - 1
        controller = self._controllers[horizon_idx]
        center = float(row[Y_HAT])
        lower = float(row[lower_col])
        upper = float(row[upper_col])
        alpha = float(row.get(CONFORMAL_ALPHA, controller.current_alpha))
        prediction = IntervalPrediction(
            center=center,
            lower=lower,
            upper=upper,
            radius=max(center - lower, upper - center),
            alpha=alpha,
        )
        result = controller.observe(y_true=float(row[Y]), prediction=prediction)
        controller.trim_scores(self.config.calibration_window)
        return float(result["score"])

    def snapshot(self) -> dict[str, Any]:
        return {
            "method": "aci",
            "horizon": self.horizon,
            "issued_count": self._issued_count,
            "controllers": [controller.get_diagnostics() for controller in self._controllers],
        }


def _policy_key(frame: pd.DataFrame) -> tuple[str, str]:
    return str(frame[UNIQUE_ID].iloc[0]), str(frame[MODEL_NAME].iloc[0])


def _validate_horizon_layout(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(H).copy()
    horizons = ordered[H].to_numpy()
    expected = np.arange(1, len(ordered) + 1, dtype=horizons.dtype)
    if not np.array_equal(horizons, expected):
        raise ValueError("Conformal runtime expects one row per horizon in ascending order")
    return ordered


class ConformalRuntime:
    def __init__(self, config: ConformalPolicyConfig) -> None:
        self.config = config
        self._policies: dict[tuple[str, str], _BasePolicyState] = {}

    def _build_policy(self, horizon: int) -> _BasePolicyState:
        if self.config.method == "mscp":
            return _MscpPolicyState(self.config, horizon)
        return _AciPolicyState(self.config, horizon)

    def _get_policy(self, key: tuple[str, str], horizon: int) -> _BasePolicyState:
        policy = self._policies.get(key)
        if policy is None:
            policy = self._build_policy(horizon)
            self._policies[key] = policy
            return policy
        if policy.horizon != horizon:
            raise ValueError(
                f"Conformal horizon changed for {key}: existing {policy.horizon}, new {horizon}"
            )
        return policy

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.copy()

        lower_col, upper_col = self.config.interval_columns
        enriched_parts: list[pd.DataFrame] = []

        for _, group in frame.groupby([UNIQUE_ID, MODEL_NAME], sort=False):
            ordered = _validate_horizon_layout(group)
            key = _policy_key(ordered)
            policy = self._get_policy(key, len(ordered))
            prediction = policy.predict(ordered[Y_HAT].to_numpy())
            lower_values = np.asarray(prediction.lower, dtype=float).copy()
            upper_values = np.asarray(prediction.upper, dtype=float).copy()
            mask = policy.emittable_mask(prediction)
            lower_values[~mask] = np.nan
            upper_values[~mask] = np.nan

            ordered[lower_col] = lower_values
            ordered[upper_col] = upper_values
            ordered[CONFORMAL_METHOD] = self.config.method
            ordered[CONFORMAL_ALPHA] = prediction.alpha
            ordered[CALIBRATION_STATE] = serialize_calibration_state(policy.snapshot())
            if NONCONFORMITY_SCORE not in ordered.columns:
                ordered[NONCONFORMITY_SCORE] = np.nan
            enriched_parts.append(ordered)

        return pd.concat(enriched_parts).sort_index()

    def observe(self, resolved: pd.DataFrame) -> pd.DataFrame:
        if resolved.empty:
            return resolved.copy()

        lower_col, upper_col = self.config.interval_columns
        if lower_col not in resolved.columns or upper_col not in resolved.columns:
            raise ValueError("Resolved rows must include conformal interval columns")

        observed = resolved.copy()
        if NONCONFORMITY_SCORE not in observed.columns:
            observed[NONCONFORMITY_SCORE] = np.nan

        for _, group in observed.groupby([UNIQUE_ID, MODEL_NAME], sort=False):
            key = _policy_key(group)
            policy = self._policies.get(key)
            if policy is None:
                raise ValueError(f"No conformal state found for key {key}")
            ordered = group.sort_values([DS, FORECAST_ORIGIN, H])
            for idx, row in ordered.iterrows():
                observed.at[idx, NONCONFORMITY_SCORE] = policy.observe_row(
                    row, lower_col, upper_col
                )

        return observed
