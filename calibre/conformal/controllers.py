from __future__ import annotations

from typing import Any

import numpy as np

from calibre.conformal.numerics import _clip_alpha, _validate_bounds
from calibre.conformal.types import IntervalPrediction


class FixedAlphaController:
    def __init__(self, alpha: float) -> None:
        self._alpha = float(alpha)
        self._observations = 0

    def observe(self, y_true, y_pred, h: int) -> None:
        del y_true, y_pred, h
        self._observations += 1

    def get_alpha(self) -> float:
        return float(self._alpha)

    def get_state(self) -> dict:
        return {"type": "fixed", "alpha": float(self._alpha), "observations": self._observations}

    def set_state(self, state: dict) -> None:
        if state.get("type", "fixed") != "fixed":
            raise ValueError("FixedAlphaController state must have type='fixed'")
        self._alpha = float(state.get("alpha", self._alpha))
        self._observations = int(state.get("observations", self._observations))


class AdaptiveAlphaController:
    def __init__(
        self,
        *,
        alpha: float,
        gamma: float,
        initial_alpha: float | None = None,
        alpha_bounds: tuple[float, float] | None = (1e-6, 1.0 - 1e-6),
    ) -> None:
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self._target_alpha = float(alpha)
        self._gamma = float(gamma)
        self._bounds = _validate_bounds(alpha_bounds)
        self._alpha = float(
            _clip_alpha(
                self._target_alpha if initial_alpha is None else float(initial_alpha),
                self._bounds,
            )
        )
        self._alpha_history = [float(self._alpha)]
        self._error_history: list[int] = []

    @property
    def current_alpha(self) -> float:
        return float(self._alpha)

    @property
    def target_alpha(self) -> float:
        return float(self._target_alpha)

    @property
    def error_history(self) -> list[int]:
        return list(self._error_history)

    def _error(self, y_true: Any, y_pred: Any) -> int:
        if isinstance(y_pred, IntervalPrediction):
            return int(not y_pred.contains(float(y_true)))
        if isinstance(y_pred, dict) and {"lower", "upper"}.issubset(y_pred):
            return int(not (float(y_pred["lower"]) <= float(y_true) <= float(y_pred["upper"])))
        return int(bool(y_pred))

    def observe(self, y_true, y_pred, h: int) -> None:
        del h
        error = self._error(y_true, y_pred)
        self._error_history.append(error)
        self._alpha = float(
            _clip_alpha(
                self._alpha + self._gamma * (self._target_alpha - error),
                self._bounds,
            )
        )
        self._alpha_history.append(float(self._alpha))

    def get_alpha(self) -> float:
        return float(self._alpha)

    def get_state(self) -> dict:
        return {
            "type": "adaptive",
            "target_alpha": self._target_alpha,
            "gamma": self._gamma,
            "current_alpha": float(self._alpha),
            "alpha_bounds": self._bounds,
            "alpha_history": np.asarray(self._alpha_history, dtype=float),
            "error_history": np.asarray(self._error_history, dtype=int),
        }

    def set_state(self, state: dict) -> None:
        if state.get("type", "adaptive") != "adaptive":
            raise ValueError("AdaptiveAlphaController state must have type='adaptive'")
        self._target_alpha = float(state.get("target_alpha", self._target_alpha))
        self._gamma = float(state.get("gamma", self._gamma))
        raw_bounds = state.get("alpha_bounds", self._bounds)
        self._bounds = _validate_bounds(tuple(raw_bounds) if raw_bounds is not None else None)
        self._alpha = float(
            _clip_alpha(float(state.get("current_alpha", self._alpha)), self._bounds)
        )
        self._alpha_history = [
            float(value) for value in state.get("alpha_history", [float(self._alpha)])
        ]
        self._error_history = [int(value) for value in state.get("error_history", [])]
