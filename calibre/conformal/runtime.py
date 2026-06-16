"""Stable, pipeline-facing conformal runtime: apply, observe, and persist.

This is the interface the backend calibration step depends on; the rest of
:mod:`calibre.conformal` is experimental low-level building blocks.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Hashable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from calibre.conformal.calibrators import RollingQuantileCalibrator
from calibre.conformal.controllers import AdaptiveAlphaController, FixedAlphaController
from calibre.conformal.partitions import GLOBAL_PARTITION, global_partition, series_partition
from calibre.conformal.protocols import Calibrator, Controller, Score, Spread
from calibre.conformal.scores import absolute_error_score
from calibre.conformal.spread import AnalyticRadius
from calibre.conformal.types import IntervalPrediction
from calibre.core.forecast_frame import (
    CALIBRATION_STATE_REF,
    CONFORMAL_ALPHA,
    CONFORMAL_METHOD,
    CONFORMAL_MODE,
    CONFORMAL_PARTITION,
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


def to_json_safe_state(state: dict[str, Any]) -> dict[str, Any]:
    """Coerce numpy/ndarray/Timestamp values into JSON-safe Python objects."""
    return json.loads(json.dumps(state, default=_json_default))


@dataclass(frozen=True, slots=True)
class StateRef:
    """Parsed components of a calibration-state reference string."""

    method: str
    mode: str
    issued_count: int
    partition: str


def state_ref_value(method: str, mode: str, issued_count: int, partition: str) -> str:
    """Encode a calibration-state reference as ``method:mode:issued_count:partition``."""
    return f"{method}:{mode}:{issued_count}:{partition}"


def parse_state_ref(text: str) -> StateRef | None:
    """Parse a state-reference string, returning ``None`` if it is malformed."""
    parts = text.split(":", 3)
    if len(parts) != 4:
        return None
    try:
        issued_count = int(parts[2])
    except ValueError:
        return None
    return StateRef(method=parts[0], mode=parts[1], issued_count=issued_count, partition=parts[3])


class ConformalRuntime(Protocol):
    """Pipeline-facing conformal runtime the backend calibration step depends on.

    Defines the apply/observe/persist surface every conformal runtime exposes:
    :meth:`apply` wraps forecasts in intervals, :meth:`observe` feeds resolved
    actuals back into calibration, and :meth:`get_resume_state` snapshots state
    for persistence.
    """

    @property
    def interval_columns(self) -> tuple[str, str]: ...

    @property
    def mode(self) -> str: ...

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame: ...

    def observe(self, resolved: pd.DataFrame) -> pd.DataFrame: ...

    def adaptive_drift(self) -> float | None: ...

    def get_resume_state(self) -> dict[str, Any]: ...


@runtime_checkable
class PartitionedConformalRuntime(Protocol):
    """A conformal runtime that exposes per-partition state for persistence.

    The backend uses this to persist one row per ``(uid, model)`` partition
    instead of a single blob. Restoration is factory-only — see
    ``SymmetricIntervalRuntime.from_partition_states`` — so this protocol has no
    in-place state setter by design, avoiding the mutate-existing-instance
    pattern (a restored instance must be a fresh object, never a mutated live
    one, so persistence and the running runtime can never alias).
    """

    def get_partition_states(self) -> dict[str, dict[str, Any]]: ...

    def get_diagnostics(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class SymmetricIntervalConfig:
    """Validated configuration for a symmetric interval conformal runtime.

    Selects the controller method (``mscp``/``aci``), target coverage,
    calibration window, partition key, and per-horizon vs. cumulative mode.
    """

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


def _validate_horizon_layout_grouped(
    horizons: np.ndarray,
    group_codes: np.ndarray,
    group_count: int,
) -> None:
    """Require every group's horizons to be exactly ``1..len(group)``."""
    order = np.lexsort((horizons, group_codes))
    sorted_horizons = horizons[order]
    counts = np.bincount(group_codes, minlength=group_count)
    starts = np.cumsum(counts) - counts
    positions_in_group = np.arange(len(horizons)) - np.repeat(starts, counts)
    if not np.array_equal(sorted_horizons, positions_in_group + 1):
        raise ValueError("Conformal runtime expects one row per horizon in ascending order")


def _hashable(value: Hashable) -> Hashable:
    try:
        hash(value)
    except TypeError:
        return str(value)
    return value


class SymmetricIntervalRuntime:
    """Conformal runtime that wraps point forecasts in symmetric intervals.

    Composes a score, calibrator, and controller to apply calibrated radii to
    a forecast frame, observe resolved actuals, and emit resumable state.
    """

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
        # The point-forecast spread is the only adapter today; it carries no
        # state, so it is owned by the runtime rather than threaded through the
        # resume-state plumbing. A config selector arrives with the next adapter.
        self.spread: Spread = AnalyticRadius()
        self.method_name = method_name or config.method
        self._issued_count = 0

    @classmethod
    def from_state(
        cls,
        config: SymmetricIntervalConfig,
        state: dict[str, Any] | None,
    ) -> SymmetricIntervalRuntime:
        """Rehydrate a runtime from a calibration-state snapshot."""
        state = state or {}
        runtime = cls(config, method_name=state.get("method", config.method))
        runtime.restore_issued_count(int(state.get("issued_count", 0)))
        if "calibrator" in state:
            runtime.calibrator.set_state(state["calibrator"])
        if "controller" in state:
            runtime.controller.set_state(state["controller"])
        return runtime

    @property
    def interval_columns(self) -> tuple[str, str]:
        return self.config.interval_columns

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def issued_count(self) -> int:
        return self._issued_count

    def restore_issued_count(self, count: int) -> None:
        if count < 0:
            raise ValueError("issued_count must be non-negative")
        self._issued_count = count

    def adaptive_drift(self) -> float | None:
        return self.controller.drift()

    def _base_partition(self, row: pd.Series) -> str:
        value = _hashable(self.config.partition_key(row))
        return str(value)

    def _partition_for_row(self, row: pd.Series, *, cumulative: bool = False) -> str:
        model_name = str(row[MODEL_NAME])
        base = self._base_partition(row)
        if cumulative:
            return f"{model_name}:cumulative:{base}"
        return f"{model_name}:h{int(row[H])}:{base}"

    def _snapshot(self, partition: str) -> dict[str, Any]:
        calibrator_state: dict[str, Any] = self.calibrator.get_state()
        return {
            "method": self.method_name,
            "mode": self.config.mode,
            "coverage": self.config.coverage,
            "partition": partition,
            "issued_count": self._issued_count,
            "controller": self.controller.get_state(),
            "calibrator": calibrator_state,
        }

    def _state_ref(self, partition: str) -> str:
        return state_ref_value(self.method_name, self.config.mode, self._issued_count, partition)

    def _controller_resume_state(self) -> dict[str, Any]:
        state = self.controller.get_state()
        controller_type = state.get("type")
        if controller_type == "adaptive":
            return {
                key: state[key]
                for key in (
                    "type",
                    "target_alpha",
                    "gamma",
                    "current_alpha",
                    "alpha_bounds",
                )
                if key in state
            }
        if controller_type == "fixed":
            return {key: state[key] for key in ("type", "alpha") if key in state}
        return state

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

    def _base_partition_values(self, frame: pd.DataFrame) -> list[str]:
        """Vectorized ``_base_partition`` for the shipped partition keys.

        Unknown callables fall back to the row-wise definition so custom
        partition keys keep their exact semantics.
        """
        key = self.config.partition_key
        if key is global_partition:
            return [GLOBAL_PARTITION] * len(frame)
        if key is series_partition:
            return [str(_hashable(value)) for value in frame[UNIQUE_ID].to_numpy()]
        return [str(_hashable(key(row))) for _, row in frame.iterrows()]

    def _partition_values(self, frame: pd.DataFrame) -> list[str]:
        models = frame[MODEL_NAME].to_numpy()
        horizons = frame[H].to_numpy()
        return [
            f"{model}:h{int(horizon)}:{base}"
            for model, horizon, base in zip(
                models, horizons, self._base_partition_values(frame), strict=True
            )
        ]

    def _apply_perhorizon(self, frame: pd.DataFrame) -> pd.DataFrame:
        lower_col, upper_col = self.config.interval_columns
        result = frame.copy()

        group_keys = pd.MultiIndex.from_arrays(
            [
                result[UNIQUE_ID].to_numpy(),
                result[MODEL_NAME].to_numpy(),
                result[FORECAST_ORIGIN].to_numpy(),
            ]
        )
        group_codes, uniques = pd.factorize(group_keys, sort=False)
        _validate_horizon_layout_grouped(result[H].to_numpy(), group_codes, len(uniques))

        partitions = self._partition_values(result)
        alpha = float(self.controller.get_alpha())
        radii, ready = self.calibrator.predict_batch(alpha, partitions)
        issue = ready & np.isfinite(radii)
        centers = result[Y_HAT].to_numpy(dtype=float)
        lower_values, upper_values = self.spread.to_interval(centers, radii, issue)

        # Mirror the per-group accounting of the row-wise path: groups are
        # numbered in first-occurrence order and every row of a group shares
        # the issued count its group was processed at.
        issued = self._issued_count + group_codes
        method = self.method_name
        mode = self.config.mode
        state_refs = [
            f"{method}:{mode}:{count}:{partition}"
            for count, partition in zip(issued, partitions, strict=True)
        ]

        result[lower_col] = lower_values
        result[upper_col] = upper_values
        result[CONFORMAL_METHOD] = method
        result[CONFORMAL_MODE] = mode
        result[CONFORMAL_ALPHA] = alpha
        result[CALIBRATION_STATE_REF] = state_refs
        result[CONFORMAL_PARTITION] = partitions
        if NONCONFORMITY_SCORE not in result.columns:
            result[NONCONFORMITY_SCORE] = np.nan
        self._issued_count += len(uniques)
        return result.sort_index()

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
        result[CALIBRATION_STATE_REF] = ""
        result[CONFORMAL_PARTITION] = ""
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
            if self.calibrator.ready(partition, alpha) and np.isfinite(radius):
                center = float(window[Y_HAT].sum())
                # The terminal cumulative row is the length-1 case of the
                # vectorised spread: the ``ready & isfinite`` guard above is the
                # scalar form of the per-horizon ``issue`` mask.
                lower, upper = self.spread.to_interval(
                    np.array([center]),
                    np.array([float(radius)]),
                    np.array([True]),
                )
                result.loc[terminal_idx, lower_col] = lower[0]
                result.loc[terminal_idx, upper_col] = upper[0]
            result.loc[terminal_idx, CALIBRATION_STATE_REF] = self._state_ref(partition)
            result.loc[terminal_idx, CONFORMAL_PARTITION] = partition
            self._issued_count += 1

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
        resolved_mask = observed[Y].notna() & observed[Y_HAT].notna()
        if not bool(resolved_mask.any()):
            return observed
        resolved = observed.loc[resolved_mask]

        # The row-wise path consumed groups in first-occurrence (unique_id,
        # model) order, each group sorted by (ds, origin, h). Calibration
        # windows and adaptive-alpha updates are order-sensitive, so the
        # vectorized path replays rows in exactly that order.
        group_keys = pd.MultiIndex.from_arrays(
            [resolved[UNIQUE_ID].to_numpy(), resolved[MODEL_NAME].to_numpy()]
        )
        group_codes, _ = pd.factorize(group_keys, sort=False)
        order = np.lexsort(
            (
                resolved[H].to_numpy(),
                resolved[FORECAST_ORIGIN].to_numpy(),
                resolved[DS].to_numpy(),
                group_codes,
            )
        )

        y_true = resolved[Y].to_numpy(dtype=float)
        y_hat = resolved[Y_HAT].to_numpy(dtype=float)
        scores = np.asarray(self.score(y_true, y_hat), dtype=float).reshape(-1)
        if scores.shape != y_true.shape:
            raise ValueError("Expected Score to return one score per row")

        lower_values = resolved[lower_col].to_numpy(dtype=float)
        upper_values = resolved[upper_col].to_numpy(dtype=float)
        if CONFORMAL_ALPHA in resolved.columns:
            alpha_values = resolved[CONFORMAL_ALPHA].to_numpy(dtype=float)
        else:
            alpha_values = np.full(len(resolved), float(self.controller.get_alpha()))
        horizons = resolved[H].to_numpy()
        partitions = self._partition_values(resolved)

        calibrator_update = self.calibrator.update
        controller_observe = self.controller.observe
        for position in order:
            score = float(scores[position])
            calibrator_update(score, partitions[position])
            center = float(y_hat[position])
            lower = float(lower_values[position])
            upper = float(upper_values[position])
            prediction = IntervalPrediction(
                center=center,
                lower=lower,
                upper=upper,
                radius=max(
                    abs(center - lower) if not np.isnan(lower) else 0.0,
                    abs(upper - center) if not np.isnan(upper) else 0.0,
                ),
                alpha=float(alpha_values[position]),
            )
            controller_observe(float(y_true[position]), prediction, int(horizons[position]))

        observed.loc[resolved_mask, NONCONFORMITY_SCORE] = scores
        return observed

    def _observe_cumulative(self, observed: pd.DataFrame) -> pd.DataFrame:
        """Score complete cumulative windows; leave incomplete ones pending.

        Precise *inner* rule: within each ``(uid, model, origin)`` window only
        horizons ``h <= protection_period`` are scored, and the window is ready
        only once all of them are present (``len >= protection_period``) with a
        non-null actual; the cumulative sum is undefined until then. The engine's
        :class:`~calibre.execution.backend.BackendEngine` deferral enforces this
        same rule on the streaming-ledger path, while
        :func:`~calibre.execution.decision_loop.observe_cumulative` feeds it
        through a conservative whole-group outer gate — keep the three aligned.
        """
        if self.config.protection_period is None:
            raise ValueError("cumulative mode requires config.protection_period")
        protection_period = int(self.config.protection_period)
        group_cols = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]

        # Completeness rule mirrored by BackendEngine's cumulative-window
        # deferral (calibre/execution/backend.py) — keep the two in sync.
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
        calibrator_state: dict[str, Any] = self.calibrator.get_state()
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

    def get_resume_state(self) -> dict[str, Any]:
        calibrator_state: dict[str, Any] = self.calibrator.get_state()
        return {
            "method": self.method_name,
            "mode": self.config.mode,
            "coverage": self.config.coverage,
            "calibration_window": self.config.calibration_window,
            "gamma": self.config.gamma,
            "quantile_rule": self.config.resolved_quantile_rule,
            "protection_period": self.config.protection_period,
            "issued_count": self._issued_count,
            "controller": self._controller_resume_state(),
            "calibrator": calibrator_state,
        }

    @property
    def partition_keys(self) -> tuple[str, ...]:
        calibrator_state: dict[str, Any] = self.calibrator.get_state()
        score_history = calibrator_state.get("score_history", {})
        if not isinstance(score_history, dict):
            return ()
        return tuple(str(partition) for partition in score_history)

    def get_partition_states(self) -> dict[str, dict[str, Any]]:
        state = self.get_resume_state()
        calibrator_state = state.get("calibrator", {})
        score_history = calibrator_state.get("score_history", {})
        if not isinstance(score_history, dict) or not score_history:
            return {}

        partition_states: dict[str, dict[str, Any]] = {}
        for partition, scores in score_history.items():
            partition_key = str(partition)
            partition_state = deepcopy(state)
            partition_state["partition"] = partition_key
            partition_state["calibrator"]["score_history"] = {partition_key: scores}
            partition_states[partition_key] = partition_state
        return partition_states

    @classmethod
    def from_partition_states(
        cls,
        config: SymmetricIntervalConfig,
        partition_states: Mapping[str, dict[str, Any]],
    ) -> SymmetricIntervalRuntime:
        if not partition_states:
            return cls(config)

        merged_state: dict[str, Any] | None = None
        score_history: dict[str, Any] = {}
        max_issued_count = 0
        for fallback_partition, state in partition_states.items():
            if merged_state is None:
                merged_state = deepcopy(state)
            issued_count = int(state.get("issued_count", 0))
            if issued_count >= max_issued_count:
                max_issued_count = issued_count
                merged_state["controller"] = deepcopy(state.get("controller", {}))
                merged_state["method"] = state.get("method", config.method)

            calibrator_state = state.get("calibrator", {})
            partition_scores = calibrator_state.get("score_history", {})
            if isinstance(partition_scores, dict) and partition_scores:
                for partition, scores in partition_scores.items():
                    score_history[str(partition)] = scores
            else:
                partition = str(state.get("partition", fallback_partition))
                score_history[partition] = []

        assert merged_state is not None
        merged_state["issued_count"] = max_issued_count
        merged_state.setdefault("calibrator", {})
        merged_state["calibrator"]["score_history"] = score_history
        return cls.from_state(config, merged_state)


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
    """Build a :class:`SymmetricIntervalRuntime` from a validated config."""
    score, calibrator, controller = _components_from_config(config)
    return SymmetricIntervalRuntime(
        config=config,
        score=score,
        calibrator=calibrator,
        controller=controller,
        method_name=config.method,
    )
