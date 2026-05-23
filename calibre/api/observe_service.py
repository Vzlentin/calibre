"""Observe pipeline: merge actuals into the last calibrated frame and update conformal state.

Pure functions taking a ``LifecycleStore`` and a runtime factory as
dependencies; the HTTP layer wires those in. Conformal runtime construction
lives here because the rebuild path (state-from-store) is observe's only
real caller.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from calibre.api.lifecycle import FitRecord, LifecycleStore
from calibre.cli.config import ConformalConfig
from calibre.conformal.runtime import SymmetricIntervalRuntime, build_symmetric_interval_runtime
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.execution.decision_loop import observe_cumulative, observe_per_horizon

logger = logging.getLogger(__name__)

RuntimeFactory = Callable[[FitRecord], SymmetricIntervalRuntime]


def conformal_config_from_dict(payload: dict) -> ConformalConfig:
    return ConformalConfig(
        method=payload["method"],
        coverage=float(payload.get("coverage", 0.9)),
        calibration_window=int(payload.get("calibration_window", 100)),
        gamma=float(payload.get("gamma", 0.05)),
        mode=payload.get("mode", "perhorizon"),
        protection_period=payload.get("protection_period"),
    )


def runtime_for_session(record: FitRecord, store: LifecycleStore) -> SymmetricIntervalRuntime:
    assert record.conformal_config is not None
    runtime_config = conformal_config_from_dict(record.conformal_config).to_runtime_config()
    saved = store.get_conformal_state(record.session_id)
    if saved:
        return SymmetricIntervalRuntime.from_partition_states(runtime_config, saved)
    return build_symmetric_interval_runtime(runtime_config)


def _frame_from_records(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    if DS in frame.columns:
        frame[DS] = pd.to_datetime(frame[DS])
    if "forecast_origin" in frame.columns:
        frame["forecast_origin"] = pd.to_datetime(frame["forecast_origin"])
    return frame


def run_observe(
    store: LifecycleStore,
    session_id: str,
    actual_records: list[dict],
    runtime_factory: RuntimeFactory,
) -> None:
    """Apply ``actual_records`` to the session's last calibrated frame.

    Logs a warning and returns without mutating state on any of the
    documented skip conditions (no fit, no conformal config, no calibrated
    frame, missing actuals, no interval columns, no rows resolved).
    """
    record = store.first_fit_for_session(session_id)
    if record is None:
        logger.warning("observe skipped: no fit for session", extra={"session_id": session_id})
        return
    if record.conformal_config is None:
        logger.warning(
            "observe skipped: session has no conformal config",
            extra={"session_id": session_id},
        )
        return
    if record.last_calibrated is None or record.last_calibrated.empty:
        logger.warning(
            "observe skipped: no calibrated frame on session (call /calibrate first)",
            extra={"session_id": session_id},
        )
        return

    actuals = _frame_from_records(actual_records)
    if actuals.empty or UNIQUE_ID not in actuals.columns or DS not in actuals.columns:
        logger.warning(
            "observe skipped: actuals empty or missing unique_id/ds",
            extra={"session_id": session_id, "rows": len(actuals)},
        )
        return

    runtime = runtime_factory(record)
    lower_col, upper_col = runtime.interval_columns
    calibrated = record.last_calibrated.copy()
    if lower_col not in calibrated.columns or upper_col not in calibrated.columns:
        logger.warning(
            "observe skipped: calibrated frame missing interval columns",
            extra={"session_id": session_id, "expected": [lower_col, upper_col]},
        )
        return

    merged = calibrated.merge(
        actuals[[UNIQUE_ID, DS, Y]].rename(columns={Y: "_y_actual"}),
        on=[UNIQUE_ID, DS],
        how="left",
    )
    if Y in merged.columns:
        merged[Y] = merged["_y_actual"].combine_first(merged[Y])
    else:
        merged[Y] = merged["_y_actual"]
    merged = merged.drop(columns=["_y_actual"])

    actuals_lookup = actuals.dropna(subset=[Y]).set_index([UNIQUE_ID, DS])[Y]
    actuals_lookup = actuals_lookup[~actuals_lookup.index.duplicated(keep="last")]
    mode = getattr(runtime, "mode", "perhorizon")
    if mode == "cumulative":
        remaining = observe_cumulative(runtime, [merged], actuals_lookup)
    else:
        remaining = observe_per_horizon(runtime, [merged], actuals_lookup, lower_col, upper_col)
    pending_rows = sum(len(frame) for frame in remaining)
    if pending_rows >= len(merged):
        logger.warning(
            "observe skipped: no rows resolved after merging actuals",
            extra={"session_id": session_id, "calibrated_rows": len(calibrated)},
        )
        return
    store.upsert_conformal_state(session_id, runtime.get_partition_states())
