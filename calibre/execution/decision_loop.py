"""Rolling decision-loop orchestrator for Calibre benchmarks.

Encodes the correct ``ConformalRuntime.observe()`` dispatch so individual
benchmarks don't have to re-implement the pending-forecast bookkeeping or the
per-horizon vs cumulative split logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from calibre.conformal.runtime import ConformalRuntime
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    Y,
)
from calibre.core.forecast_task import TaskGroups
from calibre.execution.backend import BackendEngine
from calibre.storage.postgres import PendingObservationRepo


@dataclass
class RoundResult:
    """Per-round outputs from a :class:`DecisionLoop` run."""

    round_num: int
    origin: pd.Timestamp
    ledger: pd.DataFrame
    conformal_frame: pd.DataFrame | None
    orders: dict[str, float]
    actual_demand: dict[str, float]


@dataclass
class DecisionLoopConfig:
    """Configuration for :class:`DecisionLoop`."""

    n_rounds: int
    n_delivery_rounds: int = 0
    on_round: Callable[[RoundResult], None] | None = field(default=None, compare=False)


def build_actuals_lookup(actuals: pd.DataFrame) -> pd.Series:
    """Build the ``(unique_id, ds) -> y`` observe lookup from an actuals frame.

    The canonical constructor for the Series that :func:`_fill_actuals` and the
    ``observe_*`` functions consume: ``(str, Timestamp)`` keys, dropping rows
    with no actual ``y`` to record. Vectorized (no per-row iteration); duplicate
    keys keep the last row. An empty (or all-NaN-y) frame yields an empty float
    Series.

    Args:
        actuals: Frame carrying ``UNIQUE_ID``/``DS``/``Y`` columns; may hold
            NaN-y or duplicate keys.

    Returns:
        A ``(object, datetime64[ns]) -> float`` Series for order-independent
        ``reindex`` lookup.
    """
    usable = actuals[actuals[Y].notna()]
    if usable.empty:
        return pd.Series(dtype=float)
    keys = pd.MultiIndex.from_arrays([usable[UNIQUE_ID].astype(str), pd.to_datetime(usable[DS])])
    lookup = pd.Series(usable[Y].astype(float).to_numpy(), index=keys, dtype=float)
    return lookup[~keys.duplicated(keep="last")]


def _fill_actuals(frame: pd.DataFrame, lookup: pd.Series) -> pd.DataFrame:
    """Fill NaN y values from a ``(uid, ds) → float`` lookup Series."""
    if frame.empty or not frame[Y].isna().any():
        return frame
    keys = pd.MultiIndex.from_arrays([frame[UNIQUE_ID].values, frame[DS].values])
    filled = lookup.reindex(keys).to_numpy()
    result = frame.copy()
    missing = result[Y].isna().to_numpy()
    result.loc[missing, Y] = filled[missing]
    return result


def observe_per_horizon(
    runtime: ConformalRuntime,
    pending: list[pd.DataFrame],
    actuals_lookup: pd.Series,
    lower_col: str,
    upper_col: str,
) -> list[pd.DataFrame]:
    """Observe resolved per-horizon rows; return the still-pending frames.

    A row is ready when it has a non-null actual *and* a non-null point
    forecast (``Y_HAT``) — the runtime's own readiness rule
    (``runtime._observe_perhorizon``). Bounds may be NaN: a cold non-ACI
    runtime emits NaN bounds until its first score, and filtering those rows
    out would deadlock its calibration. The structural interval-column guard
    lives in ``runtime.observe``, which raises if the columns are absent.
    """
    still_pending: list[pd.DataFrame] = []
    for frame in pending:
        if lower_col not in frame.columns or upper_col not in frame.columns:
            still_pending.append(frame)
            continue
        updated = _fill_actuals(frame, actuals_lookup)
        ready = updated[Y].notna() & updated[Y_HAT].notna()
        to_observe, unresolved = updated[ready], updated[~ready]
        if not to_observe.empty:
            runtime.observe(to_observe)
        if not unresolved.empty:
            still_pending.append(unresolved)
    return still_pending


def observe_cumulative(
    runtime: ConformalRuntime,
    pending: list[pd.DataFrame],
    actuals_lookup: pd.Series,
) -> list[pd.DataFrame]:
    """Observe complete windows for cumulative-mode conformal; return pending.

    Conservative *outer* gate: a ``(uid, model, origin)`` group is handed to
    :meth:`~calibre.conformal.runtime.ConformalRuntime.observe` only once every
    row in it has a non-null actual. The runtime then applies the precise
    *inner* rule (score only ``h <= protection_period``, ready once all of those
    horizons are present and non-null), and the engine's
    :class:`~calibre.execution.backend.BackendEngine` deferral enforces that same
    inner rule on the streaming-ledger path. For contiguous ``h = 1..horizon``
    the outer gate is a strict subset of the inner rule, so the three agree on
    which windows score — keep them aligned.
    """
    still_pending: list[pd.DataFrame] = []
    for frame in pending:
        updated = _fill_actuals(frame, actuals_lookup)
        group_keys = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]
        grouped_y = updated.groupby(group_keys, sort=False)[Y]
        window_complete = grouped_y.transform("count").eq(grouped_y.transform("size"))
        to_observe, unresolved = updated[window_complete], updated[~window_complete]
        if not to_observe.empty:
            runtime.observe(to_observe)
        if not unresolved.empty:
            still_pending.append(unresolved)
    return still_pending


def observe_pending(
    runtime: ConformalRuntime,
    pending: list[pd.DataFrame],
    actuals_lookup: pd.Series,
) -> list[pd.DataFrame]:
    """Observe resolved pending frames; return the still-pending frames.

    Single mode-keyed entry point for driver-side observe: keys on
    ``runtime.mode`` and delegates to :func:`observe_cumulative` or
    :func:`observe_per_horizon` (deriving the bound columns from
    ``runtime.interval_columns``). ``pending`` is passed through untouched —
    no re-grouping, re-sorting, or pre-processing — so the helpers' pending
    bookkeeping stays byte-identical (CRC recency weighting depends on it).
    """
    if runtime.mode == "cumulative":
        return observe_cumulative(runtime, pending, actuals_lookup)
    lower_col, upper_col = runtime.interval_columns
    return observe_per_horizon(runtime, pending, actuals_lookup, lower_col, upper_col)


class DecisionLoop:
    """Rolling inventory decision loop.

    For each decision round:

    1. Build a ``TaskGroups`` partition via ``build_round_tasks(round_num)``,
       which also returns the origin timestamp and the actuals DataFrame
       for the engine's ledger scoring.
    2. Execute the engine, optionally ensemble, optionally apply conformal.
    3. Compute orders via ``policy(frame)``.
    4. Fetch realised demand via ``get_actuals(round_num)``.
    5. Step the simulator.
    6. Feed resolved actuals back to the conformal runtime via
       :func:`observe_pending` (mode-keyed on ``runtime.mode``).
    7. Fire ``config.on_round`` with the :class:`RoundResult`.

    Delivery rounds (``config.n_delivery_rounds``) follow: they step the
    simulator with zero orders but no forecasting.

    Args:
        engine: Configured :class:`BackendEngine`.
        simulator: Simulator exposing
            ``step(period, orders, actual_demand) → Any``.
        build_round_tasks: ``round_num → (task_groups, origin, round_actuals_df)``.
            ``round_actuals_df`` is passed to ``engine.execute`` for ledger scoring.
        policy: ``forecast_frame → {uid: order_qty}``.
            Receives the conformal frame when ``runtime`` is set, otherwise the
            raw ledger frame (or ensemble output).
        get_actuals: ``round_num → {uid: realised_demand}``.  Called for both
            decision and delivery rounds — callers are responsible for the
            correct offset to the underlying data files.  For delivery rounds
            the returned dict's keys are used directly as the zero-order set
            passed to the simulator, so it must cover all tracked SKUs.
        config: Loop-level settings (n_rounds, n_delivery_rounds, on_round).
        runtime: Optional conformal runtime.  When set, its ``apply`` is called
            after (optional) ensembling and the output is appended to the
            pending-forecast queue for later :func:`observe_pending` calls.
        ensemble: Optional ``ledger_df → frame`` aggregation applied before
            conformal.  Typically ``ensemble_median``.
    """

    def __init__(
        self,
        *,
        engine: BackendEngine,
        simulator: Any,
        build_round_tasks: Callable[[int], tuple[TaskGroups, pd.Timestamp, pd.DataFrame]],
        policy: Callable[[pd.DataFrame], dict[str, float]],
        get_actuals: Callable[[int], dict[str, float]],
        config: DecisionLoopConfig,
        runtime: ConformalRuntime | None = None,
        ensemble: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        pending_observation_repo: PendingObservationRepo | None = None,
        session_id: str | None = None,
    ) -> None:
        if pending_observation_repo is not None and session_id is None:
            raise ValueError("session_id is required with pending_observation_repo")
        self._engine = engine
        self._simulator = simulator
        self._build_round_tasks = build_round_tasks
        self._policy = policy
        self._get_actuals = get_actuals
        self._config = config
        self._runtime = runtime
        self._ensemble = ensemble
        self._pending_observation_repo = pending_observation_repo
        self._session_id = session_id

    def run(self) -> list[RoundResult]:
        """Execute all decision and delivery rounds; return per-round results."""
        results: list[RoundResult] = []
        pending: list[pd.DataFrame] = []
        actuals_cache: dict[tuple[str, pd.Timestamp], float] = {}
        freq_offset = pd.tseries.frequencies.to_offset(self._engine.freq)

        lower_col = upper_col = ""
        if self._runtime is not None:
            lower_col, upper_col = self._runtime.interval_columns

        for round_num in range(1, self._config.n_rounds + 1):
            tasks, origin, round_actuals = self._build_round_tasks(round_num)
            result = self._engine.execute(tasks, actuals=round_actuals, origins=[origin])
            ledger_df = result.ledger.to_df()

            frame = self._ensemble(ledger_df) if self._ensemble else ledger_df

            conformal_frame: pd.DataFrame | None = None
            if self._runtime is not None:
                conformal_frame = self._runtime.apply(frame)
                if lower_col in conformal_frame.columns and upper_col in conformal_frame.columns:
                    if self._pending_observation_repo is not None:
                        self._pending_observation_repo.upsert_frame(
                            self._session_id or "",
                            conformal_frame,
                            lower_col=lower_col,
                            upper_col=upper_col,
                        )
                    else:
                        pending.append(conformal_frame.copy())
                policy_frame = conformal_frame
            else:
                policy_frame = frame

            orders = self._policy(policy_frame)
            actual_demand = self._get_actuals(round_num)

            self._simulator.step(round_num, orders=orders, actual_demand=actual_demand)

            rr = RoundResult(
                round_num=round_num,
                origin=origin,
                ledger=ledger_df,
                conformal_frame=conformal_frame,
                orders=orders,
                actual_demand=actual_demand,
            )
            if self._config.on_round is not None:
                self._config.on_round(rr)
            results.append(rr)

            if self._runtime is not None and actual_demand and freq_offset is not None:
                actuals_ds = origin + freq_offset
                for uid, demand in actual_demand.items():
                    actuals_cache[(uid, actuals_ds)] = demand
                cache_frame = pd.DataFrame(
                    {
                        UNIQUE_ID: [uid for uid, _ in actuals_cache],
                        DS: [ds for _, ds in actuals_cache],
                        Y: list(actuals_cache.values()),
                    }
                )
                lookup = build_actuals_lookup(cache_frame)
                if self._pending_observation_repo is not None:
                    pending_frames = self._pending_observation_repo.to_frames(
                        self._session_id or "",
                        lower_col=lower_col,
                        upper_col=upper_col,
                    )
                    remaining = observe_pending(self._runtime, pending_frames, lookup)
                    self._pending_observation_repo.replace_session(
                        self._session_id or "",
                        remaining,
                        lower_col=lower_col,
                        upper_col=upper_col,
                    )
                else:
                    pending = observe_pending(self._runtime, pending, lookup)

        for week_offset in range(1, self._config.n_delivery_rounds + 1):
            delivery_num = self._config.n_rounds + week_offset
            actual_demand = self._get_actuals(delivery_num)
            self._simulator.step(
                delivery_num,
                orders={uid: 0.0 for uid in actual_demand},
                actual_demand=actual_demand,
            )

        return results
