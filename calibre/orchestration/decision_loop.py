"""Rolling decision-loop orchestrator for Calibre benchmarks.

Encodes the correct ``ConformalRuntime.observe()`` dispatch (lessons.md §40)
so individual benchmarks don't have to re-implement the pending-forecast
bookkeeping or the per-horizon vs cumulative split logic.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from calibre.conformal.runtime import ConformalRuntime
from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y,
)
from calibre.engine.backend import BackendEngine
from calibre.tasks.forecast_task import ForecastTask


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

    A row is ready when it has a non-null actual *and* both interval columns
    present. Implements the dispatch rule from lessons.md §40 so benchmarks
    don't have to re-derive it.
    """
    still_pending: list[pd.DataFrame] = []
    for frame in pending:
        if lower_col not in frame.columns or upper_col not in frame.columns:
            still_pending.append(frame)
            continue
        updated = _fill_actuals(frame, actuals_lookup)
        ready = updated[Y].notna() & updated[lower_col].notna() & updated[upper_col].notna()
        to_observe, unresolved = updated[ready], updated[~ready]
        if not to_observe.empty:
            with contextlib.suppress(ValueError):
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

    A window is ready when every row in its ``(uid, model, origin)`` group
    has a non-null actual. Implements the dispatch rule from lessons.md §40
    for cumulative mode.
    """
    still_pending: list[pd.DataFrame] = []
    for frame in pending:
        updated = _fill_actuals(frame, actuals_lookup)
        group_keys = [UNIQUE_ID, MODEL_NAME, FORECAST_ORIGIN]
        grouped_y = updated.groupby(group_keys, sort=False)[Y]
        window_complete = grouped_y.transform("count").eq(grouped_y.transform("size"))
        to_observe, unresolved = updated[window_complete], updated[~window_complete]
        if not to_observe.empty:
            with contextlib.suppress(ValueError):
                runtime.observe(to_observe)
        if not unresolved.empty:
            still_pending.append(unresolved)
    return still_pending


class DecisionLoop:
    """Rolling inventory decision loop.

    For each decision round:

    1. Build ``ForecastTask`` objects via ``build_round_tasks(round_num)``,
       which also returns the origin timestamp and the actuals DataFrame
       for the engine's ledger scoring.
    2. Execute the engine, optionally ensemble, optionally apply conformal.
    3. Compute orders via ``policy(frame)``.
    4. Fetch realised demand via ``get_actuals(round_num)``.
    5. Step the simulator.
    6. Feed resolved actuals back to the conformal runtime via ``observe_fn``.
    7. Fire ``config.on_round`` with the :class:`RoundResult`.

    Delivery rounds (``config.n_delivery_rounds``) follow: they step the
    simulator with zero orders but no forecasting.

    Args:
        engine: Configured :class:`BackendEngine`.
        simulator: Simulator exposing
            ``step(period, orders, actual_demand) → Any``.
        build_round_tasks: ``round_num → (tasks, origin, round_actuals_df)``.
            ``round_actuals_df`` is passed to ``engine.execute`` for ledger scoring.
        policy: ``forecast_frame → {uid: order_qty}``.
            Receives the conformal frame when ``runtime`` is set, otherwise the
            raw ledger frame (or ensemble output).
        get_actuals: ``round_num → {uid: realised_demand}``.  Called for both
            decision and delivery rounds — callers are responsible for the
            correct offset to the underlying data files.
        config: Loop-level settings (n_rounds, n_delivery_rounds, on_round).
        runtime: Optional :class:`ConformalRuntime`.  When set, its ``apply``
            is called after (optional) ensembling and the output is appended to
            the pending-forecast queue for later ``observe_fn`` calls.
        ensemble: Optional ``ledger_df → frame`` aggregation applied before
            conformal.  Typically ``ensemble_median``.
        observe_fn: ``(runtime, pending, actuals_lookup) → new_pending``.
            Use :func:`observe_per_horizon` or :func:`observe_cumulative`
            (possibly via ``functools.partial``) to match the conformal mode.
            Only called when ``runtime`` is set and ``actual_demand`` is
            non-empty.
    """

    def __init__(
        self,
        *,
        engine: BackendEngine,
        simulator: Any,
        build_round_tasks: Callable[[int], tuple[list[ForecastTask], pd.Timestamp, pd.DataFrame]],
        policy: Callable[[pd.DataFrame], dict[str, float]],
        get_actuals: Callable[[int], dict[str, float]],
        config: DecisionLoopConfig,
        runtime: ConformalRuntime | None = None,
        ensemble: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
        observe_fn: Callable[[ConformalRuntime, list[pd.DataFrame], pd.Series], list[pd.DataFrame]]
        | None = None,
    ) -> None:
        self._engine = engine
        self._simulator = simulator
        self._build_round_tasks = build_round_tasks
        self._policy = policy
        self._get_actuals = get_actuals
        self._config = config
        self._runtime = runtime
        self._ensemble = ensemble
        self._observe_fn = observe_fn

    def run(self) -> list[RoundResult]:
        """Execute all decision and delivery rounds; return per-round results."""
        results: list[RoundResult] = []
        pending: list[pd.DataFrame] = []
        actuals_cache: dict[tuple[str, pd.Timestamp], float] = {}
        freq_offset = pd.tseries.frequencies.to_offset(self._engine.freq)

        lower_col = upper_col = ""
        if self._runtime is not None:
            lower_col, upper_col = self._runtime.config.interval_columns

        for round_num in range(1, self._config.n_rounds + 1):
            tasks, origin, round_actuals = self._build_round_tasks(round_num)
            result = self._engine.execute(tasks, actuals=round_actuals, origins=[origin])
            ledger_df = result.ledger.to_df()

            frame = self._ensemble(ledger_df) if self._ensemble else ledger_df

            conformal_frame: pd.DataFrame | None = None
            if self._runtime is not None:
                conformal_frame = self._runtime.apply(frame)
                if lower_col in conformal_frame.columns and upper_col in conformal_frame.columns:
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

            if (
                self._observe_fn is not None
                and actual_demand
                and freq_offset is not None
                and self._runtime is not None
            ):
                actuals_ds = origin + freq_offset
                for uid, demand in actual_demand.items():
                    actuals_cache[(uid, actuals_ds)] = demand
                lookup = pd.Series(actuals_cache, dtype=float)
                if not lookup.empty:
                    lookup.index = pd.MultiIndex.from_tuples(lookup.index)
                pending = self._observe_fn(self._runtime, pending, lookup)

        for week_offset in range(1, self._config.n_delivery_rounds + 1):
            delivery_num = self._config.n_rounds + week_offset
            actual_demand = self._get_actuals(delivery_num)
            self._simulator.step(
                delivery_num,
                orders={uid: 0.0 for uid in actual_demand},
                actual_demand=actual_demand,
            )

        return results
