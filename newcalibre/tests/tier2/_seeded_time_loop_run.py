"""Run one deterministic, seed-sensitive rolling-loop fixture in a child process."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    MODEL_NAME,
    OBSERVED_VALUE,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    TIMESTAMP,
    ActualsSemantics,
    Calendar,
    CostStructure,
    DecisionTiming,
    FittedValues,
    ForecastTask,
    HierarchyIndex,
    InventoryPosition,
    Panel,
    Scope,
    SessionIdentity,
    StockoutRule,
    TargetSupport,
    target_timestamp,
)
from newcalibre.engine import (
    CommitReceipt,
    Engine,
    InMemoryActualsSource,
    InMemoryArtifactStore,
    InMemoryCalibrationStateStore,
    InMemoryLedgerSink,
    InMemoryPanelSource,
    InProcessDispatch,
    OrderProposal,
    OrderRequest,
    OriginCommit,
    PhaseError,
)
from newcalibre.engine.time_loop import TimeLoop, TimeLoopRequest, TimeLoopResult
from newcalibre.forecasting import AdapterCapability, AdapterCapabilityError
from newcalibre.ledger import ForecastRow, OrderRow, SettlementRecord

_CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
_ORIGINS = (
    pd.Timestamp("2026-01-10"),
    pd.Timestamp("2026-01-12"),
    pd.Timestamp("2026-01-14"),
)
_FAIL_ORIGIN = _ORIGINS[1]
_MODEL_NAME = "tier2-seeded"
_SERIES = ("alpha", "zeta")
_TIMING = DecisionTiming(lead_time=2, review_period=2)


class SeededFixtureAdapter:
    """Emit integer-valued forecasts whose values visibly depend on the run seed."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        raw_seed = model_config.get("seed")
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise ValueError("tier-2 fixture seed must be an integer")
        self._seed = raw_seed
        self._points: dict[str, float] | None = None

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        return frozenset()

    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None:
        if collect_fitted_values:
            raise AdapterCapabilityError("tier-2 fixture has no fitted-values capability")
        history = task.history
        points: dict[str, float] = {}
        for series_key in task.series_keys:
            series_history = history[history[SERIES_KEY] == series_key]
            latest = float(series_history[OBSERVED_VALUE].iloc[-1])
            seed_offset = (self._seed + sum(series_key.encode("utf-8"))) % 7
            points[series_key] = latest + float(seed_offset + 1)
        self._points = points

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._points is None:
            raise RuntimeError("tier-2 fixture predict requires fit")
        rows = [
            {
                SERIES_KEY: series_key,
                TARGET_TIMESTAMP: target_timestamp(task.origin, step, calendar=task.calendar),
                ACTUAL_VALUE: float("nan"),
                POINT_FORECAST: self._points[series_key] + float(step - 1),
                HORIZON_STEP: step,
                ORIGIN: task.origin,
                MODEL_NAME: _MODEL_NAME,
            }
            for series_key in task.series_keys
            for step in range(1, task.horizon + 1)
        ]
        frame = pd.DataFrame.from_records(rows)
        frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
        frame[MODEL_NAME] = frame[MODEL_NAME].astype("string")
        frame[ACTUAL_VALUE] = frame[ACTUAL_VALUE].astype("float64")
        frame[POINT_FORECAST] = frame[POINT_FORECAST].astype("float64")
        frame[HORIZON_STEP] = frame[HORIZON_STEP].astype("int64")
        return frame

    def fitted_values(self, task: ForecastTask) -> FittedValues:
        raise AdapterCapabilityError("tier-2 fixture has no fitted-values capability")

    def dump_state(self) -> bytes:
        raise AdapterCapabilityError("tier-2 fixture has no persistence capability")

    def load_state(self, state: bytes) -> None:
        raise AdapterCapabilityError("tier-2 fixture has no persistence capability")

    def update(self, task: ForecastTask) -> None:
        raise AdapterCapabilityError("tier-2 fixture has no incremental-update capability")


class InterruptingLedgerSink(InMemoryLedgerSink):
    """Interrupt one selected commit immediately before or after publication."""

    def __init__(
        self,
        *,
        session: SessionIdentity,
        calendar: Calendar,
        fail_origin: pd.Timestamp,
        commit_before_failure: bool,
    ) -> None:
        super().__init__(session=session, calendar=calendar)
        self._fail_origin = fail_origin
        self._commit_before_failure = commit_before_failure
        self._failed = False

    def commit(self, write: OriginCommit) -> CommitReceipt:
        if write.origin != self._fail_origin or self._failed:
            return super().commit(write)
        self._failed = True
        if not self._commit_before_failure:
            raise RuntimeError("tier-2 fixture interrupted before the durable commit")
        super().commit(write)
        raise RuntimeError("tier-2 fixture lost the durable commit response")


def _panel() -> Panel:
    timestamps = pd.date_range("2026-01-01", periods=20, freq="D")
    rows = [
        {
            SERIES_KEY: series_key,
            TIMESTAMP: timestamp,
            OBSERVED_VALUE: float((index % 5) + series_index + 1),
        }
        for series_index, series_key in enumerate(reversed(_SERIES))
        for index, timestamp in enumerate(timestamps)
    ]
    frame = pd.DataFrame.from_records(rows)
    frame[SERIES_KEY] = frame[SERIES_KEY].astype("string")
    frame[OBSERVED_VALUE] = frame[OBSERVED_VALUE].astype("float64")
    return Panel.from_frame(frame, calendar=_CALENDAR, target_support=TargetSupport.REAL)


def _session(seed: int) -> SessionIdentity:
    return SessionIdentity.derive(
        tenant="tier-2",
        series_keys=_SERIES,
        calendar=_CALENDAR,
        horizon=_TIMING.protection_period,
        model_config={"backend": _MODEL_NAME, "seed": seed},
        conformal_config={
            "method": "split-per-step",
            "coverage": 0.5,
            "calibration_window": 20,
        },
        ordering_policy={"name": "newsvendor"},
        decision_series_keys=_SERIES,
        cost_structure=CostStructure(underage=3.0, overage=1.0, holding=0.5, shortage=4.0),
        decision_timing=_TIMING,
        stockout_rule=StockoutRule.LOST_SALES,
    )


def _order(request: OrderRequest) -> tuple[OrderProposal, ...]:
    assert request.timing == _TIMING
    frame = request.forecasts.frame
    orders: list[OrderProposal] = []
    for series_key in _SERIES:
        series_frame = frame[frame[SERIES_KEY] == series_key].sort_values(HORIZON_STEP)
        target = float(series_frame[POINT_FORECAST].iloc[-1])
        position = request.inventory_positions[series_key]
        orders.append(
            OrderProposal(
                series_key=series_key,
                model_name=_MODEL_NAME,
                quantity=max(0.0, target - position.value),
            )
        )
    return tuple(orders)


def _engine(
    *,
    panel: Panel,
    actuals: InMemoryActualsSource,
    sink: InMemoryLedgerSink,
    states: InMemoryCalibrationStateStore,
) -> Engine:
    return Engine(
        panel_source=InMemoryPanelSource(panel),
        actuals_source=actuals,
        artifact_store=InMemoryArtifactStore(),
        calibration_state_store=states,
        ledger_sink=sink,
        dispatch_backend=InProcessDispatch(),
        hierarchy=HierarchyIndex.flat(panel.series_keys),
        adapter_resolver=SeededFixtureAdapter,
        orderer=_order,
    )


def _request(*, session: SessionIdentity) -> TimeLoopRequest:
    return TimeLoopRequest(
        session=session,
        origins=_ORIGINS,
        settlement_end=_CALENDAR.advance(_ORIGINS[-1], _TIMING.lead_time),
        scope=Scope.GLOBAL,
        initial_inventory_positions={
            series_key: InventoryPosition(on_hand=8.0, on_order=0.0, backorders=0.0)
            for series_key in _SERIES
        },
        actuals_semantics=ActualsSemantics.DEMAND,
    )


def _run_uninterrupted(
    seed: int,
) -> tuple[InMemoryLedgerSink, TimeLoopResult, InMemoryCalibrationStateStore]:
    panel = _panel()
    session = _session(seed)
    actuals = InMemoryActualsSource(
        panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    sink = InMemoryLedgerSink(session=session, calendar=_CALENDAR)
    states = InMemoryCalibrationStateStore()
    result = TimeLoop(
        engine=_engine(panel=panel, actuals=actuals, sink=sink, states=states),
        actuals_source=actuals,
        ledger_sink=sink,
        request=_request(session=session),
    ).run()
    assert len(states.snapshot(session)) == 2
    return sink, result, states


def _run_resumed(
    seed: int,
    *,
    commit_before_failure: bool,
) -> tuple[InMemoryLedgerSink, TimeLoopResult, InMemoryCalibrationStateStore]:
    panel = _panel()
    session = _session(seed)
    actuals = InMemoryActualsSource(
        panel,
        actuals_semantics=ActualsSemantics.DEMAND,
    )
    sink = InterruptingLedgerSink(
        session=session,
        calendar=_CALENDAR,
        fail_origin=_FAIL_ORIGIN,
        commit_before_failure=commit_before_failure,
    )
    interrupted_states = InMemoryCalibrationStateStore()
    interrupted = TimeLoop(
        engine=_engine(
            panel=panel,
            actuals=actuals,
            sink=sink,
            states=interrupted_states,
        ),
        actuals_source=actuals,
        ledger_sink=sink,
        request=_request(session=session),
    )
    try:
        interrupted.run()
    except PhaseError as error:
        expected_error = (
            "lost the durable commit response"
            if commit_before_failure
            else "interrupted before the durable commit"
        )
        assert expected_error in str(error)
    else:
        raise AssertionError("tier-2 interruption fixture did not interrupt")
    assert (sink.receipt(_FAIL_ORIGIN) is not None) is commit_before_failure
    assert len(interrupted_states.snapshot(session)) == 2

    result = TimeLoop(
        engine=_engine(
            panel=panel,
            actuals=actuals,
            sink=sink,
            states=interrupted_states,
        ),
        actuals_source=actuals,
        ledger_sink=sink,
        request=_request(session=session),
    ).run()
    assert len(interrupted_states.snapshot(session)) == 2
    return sink, result, interrupted_states


def _float(value: float | None) -> str | None:
    return None if value is None else value.hex()


def _timestamp(value: pd.Timestamp) -> str:
    return f"{value.isoformat()}[{value.unit}]"


def _forecast_payload(row: ForecastRow) -> dict[str, object]:
    values: list[tuple[str, object]] = []
    for name, value in row.values.items():
        if isinstance(value, pd.Timestamp):
            normalized: object = _timestamp(value)
        elif isinstance(value, float):
            normalized = _float(value)
        else:
            normalized = value
        values.append((name, normalized))
    return {"values": values, "issuances": repr(tuple(row.issuances.items()))}


def _order_payload(row: OrderRow) -> dict[str, object]:
    return {
        "arrival_period": _timestamp(row.arrival_period),
        "model_name": row.model_name,
        "origin": _timestamp(row.origin),
        "quantity": _float(row.quantity),
        "series_key": row.series_key,
    }


def _settlement_payload(row: SettlementRecord) -> dict[str, object]:
    transition = row.transition
    return {
        "actuals_semantics": row.actuals_semantics.value,
        "arrivals": _float(row.arrivals),
        "holding": tuple(
            _float(value) for value in (row.holding.rate, row.holding.basis, row.holding.amount)
        ),
        "inventory": tuple(
            _float(value)
            for value in (
                row.inventory_position.on_hand,
                row.inventory_position.on_order,
                row.inventory_position.backorders,
            )
        ),
        "period": _timestamp(row.period),
        "series_key": row.series_key,
        "shortage": tuple(
            _float(value) for value in (row.shortage.rate, row.shortage.basis, row.shortage.amount)
        ),
        "transition": {
            "closing_backorders": _float(transition.closing_backorders),
            "closing_on_hand": _float(transition.closing_on_hand),
            "demand": _float(transition.demand),
            "fulfilled_demand": _float(transition.fulfilled_demand),
            "rule": transition.rule.value,
            "unmet_demand": _float(transition.unmet_demand),
        },
    }


def _ledger_bytes(
    sink: InMemoryLedgerSink,
    result: TimeLoopResult,
    states: InMemoryCalibrationStateStore,
) -> bytes:
    payload = {
        "annotations": [
            {
                "advanced": value.advanced_delivered_score,
                "cause": value.exclusion_cause,
                "key": repr(value.forecast_key),
                "score": _float(value.score),
            }
            for value in sink.observe_annotations
        ],
        "conformal_states": [
            (label, value.hex()) for label, value in sorted(states.snapshot(sink.session).items())
        ],
        "decision_origins": [_timestamp(origin) for origin in result.decision_origins],
        "forecasts": [_forecast_payload(row) for row in sink.forecasts],
        "observed_history": [
            {
                "availability_bound": _float(value.availability_bound),
                "censoring": (
                    None if value.censoring_assertion is None else value.censoring_assertion.value
                ),
                "recorded_value": _float(float(value.recorded_value)),
                "series_key": value.series_key,
                "timestamp": _timestamp(value.timestamp),
            }
            for value in sink.observed_history
        ],
        "orders": [_order_payload(row) for row in sink.orders],
        "receipts": [
            {
                "digest": receipt.digest,
                "origin": _timestamp(receipt.origin),
                "settlement_periods": [_timestamp(period) for period in receipt.settlement_periods],
                "state_updates": [
                    (partition, value.hex())
                    for partition, value in sorted(receipt.state_updates.items())
                ],
            }
            for receipt in result.receipts
        ],
        "pending_observations": [
            {
                "actual": (
                    None if value.resolution is None else _float(float(value.resolution.actual))
                ),
                "forecast_key": repr(value.forecast_key),
                "target_timestamp": _timestamp(value.target_timestamp),
            }
            for value in sink.pending_observations
        ],
        "schema": "newcalibre.tier2-rolling-ledger/v2",
        "session": sink.session.value,
        "settlement_periods": [_timestamp(period) for period in result.settlement_periods],
        "settlements": [_settlement_payload(row) for row in sink.settlements],
    }
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "uninterrupted",
            "resumed-before-commit",
            "resumed-after-commit",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected fixture and write its canonical ledger bytes."""
    args = _parse_args(argv)
    if args.mode == "uninterrupted":
        sink, result, states = _run_uninterrupted(args.seed)
    else:
        sink, result, states = _run_resumed(
            args.seed,
            commit_before_failure=args.mode == "resumed-after-commit",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_ledger_bytes(sink, result, states))


if __name__ == "__main__":
    main()
