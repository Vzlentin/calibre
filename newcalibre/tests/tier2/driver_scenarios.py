"""Build deterministic restartable worlds for two-driver equivalence tests."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from newcalibre.conformal import available_methods
from newcalibre.domain import (
    ACTUAL_VALUE,
    HORIZON_STEP,
    OBSERVED_VALUE,
    ORIGIN,
    POINT_FORECAST,
    SERIES_KEY,
    TARGET_TIMESTAMP,
    TIMESTAMP,
    ActualsSemantics,
    AppliedBinding,
    Calendar,
    CostStructure,
    DecisionEvidence,
    DecisionScope,
    DecisionScopeKind,
    DecisionTiming,
    EmissionScope,
    FittedValues,
    ForecastTask,
    GuaranteeClaim,
    GuaranteeDescriptor,
    GuaranteeType,
    HierarchyIndex,
    HistoryDelta,
    InventoryPosition,
    Panel,
    Scope,
    ScoredSeries,
    SessionIdentity,
    StockoutRule,
    TargetSupport,
    target_timestamp,
)
from newcalibre.domain import (
    MODEL_NAME as MODEL_NAME_COLUMN,
)
from newcalibre.engine import (
    ActualsEvent,
    Engine,
    EventDriver,
    InMemoryIndexedRunStore,
    InMemoryPanelSource,
    InProcessDispatch,
    OrderProposal,
    OrderRequest,
    OriginEvent,
    TimeLoop,
    TimeLoopRequest,
)
from newcalibre.forecasting import (
    AdapterCapability,
    AdapterCapabilityError,
    AdapterExecutionMode,
)
from newcalibre.observe import ActualRecord, ActualsSubmission

CALENDAR = Calendar("D", phase=pd.Timestamp("2026-01-01"))
SERIES_KEYS = ("alpha", "zeta")
MODEL_NAME = "driver-equivalence"
MODEL_CONFIG: Mapping[str, object] = {
    "backend": MODEL_NAME,
    "bias": 0.25,
}
TIMING = DecisionTiming(lead_time=1, review_period=2)
ORIGINS = tuple(pd.date_range("2026-01-05", periods=8, freq="D"))
SETTLEMENT_END = ORIGINS[-1]
INITIAL_INVENTORY: Mapping[str, InventoryPosition] = {
    "alpha": InventoryPosition(6.0, 0.0, 0.0),
    "zeta": InventoryPosition(7.0, 0.0, 0.0),
}
COSTS = CostStructure(
    underage=1.0,
    overage=1.0,
    holding=0.5,
    shortage=3.0,
)
EXPECTED_FINAL_INVENTORY: Mapping[str, InventoryPosition] = {
    "alpha": InventoryPosition(15.0, 0.0, 0.0),
    "zeta": InventoryPosition(14.0, 0.0, 0.0),
}
EXPECTED_BOOKED_COSTS = (41.0, 6.0, 47.0)
EXPECTED_LATE_FINAL_INVENTORY: Mapping[str, InventoryPosition] = {
    "alpha": InventoryPosition(17.0, 0.0, 0.0),
    "zeta": InventoryPosition(16.0, 0.0, 0.0),
}
EXPECTED_LATE_BOOKED_COSTS = (41.0, 42.0, 83.0)

RUNTIME_CONFIGURATIONS: Mapping[str, Mapping[str, object]] = {
    "sequential-adaptive-per-step": {
        "method": "sequential-adaptive-per-step",
        "coverage": 0.5,
        "calibration_window": 32,
        "partition_by": "global",
        "learning_rate": 0.0,
    },
    "split-per-step": {
        "method": "split-per-step",
        "coverage": 0.5,
        "calibration_window": 32,
        "partition_by": "global",
    },
    "split-window-sum": {
        "method": "split-window-sum",
        "coverage": 0.5,
        "calibration_window": 32,
        "partition_by": "global",
        "protection_period": TIMING.protection_period,
    },
    "weighted-per-step": {
        "method": "weighted-per-step",
        "coverage": 0.5,
        "calibration_window": 32,
        "partition_by": "global",
        "weight_decay": 1.0,
    },
}
RUNTIME_CASES = (None, *available_methods())


@dataclass(slots=True)
class DriverWorld:
    """Own one session's panel and transactional in-memory store."""

    runtime_name: str | None
    panel: Panel
    session: SessionIdentity
    store: InMemoryIndexedRunStore


@dataclass(frozen=True, slots=True)
class RuntimeWitness:
    """Expose positive conformal execution facts from one completed world."""

    delivered_scores: tuple[object, ...]
    state_before: tuple[tuple[str, bytes], ...]
    state_after: tuple[tuple[str, bytes], ...]
    finite_issuances: tuple[object, ...]


class DeterministicArtifactAdapter:
    """Forecast solely from retained history and persist canonical artifact bytes."""

    def __init__(self, model_config: Mapping[str, object]) -> None:
        if dict(model_config) != dict(MODEL_CONFIG):
            raise ValueError("driver-equivalence adapter configuration is invalid")
        self._points: dict[str, float] | None = None
        self._latest: dict[str, float] | None = None

    @property
    def execution_mode(self) -> AdapterExecutionMode:
        return AdapterExecutionMode.MONOLITHIC

    @property
    def capabilities(self) -> frozenset[AdapterCapability]:
        """Declare deterministic artifact persistence."""
        return frozenset(
            {
                AdapterCapability.ARTIFACT_PERSISTENCE,
                AdapterCapability.INCREMENTAL_UPDATE,
            }
        )

    @property
    def requested_capabilities(self) -> frozenset[AdapterCapability]:
        """Return no caller-requested optional outputs."""
        return frozenset()

    def fit(self, task: ForecastTask) -> None:
        """Retain one deterministic point per series from strict history."""
        task_history = task.history.materialize()
        points: dict[str, float] = {}
        latest_values: dict[str, float] = {}
        for index, series_key in enumerate(task.series_keys):
            history = task_history[task_history[SERIES_KEY] == series_key]
            latest = float(history[OBSERVED_VALUE].iloc[-1])
            prior = float(history[OBSERVED_VALUE].iloc[-2])
            points[series_key] = (latest + prior) / 2.0 + 0.25 * (index + 1)
            latest_values[series_key] = latest
        self._points = points
        self._latest = latest_values

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        """Emit a canonical multi-series frame from restored retained state."""
        if self._points is None:
            raise RuntimeError("driver-equivalence predict requires fitted state")
        rows = [
            {
                SERIES_KEY: series_key,
                TARGET_TIMESTAMP: target_timestamp(
                    task.origin,
                    step,
                    calendar=task.calendar,
                ),
                ACTUAL_VALUE: float("nan"),
                POINT_FORECAST: self._points[series_key] + 0.25 * (step - 1),
                HORIZON_STEP: step,
                ORIGIN: task.origin,
                MODEL_NAME_COLUMN: MODEL_NAME,
            }
            for series_key in task.series_keys
            for step in range(1, task.horizon + 1)
        ]
        frame = pd.DataFrame.from_records(rows)
        return frame.astype(
            {
                SERIES_KEY: "string",
                ACTUAL_VALUE: "float64",
                POINT_FORECAST: "float64",
                HORIZON_STEP: "int64",
                MODEL_NAME_COLUMN: "string",
            }
        )

    def dump_state(self) -> bytes:
        """Serialize fitted values with canonical JSON spelling."""
        if self._points is None or self._latest is None:
            raise RuntimeError("driver-equivalence dump requires fit")
        payload = {
            "latest": [[key, value.hex()] for key, value in sorted(self._latest.items())],
            "points": [[key, value.hex()] for key, value in sorted(self._points.items())],
            "schema": "newcalibre.driver-equivalence-adapter/v1",
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("ascii")

    def load_state(self, state: bytes) -> None:
        """Restore exact fitted values from canonical artifact bytes."""
        payload = json.loads(state)
        if payload.get("schema") != "newcalibre.driver-equivalence-adapter/v1":
            raise ValueError("driver-equivalence artifact schema is invalid")
        self._latest = {
            str(series_key): float.fromhex(value) for series_key, value in payload["latest"]
        }
        self._points = {
            str(series_key): float.fromhex(value) for series_key, value in payload["points"]
        }

    def fitted_values(self) -> FittedValues:
        """Reject unsupported fitted-value collection."""
        raise AdapterCapabilityError("driver-equivalence adapter has no fitted values")

    def update(self, delta: HistoryDelta) -> None:
        """Advance retained rolling points from newly admissible observations only."""
        if self._points is None or self._latest is None:
            raise RuntimeError("driver-equivalence update requires fitted state")
        history = delta.materialize()
        series_order = sorted(self._points, key=str.encode)
        for index, series_key in enumerate(series_order):
            values = history.loc[history[SERIES_KEY] == series_key, OBSERVED_VALUE]
            for raw_value in values:
                latest = float(raw_value)
                self._points[series_key] = (latest + self._latest[series_key]) / 2.0 + 0.25 * (
                    index + 1
                )
                self._latest[series_key] = latest


def make_panel() -> Panel:
    """Return the deterministic two-series demand panel."""
    timestamps = pd.date_range("2026-01-01", periods=16, freq="D")
    rows = [
        {
            SERIES_KEY: series_key,
            TIMESTAMP: timestamp,
            OBSERVED_VALUE: float(2 + series_index + (day_index % 5)),
        }
        for series_index, series_key in enumerate(reversed(SERIES_KEYS))
        for day_index, timestamp in enumerate(timestamps)
    ]
    frame = pd.DataFrame.from_records(rows).astype(
        {SERIES_KEY: "string", OBSERVED_VALUE: "float64"}
    )
    return Panel.from_frame(frame, calendar=CALENDAR, target_support=TargetSupport.REAL)


def make_session(runtime_name: str | None) -> SessionIdentity:
    """Derive one canonical session for the selected runtime."""
    if runtime_name is not None and runtime_name not in RUNTIME_CONFIGURATIONS:
        raise ValueError(f"unknown driver-equivalence runtime: {runtime_name!r}")
    conformal = None if runtime_name is None else dict(RUNTIME_CONFIGURATIONS[runtime_name])
    return SessionIdentity.derive(
        tenant="driver-equivalence",
        series_keys=SERIES_KEYS,
        calendar=CALENDAR,
        horizon=TIMING.protection_period,
        model_config=MODEL_CONFIG,
        conformal_config=conformal,
        ordering_policy={"name": "newsvendor"},
        decision_series_keys=SERIES_KEYS,
        cost_structure=COSTS,
        decision_timing=TIMING,
        stockout_rule=StockoutRule.LOST_SALES,
    )


def make_world(
    runtime_name: str | None,
    *,
    source_actuals: bool = True,
    store_factory: Callable[[SessionIdentity, Panel], InMemoryIndexedRunStore] | None = None,
) -> DriverWorld:
    """Build a fresh world while allowing a caller-owned transactional store."""
    panel = make_panel()
    session = make_session(runtime_name)
    store = (
        InMemoryIndexedRunStore(
            session=session,
            calendar=CALENDAR,
            actuals=panel if source_actuals else None,
            actuals_semantics=ActualsSemantics.DEMAND,
        )
        if store_factory is None
        else store_factory(session, panel)
    )
    return DriverWorld(
        runtime_name=runtime_name,
        panel=panel,
        session=session,
        store=store,
    )


def build_engine(world: DriverWorld) -> Engine:
    """Reconstruct the engine over one world's transactional store."""
    return Engine(
        session=world.session,
        panel_source=InMemoryPanelSource(world.panel),
        run_store=world.store,
        dispatch_backend=InProcessDispatch(),
        hierarchy=HierarchyIndex.flat(world.panel.series_keys),
        adapter_resolver=DeterministicArtifactAdapter,
        orderer=_order,
    )


def build_event_driver(world: DriverWorld) -> EventDriver:
    """Reconstruct an event driver over the caller-owned store."""
    return EventDriver(
        engine=build_engine(world),
        run_store=world.store,
        actuals_semantics=ActualsSemantics.DEMAND,
    )


def build_time_loop(
    world: DriverWorld,
    *,
    origins: Sequence[pd.Timestamp] = ORIGINS,
    settlement_end: pd.Timestamp = SETTLEMENT_END,
) -> TimeLoop:
    """Reconstruct a historical loop over the caller-owned store and schedule."""
    return TimeLoop(
        engine=build_engine(world),
        run_store=world.store,
        request=TimeLoopRequest(
            session=world.session,
            origins=origins,
            settlement_end=settlement_end,
            scope=Scope.GLOBAL,
            initial_inventory_positions=INITIAL_INVENTORY,
            actuals_semantics=ActualsSemantics.DEMAND,
        ),
    )


def origin_event(world: DriverWorld, origin: pd.Timestamp, *, seed: bool = False) -> OriginEvent:
    """Build one canonical origin event."""
    return OriginEvent(
        session=world.session,
        origin=origin,
        scope=Scope.GLOBAL,
        initial_inventory_positions=INITIAL_INVENTORY if seed else None,
    )


def actual_records(
    world: DriverWorld,
    *,
    timestamps: Iterable[pd.Timestamp],
) -> tuple[ActualRecord, ...]:
    """Project panel observations into explicit actual records."""
    selected = set(timestamps)
    records = []
    for values in world.panel.frame.to_dict("records"):
        timestamp = pd.Timestamp(values[TIMESTAMP])
        if timestamp not in selected:
            continue
        records.append(
            ActualRecord(
                series_key=str(values[SERIES_KEY]),
                timestamp=timestamp,
                recorded_value=float(values[OBSERVED_VALUE]),
            )
        )
    return tuple(sorted(records, key=lambda value: (value.series_key.encode(), value.timestamp)))


def actuals_event(
    world: DriverWorld,
    records: Sequence[ActualRecord],
    *,
    reverse: bool = False,
) -> ActualsEvent:
    """Build one atomic actual event, optionally with reversed record order."""
    supplied = tuple(reversed(records)) if reverse else tuple(records)
    return ActualsEvent(world.session, ActualsSubmission(supplied))


def seed_event_history(
    world: DriverWorld,
    driver: EventDriver,
    *,
    rechunk: bool = False,
    reverse: bool = False,
) -> None:
    """Submit every actual strictly before the first origin."""
    records = actual_records(
        world,
        timestamps=pd.date_range("2026-01-01", "2026-01-04", freq="D"),
    )
    submit_actuals(world, driver, records, rechunk=rechunk, reverse=reverse)


def submit_actuals(
    world: DriverWorld,
    driver: EventDriver,
    records: Sequence[ActualRecord],
    *,
    rechunk: bool = False,
    reverse: bool = False,
) -> None:
    """Submit atomically or in canonical sequence-preserving chunks."""
    canonical = tuple(
        sorted(records, key=lambda value: (value.series_key.encode(), value.timestamp))
    )
    if rechunk:
        for record in canonical:
            driver.handle(actuals_event(world, (record,)))
        return
    driver.handle(actuals_event(world, canonical, reverse=reverse))


def drive_origins(
    world: DriverWorld,
    driver: EventDriver,
    *,
    origins: Sequence[pd.Timestamp] = ORIGINS,
    rechunk: bool = False,
    reverse: bool = False,
) -> None:
    """Drive origins and same-period actual availability through final drain."""
    for origin in origins:
        seed = world.store.earliest_origin is None
        driver.handle(origin_event(world, origin, seed=seed))
        records = actual_records(world, timestamps=(origin,))
        submit_actuals(world, driver, records, rechunk=rechunk, reverse=reverse)


def run_time_world(runtime_name: str | None) -> DriverWorld:
    """Run one uninterrupted time-loop reference world."""
    world = make_world(runtime_name)
    build_time_loop(world).run()
    return world


def run_event_world(
    runtime_name: str | None,
    *,
    rechunk: bool = False,
    reverse: bool = False,
) -> DriverWorld:
    """Run one uninterrupted event schedule with equivalent availability."""
    world = make_world(runtime_name, source_actuals=False)
    driver = build_event_driver(world)
    seed_event_history(world, driver, rechunk=rechunk, reverse=reverse)
    drive_origins(world, driver, rechunk=rechunk, reverse=reverse)
    return world


def runtime_witness(
    world: DriverWorld,
    *,
    state_before: Mapping[str, bytes] | None = None,
) -> RuntimeWitness:
    """Return delivered scores, state movement, and finite issuance facts."""
    delivered = tuple(
        value.forecast_key
        for value in world.store.observe_annotations
        if value.advanced_delivered_score
    )
    finite = tuple(
        row.key
        for row in world.store.forecasts
        if row.observation_issuance is not None
        and row.observation_issuance.calibration_ready
        and math.isfinite(row.observation_issuance.lower_bound)
        and math.isfinite(row.observation_issuance.upper_bound)
    )
    before = tuple(sorted((state_before or {}).items(), key=lambda item: item[0].encode()))
    after = tuple(sorted(world.store.states.items(), key=lambda item: item[0].encode()))
    return RuntimeWitness(delivered, before, after, finite)


def _order(request: OrderRequest) -> tuple[OrderProposal, ...]:
    """Emit deterministic evidence-bearing orders independently of calibration readiness."""
    descriptor = GuaranteeDescriptor(
        type=GuaranteeType(
            claim=GuaranteeClaim.NONE,
            currency=None,
            declared_slack=None,
        ),
        level=0.5,
        scored_series=ScoredSeries.DEMAND_HONEST,
        window=EmissionScope.WINDOW_SUM,
        scope=DecisionScope(DecisionScopeKind.PER_DECISION_NODE, None),
    )
    frame = request.forecasts.frame
    proposals: list[OrderProposal] = []
    for series_key in SERIES_KEYS:
        rows = frame[frame[SERIES_KEY] == series_key].sort_values(HORIZON_STEP)
        raw_target = math.fsum(float(value) for value in rows[POINT_FORECAST])
        quantity = max(0.0, raw_target - request.inventory_positions[series_key].value)
        proposals.append(
            OrderProposal(
                series_key=series_key,
                model_name=MODEL_NAME,
                quantity=quantity,
                evidence=DecisionEvidence(
                    raw_target=raw_target,
                    target=raw_target,
                    source_columns=(POINT_FORECAST,),
                    source_descriptor=descriptor,
                    effective_descriptor=descriptor,
                    bindings=(AppliedBinding("fixture-scale", 1.0, False),),
                ),
            )
        )
    return tuple(proposals)


__all__ = [
    "CALENDAR",
    "COSTS",
    "EXPECTED_BOOKED_COSTS",
    "EXPECTED_FINAL_INVENTORY",
    "EXPECTED_LATE_BOOKED_COSTS",
    "EXPECTED_LATE_FINAL_INVENTORY",
    "INITIAL_INVENTORY",
    "MODEL_NAME",
    "ORIGINS",
    "RUNTIME_CASES",
    "RUNTIME_CONFIGURATIONS",
    "SERIES_KEYS",
    "SETTLEMENT_END",
    "TIMING",
    "DeterministicArtifactAdapter",
    "DriverWorld",
    "RuntimeWitness",
    "actual_records",
    "actuals_event",
    "build_engine",
    "build_event_driver",
    "build_time_loop",
    "drive_origins",
    "make_panel",
    "make_session",
    "make_world",
    "origin_event",
    "run_event_world",
    "run_time_world",
    "runtime_witness",
    "seed_event_history",
    "submit_actuals",
]
