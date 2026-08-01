"""Run validated VN2 data through the generic successor engine."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

import pandas as pd

from newcalibre.domain import (
    CENSOR_STATUS,
    OBSERVED_VALUE,
    SERIES_KEY,
    TIMESTAMP,
    UNDECLARED_CENSORING,
    CensoringAssertion,
    InventoryPosition,
    Panel,
    Scope,
    SessionIdentity,
    TargetSupport,
)
from newcalibre.engine import (
    ActualKey,
    ConfiguredPolicyOrderer,
    Engine,
    InMemoryIndexedRunStore,
    InMemoryPanelSource,
    InProcessDispatch,
    TimeLoop,
    TimeLoopRequest,
    TimeLoopResult,
)
from newcalibre.ledger import CoverageReport, ForecastRow, OrderRow, SettlementRecord
from newcalibre.protocols.vn2.forecasting import resolve_vn2_adapter
from newcalibre.protocols.vn2.loader import VN2DataError, VN2Dataset, VN2RoundInput

type VN2SeriesIdentity = tuple[int, int]

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class VN2RunResult:
    """Return raw generic-engine facts from one complete VN2 run."""

    session: SessionIdentity
    time_loop: TimeLoopResult
    input_inventory_sha256: str
    forecasts: tuple[ForecastRow, ...]
    coverage_report: CoverageReport
    orders: tuple[OrderRow, ...]
    settlements: tuple[SettlementRecord, ...]
    series_identities: Mapping[str, VN2SeriesIdentity]

    def __post_init__(self) -> None:
        if not isinstance(self.session, SessionIdentity):
            raise TypeError("VN2 run session must be a SessionIdentity")
        if not isinstance(self.time_loop, TimeLoopResult):
            raise TypeError("VN2 run time loop must be a TimeLoopResult")
        if (
            not isinstance(self.input_inventory_sha256, str)
            or _SHA256.fullmatch(self.input_inventory_sha256) is None
        ):
            raise TypeError("VN2 run input inventory identity must be a sha256 digest")
        forecasts = tuple(self.forecasts)
        if any(not isinstance(row, ForecastRow) for row in forecasts):
            raise TypeError("VN2 run forecasts must contain ForecastRow values")
        if not isinstance(self.coverage_report, CoverageReport):
            raise TypeError("VN2 run coverage report must be a CoverageReport")
        orders = tuple(self.orders)
        settlements = tuple(self.settlements)
        if any(not isinstance(row, OrderRow) for row in orders):
            raise TypeError("VN2 run orders must contain OrderRow values")
        if any(not isinstance(row, SettlementRecord) for row in settlements):
            raise TypeError("VN2 run settlements must contain SettlementRecord values")
        identities = dict(self.series_identities)
        if any(
            not isinstance(key, str)
            or not isinstance(identity, tuple)
            or len(identity) != 2
            or any(not isinstance(value, int) for value in identity)
            for key, identity in identities.items()
        ):
            raise TypeError("VN2 series identities must map strings to integer pairs")
        object.__setattr__(self, "forecasts", forecasts)
        object.__setattr__(self, "orders", orders)
        object.__setattr__(self, "settlements", settlements)
        object.__setattr__(self, "series_identities", MappingProxyType(identities))


def run_vn2(dataset: VN2Dataset) -> VN2RunResult:
    """Execute one validated VN2 dataset through the generic time-loop driver."""
    if not isinstance(dataset, VN2Dataset):
        raise TypeError("run_vn2 requires a VN2Dataset")
    config = dataset.config
    first = dataset.round_input(1)
    identities = _series_identities(first, key_columns=config.columns.series_keys)
    panel = _panel(dataset, first=first)
    positions, initial_arrivals = _initial_inventory(
        first,
        identities=identities,
        key_columns=config.columns.series_keys,
        on_hand_column=config.columns.initial_on_hand,
        pipeline_columns=config.columns.initial_pipeline,
        realized_periods=config.realized_periods,
    )
    series_keys = tuple(identities)
    session = SessionIdentity.derive(
        tenant=config.dataset,
        series_keys=series_keys,
        calendar=config.calendar,
        horizon=config.task_horizon,
        model_config=config.model_config,
        conformal_config=config.conformal_config,
        ordering_policy=config.ordering_policy,
        decision_series_keys=series_keys,
        cost_structure=config.cost_structure,
        decision_timing=config.timing,
        stockout_rule=config.stockout_rule,
    )
    store = InMemoryIndexedRunStore(
        session=session,
        calendar=config.calendar,
        actuals=panel,
        actuals_semantics=config.actuals_semantics,
        initial_arrivals=initial_arrivals,
    )
    engine = Engine(
        session=session,
        panel_source=InMemoryPanelSource(panel),
        run_store=store,
        dispatch_backend=InProcessDispatch(),
        hierarchy=None,
        adapter_resolver=resolve_vn2_adapter,
        orderer=ConfiguredPolicyOrderer(),
    )
    origins = config.realized_periods[: -config.drain_periods]
    derived_decisions = origins[:: config.timing.review_period]
    if derived_decisions != config.decision_origins:
        raise VN2DataError("VN2 realized periods do not derive the configured decision origins")
    time_loop = TimeLoop(
        engine=engine,
        run_store=store,
        request=TimeLoopRequest(
            session=session,
            origins=origins,
            settlement_end=config.realized_periods[-1],
            scope=Scope.GLOBAL,
            initial_inventory_positions=positions,
            actuals_semantics=config.actuals_semantics,
        ),
    ).run()
    if time_loop.decision_origins != config.decision_origins:
        raise VN2DataError("VN2 engine decisions do not match the configured origins")
    return VN2RunResult(
        session=session,
        time_loop=time_loop,
        input_inventory_sha256=dataset.input_inventory_sha256,
        forecasts=store.logical_forecasts,
        coverage_report=store.coverage_report(),
        orders=store.orders,
        settlements=store.settlements,
        series_identities=identities,
    )


def _series_identities(
    first: VN2RoundInput,
    *,
    key_columns: tuple[str, str],
) -> dict[str, VN2SeriesIdentity]:
    store_column, product_column = key_columns
    identities: dict[str, VN2SeriesIdentity] = {}
    for store, product in zip(
        first.sales[store_column],
        first.sales[product_column],
        strict=True,
    ):
        identity = (int(store), int(product))
        engine_key = f"{identity[0]}_{identity[1]}"
        if engine_key in identities:
            raise VN2DataError(f"duplicate VN2 engine series key: {engine_key!r}")
        identities[engine_key] = identity
    return identities


def _panel(
    dataset: VN2Dataset,
    *,
    first: VN2RoundInput,
) -> Panel:
    config = dataset.config
    key_columns = config.columns.series_keys
    history = _wide_history(
        first.sales,
        in_stock=first.in_stock,
        key_columns=key_columns,
    )
    realized: list[pd.DataFrame] = []
    for week_number in range(1, len(config.realized_periods) + 1):
        weekly = dataset.weekly_actuals(week_number)
        frame = weekly.sales.copy(deep=True)
        frame[SERIES_KEY] = _engine_keys(frame, key_columns=key_columns)
        realized.append(
            pd.DataFrame(
                {
                    SERIES_KEY: pd.Series(frame[SERIES_KEY], dtype="string"),
                    TIMESTAMP: pd.Series(
                        [weekly.period] * len(frame),
                        dtype="datetime64[ns]",
                    ),
                    OBSERVED_VALUE: pd.Series(frame["sales"], dtype="float64"),
                    CENSOR_STATUS: pd.Series(
                        [UNDECLARED_CENSORING] * len(frame),
                        dtype="string",
                    ),
                }
            )
        )
    return Panel.from_frame(
        pd.concat((history, *realized), ignore_index=True),
        calendar=config.calendar,
        target_support=TargetSupport.NONNEGATIVE,
    )


def _wide_history(
    sales: pd.DataFrame,
    *,
    in_stock: pd.DataFrame,
    key_columns: tuple[str, str],
) -> pd.DataFrame:
    date_columns = tuple(column for column in sales.columns if column not in key_columns)
    keyed_sales = sales.copy(deep=True)
    keyed_sales[SERIES_KEY] = _engine_keys(keyed_sales, key_columns=key_columns)
    history = keyed_sales[[SERIES_KEY, *date_columns]].melt(
        id_vars=SERIES_KEY,
        var_name=TIMESTAMP,
        value_name=OBSERVED_VALUE,
    )
    history[TIMESTAMP] = pd.to_datetime(history[TIMESTAMP])
    history[OBSERVED_VALUE] = history[OBSERVED_VALUE].astype("float64")

    stock_dates = tuple(column for column in in_stock.columns if column not in key_columns)
    keyed_stock = in_stock.copy(deep=True)
    keyed_stock[SERIES_KEY] = _engine_keys(keyed_stock, key_columns=key_columns)
    stock = keyed_stock[[SERIES_KEY, *stock_dates]].melt(
        id_vars=SERIES_KEY,
        var_name=TIMESTAMP,
        value_name="in_stock",
    )
    stock[TIMESTAMP] = pd.to_datetime(stock[TIMESTAMP])
    assertions = {
        (str(row.series_key), cast(pd.Timestamp, row.timestamp)): (
            CensoringAssertion.UNCENSORED.value
            if bool(row.in_stock)
            else CensoringAssertion.CENSORED.value
        )
        for row in stock.rename(
            columns={SERIES_KEY: "series_key", TIMESTAMP: "timestamp"}
        ).itertuples(index=False)
    }
    history[CENSOR_STATUS] = pd.Series(
        [
            assertions.get(
                (str(series_key), pd.Timestamp(timestamp)),
                UNDECLARED_CENSORING,
            )
            for series_key, timestamp in zip(
                history[SERIES_KEY],
                history[TIMESTAMP],
                strict=True,
            )
        ],
        dtype="string",
    )
    history[SERIES_KEY] = history[SERIES_KEY].astype("string")
    return history[[SERIES_KEY, TIMESTAMP, OBSERVED_VALUE, CENSOR_STATUS]]


def _initial_inventory(
    first: VN2RoundInput,
    *,
    identities: Mapping[str, VN2SeriesIdentity],
    key_columns: tuple[str, str],
    on_hand_column: str,
    pipeline_columns: tuple[str, ...],
    realized_periods: tuple[pd.Timestamp, ...],
) -> tuple[dict[str, InventoryPosition], dict[ActualKey, float]]:
    by_identity = {identity: engine_key for engine_key, identity in identities.items()}
    positions: dict[str, InventoryPosition] = {}
    arrivals: dict[ActualKey, float] = {}
    store_column, product_column = key_columns
    for row in first.initial_state.itertuples(index=False, name=None):
        values = dict(zip(first.initial_state.columns, row, strict=True))
        identity = (int(values[store_column]), int(values[product_column]))
        engine_key = by_identity[identity]
        pipeline = tuple(float(values[column]) for column in pipeline_columns)
        positions[engine_key] = InventoryPosition(
            on_hand=float(values[on_hand_column]),
            on_order=math.fsum(pipeline),
            backorders=0.0,
        )
        arrival_periods = realized_periods[: len(pipeline)]
        if len(arrival_periods) != len(pipeline):
            raise VN2DataError("VN2 realized periods do not cover the initial pipeline")
        for period, quantity in zip(arrival_periods, pipeline, strict=True):
            arrivals[(engine_key, period)] = quantity
    return positions, arrivals


def _engine_keys(
    frame: pd.DataFrame,
    *,
    key_columns: tuple[str, str],
) -> pd.Series:
    store_column, product_column = key_columns
    return pd.Series(
        [
            f"{int(store)}_{int(product)}"
            for store, product in zip(
                frame[store_column],
                frame[product_column],
                strict=True,
            )
        ],
        index=frame.index,
        dtype="string",
    )


__all__ = ["VN2RunResult", "run_vn2"]
