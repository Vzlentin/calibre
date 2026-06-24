"""Settle-contract fixture for the rolling decision loop.

Drives a :class:`~calibre.execution.decision_loop.DecisionLoop` over the generic
:class:`~calibre.ordering.simulation.simulator.Simulator` and a real
:class:`~calibre.conformal.cumulative_risk.CumulativeRiskRuntime` on a tiny
hand-figured fixture (two series, a few weekly origins), with a deterministic
stub engine standing in for the model. This is the same assembly
``run_config``'s settle branch builds, so it pins the contract the engine path
relies on:

* **drain count** — the simulator steps once per series for every decision round
  AND every lead-time delivery round;
* **pipeline-shift** — an order placed at round ``r`` arrives ``lead_time`` rounds
  later (the lost-sales pipeline), so a late order only relieves shortage after
  the drain;
* **cost-once** — the accumulated holding/shortage cost is the single scalar the
  simulator owns; no double counting across the loop;
* **deterministic chronological observe order** — the CRC ingests origins in
  ascending date order, so its residual records' ``sequence`` is non-decreasing
  with origin date (recency weighting depends on this).

It does NOT prove recency-weighting parity at scale — that is
``tests/benchmarks/test_vn2_regression.py``'s job on CI Linux.
"""

from __future__ import annotations

import pandas as pd
import pytest

from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeRiskRuntime,
)
from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    quantile_column,
)
from calibre.core.forecast_task import TaskGroups
from calibre.execution.decision_loop import DecisionLoop, DecisionLoopConfig
from calibre.execution.ledger import InMemoryLedger
from calibre.ordering.policy_config import RsConfig, apply_order_policy, build_rs_params
from calibre.ordering.simulation.costs import LinearCostModel
from calibre.ordering.simulation.rules import LostSalesRule
from calibre.ordering.simulation.simulator import Simulator
from calibre.ordering.simulation.state import ProductState, make_pipeline

_FREQ = "W-MON"
_HORIZON = 2
_LEAD_TIME = 1
_REVIEW_PERIOD = 1
_PROTECTION = _LEAD_TIME + _REVIEW_PERIOD
_COVERAGE = 0.5
_SERIES = ("A", "B")
# Weekly demand per series, indexed by demand week (1-based).
_DEMAND = {
    "A": {1: 4.0, 2: 5.0, 3: 6.0, 4: 5.0, 5: 4.0},
    "B": {1: 2.0, 2: 3.0, 3: 2.0, 4: 3.0, 5: 2.0},
}
_ORIGINS = [
    pd.Timestamp("2024-01-01") + i * pd.tseries.frequencies.to_offset(_FREQ) for i in range(3)
]
_N_ROUNDS = len(_ORIGINS)


class _StubResult:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._ledger = InMemoryLedger()
        if not frame.empty:
            self._ledger.append(frame)

    @property
    def ledger(self) -> InMemoryLedger:
        return self._ledger


class _StubEngine:
    """Deterministic per-origin forecaster: y_hat = a fixed base per series.

    Emits the full forecast-frame contract (REQUIRED_COLUMNS + the quantile
    column) for h=1..horizon at ``origin + k`` so the CRC runtime can apply its
    bound and resolve the protection window.
    """

    freq = _FREQ

    def __init__(self, base: dict[str, float]) -> None:
        self._base = base
        self.execute_calls = 0

    def execute(
        self, tasks: TaskGroups, actuals: pd.DataFrame, origins: list[pd.Timestamp]
    ) -> _StubResult:
        self.execute_calls += 1
        offset = pd.tseries.frequencies.to_offset(_FREQ)
        rows = []
        qcol = quantile_column(_COVERAGE)
        for origin in origins:
            for uid in _SERIES:
                for h in range(1, _HORIZON + 1):
                    rows.append(
                        {
                            UNIQUE_ID: uid,
                            DS: origin + h * offset,
                            Y: float("nan"),
                            Y_HAT: self._base[uid],
                            qcol: self._base[uid],
                            H: h,
                            FORECAST_ORIGIN: origin,
                            MODEL_NAME: "stub",
                        }
                    )
        frame = pd.DataFrame(rows)
        frame[DS] = frame[DS].astype("datetime64[ns]")
        frame[FORECAST_ORIGIN] = frame[FORECAST_ORIGIN].astype("datetime64[ns]")
        return _StubResult(frame)

    def close(self) -> None:
        pass


def _seeded_simulator() -> Simulator:
    states = {
        uid: ProductState(
            unique_id=uid,
            end_inventory=10.0,
            pipeline=make_pipeline([0.0], _LEAD_TIME),
        )
        for uid in _SERIES
    }
    return Simulator(
        states,
        LostSalesRule(_LEAD_TIME),
        LinearCostModel(rates={"holding": 0.2, "shortage": 1.0}),
    )


def _demand_at(week: int) -> dict[str, float]:
    return {uid: _DEMAND[uid].get(week, 0.0) for uid in _SERIES}


def _drive_loop() -> tuple[Simulator, CumulativeRiskRuntime, _StubEngine, list]:
    simulator = _seeded_simulator()
    runtime = CumulativeRiskRuntime(
        CumulativeConformalRiskConfig(
            coverage=_COVERAGE,
            protection_period=_PROTECTION,
            weight_decay=None,
            buffer_max=0.0,
        )
    )
    engine = _StubEngine(base={"A": 5.0, "B": 3.0})
    captured: list = []

    def build_round(round_num: int):
        origin = _ORIGINS[round_num - 1]
        return TaskGroups(), origin, pd.DataFrame()

    def policy(frame: pd.DataFrame) -> dict[str, float]:
        order_config = RsConfig(
            params=build_rs_params(simulator, _LEAD_TIME, _REVIEW_PERIOD),
            coverage=_COVERAGE,
        )
        order_result = apply_order_policy(frame, order_config)
        orders = dict.fromkeys(_SERIES, 0.0)
        for uid, qty in zip(
            order_result[UNIQUE_ID].astype(str),
            order_result["order_qty"].astype(float),
            strict=False,
        ):
            orders[uid] = float(max(qty, 0.0))
        return orders

    def get_actuals(week: int) -> dict[str, float]:
        return _demand_at(week)

    results = DecisionLoop(
        engine=engine,
        simulator=simulator,
        build_round_tasks=build_round,
        policy=policy,
        get_actuals=get_actuals,
        config=DecisionLoopConfig(
            n_rounds=_N_ROUNDS,
            n_delivery_rounds=_LEAD_TIME,
            on_round=captured.append,
        ),
        runtime=runtime,
    ).run()
    return simulator, runtime, engine, results


def test_settle_loop_drains_lead_time_rounds() -> None:
    """The simulator steps once per series for every decision AND delivery round."""
    simulator, _, engine, results = _drive_loop()
    expected_periods = _N_ROUNDS + _LEAD_TIME
    # One history record per (series, period) across decision + delivery rounds.
    assert len(simulator.history) == expected_periods * len(_SERIES)
    # The engine forecasts only on decision rounds (delivery rounds place no order).
    assert engine.execute_calls == _N_ROUNDS
    assert len(results) == _N_ROUNDS


def test_settle_loop_pipeline_shift_defers_arrivals_by_lead_time() -> None:
    """An order placed at round r only arrives lead_time rounds later."""
    simulator, _, _, _ = _drive_loop()
    history = simulator.to_dataframe()
    a_rows = history[history["unique_id"] == "A"].sort_values("period").reset_index(drop=True)
    # Period 1 cannot see any order placed this loop (the seed pipeline slot is
    # zero), so arrivals at period 1 are zero; later periods may carry arrivals.
    assert a_rows.loc[0, "arrivals"] == 0.0
    # Some later period receives the lead-time-delayed arrivals from an order.
    assert (a_rows["arrivals"] > 0).any()


def test_settle_loop_accumulates_cost_once() -> None:
    """The simulator owns a single finite scalar cost (no double counting)."""
    simulator, _, _, _ = _drive_loop()
    total = simulator.total_cost()
    breakdown = simulator.cost_breakdown()
    assert total == pytest.approx(breakdown["holding"] + breakdown["shortage"])
    assert total >= 0.0
    # The fixture demand exceeds the per-round forecast at the tail, so the lost-
    # sales rule books some shortage — a non-degenerate, finite cost.
    assert total > 0.0


def test_settle_loop_observes_origins_in_chronological_order() -> None:
    """The CRC ingests origins in ascending date order (recency weighting)."""
    _, runtime, _, _ = _drive_loop()
    records = runtime._calibrator._records
    # At least one window resolved over the loop + drain.
    assert records, "expected the CRC to record at least one resolved residual"
    sequences = [record.sequence for record in records]
    # sort=True in observe means the per-origin sequence counter is monotone in
    # ingestion (chronological) order — never reordered.
    assert sequences == sorted(sequences)
    assert sequences[0] >= 1


def test_non_ordering_run_is_single_pass_with_no_simulator(tmp_path) -> None:
    """A non-ordering `run_config` takes the single pass: no settle, no cost gauge.

    Byte-identity guard for execution case 1 — the settle branch is skipped
    entirely (no ``ordering:`` block), so no simulator is constructed and no
    order ledger or order-cost gauge is written for the run.
    """
    from prometheus_client import REGISTRY

    from calibre.cli.commands import _load_dataset, _settling, run_config
    from calibre.cli.config import load_config_from_mapping

    _WIDE_SALES = "\n".join(
        [
            "Store,Product,2024-01-01,2024-01-08,2024-01-15,2024-01-22,2024-01-29,2024-02-05",
            "1,10,10,12,11,15,9,13",
            "2,20,5,7,6,9,4,8",
        ]
    )
    (tmp_path / "week_0_sales.csv").write_text(_WIDE_SALES + "\n")
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "vn2", "path": str(tmp_path), "period": 0},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 2,
                    "config": {"backend": "statsforecast", "season_length": 2},
                }
            ],
            "origins": {"start": "2024-01-29", "end": "2024-02-05", "freq": "W-MON"},
            "output": {"streaming": False},
            "execution": {"backend": "local", "seed": 42},
        }
    )
    bundle = _load_dataset(config)
    assert bundle.inventory is None
    assert not _settling(config, bundle)

    before = REGISTRY.get_sample_value("calibre_order_cost", {"currency": "EUR", "dataset": "vn2"})
    result = run_config(config)
    after = REGISTRY.get_sample_value("calibre_order_cost", {"currency": "EUR", "dataset": "vn2"})

    # Single-pass: a forecast ledger, no order ledger, and the cost gauge is
    # untouched by this run (no ordering -> no cost recorded).
    assert result.order_ledger is None
    assert not result.ledger.to_df().empty
    assert before == after
