"""x86_64-gated VN2 CLI parity harness: `calibre run` reproduces 4992.20.

This is the per-round C1-C5 parity proof for the **production** settle path
(`run_config` -> `_run_settle_loop`), distinct from the frozen scalar gate in
:mod:`tests.benchmarks.test_vn2_regression` (which pins the bespoke
``benchmarks/vn2`` harness). The settle path shares the benchmark's
``DecisionLoop``/``Simulator``/``observe_pending``/``orders_from_policy_result``,
so the parity work is confined to the un-shared sites — config resolution,
``base_column``, the censoring-aware fit, the warmup ``y`` refill, the demand
anchor, and the per-round actuals reveal. The checkpoints localize a drift to a
specific site; C5's scalar vs the frozen ``4992.20`` is the independent oracle.

* **C1** — the censoring-aware fit history (``y``) the settle path trains on
  equals the benchmark's ``prepare_model_history`` for every decision origin.
* **C2** — the per-round realised actuals the settle loop settles equal the
  benchmark's ``_actuals_for_replay_round`` under the EXPLICIT round-index/anchor
  mapping (settle round r demand = origins[r-1] week = ``extract_new_actuals(r)``),
  not a naive same-date equality.
* **C3** — the per-round cumulative holding/shortage trajectory matches the
  benchmark replay step by step.
* **C4** — the live ``inventory_position`` the (R,S) policy reads each round
  equals the benchmark simulator's per-uid position.
* **C5** — `calibre run` reproduces ``total_cost == 4992.20`` and both buckets
  bit-for-bit off the exposed simulator (not the Prometheus gauge), via the
  settle path, while the frozen regression gate stays green.

The settle run and the benchmark oracle each train an 800-estimator global
LightGBM (~50s each), so the model-running checkpoints share two session-scoped
fixtures and carry the ``regression`` marker. The x86_64 gate mirrors
``test_vn2_regression`` (the baseline is the x86_64/Linux value; arm64/macOS
deterministically yields a different cost — cross-arch LightGBM float divergence,
not a regression).
"""

from __future__ import annotations

import platform
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.vn2.config import DECISION_ROUNDS, HORIZON
from benchmarks.vn2.data import load_instock, prepare_model_history, strip_private
from benchmarks.vn2.replay import (
    _actuals_for_replay_round,
    build_replay_cache,
    replay_cached_cost,
)
from benchmarks.vn2.simulator import load_initial_states
from calibre.cli.commands import run_config
from calibre.cli.config import load_config
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.execution.data_loading import load_period
from calibre.execution.decision_loop import RoundResult
from calibre.ordering.simulation.simulator import Simulator

_CONFIG = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "vn2" / "config" / "vn2-winning-loop.yaml"
)

BASELINE_TOTAL_COST = 4992.20
BASELINE_HOLDING_COST = 2488.20
BASELINE_SHORTAGE_COST = 2504.00
BASELINE_ABS_TOL = 0.01

_IS_X86_64 = platform.machine().lower() in {"x86_64", "amd64"}
_x86_64_gate = pytest.mark.skipif(
    not _IS_X86_64,
    reason=(
        f"VN2 parity baseline {BASELINE_TOTAL_COST} is the x86_64/Linux gate value; "
        f"on {platform.machine()} the same config deterministically yields a different "
        "cost (cross-arch LightGBM float divergence, not a regression). See CLAUDE.md."
    ),
)


class _SettleCapture:
    """Read-only capture of a settle ``run_config`` walk for the parity harness."""

    def __init__(self) -> None:
        self.rounds: dict[int, RoundResult] = {}
        self.holding: float | None = None
        self.shortage: float | None = None

    def on_round(self, rr: RoundResult) -> None:
        self.rounds[rr.round_num] = rr

    def on_complete(self, simulator: Simulator) -> None:
        breakdown = simulator.cost_breakdown()
        self.holding = float(breakdown.get("holding", 0.0))
        self.shortage = float(breakdown.get("shortage", 0.0))


@pytest.fixture(scope="module")
def settle_capture(data_dir: Path) -> _SettleCapture:
    """Run the production settle path once; capture per-round + final state.

    Drives the real :func:`calibre.cli.commands.run_config` (never a
    reconstructed loop), pointing the config at the session ``data_dir``. Module
    scoped so the single LightGBM walk feeds every checkpoint.
    """
    config = load_config(_CONFIG)
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"path": str(data_dir)})}
    )
    capture = _SettleCapture()
    run_config(
        config,
        settle_on_round=capture.on_round,
        settle_on_complete=capture.on_complete,
    )
    return capture


@pytest.fixture(scope="module")
def replay_reference(data_dir: Path):
    """Build the benchmark replay oracle once: cache, replay result, trajectory."""
    cache = build_replay_cache(data_dir=str(data_dir))
    trajectory: list[float] = []
    result = replay_cached_cost(cache, on_progress=lambda _step, cost: trajectory.append(cost))
    return cache, result, trajectory


def _benchmark_origins(data_dir: Path) -> list[pd.Timestamp]:
    """Compute the benchmark's decision origins (round r: max(week_{r-1}) + 1wk)."""
    origins: list[pd.Timestamp] = []
    for rn in range(1, DECISION_ROUNDS + 1):
        sales = load_period(str(data_dir), rn - 1)
        origins.append(pd.Timestamp(sales[DS].max()) + pd.Timedelta(weeks=1))
    return origins


@pytest.mark.regression
@_x86_64_gate
def test_c1_fit_history_matches_prepare_model_history(data_dir: Path) -> None:
    """C1 — the censoring-aware fit ``y`` equals the benchmark per origin.

    The settle path fits ``history[ds < origin]`` with ``censoring_fit`` imputing
    in-stock-censored demand; the benchmark fits
    ``prepare_model_history(week_{r-1}, instock)``. With period-8 history sliced
    ``< origin`` byte-equal to ``week_{r-1}`` (no revision) and the same
    ``add_stockout_features`` imputation, the training ``y`` must match exactly.
    """
    config = load_config(_CONFIG)
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"path": str(data_dir)})}
    )
    from calibre.execution.dataset_registry import resolve_dataset_adapter
    from calibre.forecasting.features import add_stockout_features

    bundle = resolve_dataset_adapter("vn2").load(str(data_dir), period=8)
    instock = load_instock(str(data_dir), None)
    origins = _benchmark_origins(data_dir)

    for rn, origin in enumerate(origins, start=1):
        # Settle fit window: history strictly before the origin, imputed exactly
        # as the engine imputes it under censoring_fit (add_stockout_features ->
        # y_uncensored), sliced to ds < origin like prediction._process_global_panel.
        sliced = bundle.history[bundle.history[DS] < origin]
        sliced_censoring = bundle.censoring[bundle.censoring[DS] < origin]
        settle_imputed = add_stockout_features(sliced, sliced_censoring)
        settle_y = (
            settle_imputed[[UNIQUE_ID, DS, "y_uncensored"]]
            .rename(columns={"y_uncensored": Y})
            .sort_values([UNIQUE_ID, DS])
            .reset_index(drop=True)
        )

        # Benchmark fit window: prepare_model_history(week_{r-1}, instock).
        week_sales = load_period(str(data_dir), rn - 1)
        bench_y = prepare_model_history(
            week_sales, instock, protection_period=HORIZON, cumulative_target=False
        )
        bench_y = bench_y.sort_values([UNIQUE_ID, DS]).reset_index(drop=True)

        assert settle_y[[UNIQUE_ID, DS]].equals(bench_y[[UNIQUE_ID, DS]]), (
            f"round {rn}: fit index diverged from prepare_model_history"
        )
        pd.testing.assert_series_equal(
            settle_y[Y],
            bench_y[Y],
            check_names=False,
            obj=f"round {rn} fit y",
        )


@pytest.mark.regression
@_x86_64_gate
def test_c2_per_round_actuals_match_replay_under_explicit_anchor(
    settle_capture: _SettleCapture, data_dir: Path
) -> None:
    """C2 — settle round-r actuals equal the benchmark's, by EXPLICIT anchor.

    Settle round r settles the demand at the origin's OWN revealed week
    ``origins[r-1]`` — the week the benchmark reads via ``extract_new_actuals(r)``
    inside ``_actuals_for_replay_round(r)``. This asserts that explicit
    round-index/anchor mapping (round r <-> demand week origins[r-1]), NOT a
    naive same-date equality, so the off-by-one anchor (settle r charged with
    r+1's demand) would fail here before any cost is compared.
    """
    initial_states = load_initial_states(str(data_dir / "week_0_initial_state.csv"))
    origins = _benchmark_origins(data_dir)
    assert origins[0] == pd.Timestamp("2024-04-15")
    assert origins[-1] == pd.Timestamp("2024-05-20")

    for rn in range(1, DECISION_ROUNDS + 1):
        bench_actuals = _actuals_for_replay_round(
            str(data_dir), rn, DECISION_ROUNDS, initial_states
        )
        settle_actuals = settle_capture.rounds[rn].actual_demand

        # Explicit anchor: round r's settled week is origins[r-1].
        expected_week = origins[rn - 1]
        assert settle_capture.rounds[rn].origin == expected_week, (
            f"round {rn}: settle origin {settle_capture.rounds[rn].origin} "
            f"!= expected decision origin {expected_week}"
        )

        keys = set(bench_actuals) | set(settle_actuals)
        mismatches = {
            uid: (settle_actuals.get(uid, 0.0), bench_actuals.get(uid, 0.0))
            for uid in keys
            if abs(float(settle_actuals.get(uid, 0.0)) - float(bench_actuals.get(uid, 0.0))) > 1e-9
        }
        assert not mismatches, (
            f"round {rn} (week {expected_week.date()}): settle actuals diverged from "
            f"the benchmark's _actuals_for_replay_round on {len(mismatches)} series; "
            f"sample {dict(list(mismatches.items())[:3])}"
        )


@pytest.mark.regression
@_x86_64_gate
def test_c3_cumulative_cost_trajectory_matches_replay(
    settle_capture: _SettleCapture, replay_reference
) -> None:
    """C3 — the per-round cumulative-cost trajectory matches the replay oracle.

    The settle ``on_round`` snapshot's cumulative (holding + shortage) at each
    decision round must equal the benchmark replay's per-step cumulative total,
    step by step. A drift localizes to round k (a divergent bound, anchor, or
    observe at exactly that round) before the final scalar is consulted.
    """
    _cache, _result, trajectory = replay_reference
    for rn in range(1, DECISION_ROUNDS + 1):
        rr = settle_capture.rounds[rn]
        settle_cum = float(rr.holding_cost_cum) + float(rr.shortage_cost_cum)
        bench_cum = trajectory[rn - 1]
        assert settle_cum == pytest.approx(bench_cum, abs=BASELINE_ABS_TOL), (
            f"round {rn}: settle cumulative cost {settle_cum:.4f} drifted from the "
            f"benchmark replay trajectory {bench_cum:.4f}"
        )


@pytest.mark.regression
@_x86_64_gate
def test_c4_inventory_position_matches_replay(
    settle_capture: _SettleCapture, data_dir: Path
) -> None:
    """C4 — the live (R,S) ``inventory_position`` matches the benchmark sim.

    Each round the settle policy builds (R,S) params off
    ``simulator.inventory_position``; replaying the benchmark with the SAME
    captured per-round orders and actuals must hold the identical per-uid
    position at each round, proving the two simulators step in lock-step.
    """
    from calibre.ordering.policy_config import build_rs_params
    from calibre.ordering.simulation.costs import LinearCostModel
    from calibre.ordering.simulation.rules import LostSalesRule

    config = load_config(_CONFIG)
    bundle_states = load_initial_states(str(data_dir / "week_0_initial_state.csv"))
    # Reconstruct a generic simulator seeded identically and step it with the
    # settle walk's own orders + actuals. The settle snapshot is taken in the
    # loop's on_round, AFTER that round's simulator step, so the replay's
    # post-step inventory_position — read via build_rs_params, the same accessor
    # the (R,S) policy uses — must equal the snapshot per uid/round.
    from calibre.execution.dataset_registry import resolve_dataset_adapter

    bundle = resolve_dataset_adapter("vn2").load(str(data_dir), period=8)
    lead_time = int(config.ordering.lead_time)
    simulator = Simulator(
        bundle.inventory,
        LostSalesRule(lead_time),
        LinearCostModel(rates={"holding": 0.2, "shortage": 1.0}),
    )
    review_period = int(config.ordering.review_period)

    for rn in range(1, DECISION_ROUNDS + 1):
        rr = settle_capture.rounds[rn]
        simulator.step(rn, orders=rr.orders, actual_demand=rr.actual_demand)
        params = build_rs_params(simulator, lead_time, review_period)
        position = {p.unique_id: float(p.inventory_position) for p in params}
        assert rr.inventory_position is not None
        for uid in bundle_states:
            assert position[uid] == pytest.approx(rr.inventory_position[uid]), (
                f"round {rn} uid {uid}: replayed post-step inventory_position "
                f"{position[uid]} != settle snapshot {rr.inventory_position[uid]}"
            )


@pytest.mark.regression
@_x86_64_gate
def test_c5_calibre_run_reproduces_4992_via_settle_path(
    settle_capture: _SettleCapture,
) -> None:
    """C5 — `calibre run` settle path reproduces 4992.20 + both buckets.

    The independent oracle: the production settle run's final bucketed cost,
    read off the exposed simulator (NOT the Prometheus gauge — avoids the
    global-singleton isolation hazard), is the frozen baseline bit-for-bit.
    """
    assert settle_capture.holding is not None
    assert settle_capture.shortage is not None
    total = settle_capture.holding + settle_capture.shortage

    assert settle_capture.holding == pytest.approx(BASELINE_HOLDING_COST, abs=BASELINE_ABS_TOL), (
        f"settle holding_cost {settle_capture.holding:.4f} drifted from "
        f"{BASELINE_HOLDING_COST}; do not loosen — see CLAUDE.md Gotchas."
    )
    assert settle_capture.shortage == pytest.approx(BASELINE_SHORTAGE_COST, abs=BASELINE_ABS_TOL), (
        f"settle shortage_cost {settle_capture.shortage:.4f} drifted from {BASELINE_SHORTAGE_COST}."
    )
    assert total == pytest.approx(BASELINE_TOTAL_COST, abs=BASELINE_ABS_TOL), (
        f"settle total_cost {total:.4f} drifted from the x86_64 baseline "
        f"{BASELINE_TOTAL_COST}. Do not loosen this — see CLAUDE.md Gotchas."
    )


@pytest.mark.regression
@_x86_64_gate
def test_seam_is_behavior_neutral(data_dir: Path) -> None:
    """The capturing ``on_round`` observer does not change the cost.

    A run with the read-only per-round + completion observers must produce the
    identical ``total_cost`` as one without (the observer takes no copy into the
    pending queue and never re-sorts; the CRC recency weighting is untouched).
    """
    config = load_config(_CONFIG)
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"path": str(data_dir)})}
    )

    plain: dict[str, float] = {}
    observed: dict[str, float] = {}
    run_config(config, settle_on_complete=lambda sim: plain.update(sim.cost_breakdown()))
    run_config(
        config,
        settle_on_round=lambda rr: None,
        settle_on_complete=lambda sim: observed.update(sim.cost_breakdown()),
    )

    assert sum(plain.values()) == pytest.approx(sum(observed.values()), abs=1e-9), (
        f"observer changed the cost: plain={sum(plain.values()):.4f} "
        f"observed={sum(observed.values()):.4f}"
    )


def test_config_fidelity_guard() -> None:
    """The winning-loop config resolves to ``strip_private(BEST_CONFIG)`` exactly.

    Four config-seam guards (no model run): the resolved model config equals the
    benchmark's ``strip_private(BEST_CONFIG)`` (every hyperparam + the resolved
    ``lag_transforms`` spec-dict), ``censoring_fit`` survives at the task root,
    ``base_column`` is ``q_0p59`` (divergence #2), and ``weight_decay`` is None
    (DF-90). A dropped transform or flipped flag is a silent model divergence.
    """
    from benchmarks.vn2.config import BEST_CONFIG
    from calibre.forecasting.mlforecast_adapter import _resolve_mlforecast_option

    config = load_config(_CONFIG)
    resolved = config.tasks[0].resolved_model_config()
    expected = strip_private(BEST_CONFIG)

    # censoring_fit survives at the task root (divergence #3): nested in config:
    # it would be silently popped by resolved_model_config.
    assert resolved.pop("censoring_fit") is True

    # base_column (divergence #2) and weight_decay (DF-90) on the order block.
    assert config.order_conformal is not None
    assert config.order_conformal.base_column == "q_0p59"
    assert config.order_conformal.weight_decay is None
    assert config.order_conformal.to_runtime_config().resolved_base_column == "q_0p59"

    # lag_transforms: the YAML spec-dict resolves to the same live transforms.
    resolved_lt = _resolve_mlforecast_option(resolved.pop("lag_transforms"))
    expected_lt = _resolve_mlforecast_option(expected.pop("lag_transforms"))
    assert set(resolved_lt) == set(expected_lt)
    for lag in resolved_lt:
        resolved_specs = [(type(t).__name__, t.window_size) for t in resolved_lt[lag]]
        expected_specs = [(type(t).__name__, t.window_size) for t in expected_lt[lag]]
        assert resolved_specs == expected_specs

    # Every remaining hyperparam matches strip_private(BEST_CONFIG) exactly.
    assert resolved == expected
