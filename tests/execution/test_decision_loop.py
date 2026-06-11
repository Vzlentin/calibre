"""Tests for calibre.execution.decision_loop."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    UNIQUE_ID,
    Y,
    interval_column_names,
)
from calibre.core.forecast_task import TaskGroups
from calibre.execution.decision_loop import (
    DecisionLoop,
    DecisionLoopConfig,
    RoundResult,
    observe_cumulative,
    observe_pending,
    observe_per_horizon,
)

_ORIGIN = pd.Timestamp("2023-01-09")  # Monday


def _make_frame(
    uid: str,
    origin: pd.Timestamp,
    horizon: int = 2,
    y_vals: list[float | None] | None = None,
    lower: list[float | None] | None = None,
    upper: list[float | None] | None = None,
    model_name: str = "m",
) -> pd.DataFrame:
    """Build a minimal conformal forecast frame."""
    ds_vals = [origin + pd.Timedelta(weeks=h) for h in range(1, horizon + 1)]
    df = pd.DataFrame(
        {
            UNIQUE_ID: uid,
            DS: ds_vals,
            FORECAST_ORIGIN: origin,
            MODEL_NAME: model_name,
            Y: y_vals if y_vals is not None else [None] * horizon,
        }
    )
    lo_col, hi_col = interval_column_names(0.9)
    df[lo_col] = lower if lower is not None else [None] * horizon
    df[hi_col] = upper if upper is not None else [None] * horizon
    return df


def _actuals_lookup(uid_ds_pairs: list[tuple[str, pd.Timestamp, float]]) -> pd.Series:
    idx = pd.MultiIndex.from_tuples([(uid, ds) for uid, ds, _ in uid_ds_pairs])
    return pd.Series([v for _, _, v in uid_ds_pairs], index=idx, dtype=float)


class TestDecisionLoopSmoke:
    def test_rounds_called_and_results_returned(self) -> None:
        """Loop calls each collaborator once per decision round."""
        n_rounds = 3
        calls: dict[str, list[int]] = {"build": [], "policy": [], "actuals": []}
        sim_calls: list[dict] = []

        ledger_df = pd.DataFrame()
        fake_result = MagicMock()
        fake_result.ledger.to_df.return_value = ledger_df

        fake_engine = MagicMock()
        fake_engine.freq = "W-MON"
        fake_engine.execute.return_value = fake_result

        fake_sim = MagicMock()
        fake_sim.step.side_effect = lambda period, orders, actual_demand: sim_calls.append(
            {"period": period, "orders": orders, "actual_demand": actual_demand}
        )

        def build(rn: int):
            calls["build"].append(rn)
            return (TaskGroups(), _ORIGIN + pd.Timedelta(weeks=rn), pd.DataFrame())

        def policy(frame: pd.DataFrame) -> dict[str, float]:
            calls["policy"].append(1)
            return {"A": 5.0}

        def get_actuals(rn: int) -> dict[str, float]:
            calls["actuals"].append(rn)
            return {"A": 10.0}

        config = DecisionLoopConfig(n_rounds=n_rounds)
        loop = DecisionLoop(
            engine=fake_engine,
            simulator=fake_sim,
            build_round_tasks=build,
            policy=policy,
            get_actuals=get_actuals,
            config=config,
        )
        results = loop.run()

        assert len(results) == n_rounds
        assert calls["build"] == list(range(1, n_rounds + 1))
        assert len(calls["policy"]) == n_rounds
        assert calls["actuals"] == list(range(1, n_rounds + 1))

        # The loop must forward the policy's orders and realised demand into the
        # simulator each round, and surface the same on each RoundResult.
        assert [c["period"] for c in sim_calls] == list(range(1, n_rounds + 1))
        assert all(c["orders"] == {"A": 5.0} for c in sim_calls)
        assert all(c["actual_demand"] == {"A": 10.0} for c in sim_calls)
        assert [r.round_num for r in results] == list(range(1, n_rounds + 1))
        assert all(r.orders == {"A": 5.0} for r in results)
        assert all(r.actual_demand == {"A": 10.0} for r in results)
        assert all(isinstance(r, RoundResult) for r in results)

    def test_delivery_rounds_use_zero_orders(self) -> None:
        n_delivery = 2
        sim_calls: list[dict] = []

        fake_result = MagicMock()
        fake_result.ledger.to_df.return_value = pd.DataFrame()
        fake_engine = MagicMock()
        fake_engine.freq = "W-MON"
        fake_engine.execute.return_value = fake_result
        fake_sim = MagicMock()
        fake_sim.step.side_effect = lambda period, orders, actual_demand: sim_calls.append(
            {"period": period, "orders": orders}
        )

        loop = DecisionLoop(
            engine=fake_engine,
            simulator=fake_sim,
            build_round_tasks=lambda rn: (TaskGroups(), _ORIGIN, pd.DataFrame()),
            policy=lambda f: {"A": 1.0},
            get_actuals=lambda rn: {"A": 5.0},
            config=DecisionLoopConfig(n_rounds=1, n_delivery_rounds=n_delivery),
        )
        loop.run()

        # Step 1 = decision round (orders = {"A": 1.0})
        # Steps 2, 3 = delivery rounds (orders = zero)
        assert sim_calls[0]["orders"] == {"A": 1.0}
        for call in sim_calls[1:]:
            assert all(v == 0.0 for v in call["orders"].values())

    def test_on_round_fires_once_per_round(self) -> None:
        fired: list[RoundResult] = []

        fake_result = MagicMock()
        fake_result.ledger.to_df.return_value = pd.DataFrame()
        fake_engine = MagicMock()
        fake_engine.freq = "W-MON"
        fake_engine.execute.return_value = fake_result

        loop = DecisionLoop(
            engine=fake_engine,
            simulator=MagicMock(),
            build_round_tasks=lambda rn: (TaskGroups(), _ORIGIN, pd.DataFrame()),
            policy=lambda f: {},
            get_actuals=lambda rn: {},
            config=DecisionLoopConfig(n_rounds=3, on_round=fired.append),
        )
        loop.run()

        assert len(fired) == 3
        assert [r.round_num for r in fired] == [1, 2, 3]


class TestObservePerHorizon:
    def _make_runtime(self) -> MagicMock:
        rt = MagicMock()
        rt.observe.return_value = None
        return rt

    def test_resolved_rows_are_observed(self) -> None:
        """Rows with actuals + both interval columns are forwarded to observe."""
        lo_col, hi_col = interval_column_names(0.9)
        frame = _make_frame(
            "A",
            _ORIGIN,
            horizon=1,
            y_vals=[None],
            lower=[10.0],
            upper=[20.0],
        )
        lookup = _actuals_lookup([("A", _ORIGIN + pd.Timedelta(weeks=1), 15.0)])
        rt = self._make_runtime()

        remaining = observe_per_horizon(rt, [frame], lookup, lo_col, hi_col)

        rt.observe.assert_called_once()
        assert remaining == []

    def test_partial_window_stays_pending(self) -> None:
        """Rows missing actuals remain in pending; observe is not called."""
        lo_col, hi_col = interval_column_names(0.9)
        frame = _make_frame("A", _ORIGIN, horizon=2, lower=[10.0, 11.0], upper=[20.0, 21.0])
        # Only fill actuals for h=1, not h=2
        lookup = _actuals_lookup([("A", _ORIGIN + pd.Timedelta(weeks=1), 5.0)])
        rt = self._make_runtime()

        remaining = observe_per_horizon(rt, [frame], lookup, lo_col, hi_col)

        # h=1 is resolved → observed; h=2 remains
        rt.observe.assert_called_once()
        assert len(remaining) == 1
        assert len(remaining[0]) == 1  # only h=2 row

    def test_missing_interval_columns_keeps_frame_pending(self) -> None:
        """Frame without interval columns is kept pending, observe not called."""
        lo_col, hi_col = interval_column_names(0.9)
        frame = pd.DataFrame(
            {UNIQUE_ID: ["A"], DS: [_ORIGIN], FORECAST_ORIGIN: [_ORIGIN], Y: [None]}
        )
        lookup = _actuals_lookup([("A", _ORIGIN, 5.0)])
        rt = self._make_runtime()

        remaining = observe_per_horizon(rt, [frame], lookup, lo_col, hi_col)

        rt.observe.assert_not_called()
        assert len(remaining) == 1


class TestObserveCumulative:
    def _make_runtime(self) -> MagicMock:
        rt = MagicMock()
        rt.observe.return_value = None
        return rt

    def test_complete_window_is_observed(self) -> None:
        """A window where all rows have actuals triggers observe."""
        frame = _make_frame("A", _ORIGIN, horizon=2)
        lookup = _actuals_lookup(
            [
                ("A", _ORIGIN + pd.Timedelta(weeks=1), 3.0),
                ("A", _ORIGIN + pd.Timedelta(weeks=2), 4.0),
            ]
        )
        rt = self._make_runtime()

        remaining = observe_cumulative(rt, [frame], lookup)

        rt.observe.assert_called_once()
        assert remaining == []

    def test_incomplete_window_stays_pending(self) -> None:
        """A window with partial actuals is not observed and stays pending."""
        frame = _make_frame("A", _ORIGIN, horizon=2)
        # Only h=1 resolved
        lookup = _actuals_lookup([("A", _ORIGIN + pd.Timedelta(weeks=1), 3.0)])
        rt = self._make_runtime()

        remaining = observe_cumulative(rt, [frame], lookup)

        rt.observe.assert_not_called()
        assert len(remaining) == 1


class TestObservePending:
    """The mode-keyed dispatcher routes to the matching per-mode helper."""

    def _make_runtime(self, mode: str) -> MagicMock:
        rt = MagicMock()
        rt.observe.return_value = None
        rt.mode = mode
        rt.interval_columns = interval_column_names(0.9)
        return rt

    def test_cumulative_mode_routes_to_cumulative_helper(self) -> None:
        """mode="cumulative" applies window-completeness: a partial window
        stays pending and observe is never called."""
        frame = _make_frame("A", _ORIGIN, horizon=2)
        # Only h=1 resolved → window incomplete under cumulative semantics.
        lookup = _actuals_lookup([("A", _ORIGIN + pd.Timedelta(weeks=1), 3.0)])
        rt = self._make_runtime("cumulative")

        remaining = observe_pending(rt, [frame], lookup)

        rt.observe.assert_not_called()
        assert len(remaining) == 1
        assert len(remaining[0]) == 2  # whole window retained, not split per-row

    def test_perhorizon_mode_routes_with_columns_from_interval_columns(self) -> None:
        """mode="perhorizon" applies per-row readiness using the bound columns
        derived from runtime.interval_columns."""
        lo_col, hi_col = interval_column_names(0.9)
        frame = _make_frame("A", _ORIGIN, horizon=2, lower=[10.0, 11.0], upper=[20.0, 21.0])
        # Only h=1 resolved → that single row is observed; h=2 stays pending.
        lookup = _actuals_lookup([("A", _ORIGIN + pd.Timedelta(weeks=1), 5.0)])
        rt = self._make_runtime("perhorizon")

        remaining = observe_pending(rt, [frame], lookup)

        rt.observe.assert_called_once()
        observed = rt.observe.call_args.args[0]
        assert observed[lo_col].notna().all()
        assert len(remaining) == 1
        assert len(remaining[0]) == 1  # only the h=2 row stays pending

    def test_pending_passed_through_untouched(self, monkeypatch) -> None:
        """The dispatcher must not re-group/re-sort/pre-process pending before
        delegating: the helper receives the exact list object and lookup."""
        frame = _make_frame("A", _ORIGIN, horizon=2)
        pending = [frame]
        lookup = _actuals_lookup([("A", _ORIGIN + pd.Timedelta(weeks=1), 3.0)])
        rt = self._make_runtime("cumulative")

        seen: dict[str, object] = {}

        def _spy(runtime, p, actuals_lookup):
            seen["pending"] = p
            seen["lookup"] = actuals_lookup
            return p

        monkeypatch.setattr(
            "calibre.execution.decision_loop.observe_cumulative", _spy
        )
        observe_pending(rt, pending, lookup)

        assert seen["pending"] is pending
        assert seen["lookup"] is lookup
