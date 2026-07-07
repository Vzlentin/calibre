"""Tests for the guarantee-on VN2 measurement apparatus (#286).

The realized-coverage computation is new apparatus: it must be validated on
hand-checkable fixtures, not trusted ad hoc. The config-construction test
asserts the guarantee-on variant reaches the runtime with coverage at tau and
the buffer clamp absent (field assertions, not floats). No test here trains a
model or touches the pinned baselines.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmarks.vn2.guarantee_on_coverage import (
    censoring_aware_panel,
    detect_bound_column,
    realized_coverage,
    terminal_bounds,
    window_demand,
)
from calibre.cli.config import load_config
from calibre.core.forecast_frame import DS, IN_STOCK, UNIQUE_ID, Y

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "vn2" / "config"
_GUARANTEE_ON = _CONFIG_DIR / "vn2-winning-loop-guarantee-on.yaml"
_WINNING = _CONFIG_DIR / "vn2-winning-loop.yaml"


class TestGuaranteeOnConfig:
    """The variant differs from the winning config by exactly the two knobs."""

    def test_constructs_runtime_at_tau_with_clamp_absent(self) -> None:
        config = load_config(_GUARANTEE_ON)
        assert config.order_conformal is not None
        runtime = config.order_conformal.to_runtime_config()
        assert runtime.coverage == 0.833
        assert runtime.buffer_max is None
        assert runtime.weight_decay is None
        assert runtime.base_column == "q_0p59"
        assert runtime.protection_period == 3

    def test_ordering_coverage_matches_conformal(self) -> None:
        config = load_config(_GUARANTEE_ON)
        assert config.ordering is not None
        assert config.ordering.coverage == 0.833

    def test_only_the_two_knobs_differ_from_winning(self) -> None:
        """Field-by-field: everything but coverage and buffer_max is identical."""
        variant = load_config(_GUARANTEE_ON)
        winning = load_config(_WINNING)

        v_oc = variant.order_conformal.model_dump()
        w_oc = winning.order_conformal.model_dump()
        assert v_oc.pop("coverage") == 0.833 and w_oc.pop("coverage") == 0.74
        assert v_oc.pop("buffer_max") is None and w_oc.pop("buffer_max") == 0.0
        # method_name is a cosmetic provenance label, deliberately renamed.
        v_oc.pop("method_name"), w_oc.pop("method_name")
        assert v_oc == w_oc

        v_ord = variant.ordering.model_dump()
        w_ord = winning.ordering.model_dump()
        assert v_ord.pop("coverage") == 0.833 and w_ord.pop("coverage") == 0.74
        assert v_ord == w_ord

        assert variant.tasks == winning.tasks
        assert variant.origins == winning.origins
        assert variant.execution == winning.execution


def _conformal_fixture() -> pd.DataFrame:
    """Two series x two origins; hi_0p833 present only on terminal h=3 rows."""
    rows = []
    for origin, (b_a, b_b) in {
        pd.Timestamp("2024-04-15"): (10.0, 5.0),
        pd.Timestamp("2024-04-22"): (8.0, 4.0),
    }.items():
        for uid, bound in [("A", b_a), ("B", b_b)]:
            for h in (1, 2, 3):
                rows.append(
                    {
                        UNIQUE_ID: uid,
                        "forecast_origin": origin,
                        "h": h,
                        "hi_0p833": bound if h == 3 else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


class TestTerminalBounds:
    def test_extracts_terminal_rows_only(self) -> None:
        bounds = terminal_bounds(_conformal_fixture(), protection_period=3)
        assert len(bounds) == 4
        got = bounds.set_index([UNIQUE_ID, "forecast_origin"])["bound"]
        assert got[("A", pd.Timestamp("2024-04-15"))] == 10.0
        assert got[("B", pd.Timestamp("2024-04-22"))] == 4.0

    def test_detect_bound_column_rejects_ambiguity(self) -> None:
        frame = _conformal_fixture()
        frame["hi_0p74"] = 1.0
        with pytest.raises(ValueError, match="exactly one"):
            detect_bound_column(frame)

    def test_nan_terminal_bound_raises(self) -> None:
        frame = _conformal_fixture()
        frame.loc[(frame[UNIQUE_ID] == "A") & (frame["h"] == 3), "hi_0p833"] = float("nan")
        with pytest.raises(ValueError, match="NaN bound"):
            terminal_bounds(frame, protection_period=3)


def _weekly_panel() -> pd.DataFrame:
    """Two series over five Mondays with hand-summable values."""
    weeks = pd.date_range("2024-04-15", periods=5, freq="W-MON")
    rows = []
    for uid, values in [("A", [1, 2, 3, 4, 5]), ("B", [10, 0, 10, 0, 10])]:
        rows += [{UNIQUE_ID: uid, DS: w, Y: float(v)} for w, v in zip(weeks, values, strict=True)]
    return pd.DataFrame(rows)


class TestWindowDemand:
    def test_hand_checkable_window_sums(self) -> None:
        origins = [pd.Timestamp("2024-04-15"), pd.Timestamp("2024-04-22")]
        demand = window_demand(_weekly_panel(), origins, protection_period=3)
        got = demand.set_index([UNIQUE_ID, "forecast_origin"])["demand"]
        # A: 1+2+3 = 6 from 04-15; 2+3+4 = 9 from 04-22.
        assert got[("A", origins[0])] == 6.0
        assert got[("A", origins[1])] == 9.0
        # B: 10+0+10 = 20; 0+10+0 = 10.
        assert got[("B", origins[0])] == 20.0
        assert got[("B", origins[1])] == 10.0

    def test_incomplete_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window weeks missing"):
            window_demand(_weekly_panel(), [pd.Timestamp("2024-05-06")], protection_period=3)


class TestRealizedCoverage:
    def test_hand_checkable_per_origin_and_overall(self) -> None:
        o1, o2 = pd.Timestamp("2024-04-15"), pd.Timestamp("2024-04-22")
        bounds = pd.DataFrame(
            {
                UNIQUE_ID: ["A", "B", "A", "B"],
                "forecast_origin": [o1, o1, o2, o2],
                "bound": [6.0, 19.0, 9.0, 10.0],
            }
        )
        demand = pd.DataFrame(
            {
                UNIQUE_ID: ["A", "B", "A", "B"],
                "forecast_origin": [o1, o1, o2, o2],
                "demand": [6.0, 20.0, 9.0, 10.0],
            }
        )
        # o1: A covered (6<=6), B not (20>19) -> 1/2. o2: both covered -> 2/2.
        table = realized_coverage(bounds, demand)
        per_origin = table.dropna(subset=["forecast_origin"]).set_index("forecast_origin")
        assert per_origin.loc[o1, "coverage"] == 0.5
        assert per_origin.loc[o2, "coverage"] == 1.0
        overall = table[table["forecast_origin"].isna()].iloc[0]
        assert overall["n"] == 4 and overall["covered"] == 3
        assert overall["coverage"] == 0.75

    def test_join_mismatch_raises(self) -> None:
        o1 = pd.Timestamp("2024-04-15")
        bounds = pd.DataFrame({UNIQUE_ID: ["A"], "forecast_origin": [o1], "bound": [6.0]})
        demand = pd.DataFrame({UNIQUE_ID: ["B"], "forecast_origin": [o1], "demand": [1.0]})
        with pytest.raises(ValueError, match="join mismatch"):
            realized_coverage(bounds, demand)


class TestCensoringAwarePanel:
    def test_oos_week_imputed_to_expanding_median_floor(self) -> None:
        """OOS week with observed 0 lifts to the expanding median of prior in-stock sales."""
        weeks = pd.date_range("2024-04-15", periods=4, freq="W-MON")
        sales = pd.DataFrame(
            {
                UNIQUE_ID: ["A"] * 4,
                DS: weeks,
                Y: [4.0, 2.0, 0.0, 5.0],
            }
        )
        instock = pd.DataFrame(
            {
                UNIQUE_ID: ["A"] * 4,
                DS: weeks,
                IN_STOCK: [True, True, False, True],
            }
        )
        panel = censoring_aware_panel(sales, instock)
        got = panel.set_index(DS)["y_uncensored"]
        # In-stock weeks pass through; the OOS week imputes to
        # max(observed 0, expanding median of in-stock sales so far = median(4,2) = 3).
        assert got[weeks[0]] == 4.0
        assert got[weeks[1]] == 2.0
        assert got[weeks[2]] == 3.0
        assert got[weeks[3]] == 5.0

    def test_window_sums_diverge_between_series(self) -> None:
        """The raw and censoring-aware window sums differ exactly by the imputation."""
        weeks = pd.date_range("2024-04-15", periods=3, freq="W-MON")
        sales = pd.DataFrame({UNIQUE_ID: ["A"] * 3, DS: weeks, Y: [4.0, 2.0, 0.0]})
        instock = pd.DataFrame({UNIQUE_ID: ["A"] * 3, DS: weeks, IN_STOCK: [True, True, False]})
        origin = [pd.Timestamp("2024-04-15")]
        raw = window_demand(sales, origin, protection_period=3)
        aware = window_demand(
            censoring_aware_panel(sales, instock),
            origin,
            protection_period=3,
            value_col="y_uncensored",
        )
        assert raw["demand"].iloc[0] == 6.0
        assert aware["demand"].iloc[0] == 9.0  # 4 + 2 + imputed 3
