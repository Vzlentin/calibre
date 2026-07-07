"""Drive the guarantee-on VN2 measurement run and its baseline twin (#286).

Runs the production settle loop (``calibre.cli.commands.run_config``) for the
winning-loop config and the guarantee-on variant on the same machine, captures
per-round cost state through the read-only ``settle_on_round`` hook (the CLI
does not expose it), and computes post-hoc realized coverage at tau on both the
raw sales series and the censoring-aware series. Writes all measurement
artifacts to the output directory.

Usage:
    uv run python -m benchmarks.vn2.run_guarantee_on [--out-dir DIR] [--skip-baseline]
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
from pathlib import Path

import pandas as pd

from benchmarks.vn2.guarantee_on_coverage import (
    censoring_aware_panel,
    realized_coverage,
    terminal_bounds,
    window_demand,
)
from calibre.cli.commands import run_config
from calibre.cli.config import load_config
from calibre.execution.dataset_registry import resolve_dataset_adapter
from calibre.execution.decision_loop import RoundResult
from calibre.ordering.simulation.simulator import Simulator

CONFIG_DIR = Path(__file__).resolve().parent / "config"
BASELINE_CONFIG = CONFIG_DIR / "vn2-winning-loop.yaml"
GUARANTEE_ON_CONFIG = CONFIG_DIR / "vn2-winning-loop-guarantee-on.yaml"

logger = logging.getLogger(__name__)


class SettleCapture:
    """Read-only capture of per-round state and the final cost breakdown."""

    def __init__(self) -> None:
        self.rounds: list[RoundResult] = []
        self.holding: float | None = None
        self.shortage: float | None = None

    def on_round(self, rr: RoundResult) -> None:
        self.rounds.append(rr)

    def on_complete(self, simulator: Simulator) -> None:
        breakdown = simulator.cost_breakdown()
        self.holding = float(breakdown.get("holding", 0.0))
        self.shortage = float(breakdown.get("shortage", 0.0))


def run_captured(config_path: Path) -> SettleCapture:
    """Run one settle-loop config with per-round capture."""
    config = load_config(config_path)
    capture = SettleCapture()
    logger.info("running %s ...", config_path.name)
    run_config(config, settle_on_round=capture.on_round, settle_on_complete=capture.on_complete)
    logger.info(
        "%s: holding=%.2f shortage=%.2f total=%.2f",
        config_path.name,
        capture.holding,
        capture.shortage,
        capture.holding + capture.shortage,
    )
    return capture


def per_round_frame(capture: SettleCapture) -> pd.DataFrame:
    """Tabulate per-round cumulative and incremental cost buckets."""
    rows = []
    prev_h, prev_s = 0.0, 0.0
    for rr in capture.rounds:
        h_cum = rr.holding_cost_cum if rr.holding_cost_cum is not None else float("nan")
        s_cum = rr.shortage_cost_cum if rr.shortage_cost_cum is not None else float("nan")
        rows.append(
            {
                "round": rr.round_num,
                "origin": rr.origin,
                "holding_cum": h_cum,
                "shortage_cum": s_cum,
                "cost_cum": h_cum + s_cum,
                "holding_inc": h_cum - prev_h,
                "shortage_inc": s_cum - prev_s,
                "cost_inc": (h_cum - prev_h) + (s_cum - prev_s),
            }
        )
        prev_h, prev_s = h_cum, s_cum
    return pd.DataFrame(rows)


def coverage_tables(
    capture: SettleCapture, data_path: str, protection_period: int
) -> dict[str, pd.DataFrame]:
    """Compute realized coverage on the raw and censoring-aware series."""
    frames = [rr.conformal_frame for rr in capture.rounds if rr.conformal_frame is not None]
    conformal = pd.concat(frames, ignore_index=True)
    bounds = terminal_bounds(conformal, protection_period=protection_period)
    origins = sorted(bounds["forecast_origin"].unique())

    bundle = resolve_dataset_adapter("vn2").load(data_path, period=8)
    raw_demand = window_demand(bundle.history, origins, protection_period)
    censored_panel = censoring_aware_panel(bundle.history, bundle.censoring)
    aware_demand = window_demand(
        censored_panel, origins, protection_period, value_col="y_uncensored"
    )

    return {
        "bounds": bounds,
        "coverage_raw": realized_coverage(bounds, raw_demand),
        "coverage_censoring_aware": realized_coverage(bounds, aware_demand),
    }


def main(argv: list[str] | None = None) -> None:
    """Run the measurement pair and write artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="benchmarks/vn2/results/guarantee-on")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "machine": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "configs": {
            "baseline": str(BASELINE_CONFIG.name),
            "guarantee_on": str(GUARANTEE_ON_CONFIG.name),
        },
        "reference_triple_x86_64_linux": {
            "total": 4992.20,
            "holding": 2488.20,
            "shortage": 2504.00,
        },
    }

    runs = {"guarantee_on": GUARANTEE_ON_CONFIG}
    if not args.skip_baseline:
        runs["baseline"] = BASELINE_CONFIG

    for name, cfg_path in runs.items():
        capture = run_captured(cfg_path)
        rounds = per_round_frame(capture)
        rounds.to_csv(out / f"{name}-per-round.csv", index=False)
        protection = load_config(cfg_path).order_conformal.protection_period
        tables = coverage_tables(capture, data_path="data/vn2", protection_period=protection)
        tables["bounds"].to_parquet(out / f"{name}-bounds.parquet", index=False)
        tables["coverage_raw"].to_csv(out / f"{name}-coverage-raw.csv", index=False)
        tables["coverage_censoring_aware"].to_csv(
            out / f"{name}-coverage-censoring-aware.csv", index=False
        )
        summary[name] = {
            "holding": capture.holding,
            "shortage": capture.shortage,
            "total": capture.holding + capture.shortage,
            "coverage_raw_overall": float(tables["coverage_raw"]["coverage"].iloc[-1]),
            "coverage_censoring_aware_overall": float(
                tables["coverage_censoring_aware"]["coverage"].iloc[-1]
            ),
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("summary: %s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
