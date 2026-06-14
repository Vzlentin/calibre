from __future__ import annotations

from pathlib import Path

import pandas as pd

from calibre.cli.commands import run_config
from calibre.cli.config import load_config
from calibre.conformal.runtime import SymmetricIntervalConfig
from calibre.core.forecast_frame import (
    CONFORMAL_MODE,
    H,
    interval_column_names,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WINNING_CONFIG = _REPO_ROOT / "benchmarks" / "vn2" / "config" / "winning.yaml"
_CUMULATIVE_CONFIG = _REPO_ROOT / "benchmarks" / "vn2" / "config" / "cumulative.yaml"
_SMOKE_CUMULATIVE_CONFIG = _REPO_ROOT / "benchmarks" / "vn2" / "config" / "smoke-cumulative.yaml"


def test_vn2_cumulative_config_selects_engine_internal_cumulative_mode() -> None:
    config = load_config(_CUMULATIVE_CONFIG)

    assert config.dataset.adapter == "vn2"
    assert config.dataset.path == "data/vn2"
    assert config.dataset.period == 8
    assert config.conformal is not None
    assert config.conformal.method == "mscp"
    assert config.conformal.mode == "cumulative"
    assert config.conformal.protection_period == 3
    assert config.conformal.coverage == 0.9
    assert config.conformal.calibration_window == 10
    assert config.conformal.partition == "series"
    assert config.tasks[0].horizon == 3
    assert config.output.ledger_path == "results/vn2/cumulative-ledger.parquet"
    assert config.execution.backend == "auto"

    runtime_config = config.conformal.to_runtime_config()
    # Constructing the runtime config proves the cumulative mscp + protection
    # invariant (SymmetricIntervalConfig.__post_init__) is satisfied.
    assert isinstance(runtime_config, SymmetricIntervalConfig)
    assert runtime_config.mode == "cumulative"
    assert runtime_config.protection_period == 3


def test_vn2_winning_config_baseline_carries_no_conformal_section() -> None:
    """Guard: the cumulative addition must not perturb the 4992.20 baseline.

    The winning config drives the VN2 regression baseline; it deliberately
    carries no engine-internal conformal section. Asserting its loaded shape
    here is cheap insurance that the new cumulative configs are strictly
    additive and never silently leak into the baseline run.
    """
    config = load_config(_WINNING_CONFIG)

    assert config.conformal is None
    assert config.dataset.adapter == "vn2"
    assert config.dataset.path == "data/vn2"
    assert config.dataset.period == 8
    assert config.tasks[0].model == "global_lgbm_qq_0p59"
    assert config.tasks[0].horizon == 3
    assert config.output.ledger_path == "results/vn2/ledger.parquet"
    assert config.output.order_ledger_path == "results/vn2/orders.parquet"


def test_vn2_smoke_cumulative_config_parses_as_source_cli_config() -> None:
    config = load_config(_SMOKE_CUMULATIVE_CONFIG)

    assert config.dataset.adapter == "vn2"
    assert config.conformal is not None
    assert config.conformal.mode == "cumulative"
    assert config.conformal.protection_period == 2
    assert config.tasks[0].horizon == 2


def test_vn2_smoke_cumulative_config_executes_cumulative_path(tmp_path: Path) -> None:
    ledger_path = tmp_path / "results" / "vn2" / "smoke-cumulative-ledger.parquet"
    config = load_config(_SMOKE_CUMULATIVE_CONFIG)
    config = config.model_copy(
        update={"output": config.output.model_copy(update={"ledger_path": str(ledger_path)})}
    )
    assert config.conformal is not None
    protection_period = config.conformal.protection_period
    assert protection_period == 2
    lower_col, upper_col = interval_column_names(config.conformal.coverage)

    result = run_config(config)
    ledger = result.ledger.to_df()

    assert ledger_path.exists()
    assert not ledger.empty
    written = pd.read_parquet(ledger_path)
    assert len(written) == len(ledger)

    # The cumulative branch executed (not perhorizon).
    assert set(ledger[CONFORMAL_MODE]) == {"cumulative"}

    # The cumulative bound is written only on the terminal-H row (H ==
    # protection_period) of each (uid, model, origin) group, never on earlier-H
    # rows — directly asserting _apply_cumulative's terminal-row contract.
    earlier = ledger[ledger[H] < protection_period]
    assert earlier[lower_col].isna().all()
    assert earlier[upper_col].isna().all()

    # Once the per-series calibrator has warmed up, at least one terminal-H row
    # carries a populated, finite bound.
    terminal = ledger[ledger[H] == protection_period]
    populated = terminal[terminal[lower_col].notna()]
    assert not populated.empty
    for _, row in populated.iterrows():
        assert row[lower_col] <= row[upper_col]

    # The incomplete-window deferral contract is not reproducible at smoke shape
    # (horizon == protection_period makes every window complete); it is covered
    # by the runtime's own unit tests, not asserted here.
