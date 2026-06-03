from __future__ import annotations

from pathlib import Path

from calibre.cli.commands import run_config
from calibre.cli.config import load_config_from_mapping
from calibre.execution.validation import validate_dataset_bundle

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "m5"


def test_run_config_smoke_on_m5_fixture(tmp_path: Path) -> None:
    config = load_config_from_mapping(
        {
            "config_schema": "1.0",
            "dataset": {"adapter": "m5", "path": str(_FIXTURE)},
            "tasks": [
                {
                    "model": "SeasonalNaive",
                    "horizon": 1,
                    "config": {"backend": "statsforecast", "season_length": 7},
                }
            ],
            "origins": {"start": "2011-01-30", "end": "2011-01-30", "freq": "D"},
            "output": {
                "ledger_path": str(tmp_path / "m5-ledger.parquet"),
                "streaming": False,
            },
            "execution": {"backend": "local", "seed": 42},
        }
    )
    result = run_config(config)
    ledger_df = result.ledger.to_df()
    assert not ledger_df.empty
    assert (tmp_path / "m5-ledger.parquet").exists()

    from calibre.cli.commands import _load_dataset

    bundle = _load_dataset(config)
    assert bundle.hierarchy is not None
    validate_dataset_bundle(bundle)
