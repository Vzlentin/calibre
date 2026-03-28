"""Configuration for the VN2 inventory planning benchmark."""

from __future__ import annotations

from pathlib import Path

from calibre.conformal.runtime import ConformalPolicyConfig

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vn2"

MODEL_CONFIGS: list[dict] = [
    {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 52},
    {"backend": "statsforecast", "model": "AutoETS"},
    {"backend": "statsforecast", "model": "AutoARIMA"},
    {"backend": "statsforecast", "model": "MFLES", "season_length": 52},
]

CONFORMAL_CONFIG = ConformalPolicyConfig(
    method="aci",
    coverage=0.9,
    gamma=0.05,
    calibration_window=50,
)

HORIZON: int = 3  # protection_period = lead_time(2) + review_period(1)
WARMUP_ORIGINS: int = 20
LEAD_TIME: int = 2
REVIEW_PERIOD: int = 1
DECISION_ROUNDS: int = 6
DELIVERY_WEEKS: int = 2

# Tuning
TUNE_BASE_CONFIG: dict = {
    "backend": "statsforecast",
    "model": "SeasonalNaive",
    "name": "tuned_sn",
}
TUNE_N_TRIALS: int = 20
TUNE_N_ORIGINS: int = 5
TUNE_MAX_WORKERS: int = 4
