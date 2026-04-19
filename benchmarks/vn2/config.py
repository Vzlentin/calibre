"""Configuration for the VN2 inventory planning benchmark."""

from __future__ import annotations

from pathlib import Path

from calibre.conformal.runtime import ConformalPolicyConfig

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vn2"

MODEL_CONFIGS: list[dict] = [
    {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 52},
]

# Cost-optimal service level: Cu / (Cu + Co) = 1.0 / (1.0 + 0.2) ≈ 0.833
# Aligns conformal intervals with the asymmetric cost structure (shortage 5× holding).
CONFORMAL_CONFIG = ConformalPolicyConfig(
    method="aci",
    coverage=0.833,
    gamma=0.05,
    calibration_window=50,
)

HORIZON: int = 3  # protection_period = lead_time(2) + review_period(1)
WARMUP_ORIGINS: int = 6
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
TUNE_N_TRIALS: int = 6
TUNE_N_ORIGINS: int = 3
TUNE_MAX_WORKERS: int = 4
