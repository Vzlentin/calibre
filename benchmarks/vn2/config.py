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
    {
        "backend": "mlforecast",
        "model": "lightgbm.LGBMRegressor",
        "name": "lgbm",
        "lags": [1, 2, 3, 4, 13, 26, 52],
        "n_estimators": 200,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbosity": -1,
    },
]

HORIZON: int = 3  # protection_period = lead_time(2) + review_period(1)
WARMUP_ORIGINS: int = 20
LEAD_TIME: int = 2
REVIEW_PERIOD: int = 1
DECISION_ROUNDS: int = 6
DELIVERY_WEEKS: int = 2

# Cost-optimal service level: Cu / (Cu + Co) = 1.0 / (1.0 + 0.2) ≈ 0.833
# Cumulative mode bounds Σdemand over the protection period directly,
# avoiding the per-horizon-sum inflation from independent bounds.
CONFORMAL_CONFIG = ConformalPolicyConfig(
    method="mscp",
    coverage=0.833,
    calibration_window=50,
    mode="cumulative",
    protection_period=LEAD_TIME + REVIEW_PERIOD,
)

# Tuning
TUNE_BASE_CONFIG: dict = {
    "backend": "statsforecast",
    "model": "SeasonalNaive",
    "name": "tuned_sn",
}
TUNE_N_TRIALS: int = 20
TUNE_N_ORIGINS: int = 5
TUNE_MAX_WORKERS: int = 4
