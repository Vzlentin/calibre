"""Configuration for the VN2 inventory planning benchmark."""

from __future__ import annotations

from pathlib import Path

from calibre.conformal.crc import CumulativeConformalRiskConfig
from calibre.conformal.runtime import ConformalPolicyConfig

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "vn2"

MODEL_CONFIGS: list[dict] = [
    {"backend": "statsforecast", "model": "SeasonalNaive", "season_length": 52},
]

HORIZON: int = 3  # protection_period = lead_time(2) + review_period(1)
WARMUP_ORIGINS: int = 6
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

# Order-driven conformal calibration for the tuned global LGBM benchmark.
# This uses a one-sided cumulative residual buffer and writes the calibrated
# upper target into the conformal ``hi_*`` column consumed by R,S. The global
# capped residual correction keeps the calibrated target from double-counting
# uncertainty above the cost-tuned base quantile.
CONFORMAL_ORDER_CONFIG = CumulativeConformalRiskConfig(
    coverage=0.72,
    calibration_window=5000,
    protection_period=LEAD_TIME + REVIEW_PERIOD,
    weight_decay=None,
    buffer_max=0.0,
    method_name="capped_crc",
)

# Tuning (legacy seasonal-naive smoke run; kept to exercise TuningTask wiring)
TUNE_BASE_CONFIG: dict = {
    "backend": "statsforecast",
    "model": "SeasonalNaive",
    "name": "tuned_sn",
}
TUNE_N_TRIALS: int = 6
TUNE_N_ORIGINS: int = 3
TUNE_MAX_WORKERS: int = 4

# ------------------------------------------------------------------ #
# HPO for the tuned global LGBM pipeline (run_benchmark.py).
# Panel-level Optuna sweep over walk-forward origins of week_0; the
# objective is cumulative-horizon pinball loss at the cost-optimal
# tau = Cu / (Cu + Co) = 0.833. Minimising pinball at that tau is, up
# to a constant factor, the deployed newsvendor cost on cumulative
# demand — so the HPO truly optimises what `apply_rs_policy(...,
# quantile=alpha)` deploys: alpha is just the knob the model uses to
# make the per-horizon-summed prediction approximate the cumulative
# 0.833-quantile.
# ------------------------------------------------------------------ #
HPO_N_TRIALS: int = 25
HPO_N_ORIGINS: int = 3
HPO_TIMEOUT_SEC: int = 600
HPO_COST_OPTIMAL_TAU: float = 1.0 / (1.0 + 0.2)  # 0.833 (Cu / (Cu + Co))

# Categorical lag sets explored by the HPO. Lag 52 is a given (annual
# seasonality); the alternatives differ only in how many short / mid
# lags they include.
HPO_LAG_SETS: list[list[int]] = [
    [1, 2, 3, 4, 13, 26, 52],
    [1, 2, 3, 4, 8, 13, 26, 52],
    [*range(1, 14), 26, 52],
]

# Search-space spec consumed inside `run_benchmark.run_hpo`. Each entry
# describes how the trial should sample a value for the named parameter.
# `quantile_alpha` choices were picked empirically: pinball@tau=0.833 of
# cumulative demand is best matched by per-horizon quantiles in the
# 0.45-0.59 band (higher per-horizon alphas overshoot when summed across
# the protection period).
HPO_SEARCH_SPACE: dict = {
    "quantile_alpha": {
        "type": "categorical",
        "choices": [0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59],
    },
    "n_estimators": {"type": "int", "low": 200, "high": 800, "step": 50},
    "learning_rate": {"type": "float", "low": 0.02, "high": 0.10, "log": True},
    "num_leaves": {"type": "categorical", "choices": [15, 31, 63, 127]},
    "min_child_samples": {"type": "int", "low": 10, "high": 60},
    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
    "reg_alpha": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
    "reg_lambda": {"type": "float", "low": 1e-3, "high": 1.0, "log": True},
    "lag_set_idx": {"type": "categorical", "choices": list(range(len(HPO_LAG_SETS)))},
}
