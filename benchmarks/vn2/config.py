"""Configuration for the VN2 inventory planning benchmark."""

from __future__ import annotations

from pathlib import Path

from mlforecast.lag_transforms import RollingMean, RollingStd

from calibre.conformal.cumulative_risk import CumulativeConformalRiskConfig
from calibre.conformal.partitions import global_partition, series_partition
from calibre.conformal.runtime import ConformalPolicyConfig
from calibre.core.forecast_frame import UNIQUE_ID

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
    coverage=0.74,
    calibration_window=5000,
    protection_period=LEAD_TIME + REVIEW_PERIOD,
    weight_decay=None,
    buffer_max=0.0,
    method_name="capped_crc",
)

# Explicit name for the current best verified CRC path. Keep
# ``CONFORMAL_ORDER_CONFIG`` as the historical default used by
# ``run_benchmark(tune=False)``.
TOP1_CRC_CONFIG = CONFORMAL_ORDER_CONFIG

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

# Reproducible HPO-best model used by run_benchmark(tune=False). On current
# main, resolved week_0 CRC warmup replays at EUR 4,992.20 total cost.
BEST_CONFIG: dict = {
    "backend": "mlforecast",
    "scope": "global",
    "name": "global_lgbm_qq_0p59",
    "model": "lightgbm.LGBMRegressor",
    "objective": "quantile",
    "quantiles": [0.59],
    "strategy": "direct",
    "lags": [*range(1, 14), 26, 52],
    "lag_transforms": {
        1: [
            *(RollingMean(window_size=w) for w in [4, 13, 26]),
            *(RollingStd(window_size=w) for w in [4, 13, 26]),
        ]
    },
    "n_estimators": 800,
    "learning_rate": 0.029992886777618452,
    "num_leaves": 15,
    "min_child_samples": 41,
    "subsample": 0.8010716092915446,
    "colsample_bytree": 0.6205915004999957,
    "reg_alpha": 0.006853925708853058,
    "reg_lambda": 0.5306371575220844,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
    "_quantile_alpha": 0.59,
}

# Candidate direct cumulative-demand model. The target is the trailing
# protection-period demand aligned to the terminal horizon, so the deployed
# h=3 prediction estimates demand over h=1..3 directly instead of summing
# three independent per-horizon quantiles.
CUMULATIVE_BEST_CONFIG: dict = {
    **BEST_CONFIG,
    "name": "global_lgbm_cumulative_q_0p833",
    "quantiles": [HPO_COST_OPTIMAL_TAU],
    "_quantile_alpha": HPO_COST_OPTIMAL_TAU,
    "_target_mode": "cumulative",
}

CRC_ABLATION_CONFIGS: dict[str, CumulativeConformalRiskConfig | None] = {
    "no_crc": None,
    "current_capped_global_crc": TOP1_CRC_CONFIG,
    "weighted_empirical_crc": CumulativeConformalRiskConfig(
        coverage=0.72,
        calibration_window=5000,
        protection_period=LEAD_TIME + REVIEW_PERIOD,
        partition_key=global_partition,
        weight_decay=0.85,
        method_name="weighted_crc",
    ),
    "farinhas_nonexchangeable_crc": CumulativeConformalRiskConfig(
        coverage=0.72,
        calibration_window=5000,
        protection_period=LEAD_TIME + REVIEW_PERIOD,
        partition_key=global_partition,
        weight_decay=0.85,
        weighted_quantile_mode="nonexchangeable",
        buffer_max=0.0,
        method_name="farinhas_nonexchangeable_crc_capped",
    ),
    "series_crc": CumulativeConformalRiskConfig(
        coverage=0.72,
        calibration_window=5000,
        protection_period=LEAD_TIME + REVIEW_PERIOD,
        partition_key=series_partition,
        weight_decay=None,
        fallback_buffer=0.0,
        method_name="series_crc",
    ),
    "hierarchical_shrinkage_crc": CumulativeConformalRiskConfig(
        coverage=0.72,
        calibration_window=5000,
        protection_period=LEAD_TIME + REVIEW_PERIOD,
        partition_key=lambda row: str(row[UNIQUE_ID]).split("_")[0],
        weight_decay=None,
        shrinkage_strength=0.35,
        method_name="hierarchical_shrinkage_crc",
    ),
}
