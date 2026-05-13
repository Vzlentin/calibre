# Calibre Structure Migration

Final target layout. Run from repository root. Command snippets assume Git Bash or WSL.

## Prerequisites

bash
uv run pytest          # establish baseline
git checkout -b restructure

---

## Step 1: Create Target Directories

bash
mkdir -p calibre/core
mkdir -p calibre/forecasting/features
mkdir -p calibre/ordering/simulation
mkdir -p calibre/evaluation
mkdir -p calibre/execution
mkdir -p calibre/tuning

---

## Step 2: Move Files

### Core

bash
git mv calibre/contracts/forecast_frame.py calibre/core/forecast_frame.py
git mv calibre/tasks/forecast_task.py     calibre/core/forecast_task.py
git mv calibre/order/types.py             calibre/core/order_types.py

### Conformal (renames for explicitness)

bash
git mv calibre/conformal/aci.py     calibre/conformal/adaptive.py
git mv calibre/conformal/mscp.py    calibre/conformal/split.py
git mv calibre/conformal/crc.py     calibre/conformal/cumulative_risk.py

Extract private numeric helpers from `adaptive.py` into new `calibre/conformal/numerics.py` (see Step 6).

### Evaluation

bash
git mv calibre/metrics.py          calibre/evaluation/point_metrics.py
git mv calibre/eval/metrics.py     calibre/evaluation/forecast_metrics.py

Delete placeholders:

bash
rm calibre/eval/coverage.py calibre/eval/regret.py calibre/eval/service.py
git add calibre/eval/

### Execution

bash
git mv calibre/engine/backend.py              calibre/execution/backend.py
git mv calibre/engine/ledger.py               calibre/execution/ledger.py
git mv calibre/pipeline/dataset.py            calibre/execution/dataset.py
git mv calibre/pipeline/loading.py            calibre/execution/data_loading.py
git mv calibre/pipeline/runner.py             calibre/execution/runner.py
git mv calibre/pipeline/tasks.py              calibre/execution/task_builder.py
git mv calibre/orchestration/decision_loop.py calibre/execution/decision_loop.py

### Forecasting

bash
git mv calibre/models/base.py              calibre/forecasting/adapter_base.py
git mv calibre/models/registry.py          calibre/forecasting/adapter_registry.py
git mv calibre/models/statsforecast.py     calibre/forecasting/statsforecast_adapter.py
git mv calibre/models/mlforecast.py        calibre/forecasting/mlforecast_adapter.py
git mv calibre/models/neuralforecast.py    calibre/forecasting/neuralforecast_adapter.py

Merge ensemble into a single module (see Step 6).

Features:

bash
git mv calibre/features/_helpers.py     calibre/forecasting/features/panel.py
git mv calibre/features/calendar.py     calibre/forecasting/features/calendar_features.py
git mv calibre/features/censoring.py    calibre/forecasting/features/stockout_features.py
git mv calibre/features/lags.py         calibre/forecasting/features/lag_features.py
git mv calibre/features/pipeline.py     calibre/forecasting/features/training_frame.py
git mv calibre/features/scaling.py      calibre/forecasting/features/scaling_features.py
git mv calibre/features/static.py       calibre/forecasting/features/static_features.py
git mv calibre/features/weights.py      calibre/forecasting/features/weight_features.py

### Ordering

bash
git mv calibre/order/_helpers.py     calibre/ordering/decision_frame.py
git mv calibre/order/rules.py        calibre/ordering/decision_rules.py
git mv calibre/order/config.py       calibre/ordering/policy_config.py
git mv calibre/order/protocols.py    calibre/ordering/policy_protocols.py
git mv calibre/order/rs.py           calibre/ordering/periodic_review.py
git mv calibre/order/rss.py          calibre/ordering/reorder_point.py
git mv calibre/order/newsvendor.py   calibre/ordering/newsvendor.py

Simulation (no `inventory_` prefix — namespace is self-describing):

bash
git mv calibre/simulation/costs.py     calibre/ordering/simulation/costs.py
git mv calibre/simulation/results.py   calibre/ordering/simulation/results.py 
git mv calibre/simulation/rules.py     calibre/ordering/simulation/rules.py
git mv calibre/simulation/simulator.py calibre/ordering/simulation/simulator.py
git mv calibre/simulation/state.py     calibre/ordering/simulation/state.py

### Tuning

bash
git mv calibre/tasks/tuning_task.py  calibre/tuning/task.py

Create `calibre/tuning/optimizer.py` with extracted logic (see Step 6).

---

## Step 3: Create / Move `__init__.py` Files

bash
Core: create fresh (old contracts/__init__.py was empty)
cat > calibre/core/__init__.py << 'EOF'
"""Core contracts and shared primitives."""

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    H,
    IN_STOCK,
    MODEL_NAME,
    NONCONFORMITY_SCORE,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y,
    Y_HAT,
    interval_column_names,
    is_quantile_column,
    quantile_column,
    validate_forecast_frame,
)
from calibre.core.forecast_task import ForecastTask
from calibre.core.order_types import (
    CostStruct,
    INVENTORY_POSITION,
    LEAD_TIME,
    OVERAGE_COST,
    REORDER_POINT,
    REVIEW_PERIOD,
    UNDERAGE_COST,
    NewsvendorPolicyParameters,
    RsPolicyParameters,
    RssPolicyParameters,
)

__all__ = [
    "DS",
    "FORECAST_ORIGIN",
    "H",
    "IN_STOCK",
    "MODEL_NAME",
    "NONCONFORMITY_SCORE",
    "REQUIRED_COLUMNS",
    "UNIQUE_ID",
    "Y",
    "Y_HAT",
    "interval_column_names",
    "is_quantile_column",
    "quantile_column",
    "validate_forecast_frame",
    "ForecastTask",
    "CostStruct",
    "INVENTORY_POSITION",
    "LEAD_TIME",
    "OVERAGE_COST",
    "REORDER_POINT",
    "REVIEW_PERIOD",
    "UNDERAGE_COST",
    "NewsvendorPolicyParameters",
    "RsPolicyParameters",
    "RssPolicyParameters",
]
EOF

Forecasting: create fresh
cat > calibre/forecasting/__init__.py << 'EOF'
"""Forecasting: adapters, task definitions, ensembles, feature transforms."""

from calibre.forecasting.adapter_base import ModelAdapter
from calibre.forecasting.adapter_registry import (
    get_adapter_cls,
    get_scope,
    resolve_adapter,
)
from calibre.forecasting.ensemble import (
    ensemble_inverse_error,
    ensemble_median,
    ensemble_weighted,
)

__all__ = [
    "ModelAdapter",
    "get_adapter_cls",
    "get_scope",
    "resolve_adapter",
    "ensemble_inverse_error",
    "ensemble_median",
    "ensemble_weighted",
]
EOF

Forecasting features: move old features/__init__.py and update
git mv calibre/features/__init__.py calibre/forecasting/features/__init__.py
Then edit calibre/forecasting/features/__init__.py to update internal imports.
See Step 7 snippets for the target content.

Evaluation: create fresh (old eval/__init__.py was only a docstring)
cat > calibre/evaluation/__init__.py << 'EOF'
"""Forecast evaluation: point metrics and ledger scoring."""

from calibre.evaluation.forecast_metrics import (
    compute_metrics,
    compute_row_errors,
    resolve_actuals,
)
from calibre.evaluation.point_metrics import METRICS, evaluate, evaluate_all
from calibre.evaluation.point_metrics import (
    mae,
    mape,
    mase,
    mda,
    mdae,
    me,
    mpe,
    mse,
    nrmse,
    rmse,
    rmspe,
    rmsse,
    rrse,
    smape,
    wape,
)

__all__ = [
    "compute_metrics",
    "compute_row_errors",
    "resolve_actuals",
    "METRICS",
    "evaluate",
    "evaluate_all",
    "mae",
    "mape",
    "mase",
    "mda",
    "mdae",
    "me",
    "mpe",
    "mse",
    "nrmse",
    "rmse",
    "rmspe",
    "rmsse",
    "rrse",
    "smape",
    "wape",
]
EOF

Execution: create fresh (old engine/__init__.py was empty)
cat > calibre/execution/__init__.py << 'EOF'
"""Execution layer: backend engine, ledgers, pipeline runners, decision loops."""

from calibre.execution.backend import BackendEngine, BackendResult
from calibre.execution.data_loading import (
    load_master,
    load_period, 
melt_wide_instock,
    melt_wide_sales,
)
from calibre.execution.dataset import DatasetAdapter, DatasetBundle
from calibre.execution.decision_loop import (
    DecisionLoop,
    DecisionLoopConfig,
    RoundResult,
    observe_cumulative,
    observe_per_horizon,
)
from calibre.execution.ledger import ForecastLedger, OrderLedger
from calibre.execution.runner import PipelineResult, run_backtest, run_forecast
from calibre.execution.task_builder import build_tasks

__all__ = [
    "BackendEngine",
    "BackendResult",
    "load_master",
    "load_period",
    "melt_wide_instock",
    "melt_wide_sales",
    "DatasetAdapter",
    "DatasetBundle",
    "DecisionLoop",
    "DecisionLoopConfig",
    "RoundResult",
    "observe_cumulative",
    "observe_per_horizon",
    "ForecastLedger",
    "OrderLedger",
    "PipelineResult",
    "run_backtest",
    "run_forecast",
    "build_tasks",
]
EOF

Ordering: move old order/__init__.py and update
git mv calibre/order/__init__.py calibre/ordering/__init__.py
Then edit to update internal imports. See Step 7 snippets.

Ordering simulation: move old simulation/__init__.py and update
git mv calibre/simulation/__init__.py calibre/ordering/simulation/__init__.py
Then edit to update internal imports. See Step 7 snippets.

Tuning: keep existing, update
cat > calibre/tuning/__init__.py << 'EOF'
"""Hyperparameter tuning: objectives, task specs, and optimization."""

from calibre.tuning.objectives import Accuracy, Cost, Pareto, TuningObjective
from calibre.tuning.optimizer import optimize_task
from calibre.tuning.task import TuningTask

__all__ = [
    "Accuracy",
    "Cost",
    "Pareto",
    "TuningObjective",
    "optimize_task",
    "TuningTask",
]
EOF

Conformal: update existing for renamed modules
cat > calibre/conformal/__init__.py << 'EOF'
from calibre.conformal.adaptive import (
    AdaptiveConformalInference,
    MultiStepAdaptiveConformalInference,
)
from calibre.conformal.cumulative_risk import (
    CumulativeConformalRiskConfig,
    CumulativeConformalRiskRuntime,
)
from calibre.conformal.intervals import symmetric_interval, symmetric_intervals
from calibre.conformal.partitions import (
    category_partition,
    global_partition,
    regime_partition,
    series_partition,
)
from calibre.conformal.policies import OnlineConformalController
from calibre.conformal.runtime import (
    ConformalPolicyConfig,
    ConformalRuntime,
    build_conformal_runtime,
    deserialize_calibration_state,
    serialize_calibration_state,
)
from calibre.conformal.scores import (
    AbsoluteErrorScore,
    ScaledAbsoluteErrorScore,
    absolute_error,
    absolute_error_score,
    scaled_absolute_error,
)
from calibre.conformal.split import (
    CumulativeSplitConformalInference,
    MultiStepSplitConformalInference,
)
from calibre.conformal.types import IntervalPrediction, MultiStepIntervalPrediction
EOF

---

## Step 4: Remove Empty Directories

bash
rm -rf calibre/contracts
rm -rf calibre/engine
rm -rf calibre/orchestration
rm -rf calibre/pipeline
rm -rf calibre/models
rm -rf calibre/features
rm -rf calibre/simulation
rm -rf calibre/order
rm -rf calibre/tasks
rm -rf calibre/eval

---

## Step 5: File Merges & Splits

### Merge: `ensemble/median.py` + `ensemble/weighted.py` → `forecasting/ensemble.py`

Create `calibre/forecasting/ensemble.py` first, then remove originals:

python
"""Ensemble aggregators for multi-model forecast ledgers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from calibre.core.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    MODEL_NAME,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    is_quantile_column,
    validate_forecast_frame,
)

_GROUP_KEYS = [UNIQUE_ID, FORECAST_ORIGIN, DS, H]

 
def ensemble_median(ledger_df: pd.DataFrame, name: str = "ensemble_median") -> pd.DataFrame:
    """Aggregate by median y_hat across models."""
    ...


def ensemble_weighted(
    frames: list[pd.DataFrame],
    weights: list[float],
    name: str = "ensemble_weighted",
) -> pd.DataFrame:
    """Aggregate by weighted linear combination."""
    ...


def ensemble_inverse_error(
    frames: list[pd.DataFrame],
    errors: list[float],
    name: str = "ensemble_inverse_error",
) -> pd.DataFrame:
    """Weighted ensemble with inverse-error weights."""
    ...

Copy bodies from the two original files. Then:

bash
git rm calibre/ensemble/__init__.py calibre/ensemble/median.py calibre/ensemble/weighted.py
rmdir calibre/ensemble

### Extract: `conformal/adaptive.py` → `conformal/numerics.py`

Create `calibre/conformal/numerics.py` with exact signatures from current `aci.py`:

python
"""Private numeric helpers shared across conformal modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np


def _as_scalar_score(score) -> float:
    arr = np.asarray(score, dtype=float).reshape(-1)
    if arr.size != 1:
        raise ValueError("Expected Score to return a scalar score")
    return float(arr[0])


def _as_1d_array(values, name: str, length: int | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        if length is None:
            return arr.reshape(1)
        return np.full(length, float(arr), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1D array")
    if length is not None and arr.shape[0] != length:
        raise ValueError(f"{name} must have length {length}")
    return arr.astype(float, copy=True)


def _validate_bounds(bounds) -> tuple[float, float] | None:
    if bounds is None:
        return None
    lower, upper = bounds
    if lower > upper:
        raise ValueError("alpha_bounds must satisfy lower <= upper")
    return float(lower), float(upper)


def _clip_alpha(alpha, bounds) -> np.ndarray | float:
    if bounds is None:
        arr = np.asarray(alpha, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return arr.astype(float, copy=True)
    lower, upper = bounds
    clipped = np.clip(alpha, lower, upper)
    if np.ndim(clipped) == 0:
        return float(clipped)
    return clipped


def _validate_quantile_rule(quantile_rule: str) -> Literal["conformal", "higher"]:
    if quantile_rule not in {"conformal", "higher"}:
        raise ValueError("quantile_rule must be 'conformal' or 'higher'")
    return quantile_rule  # type: ignore[return-value]


def _finite_sample_radius(
    scores: Iterable[float],
    alpha: float,
    default_radius: float,
    quantile_rule: Literal["conformal", "higher"] = "conformal",
) -> float:
    """Compute the (1-alpha) quantile of scores under the chosen rule."""
    scores_arr = np.asarray(list(scores), dtype=float)
    if scores_arr.size == 0:
        return float(default_radius)
    ordered = np.sort(scores_arr)
    quantile_rule = _validate_quantile_rule(quantile_rule)
    alpha = float(np.asarray(alpha, dtype=float))

    if quantile_rule == "higher":
        if alpha <= 1.0 / (ordered.size + 1):
            return float(np.inf)
        clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
        return float(np.quantile(ordered, 1.0 - clipped_alpha, method="higher"))

    clipped_alpha = float(np.clip(alpha, 0.0, 1.0))
    rank = int(np.ceil((ordered.size + 1) * (1.0 - clipped_alpha))) - 1
    rank = min(max(rank, 0), ordered.size - 1)
    return float(ordered[rank]) 

Update imports in `adaptive.py`, `calibrators.py`, `controllers.py`, and `split.py`:

python
from calibre.conformal.numerics import (
    _as_1d_array,
    _as_scalar_score,
    _clip_alpha,
    _finite_sample_radius,
    _validate_bounds,
    _validate_quantile_rule,
)

### Extract: `tuning/task.py` → `tuning/optimizer.py`

`calibre/tuning/task.py` keeps only the dataclass:

python
"""Tuning task dataclass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import optuna
import pandas as pd

from calibre.conformal import ConformalPolicyConfig
from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, Y
from calibre.core.forecast_task import ForecastTask
from calibre.tuning.objectives import TuningObjective


@dataclass(frozen=True)
class TuningTask:
    unique_id: str
    history: pd.DataFrame
    horizon: int
    base_model_config: dict
    search_space: Callable[[optuna.Trial], dict]
    actuals: pd.DataFrame
    origins: list[pd.Timestamp]
    objective: TuningObjective
    n_trials: int = 50
    freq: str = "W"
    conformal_config: ConformalPolicyConfig | None = None

`calibre/tuning/optimizer.py` gets the execution logic:

python
"""Tuning optimizer: runs Optuna studies for TuningTasks."""

from __future__ import annotations

import optuna
import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine
from calibre.forecasting.adapter_registry import resolve_adapter
from calibre.tuning.objectives import TuningObjective
from calibre.tuning.task import TuningTask


def optimize_task(task: TuningTask) -> dict:
    """Run HPO via Optuna. Returns best model_config dict."""
    ...

Move the two `optimize()` code paths from the old `tuning_task.py` into this function.

---

## Step 6: Constant Migration

Move `IN_STOCK` from `calibre/execution/data_loading.py` into `calibre/core/forecast_frame.py`:

python
calibre/core/forecast_frame.py
IN_STOCK = "in_stock"

Update `calibre/forecasting/features/stockout_features.py`:

python
from calibre.core.forecast_frame import IN_STOCK

Update `calibre/execution/data_loading.py`:

python
from calibre.core.forecast_frame import IN_STOCK

---

## Step 7: Remaining `__init__.py` Content

### `calibre/forecasting/features/__init__.py`

Edit the moved file to update imports:

python
"""Feature engineering transforms."""

from calibre.forecasting.features.calendar_features import add_calendar_features
from calibre.forecasting.features.lag_features import add_lag_features, add_rolling_features
from calibre.forecasting.features.panel import _sort_panel
from calibre.forecasting.features.scaling_features import add_series_scaling
from calibre.forecasting.features.static_features import add_static_features
from calibre.forecasting.features.stockout_features import add_stockout_features
from calibre.forecasting.features.training_frame import build_training_frame
from calibre.forecasting.features.weight_features import add_time_weights

__all__ = [
    "add_calendar_features",
    "add_lag_features",
    "add_rolling_features",
    "_sort_panel",
    "add_series_scaling",
    "add_static_features",
    "add_stockout_features",
    "build_training_frame",
    "add_time_weights",
]

### `calibre/ordering/__init__.py`

Edit the moved file to update imports:

python
"""Ordering policies and inventory simulation."""

from calibre.core.order_types import (
    CostStruct,
    NewsvendorPolicyParameters,
    RsPolicyParameters,
    RssPolicyParameters,
)
from calibre.ordering.decision_frame import _decision_columns, _validate_interval_columns
from calibre.ordering.decision_rules import (
    CumulativeBoundRule, 
QuantileInterpolationRule,
    RSArithmetic,
    RSSArithmetic,
    UpperBoundRule,
)
from calibre.ordering.newsvendor import apply_newsvendor_policy
from calibre.ordering.periodic_review import apply_rs_policy
from calibre.ordering.policy_config import OrderPolicyConfig, apply_order_policy
from calibre.ordering.policy_protocols import DecisionRule, OrderingArithmetic
from calibre.ordering.reorder_point import apply_rss_policy
from calibre.ordering.simulation.simulator import Simulator
from calibre.ordering.simulation.state import ProductState, make_pipeline
from calibre.ordering.simulation.results import PeriodResult
from calibre.ordering.simulation.costs import CostModel, LinearCostModel
from calibre.ordering.simulation.rules import InventoryRule, LostSalesRule

__all__ = [
    "CostStruct",
    "NewsvendorPolicyParameters",
    "RsPolicyParameters",
    "RssPolicyParameters",
    "DecisionRule",
    "OrderingArithmetic",
    "OrderPolicyConfig",
    "apply_order_policy",
    "apply_newsvendor_policy",
    "apply_rs_policy",
    "apply_rss_policy",
    "QuantileInterpolationRule",
    "UpperBoundRule",
    "CumulativeBoundRule",
    "RSArithmetic",
    "RSSArithmetic",
    "Simulator",
    "ProductState",
    "make_pipeline",
    "PeriodResult",
    "CostModel",
    "LinearCostModel",
    "InventoryRule",
    "LostSalesRule",
]

### `calibre/ordering/simulation/__init__.py`

Edit the moved file to update imports:

python
"""Inventory simulation."""

from calibre.ordering.simulation.costs import CostModel, LinearCostModel
from calibre.ordering.simulation.results import PeriodResult
from calibre.ordering.simulation.rules import InventoryRule, LostSalesRule
from calibre.ordering.simulation.simulator import Simulator
from calibre.ordering.simulation.state import ProductState, make_pipeline

__all__ = [
    "CostModel",
    "LinearCostModel",
    "PeriodResult",
    "InventoryRule",
    "LostSalesRule",
    "Simulator",
    "ProductState",
    "make_pipeline",
]

### `calibre/__init__.py`

Keep empty:

python

---

## Step 8: Global Import Replacement

Use this Python codemod script instead of `sed` to avoid corrupting earlier replacements:

python
#!/usr/bin/env python3
"""Rewrite Calibre internal imports after restructure."""

from __future__ import annotations

import sys
from pathlib import Path

# Most specific replacements first.
REPLACEMENTS: list[tuple[str, str]] = [
    # core
    ("calibre.contracts.forecast_frame", "calibre.core.forecast_frame"),
    ("calibre.tasks.forecast_task", "calibre.core.forecast_task"),
    ("calibre.tasks.tuning_task", "calibre.tuning.task"),
    ("calibre.order.types", "calibre.core.order_types"),

    # conformal renames
    ("calibre.conformal.aci", "calibre.conformal.adaptive"),
    ("calibre.conformal.mscp", "calibre.conformal.split"),
    ("calibre.conformal.crc", "calibre.conformal.cumulative_risk"),

    # models → forecasting
    ("calibre.models.statsforecast", "calibre.forecasting.statsforecast_adapter"),
    ("calibre.models.mlforecast", "calibre.forecasting.mlforecast_adapter"),
    ("calibre.models.neuralforecast", "calibre.forecasting.neuralforecast_adapter"),
    ("calibre.models.registry", "calibre.forecasting.adapter_registry"),
    ("calibre.models.base", "calibre.forecasting.adapter_base"),

    # features → forecasting.features
    ("calibre.features._helpers", "calibre.forecasting.features.panel"),
    ("calibre.features.calendar", "calibre.forecasting.features.calendar_features"),
    ("calibre.features.censoring", "calibre.forecasting.features.stockout_features"),
    ("calibre.features.lags", "calibre.forecasting.features.lag_features"),
    ("calibre.features.pipeline", "calibre.forecasting.features.training_frame"), 
    ("calibre.features.scaling", "calibre.forecasting.features.scaling_features"),
    ("calibre.features.static", "calibre.forecasting.features.static_features"),
    ("calibre.features.weights", "calibre.forecasting.features.weight_features"),

    # order → ordering
    ("calibre.order.config", "calibre.ordering.policy_config"),
    ("calibre.order._helpers", "calibre.ordering.decision_frame"),
    ("calibre.order.rules", "calibre.ordering.decision_rules"),
    ("calibre.order.protocols", "calibre.ordering.policy_protocols"),
    ("calibre.order.newsvendor", "calibre.ordering.newsvendor"),
    ("calibre.order.rs", "calibre.ordering.periodic_review"),
    ("calibre.order.rss", "calibre.ordering.reorder_point"),

    # evaluation
    ("calibre.eval.metrics", "calibre.evaluation.forecast_metrics"),
    ("calibre.metrics", "calibre.evaluation.point_metrics"),

    # simulation → ordering.simulation
    ("calibre.simulation.costs", "calibre.ordering.simulation.costs"),
    ("calibre.simulation.results", "calibre.ordering.simulation.results"),
    ("calibre.simulation.rules", "calibre.ordering.simulation.rules"),
    ("calibre.simulation.simulator", "calibre.ordering.simulation.simulator"),
    ("calibre.simulation.state", "calibre.ordering.simulation.state"),

    # engine / pipeline / orchestration → execution
    ("calibre.engine.backend", "calibre.execution.backend"),
    ("calibre.engine.ledger", "calibre.execution.ledger"),
    ("calibre.orchestration.decision_loop", "calibre.execution.decision_loop"),
    ("calibre.pipeline.dataset", "calibre.execution.dataset"),
    ("calibre.pipeline.loading", "calibre.execution.data_loading"),
    ("calibre.pipeline.runner", "calibre.execution.runner"),
    ("calibre.pipeline.tasks", "calibre.execution.task_builder"),

    # ensemble → forecasting
    ("calibre.ensemble.median", "calibre.forecasting.ensemble"),
    ("calibre.ensemble.weighted", "calibre.forecasting.ensemble"),

    # broad package-level rewrites (must be last)
    ("calibre.models", "calibre.forecasting"),
    ("calibre.features", "calibre.forecasting.features"),
    ("calibre.simulation", "calibre.ordering.simulation"),
    ("calibre.pipeline", "calibre.execution"),
    ("calibre.orchestration", "calibre.execution"),
    ("calibre.ensemble", "calibre.forecasting.ensemble"),
]


def rewrite_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    text = original
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def main() -> int:
    changed = 0
    for pattern in ("calibre", "tests", "benchmarks"):
        base = Path(pattern)
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if rewrite_file(path):
                print(f"  rewritten: {path}")
                changed += 1
    print(f"\nDone. {changed} files changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

Save as `rewrite_imports.py` and run:

bash
python rewrite_imports.py
Check for leftovers:

bash
rg "calibre\.(contracts|tasks|models|features|order|eval|metrics|simulation|engine|pipeline|orchestration|ensemble)\b" calibre tests benchmarks
---

## Step 9: Verify

bash
uv run ruff check .
uv run mypy calibre/
uv run pytest
Fix any remaining import errors manually. The codemod should catch >95% of renames.

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Top-level subpackages | 13 | 8 |
| Files renamed | 0 | ~22 |
| Files merged | 0 | 2 (ensemble) |
| Files extracted | 0 | 2 (numerics, optimizer) |
| Files deleted (placeholders) | 0 | 3 |
| Directories removed | 0 | 9 |
| Import cycles introduced | — | 0 |
