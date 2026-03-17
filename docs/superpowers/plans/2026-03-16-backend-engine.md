# BackendEngine Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the walk-forward backtest runtime (BackendEngine) with Fugue execution, Nixtla SeasonalNaive adapter, forecast ledger, and post-hoc scoring.

**Architecture:** Walk-forward loop steps through caller-provided time origins. At each origin, Fugue partitions series by `unique_id` and dispatches fit/predict per model via a `ModelAdapter` protocol. Forecasts accumulate in a `Ledger`; actuals are resolved incrementally (partial observability) and row-level errors are computed on resolution. Aggregate metrics are computed post-hoc.

**Tech Stack:** Python 3.11+, pandas, numpy, Fugue (pandas engine), statsforecast (Nixtla), pytest, uv

**Spec:** `C:\Users\a933186\Vault\calibre\agents\specs\2026-03-16-backend-engine-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `pyproject.toml` | Project metadata, dependencies, tool config |
| `calibre/__init__.py` | Package root |
| `calibre/metrics.py` | Existing metric functions (moved from root) |
| `calibre/contracts/__init__.py` | Contracts subpackage |
| `calibre/contracts/forecast_frame.py` | Column constants + `validate_forecast_frame()` |
| `calibre/tasks/__init__.py` | Tasks subpackage |
| `calibre/tasks/forecast_task.py` | `ForecastTask` frozen dataclass |
| `calibre/models/__init__.py` | Models subpackage |
| `calibre/models/base.py` | `ModelAdapter` Protocol |
| `calibre/models/registry.py` | `ADAPTER_REGISTRY` + `resolve_adapter()` |
| `calibre/models/nixtla.py` | `StatsForecastAdapter` wrapping Nixtla |
| `calibre/engine/__init__.py` | Engine subpackage |
| `calibre/engine/ledger.py` | `Ledger` accumulator + Parquet export |
| `calibre/engine/scoring.py` | `resolve_actuals()`, `compute_row_errors()`, `compute_metrics()` |
| `calibre/engine/backend.py` | `BackendEngine` walk-forward loop |
| `tests/__init__.py` | Tests package |
| `tests/conftest.py` | Shared fixtures |
| `tests/test_forecast_frame.py` | Forecast-frame contract tests |
| `tests/test_forecast_task.py` | ForecastTask tests |
| `tests/test_registry.py` | Registry resolution tests |
| `tests/test_nixtla_adapter.py` | StatsForecast adapter tests |
| `tests/test_ledger.py` | Ledger tests |
| `tests/test_scoring.py` | Scoring tests |
| `tests/test_engine.py` | BackendEngine integration tests |

---

## Chunk 1: Foundation

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `calibre/__init__.py`, `calibre/contracts/__init__.py`, `calibre/tasks/__init__.py`, `calibre/models/__init__.py`, `calibre/engine/__init__.py`
- Create: `tests/__init__.py`
- Move: `metrics.py` -> `calibre/metrics.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "calibre"
version = "0.1.0"
description = "Demand planning engine"
requires-python = ">=3.11"
dependencies = [
    "numpy",
    "pandas",
    "pyarrow",
    "fugue",
    "statsforecast",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
    "mypy",
]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

- [ ] **Step 2: Create package directories with `__init__.py` files**

All `__init__.py` files are empty initially:

```python
# calibre/__init__.py
# calibre/contracts/__init__.py
# calibre/tasks/__init__.py
# calibre/models/__init__.py
# calibre/engine/__init__.py
# tests/__init__.py
```

- [ ] **Step 3: Move metrics.py into the package**

Copy `metrics.py` to `calibre/metrics.py` (identical content). Delete the root `metrics.py`.

- [ ] **Step 4: Install dependencies**

```bash
uv sync
```

- [ ] **Step 5: Verify package imports**

```bash
uv run python -c "from calibre import metrics; print(metrics.mae.__name__)"
```

Expected: `mae`

- [ ] **Step 6: Commit**

```bash
rm metrics.py
git add pyproject.toml calibre/ tests/__init__.py
git commit -m "feat: scaffold calibre package structure with dependencies"
```

---

### Task 2: Forecast-Frame Contract

**Files:**
- Create: `calibre/contracts/forecast_frame.py`
- Create: `tests/test_forecast_frame.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forecast_frame.py
import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME,
    REQUIRED_COLUMNS,
    validate_forecast_frame,
)


def _make_valid_frame(n: int = 3) -> pd.DataFrame:
    return pd.DataFrame({
        UNIQUE_ID: ["SKU_001"] * n,
        DS: pd.date_range("2024-01-07", periods=n, freq="W"),
        Y: np.nan,
        Y_HAT: [10.0, 20.0, 30.0][:n],
        H: list(range(1, n + 1)),
        FORECAST_ORIGIN: pd.Timestamp("2024-01-01"),
        MODEL_NAME: ["SeasonalNaive"] * n,
    })


def test_valid_frame_passes():
    df = _make_valid_frame()
    validate_forecast_frame(df)  # should not raise


def test_missing_column_raises():
    df = _make_valid_frame().drop(columns=[Y_HAT])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_forecast_frame(df)


def test_wrong_dtype_raises():
    df = _make_valid_frame()
    df[H] = df[H].astype(float)  # should be int64
    with pytest.raises(ValueError, match="Column 'h'"):
        validate_forecast_frame(df)


def test_y_allows_nan():
    df = _make_valid_frame()
    df[Y] = np.nan
    validate_forecast_frame(df)  # NaN is float64, should pass


def test_constants_are_strings():
    assert UNIQUE_ID == "unique_id"
    assert DS == "ds"
    assert Y == "y"
    assert Y_HAT == "y_hat"
    assert H == "h"
    assert FORECAST_ORIGIN == "forecast_origin"
    assert MODEL_NAME == "model_name"
    assert len(REQUIRED_COLUMNS) == 7
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_forecast_frame.py -v
```

Expected: ImportError (module doesn't exist yet)

- [ ] **Step 3: Implement forecast_frame.py**

```python
# calibre/contracts/forecast_frame.py
from __future__ import annotations

import pandas as pd

UNIQUE_ID = "unique_id"
DS = "ds"
Y = "y"
Y_HAT = "y_hat"
H = "h"
FORECAST_ORIGIN = "forecast_origin"
MODEL_NAME = "model_name"

REQUIRED_COLUMNS = [UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]

_EXPECTED_DTYPES = {
    UNIQUE_ID: "object",
    DS: "datetime64[ns]",
    Y: "float64",
    Y_HAT: "float64",
    H: "int64",
    FORECAST_ORIGIN: "datetime64[ns]",
    MODEL_NAME: "object",
}


def validate_forecast_frame(df: pd.DataFrame) -> None:
    """Validate that a DataFrame conforms to the forecast-frame contract.

    Raises ValueError if validation fails.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col, expected in _EXPECTED_DTYPES.items():
        actual = str(df[col].dtype)
        if expected == "datetime64[ns]":
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                raise ValueError(f"Column '{col}' expected datetime64, got {actual}")
        elif actual != expected:
            raise ValueError(f"Column '{col}' expected {expected}, got {actual}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_forecast_frame.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add calibre/contracts/forecast_frame.py tests/test_forecast_frame.py
git commit -m "feat: add forecast-frame contract with validation"
```

---

### Task 3: ForecastTask Dataclass

**Files:**
- Create: `calibre/tasks/forecast_task.py`
- Create: `tests/test_forecast_task.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_forecast_task.py
import pandas as pd
import pytest

from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def history():
    return pd.DataFrame({
        "ds": pd.date_range("2024-01-07", periods=10, freq="W"),
        "y": range(10),
    })


def test_create_task(history):
    task = ForecastTask(
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "season_length": 4},
    )
    assert task.unique_id == "SKU_001"
    assert task.horizon == 4
    assert task.forecast_origin is None
    assert task.future_x is None


def test_frozen(history):
    task = ForecastTask(
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive"},
    )
    with pytest.raises(AttributeError):
        task.unique_id = "other"


def test_model_name_from_model_key(history):
    task = ForecastTask(
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "season_length": 4},
    )
    assert task.model_name == "SeasonalNaive"


def test_model_name_from_name_key(history):
    task = ForecastTask(
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "name": "SN_52", "season_length": 52},
    )
    assert task.model_name == "SN_52"


def test_with_forecast_origin(history):
    origin = pd.Timestamp("2024-03-01")
    task = ForecastTask(
        unique_id="SKU_001",
        history=history,
        horizon=4,
        model_config={"model": "SeasonalNaive"},
        forecast_origin=origin,
    )
    assert task.forecast_origin == origin
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_forecast_task.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement forecast_task.py**

```python
# calibre/tasks/forecast_task.py
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ForecastTask:
    unique_id: str
    history: pd.DataFrame
    horizon: int
    model_config: dict
    forecast_origin: pd.Timestamp | None = None
    future_x: pd.DataFrame | None = None

    @property
    def model_name(self) -> str:
        return self.model_config.get("name", self.model_config["model"])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_forecast_task.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add calibre/tasks/forecast_task.py tests/test_forecast_task.py
git commit -m "feat: add ForecastTask frozen dataclass"
```

---

## Chunk 2: Models

### Task 4: ModelAdapter Protocol + Registry

**Files:**
- Create: `calibre/models/base.py`
- Create: `calibre/models/registry.py`
- Create: `tests/test_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_registry.py
import pytest

from calibre.models.registry import resolve_adapter


def test_resolve_seasonal_naive():
    adapter = resolve_adapter({"model": "SeasonalNaive", "season_length": 4})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_auto_arima():
    adapter = resolve_adapter({"model": "AutoARIMA"})
    assert hasattr(adapter, "fit")
    assert hasattr(adapter, "predict")


def test_resolve_unknown_model_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_adapter({"model": "NonExistentModel"})
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement base.py**

```python
# calibre/models/base.py
from __future__ import annotations

from typing import Protocol

import pandas as pd

from calibre.tasks.forecast_task import ForecastTask


class ModelAdapter(Protocol):
    def fit(self, task: ForecastTask) -> None: ...
    def predict(self, task: ForecastTask) -> pd.DataFrame: ...
```

- [ ] **Step 4: Implement registry.py (stub — references nixtla.py which is Task 5)**

```python
# calibre/models/registry.py
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from calibre.models.base import ModelAdapter

_ADAPTER_REGISTRY: dict[str, type] = {}


def _ensure_registry() -> None:
    if _ADAPTER_REGISTRY:
        return
    from calibre.models.nixtla import StatsForecastAdapter

    _ADAPTER_REGISTRY.update({
        "SeasonalNaive": StatsForecastAdapter,
        "AutoARIMA": StatsForecastAdapter,
    })


def resolve_adapter(model_config: dict) -> "ModelAdapter":
    _ensure_registry()
    model_name = model_config["model"]
    if model_name not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    adapter_cls = _ADAPTER_REGISTRY[model_name]
    return adapter_cls(model_config)
```

- [ ] **Step 5: Create a minimal nixtla.py stub so registry can import**

```python
# calibre/models/nixtla.py (stub — full implementation in Task 5)
from __future__ import annotations


class StatsForecastAdapter:
    def __init__(self, model_config: dict) -> None:
        self._config = model_config

    def fit(self, task):
        raise NotImplementedError

    def predict(self, task):
        raise NotImplementedError
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
uv run pytest tests/test_registry.py -v
```

Expected: all 3 tests PASS

- [ ] **Step 7: Commit**

```bash
git add calibre/models/base.py calibre/models/registry.py calibre/models/nixtla.py tests/test_registry.py
git commit -m "feat: add ModelAdapter protocol and adapter registry"
```

---

### Task 5: StatsForecast Adapter

**Files:**
- Modify: `calibre/models/nixtla.py` (replace stub)
- Create: `tests/test_nixtla_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_nixtla_adapter.py
import numpy as np
import pandas as pd
import pytest

from calibre.models.nixtla import StatsForecastAdapter
from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def repeating_history():
    """24 weeks of repeating [10, 20, 30, 40] pattern."""
    dates = pd.date_range("2024-01-07", periods=24, freq="W")
    pattern = [10.0, 20.0, 30.0, 40.0] * 6
    return pd.DataFrame({"ds": dates, "y": pattern})


@pytest.fixture
def sn_task(repeating_history):
    return ForecastTask(
        unique_id="SKU_001",
        history=repeating_history,
        horizon=4,
        model_config={"model": "SeasonalNaive", "season_length": 4, "freq": "W"},
        forecast_origin=pd.Timestamp("2024-06-23"),
    )


def test_fit_predict_returns_correct_columns(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    adapter.fit(sn_task)
    result = adapter.predict(sn_task)

    assert list(result.columns) == ["ds", "y_hat", "h"]
    assert len(result) == 4
    assert result["h"].tolist() == [1, 2, 3, 4]


def test_seasonal_naive_repeats_pattern(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    adapter.fit(sn_task)
    result = adapter.predict(sn_task)

    # SeasonalNaive repeats the last season_length values
    np.testing.assert_array_almost_equal(
        result["y_hat"].values, [10.0, 20.0, 30.0, 40.0]
    )


def test_predict_before_fit_raises(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    with pytest.raises(RuntimeError, match="fit"):
        adapter.predict(sn_task)


def test_y_hat_dtype_is_float64(sn_task):
    adapter = StatsForecastAdapter(sn_task.model_config)
    adapter.fit(sn_task)
    result = adapter.predict(sn_task)
    assert result["y_hat"].dtype == np.float64
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_nixtla_adapter.py -v
```

Expected: FAIL (stub raises NotImplementedError)

- [ ] **Step 3: Implement nixtla.py (replace stub)**

```python
# calibre/models/nixtla.py
from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive

from calibre.tasks.forecast_task import ForecastTask

_NIXTLA_MODELS: dict[str, type] = {
    "SeasonalNaive": SeasonalNaive,
    "AutoARIMA": AutoARIMA,
}


class StatsForecastAdapter:
    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._sf: StatsForecast | None = None

    def fit(self, task: ForecastTask) -> None:
        model_name = self._config["model"]
        model_cls = _NIXTLA_MODELS[model_name]
        params = {
            k: v for k, v in self._config.items() if k not in ("model", "name", "freq")
        }
        model = model_cls(**params)

        freq = self._config.get("freq", "W")
        sf_df = pd.DataFrame({
            "unique_id": task.unique_id,
            "ds": task.history["ds"].values,
            "y": task.history["y"].values.astype("float32"),
        })

        self._sf = StatsForecast(models=[model], freq=freq)
        self._sf.fit(sf_df)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._sf is None:
            raise RuntimeError("Call fit() before predict()")

        result = self._sf.predict(h=task.horizon).reset_index()
        model_cols = [c for c in result.columns if c not in ("unique_id", "ds")]

        return pd.DataFrame({
            "ds": result["ds"],
            "y_hat": result[model_cols[0]].astype("float64"),
            "h": range(1, task.horizon + 1),
        })
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_nixtla_adapter.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add calibre/models/nixtla.py tests/test_nixtla_adapter.py
git commit -m "feat: implement StatsForecast adapter for SeasonalNaive/AutoARIMA"
```

---

## Chunk 3: Engine

### Task 6: Ledger

**Files:**
- Create: `calibre/engine/ledger.py`
- Create: `tests/test_ledger.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ledger.py
import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME,
    REQUIRED_COLUMNS,
)
from calibre.engine.ledger import Ledger


def _make_frame(n: int = 3, origin: str = "2024-01-01") -> pd.DataFrame:
    return pd.DataFrame({
        UNIQUE_ID: ["SKU_001"] * n,
        DS: pd.date_range("2024-01-07", periods=n, freq="W"),
        Y: np.nan,
        Y_HAT: [10.0, 20.0, 30.0][:n],
        H: list(range(1, n + 1)),
        FORECAST_ORIGIN: pd.Timestamp(origin),
        MODEL_NAME: ["SeasonalNaive"] * n,
    })


def test_empty_ledger():
    ledger = Ledger()
    df = ledger.to_df()
    assert list(df.columns) == REQUIRED_COLUMNS
    assert len(df) == 0


def test_append_and_to_df():
    ledger = Ledger()
    ledger.append(_make_frame(3))
    df = ledger.to_df()
    assert len(df) == 3


def test_append_multiple():
    ledger = Ledger()
    ledger.append(_make_frame(2, origin="2024-01-01"))
    ledger.append(_make_frame(2, origin="2024-02-01"))
    df = ledger.to_df()
    assert len(df) == 4


def test_append_validates_schema():
    ledger = Ledger()
    bad_df = pd.DataFrame({"x": [1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        ledger.append(bad_df)


def test_update_resolved():
    ledger = Ledger()
    ledger.append(_make_frame(3))
    updated = ledger.to_df().copy()
    updated.loc[0, Y] = 11.0
    updated["error"] = np.nan
    updated.loc[0, "error"] = 1.0
    ledger.update_resolved(updated)
    df = ledger.to_df()
    assert df.loc[0, Y] == 11.0
    assert df.loc[0, "error"] == 1.0


def test_to_parquet(tmp_path):
    ledger = Ledger()
    ledger.append(_make_frame(3))
    path = str(tmp_path / "test.parquet")
    ledger.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 3
    assert set(REQUIRED_COLUMNS).issubset(set(loaded.columns))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_ledger.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement ledger.py**

```python
# calibre/engine/ledger.py
from __future__ import annotations

import pandas as pd

from calibre.contracts.forecast_frame import REQUIRED_COLUMNS, validate_forecast_frame


class Ledger:
    def __init__(self) -> None:
        self._frames: list[pd.DataFrame] = []

    def append(self, df: pd.DataFrame) -> None:
        validate_forecast_frame(df)
        self._frames.append(df)

    def to_df(self) -> pd.DataFrame:
        if not self._frames:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)
        return pd.concat(self._frames, ignore_index=True)

    def update_resolved(self, df: pd.DataFrame) -> None:
        self._frames = [df]

    def to_parquet(self, path: str) -> None:
        self.to_df().to_parquet(path, index=False)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_ledger.py -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add calibre/engine/ledger.py tests/test_ledger.py
git commit -m "feat: add Ledger accumulator with validation and Parquet export"
```

---

### Task 7: Scoring

**Files:**
- Create: `calibre/engine/scoring.py`
- Create: `tests/test_scoring.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scoring.py
from functools import partial

import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME,
)
from calibre.engine.scoring import (
    compute_metrics,
    compute_row_errors,
    resolve_actuals,
)
from calibre.metrics import mae, mase


def _make_ledger_df() -> pd.DataFrame:
    """Ledger with 4 forecast rows, y=NaN (unresolved)."""
    return pd.DataFrame({
        UNIQUE_ID: ["SKU_001"] * 4,
        DS: pd.date_range("2024-02-25", periods=4, freq="W"),
        Y: np.nan,
        Y_HAT: [40.0, 10.0, 20.0, 30.0],
        H: [1, 2, 3, 4],
        FORECAST_ORIGIN: pd.Timestamp("2024-02-25"),
        MODEL_NAME: ["SeasonalNaive"] * 4,
    })


def _make_actuals() -> pd.DataFrame:
    """Actuals covering weeks 2024-01-07 through 2024-03-31."""
    dates = pd.date_range("2024-01-07", periods=13, freq="W")
    return pd.DataFrame({
        UNIQUE_ID: ["SKU_001"] * 13,
        DS: dates,
        Y: ([10.0, 20.0, 30.0, 40.0] * 4)[:13],
    })


# --- resolve_actuals ---

def test_resolve_fills_y_for_past_dates():
    ledger_df = _make_ledger_df()
    actuals = _make_actuals()
    origin = pd.Timestamp("2024-03-17")  # covers ds for h=1,2,3

    updated, newly_resolved = resolve_actuals(ledger_df, actuals, origin)

    # h=1 (2024-02-25), h=2 (2024-03-03), h=3 (2024-03-10) should be resolved
    # h=4 (2024-03-17) also <= origin, so resolved too
    resolved_mask = updated[Y].notna()
    assert resolved_mask.sum() == 4  # all 4 are <= 2024-03-17


def test_resolve_leaves_future_as_nan():
    ledger_df = _make_ledger_df()
    actuals = _make_actuals()
    origin = pd.Timestamp("2024-02-25")  # only h=1 is <= origin

    updated, newly_resolved = resolve_actuals(ledger_df, actuals, origin)

    assert updated.loc[0, Y] == 40.0  # h=1, ds=2024-02-25
    assert pd.isna(updated.loc[1, Y])  # h=2, ds=2024-03-03, future
    assert len(newly_resolved) == 1


def test_resolve_handles_sparse_actuals():
    ledger_df = _make_ledger_df()
    # Actuals missing for some dates
    actuals = pd.DataFrame({
        UNIQUE_ID: ["SKU_001"],
        DS: [pd.Timestamp("2024-02-25")],
        Y: [40.0],
    })
    origin = pd.Timestamp("2024-03-17")

    updated, newly_resolved = resolve_actuals(ledger_df, actuals, origin)

    assert updated.loc[0, Y] == 40.0  # resolved
    assert pd.isna(updated.loc[1, Y])  # no actual available
    assert len(newly_resolved) == 1


def test_resolve_no_pending_returns_unchanged():
    ledger_df = _make_ledger_df()
    ledger_df[Y] = [40.0, 10.0, 20.0, 30.0]  # all resolved already
    actuals = _make_actuals()
    origin = pd.Timestamp("2024-03-31")

    updated, newly_resolved = resolve_actuals(ledger_df, actuals, origin)

    assert len(newly_resolved) == 0


# --- compute_row_errors ---

def test_row_errors():
    df = pd.DataFrame({
        Y: [40.0, 10.0],
        Y_HAT: [38.0, 12.0],
    })
    result = compute_row_errors(df)

    assert "error" in result.columns
    assert "abs_error" in result.columns
    assert "pct_error" in result.columns
    np.testing.assert_array_almost_equal(result["error"].values, [2.0, -2.0])
    np.testing.assert_array_almost_equal(result["abs_error"].values, [2.0, 2.0])


# --- compute_metrics ---

def test_compute_metrics_groups_by_uid_and_h():
    df = pd.DataFrame({
        UNIQUE_ID: ["A", "A", "B", "B"],
        H: [1, 1, 1, 1],
        Y: [10.0, 12.0, 20.0, 22.0],
        Y_HAT: [11.0, 11.0, 21.0, 21.0],
    })
    result = compute_metrics(df, metrics=[mae], group_by=[UNIQUE_ID])

    assert len(result) == 2
    assert "mae" in result.columns


def test_compute_metrics_with_partial():
    df = pd.DataFrame({
        UNIQUE_ID: ["A"] * 10,
        H: [1] * 10,
        Y: list(range(10)),
        Y_HAT: [x + 1 for x in range(10)],
    })
    mase_52 = partial(mase, seasonality=1)
    result = compute_metrics(df, metrics=[mae, mase_52], group_by=[UNIQUE_ID])

    assert "mae" in result.columns
    assert "mase" in result.columns


def test_compute_metrics_skips_unresolved():
    df = pd.DataFrame({
        UNIQUE_ID: ["A", "A"],
        H: [1, 2],
        Y: [10.0, np.nan],
        Y_HAT: [11.0, 12.0],
    })
    result = compute_metrics(df, metrics=[mae], group_by=[UNIQUE_ID, H])

    # Only h=1 has resolved y, so only 1 group
    assert len(result) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_scoring.py -v
```

Expected: ImportError

- [ ] **Step 3: Implement scoring.py**

```python
# calibre/engine/scoring.py
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from calibre.contracts.forecast_frame import (
    DS,
    UNIQUE_ID,
    Y,
    Y_HAT,
    H,
)


def resolve_actuals(
    ledger_df: pd.DataFrame,
    actuals: pd.DataFrame,
    current_origin: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill y where ds <= current_origin and y is currently NaN.

    Returns (updated_ledger, newly_resolved_rows).
    """
    updated = ledger_df.copy()

    mask_pending = updated[Y].isna() & (updated[DS] <= current_origin)
    if not mask_pending.any():
        return updated, pd.DataFrame(columns=updated.columns)

    # Build actuals lookup indexed by (unique_id, ds)
    lookup = (
        actuals
        .drop_duplicates(subset=[UNIQUE_ID, DS])
        .set_index([UNIQUE_ID, DS])[Y]
    )

    # Map pending rows to actuals
    pending_idx = updated.index[mask_pending]
    pending_keys = pd.MultiIndex.from_arrays(
        [updated.loc[pending_idx, UNIQUE_ID].values,
         updated.loc[pending_idx, DS].values]
    )
    resolved_y = lookup.reindex(pending_keys).values
    updated.loc[pending_idx, Y] = resolved_y

    newly_resolved = updated.loc[pending_idx[updated.loc[pending_idx, Y].notna()]].copy()

    return updated, newly_resolved


def compute_row_errors(resolved_df: pd.DataFrame) -> pd.DataFrame:
    """Add error, abs_error, pct_error columns to resolved rows."""
    df = resolved_df.copy()
    df["error"] = df[Y] - df[Y_HAT]
    df["abs_error"] = df["error"].abs()
    df["pct_error"] = df["error"] / df[Y].replace(0, np.nan)
    return df


def compute_metrics(
    ledger_df: pd.DataFrame,
    metrics: list[Callable],
    group_by: list[str] | None = None,
) -> pd.DataFrame:
    """Compute aggregate metrics on resolved rows, grouped by specified columns."""
    if group_by is None:
        group_by = [UNIQUE_ID, H]

    resolved = ledger_df.dropna(subset=[Y, Y_HAT])
    if resolved.empty:
        return pd.DataFrame()

    results = []
    for keys, group in resolved.groupby(group_by):
        if not isinstance(keys, tuple):
            keys = (keys,)
        actual = group[Y].to_numpy()
        predicted = group[Y_HAT].to_numpy()

        row = dict(zip(group_by, keys))
        for metric_fn in metrics:
            name = getattr(metric_fn, "__name__", None)
            if name is None:
                name = getattr(metric_fn.func, "__name__", str(metric_fn))
            row[name] = metric_fn(actual, predicted)
        results.append(row)

    return pd.DataFrame(results)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_scoring.py -v
```

Expected: all 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add calibre/engine/scoring.py tests/test_scoring.py
git commit -m "feat: add scoring module with resolution, row errors, and metrics"
```

---

### Task 8: BackendEngine + Integration Test

**Files:**
- Create: `calibre/engine/backend.py`
- Create: `tests/conftest.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write conftest.py with shared fixtures**

```python
# tests/conftest.py
import pandas as pd
import pytest


@pytest.fixture
def weekly_dates():
    """20 weeks of weekly dates starting 2024-01-07."""
    return pd.date_range("2024-01-07", periods=20, freq="W")


@pytest.fixture
def repeating_pattern():
    """Repeating [10, 20, 30, 40] for 20 weeks."""
    return [10.0, 20.0, 30.0, 40.0] * 5
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_engine.py
import numpy as np
import pandas as pd
import pytest

from calibre.contracts.forecast_frame import (
    UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME,
)
from calibre.engine.backend import BackendEngine
from calibre.tasks.forecast_task import ForecastTask


@pytest.fixture
def single_series_setup(weekly_dates, repeating_pattern):
    """Single series, single model, two origins."""
    actuals = pd.DataFrame({
        "unique_id": "SKU_001",
        "ds": weekly_dates,
        "y": repeating_pattern,
    })

    task = ForecastTask(
        unique_id="SKU_001",
        history=pd.DataFrame({"ds": weekly_dates, "y": repeating_pattern}),
        horizon=4,
        model_config={"model": "SeasonalNaive", "season_length": 4},
    )

    # Origin at week 8 and week 12
    origins = [weekly_dates[7], weekly_dates[11]]

    return task, actuals, origins


def test_execute_returns_ledger(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()
    assert len(df) == 8  # 4 per origin x 2 origins


def test_forecast_frame_columns_present(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()
    for col in [UNIQUE_ID, DS, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]:
        assert col in df.columns


def test_partial_resolution(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()

    # After origin 1 (week 8): h=1 resolved, h=2-4 pending
    # After origin 2 (week 12): h=2-4 from origin 1 resolved, h=1 from origin 2 resolved
    #   h=2-4 from origin 2: pending (ds > week 12)
    resolved = df[Y].notna().sum()
    unresolved = df[Y].isna().sum()
    assert resolved == 5  # h=1 from origin 1, h=2-4 from origin 1, h=1 from origin 2
    assert unresolved == 3  # h=2-4 from origin 2


def test_error_columns_on_resolved(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()
    resolved = df[df[Y].notna()]

    assert "error" in resolved.columns
    assert "abs_error" in resolved.columns
    # SeasonalNaive on perfect periodic data -> error = 0
    np.testing.assert_array_almost_equal(resolved["error"].dropna().values, 0.0)


def test_model_name_stamped(single_series_setup):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    df = ledger.to_df()
    assert (df[MODEL_NAME] == "SeasonalNaive").all()


def test_multi_series():
    """Two series, single model, single origin."""
    dates = pd.date_range("2024-01-07", periods=20, freq="W")
    pattern_a = [10.0, 20.0, 30.0, 40.0] * 5
    pattern_b = [5.0, 15.0, 25.0, 35.0] * 5

    actuals = pd.concat([
        pd.DataFrame({"unique_id": "A", "ds": dates, "y": pattern_a}),
        pd.DataFrame({"unique_id": "B", "ds": dates, "y": pattern_b}),
    ], ignore_index=True)

    tasks = [
        ForecastTask(
            unique_id="A",
            history=pd.DataFrame({"ds": dates, "y": pattern_a}),
            horizon=4,
            model_config={"model": "SeasonalNaive", "season_length": 4},
        ),
        ForecastTask(
            unique_id="B",
            history=pd.DataFrame({"ds": dates, "y": pattern_b}),
            horizon=4,
            model_config={"model": "SeasonalNaive", "season_length": 4},
        ),
    ]

    engine = BackendEngine(freq="W")
    ledger = engine.execute(tasks, actuals, origins=[dates[11]])

    df = ledger.to_df()
    assert len(df) == 8  # 4 per series x 2 series
    assert set(df[UNIQUE_ID].unique()) == {"A", "B"}


def test_to_parquet_roundtrip(single_series_setup, tmp_path):
    task, actuals, origins = single_series_setup
    engine = BackendEngine(freq="W")
    ledger = engine.execute([task], actuals, origins)

    path = str(tmp_path / "backtest.parquet")
    ledger.to_parquet(path)
    loaded = pd.read_parquet(path)
    assert len(loaded) == 8
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
uv run pytest tests/test_engine.py -v
```

Expected: ImportError

- [ ] **Step 4: Implement backend.py**

```python
# calibre/engine/backend.py
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import fugue.api as fa

from calibre.contracts.forecast_frame import (
    DS,
    FORECAST_ORIGIN,
    H,
    MODEL_NAME,
    REQUIRED_COLUMNS,
    UNIQUE_ID,
    Y,
    Y_HAT,
)
from calibre.engine.ledger import Ledger
from calibre.engine.scoring import compute_row_errors, resolve_actuals
from calibre.models.registry import resolve_adapter
from calibre.tasks.forecast_task import ForecastTask


class BackendEngine:
    def __init__(
        self,
        freq: str = "W",
        metrics: list[Callable] | None = None,
        engine: Any = None,
    ) -> None:
        self.freq = freq
        self.metrics = metrics
        self.engine = engine

    def execute(
        self,
        tasks: list[ForecastTask],
        actuals: pd.DataFrame,
        origins: list[pd.Timestamp],
    ) -> Ledger:
        ledger = Ledger()

        tasks_by_uid: dict[str, list[ForecastTask]] = {}
        for task in tasks:
            tasks_by_uid.setdefault(task.unique_id, []).append(task)

        for origin in origins:
            origin_preds = self._run_origin(tasks_by_uid, origin)

            if not origin_preds.empty:
                ledger.append(origin_preds)

            current = ledger.to_df()
            if current.empty:
                continue

            updated, newly_resolved = resolve_actuals(current, actuals, origin)

            if newly_resolved.empty:
                continue

            scored = compute_row_errors(newly_resolved)
            for col in ("error", "abs_error", "pct_error"):
                if col not in updated.columns:
                    updated[col] = np.nan
                updated.loc[scored.index, col] = scored[col]

            ledger.update_resolved(updated)

        return ledger

    def _run_origin(
        self,
        tasks_by_uid: dict[str, list[ForecastTask]],
        origin: pd.Timestamp,
    ) -> pd.DataFrame:
        uids = list(tasks_by_uid.keys())
        if not uids:
            return pd.DataFrame(columns=REQUIRED_COLUMNS)

        uid_df = pd.DataFrame({UNIQUE_ID: uids})
        freq = self.freq

        def _process_partition(df: pd.DataFrame) -> pd.DataFrame:
            uid = df[UNIQUE_ID].iloc[0]
            results: list[pd.DataFrame] = []

            for task in tasks_by_uid[uid]:
                history = task.history[task.history["ds"] < origin]
                if history.empty:
                    continue

                origin_task = ForecastTask(
                    unique_id=task.unique_id,
                    history=history,
                    horizon=task.horizon,
                    model_config={**task.model_config, "freq": freq},
                    forecast_origin=origin,
                    future_x=task.future_x,
                )

                adapter = resolve_adapter(origin_task.model_config)
                adapter.fit(origin_task)
                preds = adapter.predict(origin_task)

                preds[UNIQUE_ID] = uid
                preds[FORECAST_ORIGIN] = origin
                preds[MODEL_NAME] = origin_task.model_name
                preds[Y] = np.nan

                results.append(
                    preds[[UNIQUE_ID, DS, Y, Y_HAT, H, FORECAST_ORIGIN, MODEL_NAME]]
                )

            if not results:
                return pd.DataFrame(columns=REQUIRED_COLUMNS)

            return pd.concat(results, ignore_index=True)

        schema = (
            f"{UNIQUE_ID}:str,{DS}:datetime,{Y}:double,"
            f"{Y_HAT}:double,{H}:long,"
            f"{FORECAST_ORIGIN}:datetime,{MODEL_NAME}:str"
        )

        result = fa.transform(
            uid_df,
            _process_partition,
            schema=schema,
            partition={"by": UNIQUE_ID},
            engine=self.engine,
        )

        return pd.DataFrame(result)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/test_engine.py -v
```

Expected: all 7 tests PASS

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS across all test files

- [ ] **Step 7: Lint and format**

```bash
uv run ruff check .
uv run ruff format .
```

- [ ] **Step 8: Commit**

```bash
git add calibre/engine/backend.py tests/conftest.py tests/test_engine.py
git commit -m "feat: implement BackendEngine walk-forward loop with Fugue"
```

- [ ] **Step 9: Final commit — all passing, clean**

```bash
uv run pytest -v
git add -A
git status
git commit -m "feat: complete BackendEngine vertical slice — walk-forward backtest runtime"
```
