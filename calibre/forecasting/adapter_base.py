from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from calibre.core.forecast_frame import (
    DS,
    FITTED_VALUE_COLUMNS,
    FITTED_Y_HAT,
    MODEL_NAME,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    validate_fitted_values_frame,
)
from calibre.core.forecast_task import ForecastTask

if TYPE_CHECKING:
    from calibre.forecasting.cache import ModelArtifactCache


def build_predict_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Nixtla-format predict result to [unique_id, ds, y_hat, h]."""
    model_col = next(c for c in raw.columns if c not in (UNIQUE_ID, DS))
    out = raw[[UNIQUE_ID, DS]].reset_index(drop=True)
    out[Y_HAT] = raw[model_col].astype("float64").values
    out[H] = out.groupby(UNIQUE_ID).cumcount() + 1
    return out


def build_fitted_values_frame(raw: pd.DataFrame, *, model_name: str) -> pd.DataFrame:
    """Normalize a fitted-values result to Calibre's long sidecar contract."""
    model_cols = [c for c in raw.columns if c not in (UNIQUE_ID, DS, Y, H)]
    if len(model_cols) != 1:
        raise ValueError(
            "Expected exactly one fitted-value model column, "
            f"got {model_cols} in columns {list(raw.columns)}"
        )
    model_col = model_cols[0]
    if H in raw.columns:
        raw = raw.sort_values([UNIQUE_ID, DS, H], kind="stable").drop_duplicates(
            [UNIQUE_ID, DS],
            keep="first",
        )
    source_cols = [UNIQUE_ID, DS, Y, model_col]
    out = raw[source_cols].copy()
    out = out.rename(columns={model_col: FITTED_Y_HAT})
    out[UNIQUE_ID] = out[UNIQUE_ID].astype("object")
    out[DS] = pd.to_datetime(out[DS]).astype("datetime64[ns]")
    out[Y] = out[Y].astype("float64")
    out[FITTED_Y_HAT] = out[FITTED_Y_HAT].astype("float64")
    out[MODEL_NAME] = str(model_name)
    out[MODEL_NAME] = out[MODEL_NAME].astype("object")
    out = out.dropna(subset=[Y, FITTED_Y_HAT]).reset_index(drop=True)
    out = out[FITTED_VALUE_COLUMNS]
    validate_fitted_values_frame(out)
    return out


@dataclass(frozen=True)
class PredictionResult:
    forecast: pd.DataFrame
    fitted_values: pd.DataFrame | None = None
    artifact_key: str | None = None


class AdapterCapabilityError(RuntimeError):
    """Raised when an adapter cannot provide a requested pipeline capability."""


class ModelAdapter(ABC):
    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}

    @abstractmethod
    def fit(self, task: ForecastTask, *, collect_fitted_values: bool = False) -> None: ...

    @abstractmethod
    def predict(self, task: ForecastTask) -> pd.DataFrame: ...

    def fitted_values(self, task: ForecastTask) -> pd.DataFrame:
        del task
        raise AdapterCapabilityError(
            f"{type(self).__name__} does not support in-sample fitted values"
        )

    def cache_key(self, task: ForecastTask) -> str:
        """Default identity-hash key over history + horizon + model_config.

        Subclasses can override to incorporate additional adapter-specific
        state (e.g. registered exogenous columns).
        """
        payload = {
            "history": task.history.to_json(
                orient="split",
                date_format="iso",
                double_precision=15,
            ),
            "horizon": int(task.horizon),
            "model_config": task.model_config,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode()).hexdigest()

    def dump_state(self) -> bytes:
        """Serialize the fitted adapter state for caching.

        Default implementation refuses; subclasses must override to opt
        into the artifact cache.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement dump_state; "
            "subclass must override to use ModelArtifactCache."
        )

    def load_state(self, blob: bytes) -> None:
        """Restore adapter state previously produced by ``dump_state``."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement load_state; "
            "subclass must override to use ModelArtifactCache."
        )

    def fit_with_cache(
        self,
        task: ForecastTask,
        cache: ModelArtifactCache | None,
        *,
        collect_fitted_values: bool = False,
    ) -> tuple[bool, str | None]:
        """Fit the adapter, consulting ``cache`` first when supplied.

        Returns ``(fit_ran, key)`` where ``fit_ran`` is ``True`` when ``fit``
        actually ran and ``False`` on a cache hit (state restored from
        ``cache``), and ``key`` is the cache key used, or ``None`` when
        ``cache`` is ``None``.
        """
        if collect_fitted_values:
            # Cached adapter state does not include fitted-value sidecars, so
            # residual reconciliation must run a fresh fit that collects them.
            self.fit(task, collect_fitted_values=True)
            return True, None
        if cache is None:
            self.fit(task)
            return True, None
        key = self.cache_key(task)
        blob = cache.get(key)
        if blob is not None:
            self.load_state(blob)
            return False, key
        self.fit(task)
        cache.put(key, self.dump_state())
        return True, key
