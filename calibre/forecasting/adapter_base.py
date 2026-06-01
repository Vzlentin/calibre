from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H
from calibre.core.forecast_task import ForecastTask

if TYPE_CHECKING:
    from calibre.forecasting.cache import ModelArtifactCache


def _build_predict_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Nixtla-format predict result to [unique_id, ds, y_hat, h]."""
    model_col = next(c for c in raw.columns if c not in (UNIQUE_ID, DS))
    out = raw[[UNIQUE_ID, DS]].reset_index(drop=True)
    out[Y_HAT] = raw[model_col].astype("float64").values
    out[H] = out.groupby(UNIQUE_ID).cumcount() + 1
    return out


class ModelAdapter(ABC):
    def __init__(self, model_config: dict | None = None) -> None:
        self.model_config = model_config or {}

    @abstractmethod
    def fit(self, task: ForecastTask) -> None: ...

    @abstractmethod
    def predict(self, task: ForecastTask) -> pd.DataFrame: ...

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
    ) -> tuple[bool, str | None]:
        """Fit the adapter, consulting ``cache`` first when supplied.

        Returns ``(fit_ran, key)`` where ``fit_ran`` is ``True`` when ``fit``
        actually ran and ``False`` on a cache hit (state restored from
        ``cache``), and ``key`` is the cache key used, or ``None`` when
        ``cache`` is ``None``.
        """
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
