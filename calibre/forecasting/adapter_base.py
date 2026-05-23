from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

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
    @abstractmethod
    def __init__(self, model_config: dict) -> None: ...

    @abstractmethod
    def fit(self, task: ForecastTask) -> None: ...

    @abstractmethod
    def predict(self, task: ForecastTask) -> pd.DataFrame: ...

    def cache_key(self, task: ForecastTask) -> str:
        """Default identity-hash key over fit-affecting task fields.

        Subclasses can override to incorporate additional adapter-specific
        state (e.g. registered exogenous columns). ``cache_key`` is also
        used by adapters that do not opt into persistent caching — it only
        identifies the task, not how to serialize state.
        """
        payload = json.dumps(
            {
                "forecast_origin": _json_safe(task.forecast_origin),
                "future_x": _frame_payload(task.future_x),
                "history": _frame_payload(task.history),
                "horizon": int(task.horizon),
                "model_config": _json_safe(task.model_config),
                "task_group": task.task_group,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()


def _frame_payload(frame: pd.DataFrame | None) -> dict[str, Any] | None:
    if frame is None:
        return None
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    return json.loads(normalized.to_json(orient="split", date_format="iso"))


def _json_safe(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float | int | str | bool) or value is None:
        return value
    return repr(value)


class CacheableAdapter:
    """Opt-in mixin that adds pickle-based state persistence to ``ModelAdapter``.

    Cache persistence is **not** a property of every adapter. Subclasses that
    inherit ``CacheableAdapter`` are explicitly declaring that their
    ``__dict__`` (or their overridden ``dump_state`` / ``load_state``) safely
    captures every value needed to skip ``fit``. Adapters with framework-
    native state (e.g. compiled CUDA kernels, file-backed booster handles)
    must override ``dump_state`` / ``load_state`` to keep that invariant.
    """

    def dump_state(self) -> bytes:
        """Serialize the fitted adapter state for caching.

        Override for a compact or framework-native format. The default
        pickles ``self.__dict__`` and works when every attribute is
        picklable and fully describes the fitted state.
        """
        return pickle.dumps(self.__dict__)

    def load_state(self, blob: bytes) -> None:
        """Restore adapter state previously produced by ``dump_state``."""
        state = pickle.loads(blob)
        if not isinstance(state, dict):
            raise TypeError(f"Invalid adapter state for {type(self).__name__}")
        self.__dict__.update(state)

    def fit_with_cache(
        self,
        task: ForecastTask,
        cache: ModelArtifactCache | None,
    ) -> bool:
        """Fit the adapter, consulting ``cache`` first when supplied.

        Returns ``True`` when ``fit`` actually ran, ``False`` on a cache
        hit (state restored from ``cache``).
        """
        if cache is None:
            self.fit(task)  # type: ignore[attr-defined]
            return True
        key = self.cache_key(task)  # type: ignore[attr-defined]
        blob = cache.get(key)
        if blob is not None:
            self.load_state(blob)
            return False
        self.fit(task)  # type: ignore[attr-defined]
        cache.put(key, self.dump_state())
        return True
