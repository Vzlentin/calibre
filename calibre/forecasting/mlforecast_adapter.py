from __future__ import annotations

import importlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from calibre.core.forecast_frame import (
    DS,
    UNIQUE_ID,
    Y_HAT,
    H,
    Y,
    exogenous_columns,
    quantile_column,
)
from calibre.core.forecast_task import ForecastTask
from calibre.forecasting.adapter_base import ModelAdapter, _build_predict_frame
from calibre.forecasting.native_persistence import load_dir_from_bytes, save_dir_to_bytes

_RESERVED_KEYS = frozenset(
    {
        "model",
        "name",
        "freq",
        "lags",
        "lag_transforms",
        "target_transforms",
        "backend",
        "scope",
        "quantiles",
        "strategy",
    }
)

_VALID_STRATEGIES = frozenset({"recursive", "direct"})
_METADATA_FILE = "calibre_adapter_state.json"
_TRANSFORM_ALIASES = {
    "RollingMean": "mlforecast.lag_transforms.RollingMean",
    "RollingStd": "mlforecast.lag_transforms.RollingStd",
}
MLForecast: type[Any] | None = None


def _load_mlforecast_cls() -> type[Any]:
    global MLForecast
    if MLForecast is not None:
        return MLForecast
    try:
        from mlforecast import MLForecast as mlforecast_cls
    except ModuleNotFoundError as exc:
        raise ImportError(
            "MLForecastAdapter requires the 'mlforecast' package. "
            "Install calibre with the 'ml' extra to use backend='mlforecast'."
        ) from exc
    MLForecast = mlforecast_cls
    return mlforecast_cls


def _resolve_dotted_cls(dotted_path: str, *, field: str) -> type:
    try:
        module_path, class_name = dotted_path.rsplit(".", 1)
    except ValueError as err:
        raise ValueError(
            f"{field} must be a dotted import path "
            f"(e.g. 'lightgbm.LGBMRegressor'), got: {dotted_path!r}"
        ) from err
    try:
        mod = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module {module_path!r}: {e}") from e
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ValueError(f"Module {module_path!r} has no attribute {class_name!r}")
    return cls


def _resolve_model_cls(dotted_path: str) -> type:
    """Resolve a dotted import path like 'lightgbm.LGBMRegressor' to its class."""
    return _resolve_dotted_cls(dotted_path, field="model")


def _resolve_transform_spec(spec: Any) -> Any:
    if not isinstance(spec, Mapping):
        return spec
    transform_name = spec.get("transform") or spec.get("class") or spec.get("name")
    if transform_name is None:
        return dict(spec)
    dotted_path = _TRANSFORM_ALIASES.get(str(transform_name), str(transform_name))
    cls = _resolve_dotted_cls(dotted_path, field="transform")
    params = {
        key: value for key, value in spec.items() if key not in {"transform", "class", "name"}
    }
    return cls(**params)


def _resolve_mlforecast_option(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            int(key) if isinstance(key, str) and key.isdigit() else key: _resolve_mlforecast_option(
                item
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_transform_spec(item) for item in value]
    return _resolve_transform_spec(value)


def _validate_quantiles(quantiles: list[float]) -> list[float]:
    if not quantiles:
        raise ValueError("quantiles must be a non-empty list of values in (0, 1)")
    for q in quantiles:
        if not 0.0 < float(q) < 1.0:
            raise ValueError(f"quantile {q!r} must lie in (0, 1)")
    return [float(q) for q in quantiles]


def _build_quantile_predict_frame(
    raw: pd.DataFrame,
    name_to_quantile: dict[str, float],
) -> pd.DataFrame:
    """Normalise a Nixtla multi-model predict result to a quantile forecast frame."""
    out = raw[[UNIQUE_ID, DS]].reset_index(drop=True)
    for name, q in name_to_quantile.items():
        out[quantile_column(q)] = raw[name].astype("float64").values

    # Median if requested, else the quantile closest to it
    quantiles = list(name_to_quantile.values())
    point_q = 0.5 if 0.5 in quantiles else min(quantiles, key=lambda q: abs(q - 0.5))
    out[Y_HAT] = out[quantile_column(point_q)].astype("float64").values
    out[H] = out.groupby(UNIQUE_ID).cumcount() + 1
    return out


class MLForecastAdapter(ModelAdapter):
    def __init__(self, model_config: dict) -> None:
        self._config = model_config
        self._mlf: Any | None = None
        self._name_to_quantile: dict[str, float] = {}

    def fit(self, task: ForecastTask) -> None:
        mlforecast_cls = _load_mlforecast_cls()
        model_cls = _resolve_model_cls(self._config["model"])
        params = {k: v for k, v in self._config.items() if k not in _RESERVED_KEYS}

        quantiles = self._config.get("quantiles")
        strategy = self._config.get("strategy", "recursive")
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"strategy must be one of {sorted(_VALID_STRATEGIES)}, got {strategy!r}"
            )

        # mlforecast accepts either a list of estimators (auto-named after the
        # class) or a {name: estimator} dict (used here so each quantile gets
        # a stable column name).
        mlf_models: list[Any] | dict[str, Any]
        if quantiles is not None:
            qs = _validate_quantiles(list(quantiles))
            self._name_to_quantile = {quantile_column(q): q for q in qs}
            mlf_models = {
                col: model_cls(**{**params, "alpha": q})
                for col, q in self._name_to_quantile.items()
            }
        else:
            self._name_to_quantile = {}
            mlf_models = [model_cls(**params)]

        freq = self._config.get("freq", "W")
        lags = self._config.get("lags", list(range(1, task.horizon + 1)))

        mlf_kwargs: dict[str, Any] = {"models": mlf_models, "freq": freq, "lags": lags}
        for opt_key in ("lag_transforms", "target_transforms"):
            if (val := self._config.get(opt_key)) is not None:
                mlf_kwargs[opt_key] = _resolve_mlforecast_option(val)

        mlf_df = task.history[[UNIQUE_ID, DS, Y, *exogenous_columns(task.history)]].copy()
        mlf_df[Y] = mlf_df[Y].astype("float32")

        self._mlf = mlforecast_cls(**mlf_kwargs)
        fit_kwargs: dict[str, Any] = {}
        if strategy == "direct":
            fit_kwargs["max_horizon"] = task.horizon
        self._mlf.fit(mlf_df, **fit_kwargs)

    def dump_state(self) -> bytes:
        if self._mlf is None:
            raise RuntimeError("Call fit() before dump_state()")
        mlf = self._mlf

        def save(path: Path) -> None:
            mlf.save(str(path))
            (path / _METADATA_FILE).write_text(
                json.dumps({"name_to_quantile": self._name_to_quantile}, sort_keys=True),
                encoding="utf-8",
            )

        return save_dir_to_bytes(save)

    def load_state(self, blob: bytes) -> None:
        mlforecast_cls = _load_mlforecast_cls()

        def load(path: Path) -> tuple[Any, dict[str, float]]:
            model = mlforecast_cls.load(str(path))
            metadata_path = path / _METADATA_FILE
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                name_to_quantile = {
                    str(name): float(q) for name, q in metadata.get("name_to_quantile", {}).items()
                }
            else:
                name_to_quantile = {}
            return model, name_to_quantile

        self._mlf, self._name_to_quantile = load_dir_from_bytes(blob, load)

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        if self._mlf is None:
            raise RuntimeError("Call fit() before predict()")
        predict_kwargs: dict[str, Any] = {"h": task.horizon}
        if task.future_x is not None and not task.future_x.empty:
            predict_kwargs["X_df"] = task.future_x
        raw = self._mlf.predict(**predict_kwargs)
        if self._name_to_quantile:
            return _build_quantile_predict_frame(raw, self._name_to_quantile)
        return _build_predict_frame(raw)
