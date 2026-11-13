"""Forecasts: the points of each model, series, step and origin."""

from collections.abc import Sequence
from dataclasses import dataclass

import narwhals as nw
import numpy as np
import pandas as pd
from conformalforecast.adapters._nixtla_common import reshape_cv_many


def _labels(value: object, name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy array")
    if value.dtype.kind not in "UO" or value.ndim != 1:
        raise TypeError(f"{name} must be a 1-D array of str")
    if value.dtype.kind == "O" and not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must hold str values")
    labels = np.array(value, dtype=str)
    if len(np.unique(labels)) != len(labels):
        raise ValueError(f"{name} are not unique")
    return labels


@dataclass(frozen=True)
class Forecasts:
    """`points[m, s, h - 1, o]` is the point of model m, series s, step h, origin o."""

    points: np.ndarray  # [M, S, H, O] float32
    models: np.ndarray  # [M] str
    series: np.ndarray  # [S] str: series, or node labels after reconcile
    origins: pd.DatetimeIndex  # [O] strictly increasing

    def __post_init__(self) -> None:
        if not isinstance(self.points, np.ndarray) or self.points.dtype.kind not in "fiu":
            raise TypeError("points must be a numeric numpy array")
        models = _labels(self.models, "models")
        series = _labels(self.series, "series")
        if not isinstance(self.origins, pd.DatetimeIndex):
            raise TypeError("origins must be a pandas DatetimeIndex")
        points = np.array(self.points, dtype=np.float32)
        origins = self.origins.copy(deep=True)
        origins.freq = None
        if points.ndim != 4:
            raise ValueError(f"points must be [M, S, H, O], got shape {points.shape}")
        if points.shape != (len(models), len(series), points.shape[2], len(origins)):
            raise ValueError("points must have one row per model, series and origin")
        if len(models) == 0 or len(series) == 0 or points.shape[2] == 0 or len(origins) == 0:
            raise ValueError("forecasts need at least one model, series, step and origin")
        if origins.hasnans or not (origins[1:] > origins[:-1]).all():
            raise ValueError("origins must be strictly increasing")
        for array in (points, models, series):
            array.flags.writeable = False
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "series", series)
        object.__setattr__(self, "origins", origins)

    @classmethod
    def from_cv(
        cls,
        cv: pd.DataFrame,
        *,
        series: np.ndarray,
        horizon: int,
        models: Sequence[str] | None = None,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
    ) -> "Forecasts":
        """Read a Nixtla cross-validation frame. The origins are its sorted cutoffs."""
        if not isinstance(cv, pd.DataFrame):
            raise TypeError("cv must be a pandas DataFrame")
        labels = _labels(series, "series")
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise TypeError("horizon must be an int")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")
        for name in (id_col, time_col, target_col):
            if not isinstance(name, str):
                raise TypeError("column names must be str")
        reserved = {id_col, time_col, "cutoff", target_col}
        if models is None:
            names = [
                str(column)
                for column in cv.columns
                if column not in reserved
                and "-lo-" not in str(column)
                and "-hi-" not in str(column)
            ]
        else:
            if isinstance(models, str) or not isinstance(models, Sequence):
                raise TypeError("models must be a sequence of str or None")
            if not all(isinstance(name, str) for name in models):
                raise TypeError("models must hold str values")
            names = list(models)
        if not names:
            raise ValueError("cv has no model column")
        if len(set(names)) != len(names):
            raise ValueError("models are not unique")
        missing = [name for name in [id_col, time_col, "cutoff", *names] if name not in cv.columns]
        if missing:
            raise ValueError(f"cv has no column {missing}")
        if names and set(names) & reserved:
            raise ValueError("a model column cannot be an id, time, cutoff or target column")
        if not pd.api.types.is_datetime64_any_dtype(cv["cutoff"]):
            raise ValueError("the cutoff column must hold datetimes")
        if len(cv) == 0:
            raise ValueError("cv has no rows")
        ids = cv[id_col].astype(str).to_numpy()
        if not np.isin(ids, labels).all():
            raise ValueError("cv holds an id that is not in series")
        frame = cv[[id_col, time_col, "cutoff", *names]].copy()
        frame[id_col] = ids
        cubes = reshape_cv_many(
            nw.from_native(frame, eager_only=True),
            names,
            series_ids=tuple(labels.tolist()),
            id_col=id_col,
            time_col=time_col,
            horizon=horizon,
            aligned=True,
        )
        cutoffs = pd.DatetimeIndex(np.unique(cv["cutoff"].to_numpy()))
        points = np.stack(cubes).transpose(0, 1, 3, 2).astype(np.float32)
        return cls(points, np.array(names, dtype=str), labels, cutoffs)

    def select(self, model: str) -> "Forecasts":
        """The forecasts of one model, with M = 1."""
        if not isinstance(model, str):
            raise TypeError("model must be a str")
        where = np.flatnonzero(self.models == model)
        if len(where) == 0:
            raise ValueError(f"unknown model {model!r}")
        index = int(where[0])
        return Forecasts(
            self.points[index : index + 1],
            self.models[index : index + 1],
            self.series,
            self.origins,
        )

    @classmethod
    def concat(cls, forecasts: Sequence["Forecasts"]) -> "Forecasts":
        """Join forecasts along the one axis whose labels differ: models, series or origins.

        The parts have equal labels on the other two axes and the same steps. The origins of
        the result are sorted. The models and the series follow the order of the parts.
        """
        if isinstance(forecasts, str) or not isinstance(forecasts, Sequence):
            raise TypeError("forecasts must be a sequence of Forecasts")
        if not all(isinstance(part, Forecasts) for part in forecasts):
            raise TypeError("forecasts must hold Forecasts")
        if len(forecasts) == 0:
            raise ValueError("concat needs at least one Forecasts")
        first = forecasts[0]
        if len(forecasts) == 1:
            return first
        if any(part.points.shape[2] != first.points.shape[2] for part in forecasts):
            raise ValueError("the parts must have the same number of steps")
        differ = [
            name
            for name in ("models", "series", "origins")
            if any(
                not np.array_equal(getattr(part, name), getattr(first, name)) for part in forecasts
            )
        ]
        if len(differ) != 1:
            raise ValueError("the parts must differ on exactly one axis")
        name = differ[0]
        labels = [getattr(part, name) for part in forecasts]
        joined = labels[0].append(labels[1:]) if name == "origins" else np.concatenate(labels)
        if pd.Index(joined).has_duplicates:
            raise ValueError(f"two parts hold the same {name}")
        axis = {"models": 0, "series": 1, "origins": 3}[name]
        points = np.concatenate([part.points for part in forecasts], axis=axis)
        if name == "origins":
            order = np.argsort(np.asarray(joined), kind="stable")
            points, joined = points[..., order], joined[order]
        fields = {"models": first.models, "series": first.series, "origins": first.origins}
        fields[name] = joined
        return cls(points, **fields)
