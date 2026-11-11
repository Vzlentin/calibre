"""Expected values from statsforecast, conformalforecast and hierarchicalforecast.

Run as `python -I references.py TASK IN.npz OUT.npz`. This script imports no file of the
repository and never imports calibre, so its values are independent of the code under test.

Tasks:
- cv: `StatsForecast.cross_validation` with `refit=True`, one result per job.
- windows: `model.forecast(y=window, h=h)` of a model on each window, with its fallback.
- quantile: `conformal_quantile` of the scores of each cell, one cell at a time, +inf for a
  cell below its minimum.
- reconcile: `BottomUpSparse` and `MinTrace(method="wls_struct")`, bottoms last.
"""

import json
import sys
from typing import Any, cast

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsforecast.models
from conformalforecast._quantiles import conformal_quantile
from conformalforecast.base import InsufficientCalibrationSamples
from hierarchicalforecast.methods import BottomUpSparse, MinTrace
from statsforecast import StatsForecast


def model(spec):
    if spec is None:
        return None
    name, kwargs = spec
    return getattr(statsforecast.models, name)(**kwargs)


def frame(arrays, prefix):
    return pd.DataFrame(
        {
            "unique_id": arrays[f"{prefix}/unique_id"],
            "ds": pd.to_datetime(arrays[f"{prefix}/ds"]),
            "y": arrays[f"{prefix}/y"].astype(np.float32),
        }
    )


def cv(arrays):
    out = {}
    for k, job in enumerate(json.loads(str(arrays["jobs"]))):
        engine = StatsForecast(
            models=[model(spec) for spec in job["models"]],
            freq=job["freq"],
            fallback_model=model(job["fallback"]),
            n_jobs=1,
        )
        result = engine.cross_validation(
            df=frame(arrays, job["frame"]),
            h=job["h"],
            step_size=job["step_size"],
            n_windows=job["n_windows"],
            input_size=job["input_size"],
            refit=True,
        )
        assert isinstance(result, pd.DataFrame)
        if "unique_id" not in result.columns:
            result = result.reset_index()
        out[f"{k}/columns"] = np.array([str(c) for c in result.columns])
        for column in result.columns:
            values = result[column].to_numpy()
            if column in ("ds", "cutoff"):
                values = values.astype("datetime64[ns]")
            elif column == "unique_id":
                values = values.astype(str)
            out[f"{k}/col/{column}"] = values
    return out


def windows(arrays):
    out = {}
    for k, job in enumerate(json.loads(str(arrays["jobs"]))):
        primary = model(job["model"])
        fallback = model(job["fallback"])
        data = arrays[f"{k}/y"]
        offsets = arrays[f"{k}/offsets"]
        means = np.empty((len(offsets) - 1, job["h"]), np.float64)
        for w in range(len(offsets) - 1):
            y = data[offsets[w] : offsets[w + 1]]
            try:
                res = primary.forecast(y=y, h=job["h"])
            except Exception:
                if fallback is None:
                    raise
                res = fallback.forecast(y=y, h=job["h"])
            means[w] = res["mean"]
        out[f"{k}/mean"] = means
    return out


def quantile(arrays):
    scores = arrays["scores"]
    offsets = arrays["offsets"]
    alphas = arrays["alphas"]
    threshold = np.full(len(alphas), np.inf, np.float32)
    ready = np.zeros(len(alphas), bool)
    for i, alpha in enumerate(alphas):
        cell = scores[offsets[i] : offsets[i + 1]]
        if cell.size == 0:
            continue
        try:
            value = conformal_quantile(cell[None, :, None], float(alpha))
        except InsufficientCalibrationSamples:
            continue
        assert value.dtype == np.float32
        threshold[i] = value[0, 0]
        ready[i] = True
    return {"threshold": threshold, "ready": ready}


def reconcile(arrays):
    summing = arrays["summing"].astype(np.float64)
    y_hat = arrays["y_hat"].astype(np.float64)
    sparse = cast(Any, sp.csr_matrix(summing))  # BottomUpSparse takes a sparse S
    bottom_up = BottomUpSparse().fit_predict(S=sparse, y_hat=y_hat)["mean"]
    wls = MinTrace(method="wls_struct").fit_predict(S=summing, y_hat=y_hat)["mean"]
    return {"bottom_up": np.asarray(bottom_up), "wls_struct": np.asarray(wls)}


TASKS = {"cv": cv, "windows": windows, "quantile": quantile, "reconcile": reconcile}


def main():
    task, source, target = sys.argv[1:]
    with np.load(source, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    out = TASKS[task](arrays)
    out["calibre_loaded"] = np.array(any(m.split(".")[0] == "calibre" for m in sys.modules))
    np.savez(target, **out)


if __name__ == "__main__":
    main()
