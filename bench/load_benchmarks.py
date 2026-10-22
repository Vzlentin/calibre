"""The additional benchmark files to Calibre arrays; no dataset logic in calibre.

`load` accepts raw files; `load_prepared` reads the download script's local output.
Undated matrices require explicit positional-time consent. Missing targets fail
unless the caller explicitly chooses causal forward-fill and warm-up trimming.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from calibre.data import Hierarchy, Panel

MATRICES = {
    "electricity": ("electricity.txt.gz", "h", 26304, 321),
    "traffic": ("traffic.txt.gz", "h", 17544, 862),
    "solar": ("solar_AL.txt.gz", "10min", 52560, 137),
    "exchange": ("exchange_rate.txt.gz", "D", 7588, 8),
}
FRESH_ATTRIBUTES = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]
LEVELS = {
    "favorita": ["state", ["state", "city"], "store_nbr", "family", ["state", "family"]],
    "freshretail": [
        "city_id",
        "store_id",
        "management_group_id",
        "first_category_id",
        "second_category_id",
        "third_category_id",
        "product_id",
        ["city_id", "product_id"],
    ],
}


@dataclass(frozen=True)
class Benchmark:
    """Ready panel, matching hierarchy, static attributes, and original-target mask.

    `observed` [B, T] is False for imputed cells: exclude them from evaluation.
    `panel.censored` marks observed sales as lower bounds, not recovered demand.
    """

    panel: Panel
    hierarchy: Hierarchy
    attributes: pd.DataFrame
    observed: np.ndarray
    metadata: dict


def _attributes(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    attributes = frame[["series", *columns]].drop_duplicates().set_index("series")
    if not attributes.index.is_unique or attributes.isna().to_numpy().any():
        raise ValueError("static series attributes must be unique and nonmissing")
    return attributes


def _wide(frame: pd.DataFrame, value: str = "value") -> pd.DataFrame:
    # pivot rejects duplicate (series, period) cells rather than silently summing them.
    return frame.pivot(index="period", columns="series", values=value)


def _hierarchy(name: str, panel: Panel, attributes: pd.DataFrame) -> Hierarchy:
    if name in LEVELS:
        return Hierarchy.from_attributes(panel.series, attributes, LEVELS[name])
    return Hierarchy.flat(panel.series)


def load(
    name: str,
    path: str | Path,
    *,
    missing: str = "raise",
    positional_time: bool = False,
    limit_series: int | None = None,
) -> Benchmark:
    """Read one exact benchmark variant. All target preprocessing is explicit.

    FreshRetailNet joins its 90 train days and 7 eval days but does not fit anything;
    respect the stored split when choosing origins. No future covariates are supplied.
    """
    if missing not in ("raise", "ffill"):
        raise ValueError("missing must be 'raise' or 'ffill'")
    if limit_series is not None and limit_series < 1:
        raise ValueError("limit_series must be positive")
    path = Path(path)
    attributes = pd.DataFrame()
    censored = None
    metadata = {
        "dataset": name,
        "missing_policy": missing,
        "time_axis": "calendar",
        "warmup_periods_dropped": 0,
    }
    if name in MATRICES:
        if not positional_time:
            raise ValueError(
                f"{name} has no timestamps. Pass positional_time=True / --positional-time; "
                "the synthetic dates must not be used as real calendar features."
            )
        filename, freq, rows, columns = MATRICES[name]
        values = np.loadtxt(path / filename, delimiter=",", dtype=np.float32, ndmin=2)
        if values.shape != (rows, columns):
            raise ValueError(f"unexpected {name} matrix shape: {values.shape}")
        # Panel requires dates; these are positional labels, not reconstructed source dates.
        wide = pd.DataFrame(
            values,
            index=pd.date_range("2000-01-01", periods=rows, freq=freq),
            columns=[f"column_{i + 1:03d}" for i in range(columns)],
        )
        metadata["time_axis"] = "positional (synthetic dates; no calendar features)"
        metadata["step"] = "one observation" if name == "exchange" else freq
        metadata["target"] = {
            "electricity": "published client load (processed units not asserted)",
            "traffic": "sensor occupancy proportion, not vehicle counts",
            "solar": "published Solar-AL production (physical units not asserted)",
            "exchange": "published exchange-rate levels, original column order",
        }[name]
    elif name == "etth1":
        frame = pd.read_csv(path / "ETTh1.csv", parse_dates=["date"])
        wide = frame.set_index("date")[["OT"]]
        freq = "h"
        metadata["target"] = "OT (published oil temperature)"
    elif name == "beijing":
        frames = []
        with ZipFile(path / "beijing.zip") as archive:
            members = sorted(n for n in archive.namelist() if n.endswith(".csv"))
            if len(members) != 12:
                raise ValueError("expected the 12-station PRSA2017 UCI archive")
            for member in members:
                with archive.open(member) as stream:
                    frame = pd.read_csv(stream)
                frame["period"] = pd.to_datetime(frame[["year", "month", "day", "hour"]])
                frames.append(frame.rename(columns={"station": "series", "PM2.5": "value"}))
        wide = _wide(pd.concat(frames, ignore_index=True))
        freq = "h"
        metadata["target"] = "PM2.5 (micrograms/m3)"
    elif name == "bike":
        with ZipFile(path / "bike.zip") as archive:
            members = [n for n in archive.namelist() if Path(n).name == "hour.csv"]
            if len(members) != 1:
                raise ValueError("expected exactly one hour.csv in Bike Sharing archive")
            with archive.open(members[0]) as stream:
                frame = pd.read_csv(stream)
        periods = pd.to_datetime(frame["dteday"]) + pd.to_timedelta(frame["hr"], unit="h")
        wide = frame[["cnt"]].set_axis(periods)
        freq = "h"
        metadata["target"] = "cnt (recorded rentals, not unconstrained demand)"
    elif name == "favorita":
        frame = pd.read_csv(path / "train.csv", usecols=["date", "store_nbr", "family", "sales"])
        frame["series"] = frame["store_nbr"].astype(str) + ":" + frame["family"]
        frame["period"] = pd.to_datetime(frame["date"])
        stores = pd.read_csv(path / "stores.csv")
        frame = frame.merge(stores, on="store_nbr", how="left", validate="many_to_one")
        attributes = _attributes(frame, ["store_nbr", "family", "state", "city", "type", "cluster"])
        wide = _wide(frame, "sales")
        freq = "D"
        metadata["target"] = "store-family sales (fractional quantities preserved)"
        metadata["unlabelled_test"] = "2017-08-16 through 2017-08-31; not included"
    elif name == "freshretail":
        columns = [*FRESH_ATTRIBUTES, "dt", "sale_amount", "stock_hour6_22_cnt"]
        try:
            frame = pd.concat(
                [
                    pd.read_parquet(path / f"{split}.parquet", columns=columns, engine="pyarrow")
                    for split in ("train", "eval")
                ],
                ignore_index=True,
            )
        except ImportError as error:
            raise ImportError(
                "FreshRetailNet needs `uv run --no-dev --with pyarrow ...`"
            ) from error
        frame["series"] = frame["store_id"].astype(str) + ":" + frame["product_id"].astype(str)
        frame["period"] = pd.to_datetime(frame["dt"])
        attributes = _attributes(frame, FRESH_ATTRIBUTES)
        wide = _wide(frame, "sale_amount")
        stockouts = frame["stock_hour6_22_cnt"]
        if stockouts.isna().any() or not stockouts.between(0, 17).all():
            raise ValueError("invalid FreshRetailNet stockout-hour counts")
        frame["censored"] = stockouts > 0
        censored = _wide(frame, "censored")
        freq = "D"
        metadata.update(
            {
                "target": "globally normalized daily sales, not latent demand or physical units",
                "eval_start": "2024-06-26",
                "train_end": "2024-06-25",
                "censoring": "stock_hour6_22_cnt > 0 (annotated stockouts between 06:00 and 22:00)",
            }
        )
    else:
        raise ValueError(
            f"unknown additional benchmark: {name!r}; M5/VN2 use their existing loaders"
        )

    wide = wide.sort_index().sort_index(axis=1)
    if not wide.index.is_unique:
        raise ValueError("duplicate periods in target frame")
    if limit_series is not None:
        wide = wide.iloc[:, :limit_series]
    calendar = pd.date_range(wide.index[0], wide.index[-1], freq=freq)
    if not wide.index.isin(calendar).all():
        raise ValueError(f"off-grid timestamps for frequency {freq}")
    wide = wide.reindex(calendar)
    observed = wide.notna()
    metadata["missing_cells_before_fill"] = int((~observed).to_numpy().sum())
    if wide.isna().to_numpy().any():
        if missing == "raise":
            raise ValueError(
                f"{name}: {metadata['missing_cells_before_fill']} missing targets/calendar cells; "
                "choose missing='ffill' / --missing ffill explicitly; never silently fill zero"
            )
        wide = wide.ffill()
        valid = wide.notna().all(axis=1).to_numpy()
        if not valid.any():
            raise ValueError("no period has a causal observation for every selected series")
        first = int(np.flatnonzero(valid)[0])
        metadata["warmup_periods_dropped"] = first
        wide = wide.iloc[first:]
    observed = observed.loc[wide.index, wide.columns].to_numpy(dtype=bool).T
    if attributes.empty:
        attributes = pd.DataFrame(index=pd.Index(wide.columns, name="series"))
    else:
        attributes = attributes.loc[wide.columns]
    mask = None
    if censored is not None:
        mask = censored.reindex(index=wide.index, columns=wide.columns).fillna(False)
        mask = mask.to_numpy(dtype=bool).T
    panel = Panel(
        wide.columns.to_numpy(),
        pd.DatetimeIndex(wide.index),
        wide.to_numpy(dtype=np.float32).T,
        freq,
        mask,
    )
    metadata.update(
        {
            "series": len(panel.series),
            "periods": len(panel.periods),
            "limit_series": limit_series,
            "imputed_cells": int((~observed).sum()),
        }
    )
    return Benchmark(panel, _hierarchy(name, panel, attributes), attributes, observed, metadata)


def save_prepared(data: Benchmark, path: str | Path) -> None:
    """Save typed arrays (no pickle), static metadata, and a preparation audit locally."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    panel = data.panel
    arrays = {
        "series": panel.series,
        "periods": panel.periods.to_numpy(dtype="datetime64[ns]"),
        "values": panel.values,
        "freq": np.asarray(panel.freq),
        "observed": data.observed,
    }
    if panel.censored is not None:
        arrays["censored"] = panel.censored
    np.savez_compressed(path / "panel.npz", allow_pickle=False, **arrays)
    data.attributes.to_csv(path / "attributes.csv", index_label="series")
    (path / "preparation.json").write_text(json.dumps(data.metadata, indent=2) + "\n")


def load_prepared(path: str | Path) -> Benchmark:
    """Restore a prepared benchmark with exactly the saved series order and masks."""
    path = Path(path)
    metadata = json.loads((path / "preparation.json").read_text())
    with np.load(path / "panel.npz", allow_pickle=False) as arrays:
        panel = Panel(
            arrays["series"],
            pd.DatetimeIndex(arrays["periods"]),
            arrays["values"],
            str(arrays["freq"]),
            arrays.get("censored"),
        )
        observed = arrays["observed"].copy()
    attributes = pd.read_csv(path / "attributes.csv", index_col="series")
    attributes.index = attributes.index.astype(str)
    attributes = attributes.loc[panel.series]
    return Benchmark(
        panel, _hierarchy(metadata["dataset"], panel, attributes), attributes, observed, metadata
    )
