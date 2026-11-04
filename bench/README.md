# Forecasting benchmarks

Run from the repository root. Raw files and forecast outputs belong in ignored
`data/`, not Git. Nothing is downloaded on import.

## 1. Download raw data

```sh
uv run --no-dev python bench/download_benchmarks.py --list
uv run --no-dev python bench/download_benchmarks.py beijing
# All nine additional datasets (roughly 300 MB downloaded):
uv run --no-dev python bench/download_benchmarks.py all
```

Replace `all` with a name from `--list` to download a single dataset.

Downloads stream to a temporary file, verify SHA-256, then rename atomically.
Existing files are also verified. GitHub/Hugging Face revisions are pinned; UCI
archives are pinned by hash. Corruption or upstream changes fail closed: remove a
corrupt local file before downloading again, and review any source/hash update.
ZIP archives are read in place, never extracted using upstream paths.

## 2. Load with automatic preprocessing

Each module exposes `load(path)`, reading and hash-verifying raw files on every
call, with no prepared format or cache. For example, Beijing:

```sh
PYTHONPATH=bench uv run --no-dev python - <<'PY'
import beijing

data = beijing.load("data/beijing")
panel, hierarchy = data.panel, data.hierarchy
print(panel.values.shape)
PY

# Optional Parquet reader for FreshRetailNet only; not needed to download it:
PYTHONPATH=bench uv run --no-dev --with pyarrow python - <<'PY'
import freshretail
data = freshretail.load("data/freshretail")
print(data.panel.values.shape)  # (50000, 97)
PY
```

Results are dataset-specific dataclasses with `panel`, `hierarchy`, and
`observed[B, T]` (False on imputed targets). Favorita and FreshRetailNet also
return static `attributes`. FreshRetailNet reads only needed Parquet columns,
but the full panel can still require a few GB of RAM.

M5 and VN2 are unchanged and deliberately **not** downloaded here: use
`m5.load("data/m5")` and `vn2.load("data/vn2")` with the official releases.

### Preprocessing and evaluation

- Beijing, Bike and Favorita reindex their calendar and always forward-fill from
  the past, then trim leading periods until every series has an observation.
  `warmup_periods_dropped` records that count. No backward-fill, interpolation or
  automatic zero-fill. ETTh1, LSTNet and FreshRetailNet do not impute targets.
- Exclude imputed cells from calibration/evaluation, including any aggregate or
  lead-time target containing one. Imputed values are not ground truth.
- LSTNet matrices have **no timestamps**. Dates starting at 2000-01-01 are
  positional labels only: no real calendar features or external-weather joins.
  Exchange horizons count observations, not calendar days.
- No future covariates are supplied. Bike's `casual + registered` reveals `cnt`,
  not a usable predictor of that same target.
- Favorita and FreshRetailNet construct hierarchies from static attributes, not
  official scoring hierarchies. Other datasets have flat hierarchies, without
  invented sums of temperature, occupancy or exchange rates.
- Subset series after loading if needed; loaders always use the full panel to
  determine warm-up trimming. Rebuild a matching hierarchy for any subset.

## 3. Forecast, calibrate and evaluate

Use `rolling_forecasts(...)` to generate forecasts, `replay(...)` to apply
calibrators to those forecasts, and `calibre.metrics` to compare coverage, width
and other metrics.

This complete example continues with Beijing; change the loader, season and
horizon for other datasets. Imputed and annotated-censored targets are excluded
from calibration/evaluation. On retail data, this does not establish latent-demand
coverage.

```sh
PYTHONPATH=bench uv run --no-dev python - <<'PY'
import numpy as np
import beijing
from calibre import Absolute, BottomUp, SeasonalNaive, SplitQuantile, Step
from calibre import replay, rolling_forecasts
from calibre.metrics import coverage

folder = "data/beijing"
data = beijing.load(folder)
panel, hierarchy = data.panel, data.hierarchy
horizon, season = 24, 24  # hourly; use e.g. 7, 7 for daily retail
origins = np.arange(len(panel.periods) - horizon - 48, len(panel.periods) - horizon)
assert origins[0] + 1 >= season
run = rolling_forecasts(panel, hierarchy, SeasonalNaive(season),
                        BottomUp(hierarchy), origins, horizon)
actuals = hierarchy.aggregate(panel.values).copy()
invalid = ~data.observed
if panel.censored is not None:
    invalid = invalid | panel.censored
actuals[hierarchy.any_bottom(invalid)] = np.nan
bands = replay(run, actuals, target=Step(), score=Absolute(),
               calibrator=SplitQuantile(0.9, window=28))
print(run.points.shape)  # [origins, nodes, horizon]
print(coverage(bands.target, bands.lower, bands.upper))
np.savez_compressed(f"{folder}/forecasts.npz", origins=run.origins, points=run.points,
                    nodes=hierarchy.nodes)
PY
```

Choose train/tuning/calibration/test windows before fitting transformations.
FreshRetailNet joins 90 train days and 7 labelled eval days; its module exposes
`TRAIN_START`, `TRAIN_END`, `EVAL_START` and `EVAL_END`. Respect that split.
Favorita's future Kaggle test days have no labels here; its module exposes
`UNLABELLED_TEST_START` and `UNLABELLED_TEST_END`.

## Variants, sources and data-use conditions

Selection: the audited 2026-10-13 document
`OVENTI_Benchmarks_TimeSeries_Demande_Hierarchique.html`, excluding M5; VN2 is also
left unchanged. SHA-256 checks cover the exact downloaded bytes, not decompressed
CSV content. URLs and full expected hashes belong to each dataset module.

| Name | Variant and target | Original shape | Access / attribution / caveat |
|---|---|---|---|
| `etth1` | ETT-small ETTh1, `OT` | 17,420 hours, 7 source variables; 1 selected target | [ETT / Informer](https://github.com/zhouhaoyi/ETDataset), [CC BY-ND 4.0](https://github.com/zhouhaoyi/ETDataset/blob/1d16c8f4f943005d613b5bc962e9eeb06058cf07/LICENSE); private adaptations permitted, but the licence does not permit sharing adapted data; retain attribution |
| `electricity` | LSTNet ECL hourly charge | 26,304 × 321 | [UCI](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), CC BY 4.0; cite [LSTNet processing](https://github.com/laiguokun/multivariate-time-series-data); not the raw 370-client, 15-minute file; processed units not asserted |
| `traffic` | LSTNet PeMS occupancy proportions | 17,544 × 862 | [PeMS](https://pems.dot.ca.gov/) / LSTNet; data licence not explicit; not METR-LA/PEMS-BAY or vehicle counts |
| `solar` | LSTNet Solar-AL, published PV values | 52,560 × 137 at 10 min | NREL provenance via LSTNet; data licence/physical units not explicit; not Solar Atlanta; nighttime zeros retained |
| `exchange` | LSTNet exchange levels, original column order | 7,588 × 8 observations | LSTNet; primary quotation metadata and licence not explicit; no inferred country names or calendar |
| `beijing` | UCI PRSA2017 multi-site, `PM2.5` | 12 × 35,064 hours | [UCI](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data), CC BY 4.0, Zhang et al. (2017), DOI 10.24432/C5RK5G; real target NAs |
| `bike` | UCI Bike Sharing, `hour.csv`, `cnt` | 17,379 recorded / 17,544 calendar hours | [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), CC BY 4.0, DOI 10.24432/C5W894; 165 absent hours are not assumed zero |
| `favorita` | Store Sales, store-family `sales` and store metadata | 1,782 series × 1,684 recorded dates | [Kaggle terms](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data), public [community mirror](https://huggingface.co/datasets/t4tiana/store-sales-time-series-forecasting); no explicit mirror licence; four missing Christmas dates, fractional sales preserved; not the item-level Grocery Sales competition |
| `freshretail` | Official FreshRetailNet-50K daily `sale_amount`, train + eval | 50,000 series × (90 + 7) days | [Dingdong / Wang et al. (2025)](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K), CC BY 4.0; globally scaled sales, not physical quantities |

Public download is not legal clearance: check original terms for the intended use,
keep attribution, and do not assume that the repository's MIT licence covers data.
FreshRetailNet's daily censoring mask is `stock_hour6_22_cnt > 0`, restricted to the
annotated 06:00–22:00 window; raw files retain the hourly status/sales for other
analyses. Censoring marks sales as lower bounds; it does not reveal lost demand,
stock levels, orders, costs, or a full inventory system.

## Local validation

```sh
git check-ignore data/etth1/ETTh1.csv data/m5/probe.csv data/vn2/probe.csv
git ls-files data/  # must print nothing
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

Benchmark scripts are outside the library's test suite. Validate loaders manually
on actual verified downloads and run the forecast example. Scratch checks and
outputs also stay in ignored `data/`, never upstream-data fixtures in Git.
