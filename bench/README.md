# Download and prepare forecasting benchmarks

Run from the repository root. Only these scripts and documentation belong in Git.
Raw files, download manifests, prepared arrays, and forecast outputs belong in the
ignored root `data/` directory. Nothing is downloaded on import.

## 1. Download an exact version

```sh
uv run --no-dev python bench/download_benchmarks.py --list
uv run --no-dev python bench/download_benchmarks.py etth1
# All nine additional datasets (roughly 300 MB downloaded):
uv run --no-dev python bench/download_benchmarks.py all
```

Downloads stream to a temporary file, verify SHA-256, then rename atomically.
Existing files are also verified; corrupted files fail rather than being reused.
GitHub/Hugging Face revisions are pinned. UCI archives are pinned by hash; an
upstream update fails closed and requires reviewing the version and hash together.
Each `data/<name>/download.json` records URLs, hashes, sizes, and verification time.
Archives are read in place, never extracted using upstream paths. Network failures
leave no published partial file; rerun the same command to reuse verified files.

M5 and VN2 are deliberately **not** downloaded or reimplemented here: use the
existing `bench/m5.py` and `bench/vn2.py` with files in `data/m5/` and `data/vn2/`.
In particular the M5 loader expects the official evaluation layout, not an
interchangeable mirror with different columns.

## 2. Adapt to Calibre (one command)

```sh
# Small, complete, timestamped target: no imputation needed.
uv run --no-dev python bench/download_benchmarks.py etth1 --prepare

# Explicitly permit causal imputation for missing targets/calendar gaps.
uv run --no-dev python bench/download_benchmarks.py beijing bike favorita \
  --prepare --missing ffill --limit-series 64

# Undated matrices need explicit acknowledgement of a positional time axis.
uv run --no-dev python bench/download_benchmarks.py electricity traffic solar exchange \
  --prepare --positional-time --limit-series 64

# Optional Parquet reader, scoped to this command; no library dependency added.
uv run --no-dev --with pyarrow python bench/download_benchmarks.py freshretail \
  --prepare --limit-series 64

# Or prepare all nine, with both choices made explicitly:
uv run --no-dev --with pyarrow python bench/download_benchmarks.py all \
  --prepare --missing ffill --positional-time --limit-series 64
```

Each dataset gets `panel.npz` (typed arrays, no pickle), `attributes.csv`, and
`preparation.json` with target, time-axis, subset, imputation and split information.
`load_prepared(path)` returns a `Benchmark` with `panel`, `hierarchy`, `attributes`,
`observed`, and `metadata`. `load(name, path, ...)` instead reads raw files directly.
Repeat preparation without `--limit-series` for the full panel. The limit chooses
sorted identifiers, **not** volume ranked on future targets and not a representative
sample. Raw downloads are still complete. FreshRetailNet preparation reads only
needed columns but can use a few GB of RAM even for a small output subset.

The dataframe contract is `series`, `period`, `value`. For example, the equivalent
minimal adaptation of the ETTh1 dataframe is:

```python
import pandas as pd
from calibre import Panel

raw = pd.read_csv("data/etth1/ETTh1.csv", parse_dates=["date"])
frame = raw.rename(columns={"date": "period", "OT": "value"}).assign(series="OT")
panel = Panel.from_long(frame[["series", "period", "value"]], freq="h")
```

The loaders do this efficiently through wide arrays for large panels. Calibre needs
finite float32 `[B, T]` targets on a complete regular calendar, not a dataframe
passed directly to a model. Its forecast adapters construct their vendor frames.
Duplicate keys, nonfinite values and malformed calendars fail validation.

### Missing data, time axes and leakage

- Default `missing='raise'` never silently invents targets. Explicit `ffill` fills
  from the past only, then trims the common leading warm-up until every selected
  series has a first observation. No backward-fill, interpolation or automatic
  zero-fill. This is a baseline policy, not a claim about the unobserved target.
- `observed[B, T]` stays False on imputed cells. Exclude these cells from evaluation
  and calibration feedback, including any aggregate/lead-time target containing
  one. The example below masks them. Do not score an imputed value as ground truth.
- LSTNet matrices have **no timestamps**. Synthetic dates starting at 2000-01-01
  satisfy the `Panel` contract only; do not use them for real calendar features or
  join external weather. Exchange horizons count observations, not calendar days.
- No covariates are automatically supplied: future realized weather, pollution,
  load, transactions and stock status are not known-ahead forecasts. Bike's
  `casual + registered` reveals `cnt` and is not a predictor of that same target.
- Favorita hierarchy: store-family bottoms, state/city/store/family aggregates and
  state-family crossings. FreshRetailNet: store-product bottoms with geography,
  product categories and city-product crossings. These are constructed subsets,
  not claims to reproduce an official scoring hierarchy. Subset totals cover only
  selected bottoms. The other datasets use a flat hierarchy, not invented business
  sums of temperatures, exchange rates or occupancy proportions.

## 3. Forecast immediately

This runs against any prepared folder; change season/horizon for its frequency.
The example keeps both imputed and annotated-censored targets out of sales
calibration/evaluation. It does not claim latent-demand coverage.

```sh
PYTHONPATH=bench uv run --no-dev python - <<'PY'
import numpy as np
from load_benchmarks import load_prepared
from calibre import Absolute, BottomUp, SeasonalNaive, SplitQuantile, Step
from calibre import replay, rolling_forecasts
from calibre.metrics import coverage

folder = "data/etth1"
data = load_prepared(folder)
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

For chronological evaluation choose train/tuning/calibration/test windows before
fitting transformations. FreshRetailNet combines the 90 train days (2024-03-28 to
2024-06-25) and 7 labelled eval days (2024-06-26 to 2024-07-02); the split is recorded
in preparation metadata. Never train on eval to predict train. Favorita's 16 future
Kaggle test days have no labels here: evaluate on labelled historical windows.

## Variants, sources and data-use conditions

Selection: the audited 2026-10-13 document
`OVENTI_Benchmarks_TimeSeries_Demande_Hierarchique.html`, excluding M5; VN2 is also
left unchanged. SHA-256 checks cover the exact downloaded bytes, not decompressed
CSV content. URLs and full expected hashes are in `download_benchmarks.py`.

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

## Local validation (no datasets in the diff)

```sh
git check-ignore data/etth1/ETTh1.csv data/m5/probe.csv data/vn2/probe.csv
git ls-files data/  # must print nothing
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

Benchmark scripts are outside the library's test suite. Validate loaders manually
on the actual verified downloads, round-trip prepared arrays and run the example
above. Any scratch checks and outputs also stay in ignored `data/`, never fixtures
containing upstream datasets.
