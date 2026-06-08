# M5 Benchmark Harness

This directory is a config-only harness for running Calibre on the M5 retail
dataset through the source CLI:

```bash
uv run calibre run --config benchmarks/m5/config/smoke.yaml
uv run calibre validate --config benchmarks/m5/config/full.yaml
```

There is no `benchmarks.m5` Python entrypoint. The harness is the YAML contract
plus this runbook; execution stays in `calibre run --config`.

## Configs

- `config/smoke.yaml` uses `tests/fixtures/m5`, runs one cheap daily origin, and
  writes `results/m5/smoke/forecast-ledger.parquet`. This is a CI smoke fixture
  only, not statistical evidence for M5 coverage.
- `config/full.yaml` uses local full M5 data under `data/m5`, runs the canonical
  `evaluation` phase with 28-day horizons, point reconciliation, and MSCP
  per-horizon conformal intervals. It is structurally valid today without
  benchmark-local partition logic; the checked-in YAML carries the source-level
  partition-selection TODO. The full config streams ledger output and uses
  `execution.backend: auto` because full-M5 hierarchy expansion is large.

For full-M5 work with hierarchy-aware reconciliation installed:

```bash
uv sync --extra dev --extra benchmarks --extra hierarchy
uv run calibre validate --config benchmarks/m5/config/full.yaml
uv run calibre run --config benchmarks/m5/config/full.yaml
```

The full run is a local acceptance run only: it requires `data/m5`, is expensive,
and is not part of CI.

## Local Data Layout

Full M5 data is intentionally not committed. Place the public M5 Forecasting -
Accuracy release files under the ignored local directory:

```text
data/m5/
  sales_train_evaluation.csv
  calendar.csv
  sell_prices.csv              # optional today; the adapter does not consume it
```

The canonical acceptance variant is `sales_train_evaluation.csv` with
`dataset.phase: evaluation`. The adapter also accepts `sales_train_validation.csv`
or the `_1` suffixed release variants, but validation has about 28 fewer days of
history; recompute the origin window before treating a run as acceptance
evidence.

Cheap pre-run integrity checks:

```bash
uv run python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("data/m5")
sales_path = root / "sales_train_evaluation.csv"
sales_header = pd.read_csv(sales_path, nrows=0)
calendar = pd.read_csv(root / "calendar.csv")
day_cols = [col for col in sales_header.columns if col.startswith("d_")]
dates = pd.to_datetime(calendar.loc[calendar["d"].isin(day_cols), "date"])
with sales_path.open(encoding="utf-8") as fh:
    sales_rows = sum(1 for _ in fh) - 1

print(f"sales rows: {sales_rows}")
print(f"day columns: {len(day_cols)} ({day_cols[0]}..{day_cols[-1]})")
print(f"date range: {dates.min().date()}..{dates.max().date()}")
PY
```

For the canonical evaluation release, expect 30,490 bottom-level rows, 1,941 day
columns (`d_1..d_1941`), and dates from 2011-01-29 through 2016-05-22.

## Artifact Layout

Every M5 run writes into an explicit per-run directory:

```text
results/m5/<run-name>/
  forecast-ledger.parquet
  order-ledger.parquet              # only when ordering is configured
  coverage-by-node.parquet          # reserved for #85
  report.md                         # reserved for #85
  hierarchical-interval-baseline/   # reserved comparator lane for #85
```

Output paths are part of the YAML. When changing the strategy, model, origin
window, or conformal settings, copy the config or edit the run directory to avoid
overwriting earlier artifacts.

The future hierarchical-interval comparator is a baseline lane only. Calibre's
Nixtla hierarchical interval path emits coherent point forecasts and marginal
intervals; published per-node interval boxes are not additive conditional
coverage bands. Issue #85 owns the scoring semantics and thresholds.

## Origin Window

The full config uses 37 daily origins:

```text
origins >= warmup + scored + (H - 1)
        >= 9      + 1      + 27
```

For 90% per-horizon MSCP, the runtime's higher-quantile finite-sample rule is
infinite while `alpha <= 1 / (n + 1)`, so the first finite `(series, horizon)`
partition has 10 resolved scores. The runbook expresses that as 9 warmup scores
plus 1 scored origin, then adds the 27-day settlement delay for the longest
`h=28` score. Issue #85 owns the coverage rows, tolerances, and reports computed
from the produced ledger.

## Handoff Manifest

When a full local run is used as the handoff surface for #85, record:

- Config path and git commit.
- `dataset.phase` and sales file variant.
- Calibration partition recorded by the config/run (`global` until the source
  prerequisite lands and the config is updated).
- Origin start, end, frequency, and horizon.
- Produced artifact paths under `results/m5/<run-name>/`.
