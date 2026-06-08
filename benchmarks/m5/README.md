# M5 Benchmark Harness

This directory is a config-only harness for running Calibre on the M5 retail
dataset through the source CLI:

```bash
uv run calibre run --config benchmarks/m5/config/smoke.yaml
uv run calibre validate --config benchmarks/m5/config/full.yaml
uv run calibre score-m5-coverage \
  --ledger results/m5/full-mscp-bottom-up/forecast-ledger.resolved.parquet
```

There is no benchmark-local forecast execution entrypoint. The harness is the
YAML contract plus this runbook; execution stays in `calibre run --config`, and
coverage scoring is a post-run artifact command over a resolved ledger.

## Configs

- `config/smoke.yaml` uses `tests/fixtures/m5`, runs one cheap daily origin, and
  writes `results/m5/smoke/forecast-ledger.parquet`. This is a CI smoke fixture
  only, not statistical evidence for M5 coverage.
- `config/full.yaml` uses local full M5 data under `data/m5`, runs the canonical
  `evaluation` phase with 28-day horizons, point reconciliation, and MSCP
  per-horizon conformal intervals with `conformal.partition: series`,
  `calibration_window: 10`, and an explicit `max_partitions` conformal-state
  guard. The full config streams ledger output and uses `execution.backend:
  auto`, but streaming only reduces output buffering; it does not avoid the
  input-side node-history materialization needed for hierarchical actuals.

For full-M5 work with hierarchy-aware reconciliation installed:

```bash
uv sync --extra dev --extra benchmarks --extra hierarchy
uv run calibre validate --config benchmarks/m5/config/full.yaml
uv run calibre run --config benchmarks/m5/config/full.yaml
uv run calibre score-m5-coverage \
  --ledger results/m5/full-mscp-bottom-up/forecast-ledger.resolved.parquet
```

The full run is a local acceptance run only: it requires `data/m5`, is expensive,
and is not part of CI. On hosts whose available memory cannot support the eager
node-history expansion, `calibre run` now fails before `build_node_history` with
the estimated bottom rows, node count, projected node-history rows, forecast
partitions, and detected memory envelope instead of disappearing as an OS kill.

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
  forecast-ledger.parquet           # configured ledger path
  forecast-ledger.resolved.parquet  # resolved materialized ledger for streaming runs
  order-ledger.parquet              # only when ordering is configured
  coverage-by-node.parquet          # per-node coverage diagnostics
  coverage-summary.json             # structured gate status for agents/CI
  report.md                         # human-readable coverage report
  hierarchical-interval-baseline/   # reserved comparator lane for #85
```

The full config uses `output.streaming: true`, so `forecast-ledger.parquet` is
the raw append stream and `forecast-ledger.resolved.parquet` is the materialized
ledger with resolved actuals and nonconformity scores. Use the resolved ledger
for #85 scoring and reporting.

Output paths are part of the YAML. When changing the strategy, model, origin
window, or conformal settings, copy the config or edit the run directory to avoid
overwriting earlier artifacts.

The future hierarchical-interval comparator is a baseline lane only. Calibre's
Nixtla hierarchical interval path emits coherent point forecasts and marginal
intervals; published per-node interval boxes are not additive conditional
coverage bands. Issue #85 owns the scoring semantics and thresholds.

The memory-efficient full-M5 path remains deferred engineering work: sparse or
lazy aggregate actual resolution, a `bottom_up` path that can synthesize
aggregate forecast rows from bottom forecasts, and preservation of independent
aggregate base forecasts for MinT-style strategies. Until that lands, the
preflight guard is a deterministic stop, not the full scalability solution.

## Coverage Scoring

Score only the resolved materialized ledger from a full-M5 streaming run:

```bash
uv run calibre score-m5-coverage \
  --ledger results/m5/full-mscp-bottom-up/forecast-ledger.resolved.parquet \
  --coverage 0.9
```

The command writes `coverage-by-node.parquet`, `coverage-summary.json`, and
`report.md` into the run directory by default. It does not run forecasting,
download M5, or mutate the source YAML. A non-`PASS` acceptance gate exits
non-zero after writing artifacts; pass `--report-only` when you only want to
refresh diagnostics.

The local acceptance gate is:

- full-population marginal coverage within +/- 3.0 percentage points of target;
- per-level average node coverage within +/- 5.0 percentage points of target;
- enough scored rows to avoid accepting a tiny finite-bound subset
  (`minimum_scored_ratio` in the summary);
- per-node outliers are emitted and counted as diagnostics only, not as
  conditional-coverage failures.

CI tests this scorer with synthetic M5-shaped ledgers. The checked-in
`tests/fixtures/m5` smoke fixture remains a contract fixture only and is not M5
coverage validation.

## Origin Window

The full config uses 64 daily origins:

```text
origins >= ready-to-issue-h28 + h=28 settlement
        >= (10 + 27)         + 27
```

For 90% per-horizon MSCP, the runtime's higher-quantile finite-sample rule is
infinite while `alpha <= 1 / (n + 1)`, so the first finite `(series, horizon)`
partition has 10 resolved scores. For `h=28`, those scores themselves need the
27-day settlement delay before a finite interval can be issued, and the first
finite `h=28` interval then needs another 27 days to resolve. Coverage rows,
tolerances, and reports are computed from the produced resolved ledger.

## Handoff Manifest

When a full local run is used as the handoff surface for #85, record:

- Config path and git commit.
- `dataset.phase` and sales file variant.
- Calibration partition recorded by the config/run (`series` for the canonical
  #85 handoff).
- Origin start, end, frequency, and horizon.
- Produced artifact paths under `results/m5/<run-name>/`, including the resolved
  ledger path for streaming runs, `coverage-by-node.parquet`,
  `coverage-summary.json`, and `report.md`.
