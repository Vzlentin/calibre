# newcalibre

`newcalibre` is the isolated greenfield successor to Calibre. The name is
temporary: the successor takes the `calibre` name at the repository-wide hard
cutover. Until then, it is an independent Python 3.12 uv project beside the
frozen engine and never imports `calibre` or `benchmarks`.

The public architecture specification is the design authority. Read these
three files in order before writing first-brick code:

1. [`00-overview.md`](../docs/spec/00-overview.md)
2. [`02-domain-model.md`](../docs/spec/02-domain-model.md)
3. [`04-forecasting-plugins.md`](../docs/spec/04-forecasting-plugins.md)

[`60-onboarding.md`](../docs/spec/60-onboarding.md) turns those contracts into
the first-contributor walkthrough. The first brick must remain buildable from
the three-file reading path without frozen code or private rationale.

## Package layers

- `src/newcalibre/domain/` owns the chapter 02 vocabulary and contracts.
- `src/newcalibre/forecasting/` owns the chapter 04 adapter surface and may
  depend on `domain`.

Additional layers arrive only with their owning chapter. A tier-1 AST test
rejects imports from the frozen package and benchmark tree.

## First-brick domain decisions

The forecast frame uses the literal required names `series_key`,
`target_timestamp`, `actual_value`, `point_forecast`, `horizon_step`, `origin`,
and `model_name`. Optional interval pairs are `lower_<decimal>` /
`upper_<decimal>` and native quantiles are `quantile_<decimal>`; suffixes use a
canonical plain-decimal spelling such as `0.9`. These helpers own spelling and
interval pairing only; the configuration requesting a level owns its
admissible range.

All value columns are `float64`; accepted dense, nullable, Arrow-backed, and
sparse real numerics are copied and normalized at frame validation. Horizon
steps retain any non-boolean integer dtype. Timestamps are
timezone-naive NumPy `datetime64` values at `s`, `ms`, `us`, or `ns`
resolution. A `Calendar` owns the normalized dataset frequency; weekly data
uses an anchor such as `W-MON`, never bare `W`. Every frequency binds its
stride and clock phase to the panel's earliest canonical timestamp, and the
same value validates every observation, task origin, and forecast target.

Panels own canonical row and column order, deterministic series enumeration,
and local/global task partitioning. Optional censoring facts use the two
assertions `censored` and `uncensored`, with the separate literal `undeclared`
as their recorded default. If neither censor metadata field is supplied, the
canonical panel keeps the optional surface absent. Task transport uses a
versioned, digested Arrow envelope with an explicit logical-dtype manifest and
no pandas metadata; materialization reproduces the public task frames exactly.

The adapter protocol exposes `fit`, `predict`, `fitted_values`, `dump_state`,
`load_state`, and `update`, with fitted values, native quantiles,
censoring-aware fit, incremental update, and artifact persistence represented
as explicit capabilities. The seasonal-naive backend declares none of those
optional capabilities and rejects every request loudly. Backend selection is
always the explicit `backend` field; the registry has no default and reports
its available identifiers on a missing or unknown selection. For this brick,
non-empty `quantile_levels` and `censoring_aware: true` are capability requests
and are rejected before prediction.

The seasonal-naive retention rule is deliberately narrow: `fit` keeps
only the non-missing observations in the final `m` pre-origin calendar periods
for each series. It retains no earlier history, whole task, fitted-value
sidecar, or forecast rows. `predict` therefore fails loudly when that retained
season is short or has a missing phase, and otherwise repeats the phase lookup
deterministically for every horizon step.

## Development

Run every command from this directory against the successor lockfile:

```console
uv sync --locked --group dev
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked ty check src/newcalibre/
uv run --locked pytest tests/tier1
```

## Test tiers

| Tier | Content | Cadence |
|---|---|---|
| 0 | lint, types, and schema validation | every commit |
| 1 | synthetic conformance, without oracle or network | every commit |
| 2 | self-consistency | every merge |
| 3 | conditional replay against promoted captures | scheduled or manual |
| 4 | protocol acceptance at scale | scheduled or manual |

Tier 2 keeps the four chapter-50 class-4 contracts visible. U5c instantiates
the Gate-A subset (same seed and resume); serialization and distribution
invariance stay explicitly pending for U10 and U16.
