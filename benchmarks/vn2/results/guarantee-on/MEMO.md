# Guarantee-on VN2 measurement run (#286)

First committed VN2 run with the conformal guarantee on at tau, as evidence for
the flagship-metric decision (R4). Freeze-compatible apparatus under the
reference-maintenance carve-out; zero engine code touched.

## Config provenance

- Variant: `benchmarks/vn2/config/vn2-winning-loop-guarantee-on.yaml`. Exactly
  two decision knobs differ from `vn2-winning-loop.yaml`, enforced
  field-by-field by `tests/benchmarks/test_guarantee_on_coverage.py::TestGuaranteeOnConfig::test_only_the_two_knobs_differ_from_winning`:
  1. `coverage: 0.74 → 0.833` (tau = 1/1.2, repo convention 0.833; set in both
     `order_conformal` and `ordering` — the parse-time backstop requires them
     equal for (R,S) without a quantile).
  2. `buffer_max: 0.0 → omitted` (the residual buffer unclamped; for
     `buffer_max`, omitted ≡ null — threaded unconditionally).
  `weight_decay` stays explicitly null (the unweighted split-conformal branch
  carrying the exchangeability guarantee); the four YAML-unreachable runtime
  fields keep their guarantee-correct defaults; `method_name` is a cosmetic
  provenance label renamed to `guarantee_on_crc`.
- Driver: `benchmarks/vn2/run_guarantee_on.py` — runs the production settle
  loop (`calibre.cli.commands.run_config`) for both configs on the same
  machine, capturing per-round cost state via the read-only `settle_on_round`
  hook. Coverage analysis: `benchmarks/vn2/guarantee_on_coverage.py`, validated
  on hand-checkable fixtures (12 tests, no model runs).

## Machine

Windows 11 (10.0.22631), x86_64 (AMD64), Intel i7-1370P (14C/20T), 31.7 GB RAM,
Python 3.12.13, single process, `execution.backend: local`, seed 42.

**Baseline reproduction: exact.** The winning-loop config on this machine
produced holding 2488.20 / shortage 2504.00 / total 4992.20 — bit-parity with
the x86_64/Linux CI reference triple at the gate's 0.01 tolerance. Two
consequences: (a) the guarantee-on delta below reads directly against the
canonical baseline (same machine, same run process); (b) Windows-on-x86_64 is
now characterized for this config: it matches the Linux value (the known
divergence is arch-driven — arm64/macOS ≈ 5011.20 — not OS-driven).

## Results — cost

| run | holding | shortage | total | delta vs 4992.20 |
|---|---|---|---|---|
| baseline (0.74, clamped) | 2488.20 | 2504.00 | 4992.20 | — |
| guarantee-on (0.833, unclamped) | 3043.00 | 2156.00 | 5199.00 | **+206.80 (+4.14%)** |

The guarantee shifts cost composition as theory predicts: holding up (+554.80),
shortage down (−348.00). Per-round cumulative costs (decision rounds 1–6) are
in `guarantee_on-per-round.csv` / `baseline-per-round.csv`; the two lead-time
drain rounds accrue the remainder to the final breakdown (captured via
`settle_on_complete`, not per-round).

## Results — realized coverage at tau

Coverage event per (series, origin): realized 3-week protection-window demand
sum ≤ the calibrated one-sided bound (`hi_<coverage>` terminal-h row = the
order-up-to level). 599 series × 6 origins = 3,594 events.

| origin | guarantee-on | baseline |
|---|---|---|
| 2024-04-15 | 0.679 | 0.548 |
| 2024-04-22 | 0.603 | 0.511 |
| 2024-04-29 | 0.638 | 0.521 |
| 2024-05-06 | 0.758 | 0.619 |
| 2024-05-13 | **0.858** | 0.716 |
| 2024-05-20 | **0.891** | 0.741 |
| pooled | 0.738 | 0.609 |

Reading: with `warmup_origins: 3` and a calibration set that grows with each
origin, the first rounds under-cover; the post-warmup steady state (rounds 5–6)
straddles the 0.833 target (0.858 / 0.891). The pooled 6-round number (0.738)
is dominated by the warmup transient — a flagship-metric definition must say
which window it pools over. The clamped baseline never approaches its nominal
0.74 in-window (pooled 0.609): the clamp, not calibration, was binding.

## Both-series requirement (raw vs censoring-aware)

The censoring-aware series was recovered from run-adjacent data via the
engine's own imputation (in-stock matrix → `y_uncensored`). Finding: the
evaluation windows (2024-04-15 … 2024-06-03) contain **zero out-of-stock
observations** (last OOS week in the data: 2024-03-25), so the raw and
censoring-aware series coincide on this horizon and both coverage columns are
identical by construction. Recorded gap statement: **this VN2 horizon cannot
discriminate sales-scored from demand-honest coverage** — the guarantee-on
evidence is demand-honest on this window trivially, and U5 must weigh that the
raw-vs-honest distinction remains unmeasured on a stockout-bearing horizon.

## Gate status

- VN2 regression + CLI-parity: untouched (variant is a sibling file; the
  config-parity test pins the two-knob diff). The exact baseline reproduction
  above is direct evidence the 4992.20 gate holds on this machine.
- New apparatus tests: 12 passed; `tests/benchmarks` non-regression family:
  102 passed. `ruff check` / `ruff format` / `ty check calibre/` clean.

## Implications for U5 (evidence, not decision)

- Guarantee-on cost is near parity: +4.14% total for realized steady-state
  coverage at/above tau. This keeps guarantee-at-parity on the table as an
  honest headline; it is not a cost blow-up.
- The metric definition must pin: pooling window (pooled vs post-warmup),
  the scored series (raw sales vs demand-honest — undiscriminated here), the
  guarantee-on configuration (this variant), and the cost accounting (holding +
  shortage as booked by the simulator).
