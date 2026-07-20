# M5 Gate C acceptance-band method

## Recommendation

- Use a **joint target-date stationary block bootstrap of estimator influence
  functions** (`m5-target-date-sb-max-t/v1`). Keep every node and every
  origin/horizon row for a target date together; resample the eight criterion
  series jointly with expected block length 28 days.
- Form **95% familywise, target-centered max-*t* acceptance bands** for the
  population statistic and all seven level statistics. A wide band cannot
  manufacture a pass: require at least 80% approximate power for a
  predeclared **2 percentage-point drift** before an in-band criterion is
  determinate.
- Derive completeness from a **result-blind eligibility mask**: for each scope,
  the floor is `eligible_after_declared_readiness / resolved`. Require exact
  equality between that mask and the scored mask. Thus valid warm-up is
  allowed, but deleting even one post-readiness row cannot improve acceptance.
- Fix before the run: 19,999 replicates, NumPy 2.5.1 `PCG64DXSM`, the seed
  derived below, `method="higher"` quantiles, no data-driven block length, no
  rerun with different constants, and explicit zero-variance/few-date rules.
- Ship two different witnesses: a balanced synthetic ledger shifted by exactly
  ±0.02 must be rejected by the statistical gate; a one-row count/key mutation
  must be rejected by exact accounting validation, not by pretending one row
  is a scientifically meaningful coverage drift.

This is the recommended implementation decision for
[`Vzlentin/calibre#433`](https://github.com/Vzlentin/calibre/issues/433). It is
research only: it does not change the scorer or the protocol specification.

## Why this is the estimand

The protocol fixes two different sales-coverage estimands and one completeness
quantity: pooled covered/scored at population, the unweighted mean of per-node
covered/scored rates at each level, and scored/resolved at population and every
level. Pooled level coverage is diagnostic only. It also requires actual run
counts, dependence-aware derivation, and an `UNDETERMINED` result below the
completeness floor ([M5 protocol `[M5-A1]`–`[M5-A6]`, revision
`351b1fa`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/21-protocol-m5.md#L145-L191)).
The shared row predicate says that a row is resolved when its actual is finite,
scored when both bounds are also finite, and covered when the actual lies in
those bounds; coverage denominators contain scored rows only ([domain model
`[LED-4]`–`[LED-5]`, same
revision](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/02-domain-model.md#L310-L342)).

For one declared model and coverage target `p0 = 0.90`, let:

- `r` index a ledger row, `i(r)` its node, `l(r)` its level, `o(r)` its
  forecast origin, and `t(r)` its target calendar date;
- `R_r` be its resolved indicator;
- `M_r` be its scored indicator under the shared predicate;
- `C_r` be its covered indicator, defined only when `M_r = 1`;
- `S = Σ_r M_r`, `A = Σ_r M_r C_r`, and, for node `i`,
  `S_i = Σ_{r:i(r)=i} M_r`, `A_i = Σ_{r:i(r)=i} M_r C_r`.

The gated point estimates are therefore

```text
population:  p_hat_pop = A / S
node i:      p_hat_i   = A_i / S_i
level l:     p_hat_l   = (1 / N_l) Σ_{i in l} p_hat_i
```

where `N_l` is the number of scored nodes in level `l`. The implementation must
also require every declared lattice node to have positive eligible and scored
counts before a level can be determinate; otherwise silently changing `N_l`
would change the estimand. The pooled diagnostic
`Σ_{i in l} A_i / Σ_{i in l} S_i` is not substituted for `p_hat_l`.

### Dependence units

A row is not a Bernoulli trial independent of the other rows:

1. All rows with one **target date** can contain the same realized sales value
   predicted from several origins/horizons. They must never be split across
   resamples.
2. The 28 forecasts issued at one **origin** span 28 consecutive target dates;
   adjacent origin windows overlap. Consecutive target dates therefore need
   block treatment rather than iid date resampling.
3. Every target-date cross-section contains all nodes. Aggregate actuals are
   exact sums of shared bottom series, and the reference lattice has 33,563
   nodes, so node resampling would break the hierarchy and misrepresent
   cross-node covariance ([M5 hierarchy `[M5-H2]` and `[M5-H5]`, revision
   `351b1fa`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/21-protocol-m5.md#L68-L111)).
4. Within-series dependence across nearby target dates remains after grouping
   by date and is carried by contiguous calendar blocks.

The rolling daily, 28-step protocol is fixed by `[M5-B1]`–`[M5-B4]`; the
64-origin reference shape is an example derived from readiness, not an
inherited scorer constant ([M5 protocol, revision
`351b1fa`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/21-protocol-m5.md#L112-L147)).
For a tail window whose future rows cannot resolve past the phase end, the
reference run exposes only 64 resolved target-date clusters. This small time
axis, not the tens of millions of rows, is the limiting information source.

## Selected method

### 1. Reduce rows to target-date influence series

Build the contiguous resolved target-date grid in calendar order. Preserve
the actual score mask and all zero-contribution dates. For each date `t`, form
an influence contribution for every gated criterion:

```text
psi_pop,t = (1 / S) Σ_{r:t(r)=t} M_r (C_r - p_hat_pop)

psi_l,t = (1 / N_l) Σ_{i in l} (1 / S_i)
          Σ_{r:i(r)=i,t(r)=t} M_r (C_r - p_hat_i)
```

Each series sums to zero over dates. These are the ratio-estimator influence
functions: actual `S` and every actual `S_i` enter directly, so unequal node
masks do not silently turn the level statistic into a pooled statistic. This
linearization avoids recomputing all 33,563 node ratios in every replicate;
its first-order nature is a stated limitation rather than hidden precision.

### 2. Apply one joint stationary bootstrap

Politis and Romano's stationary bootstrap resamples a stationary sequence in
random-length contiguous blocks with geometrically distributed lengths
([Politis & Romano 1994](https://doi.org/10.1080/01621459.1994.10476870)); it
builds on block resampling for dependent stationary observations
([Künsch 1989](https://doi.org/10.1214/aos/1176347265)). Apply it to the matrix
whose columns are dates and whose eight rows are the population plus seven
level influence series:

- resampling unit: the **entire target-date cross-section** after reduction;
- all eight series use the **same bootstrap date indices**;
- restart probability: `1 / 28`;
- expected block length: `L = 28` calendar days, fixed from the protocol
  horizon before results;
- replicate length: the observed number `T` of resolved target dates;
- replicates: `B = 19_999`.

A target date is therefore indivisible, consecutive dates usually travel
inside the same block, and the same draw preserves covariance among population
and level statistics. No node, row, origin, or horizon is resampled
independently.

### 3. Build a simultaneous target-centered band

For bootstrap replicate `b` and criterion `j`, calculate

```text
d_bj = Σ_{k=1..T} psi_j,I[b,k]
s_j  = sample_sd_b(d_bj), denominator B - 1
z_b  = max over nondegenerate j of |(d_bj - mean_b(d_bj)) / s_j|
c    = higher_quantile_0.95({z_b})
```

The shared max statistic makes the eight bands a single 95% family rather than
eight unadjusted 95% checks. Resampling-based max-statistic control is the
relevant multiple-testing construction; Romano and Wolf give the primary
resampling framework for joint test statistics
([Romano & Wolf 2005](https://doi.org/10.1198/016214504000000539)).

Set

```text
h_pop       = c * s_pop
h_level,l   = max(c * s_l, h_pop)
band_j      = [max(0, p0 - h_j), min(1, p0 + h_j)]
```

The `max` enforces the protocol's requirement that a per-level band is never
narrower than the population band. The gate compares `p_hat_j` with this band
centered on the declared target. It does **not** publish a percentile interval
centered on the observed estimate and call target inclusion evidence of
acceptance.

For a symmetric standard-error interval, “an estimate-centered confidence
interval contains the target” is algebraically the same inequality as “the
estimate lies in a target-centered band.” The operational safeguard is
therefore the power rule below: uncertainty may make the result
`UNDETERMINED`, but may not widen the acceptance region until a noisy result
passes.

### 4. Require power before acceptance

Declare the smallest scientifically meaningful drift before the run as

```text
delta = 0.20 * (1 - p0) = 0.02 at p0 = 0.90.
```

This is a policy choice, not an estimate from a predecessor: a two-point loss
uses 20% more than the nominal 10% miss budget, while a two-point gain is the
symmetric overcoverage drift required by `[M5-A1]`/`[M5-A2]`. Changing the
Gate C target invalidates this preregistration and requires a new one; the
scorer must not silently recompute a different policy after seeing results.

With desired power `1 - beta = 0.80` and
`z_0.80 = 0.8416212335729143`, define the one-direction normal-approximation
minimum detectable drift

```text
MDD_j = h_j + z_0.80 * s_j.
```

An in-band criterion is determinate only when `MDD_j < delta`. This follows
from `P(p_hat < p0 - h | p = p0 - delta) ≈ Phi((delta - h)/s)` for the lower
alternative; the upper alternative is symmetric. This is a declared design
power check, not a finite-sample guarantee. An observed estimate outside its
band is `FAIL` even when the design-power check is weak; an estimate inside a
band with `MDD_j >= delta` is `UNDETERMINED`, never `PASS`.

## Exact preregistration

| Item | Fixed decision |
|---|---|
| Method ID | `m5-target-date-sb-max-t/v1` |
| Target | `p0 = 0.90` sales-coverage |
| Criterion family | population plus all seven marginal-lattice levels |
| Confidence | 0.95 simultaneous familywise |
| Resampling unit | complete target-date cross-section |
| Time order | contiguous calendar order; missing internal dates represented explicitly |
| Expected block length | 28 days |
| Restart probability | `1/28` at each transition |
| Replicates | 19,999, never extended after inspecting stability or verdict |
| RNG | NumPy 2.5.1 `Generator(PCG64DXSM(seed))` |
| Seed material | UTF-8 `calibre-m5-gate-c-stationary-bootstrap-v1` |
| Seed derivation | first 128 bits of SHA-256, big-endian |
| Seed | hex `3aef0a1f1a9127a7947864dc1dd4e132`; integer `78336387993106110653334900045528752434` |
| Bootstrap centering | subtract each criterion's replicate mean before studentization |
| Standard error | sample standard deviation with `ddof=1` |
| Critical quantile | `q=0.95`, NumPy `method="higher"`; with `B=19_999`, sorted zero-based index 18,999 |
| Per-level width | `max(raw level half-width, population half-width)` |
| Smallest meaningful drift | absolute 0.02 |
| Design power | 0.80; require `MDD < 0.02` to accept |
| Finite-sample correction | no iid/row-count correction; max-*t* studentization and the explicit power refusal are the only corrections |
| Minimum date support | fewer than 56 resolved target dates (`2 * horizon`) is `UNDETERMINED` |
| Post-result tuning | forbidden: no changed block length, seed, replicate count, confidence, power, drift, family, level roster, or quantile rule |

NumPy documents that `PCG64DXSM` guarantees the same random integer stream for
a fixed seed, while the general `Generator` API does not promise version-stable
transforms; this is why the NumPy version and draw schedule are part of the
record ([NumPy 2.5 `PCG64DXSM`](https://numpy.org/doc/2.5/reference/random/bit_generators/pcg64dxsm.html#numpy.random.PCG64DXSM),
[NumPy 2.5 `Generator`](https://numpy.org/doc/2.5/reference/random/generator.html#numpy.random.Generator)).
NumPy also defines `method="higher"` as a discontinuous order-statistic choice,
so no interpolation ambiguity remains
([NumPy 2.5 `quantile`](https://numpy.org/doc/2.5/reference/generated/numpy.quantile.html)).

The exact RNG draw schedule is:

```python
rng = Generator(PCG64DXSM(SEED))
starts = rng.integers(0, T, size=(19_999, T), dtype=int32)
restart_u = rng.random(size=(19_999, T - 1), dtype=float64)
indices[:, 0] = starts[:, 0]
for k in range(1, T):
    indices[:, k] = where(
        restart_u[:, k - 1] < 1 / 28,
        starts[:, k],
        (indices[:, k - 1] + 1) % T,
    )
```

Generate the full `starts` array before `restart_u`; changing draw order changes
the reproducible bootstrap and is not conforming. As an executable conformance
fixture at `T=64`, the first 16 indices must be
`[34, 35, 36, 37, 38, 39, 40, 10, 11, 12, 13, 14, 15, 16, 17, 18]`, and the
SHA-256 of the complete C-order index matrix encoded as little-endian `int32`
must be
`cf11f598ca53f56c74ff72bd57bb2d4a4fb9a6c0230c2d940ab24a62b467cfa2`.
This fixture was calculated from the declared algorithm under the pinned NumPy
2.5.1 API, not measured from an engine result.

## Completeness without selective missingness

A scalar scored/resolved threshold alone is unsafe: a method could omit hard
rows, remain above the scalar, and move coverage toward the target. The floor
must come from a mask that cannot see coverage outcomes.

For every row, independently derive an eligibility indicator `E_r` from:

1. finite actual / resolved status;
2. forecast origin, target date, horizon, and calibration partition;
3. the configured method's declared `n_first` and readiness rule; and
4. the number of earlier resolved calibration scores available **before issue**.

Do not inspect interval values, `C_r`, or observed coverage while deriving
`E_r`. The protocol already requires readiness to be declared and the origin
window to be recomputed from it, rather than copying `64/28/10` as constants
([`[M5-B3]`–`[M5-B4]`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/21-protocol-m5.md#L121-L147)).

For scope `g` (population or one level), calculate from the run's actual row
availability:

```text
R_g       = Σ_{r in g} R_r
E_g       = Σ_{r in g} E_r
S_g       = Σ_{r in g} M_r
floor_g   = E_g / R_g
observed_g = S_g / R_g
```

The completeness result is:

- `M_r = 1` where `E_r = 0`: hard protocol/readiness `FAIL`;
- any `E_r = 1` where `M_r = 0`: `UNDETERMINED` for that scope;
- any declared node with `E_i = 0` or `S_i = 0`: its level is
  `UNDETERMINED`;
- otherwise the masks are identical, `observed_g = floor_g`, and completeness
  is `PASS`.

Thus the numeric floor is re-derived for the actual resolved counts and the
method's legitimate warm-up; it may be much lower than one without licensing
one row of unplanned post-readiness missingness. The actual scored counts then
enter every influence denominator and the power check. This is preferable to a
historical `0.50`, a binomial count formula, or a missing-at-random assumption.
It also preserves the required semantics that below-floor is
`UNDETERMINED`, not statistical failure.

Emit the exact mismatch counts (`early_scored_rows`,
`missing_eligible_rows`) per scope. The same scorer invocation derives these
counts and the bands; no separate mask receipt, digest chain, or promotion
mechanism is needed.

## Verdict algorithm and edge cases

For each criterion, apply the following order:

1. **Validate exactly.** Unique canonical row keys, target-date arithmetic,
   covered/scored consistency, lattice roster, and integer counters must agree.
   Structural corruption is a hard validation failure, not sampling noise.
2. **Check completeness.** Missing eligible rows or a node with no eligible
   score makes the criterion `UNDETERMINED`; an early scored row is a protocol
   failure.
3. **Check date support.** `T < 56`, a non-contiguous calendar that cannot be
   represented, or non-finite bootstrap output makes the family
   `UNDETERMINED`.
4. **Derive the joint family once.** Do not derive separate critical values per
   level.
5. **Check location.** Outside the target-centered band is `FAIL`.
6. **Check power.** In-band with `MDD >= 0.02` is `UNDETERMINED`; in-band with
   `MDD < 0.02` is `PASS`.

Degenerate rules are fixed:

- If all bootstrap deviations for criterion `j` are exactly identical,
  set `s_j = 0`, exclude it from the max-*t* denominator, and use its inherited
  population width rule. If all criteria are degenerate, set `c = 0`.
- A zero-variance criterion passes only at the target (or inside a wider
  population band inherited by a level); any nonzero target departure is not
  hidden by an arbitrary epsilon.
- Clip displayed bands to `[0, 1]`, but retain un-clipped `h_j` for power and
  reproducibility fields.
- If any criterion is `UNDETERMINED`, the overall verdict is
  `UNDETERMINED`; only all-pass passes. Preserve any simultaneous failure as a
  criterion-level reason. This follows `[M5-A6]`, which makes
  `UNDETERMINED` a non-pass rather than a green fallback
  ([M5 protocol](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/21-protocol-m5.md#L177-L191)).

## Why not the alternatives

| Alternative | Why it is weaker for this gate |
|---|---|
| iid binomial/Wilson | Treats tens of millions of repeated, hierarchical rows as independent. Tolerance class 5 names sample-size-derived Wilson/binomial bands generally, but M5 separately requires within-series and cross-node dependence; the M5 requirement controls here ([class 5 and M5 binding, revision `351b1fa`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/50-test-and-oracle-strategy.md#L76-L104)). |
| iid row bootstrap | Breaks shared actuals, overlapping windows, and deterministic aggregate sums. |
| node/series bootstrap | Treats aggregate and bottom nodes as exchangeable independent units even though aggregate actuals are sums of shared bottoms; it also changes the fixed level roster. |
| origin-only cluster/bootstrap | Keeps one issued 28-step path together but splits rows that predict the exact same target from neighboring origins. Target-date blocks preserve that exact dependence and carry same-origin dependence through consecutive dates. |
| two-/three-way cluster-robust variance | Multiway clustering is designed for non-nested clustering dimensions ([Cameron, Gelbach & Miller 2011](https://doi.org/10.1198/jbes.2010.07136)), so origin × target date is a useful diagnostic. It still treats rows sharing neither cluster as independent; adding node clusters becomes undefined or very fragile for the one-node total and three-/seven-node levels and does not model serial common shocks across dates. |
| cross-sectional HAC / Driscoll–Kraay | Time-aggregated HAC is robust to broad cross-sectional dependence ([Driscoll & Kraay 1998](https://doi.org/10.1162/003465398557825)) and is a plausible variance diagnostic, but a 27-lag estimate on only 64 dates leaves few lag pairs and does not directly provide the joint max-*t* family. The selected block bootstrap uses the same calendar reduction while producing the joint reference distribution. |
| data-selected block length | It would let the full result choose a tuning constant. Fixing 28 from the protocol's overlap scale is reproducible and auditable; a miss remains a miss rather than triggering a more favorable length. |

## Smallest-meaningful-drift and accounting witnesses

The numeric-gate doctrine requires a cheap witness at the smallest drift the
gate exists to catch; a gate that cannot reject that drift is decoration
([test/oracle strategy, revision
`351b1fa`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/50-test-and-oracle-strategy.md#L211-L221)).
Ship both directions because the protocol uses symmetric target deviation.

### Statistical witness: exactly two percentage points

Use sufficient counts or a 10,000-row synthetic ledger with 100 nodes × 100
calendar dates. Define

```text
baseline: C[i,t] = 1[(i + t) mod 100 < 90]
low:      C[i,t] = 1[(i + t) mod 100 < 88]
high:     C[i,t] = 1[(i + t) mod 100 < 92]
```

Assign each synthetic level a cyclic copy of the same construction; keep row
keys, `R`, `E`, and `M` identical. Every node, date cross-section, level mean,
and the population are exactly `0.90`, `0.88`, or `0.92`. The baseline must
`PASS`; both changed ledgers perturb coverage by exactly `0.02` and must
`FAIL`, not merely become `UNDETERMINED`. Because the daily influence series
is balanced, this witness tests the acceptance boundary rather than random
Monte Carlo luck.

Also test a noisy-but-adequately-powered fixture for which `MDD < 0.02`; the
same exact drift must remain outside the derived band. This prevents a special
zero-variance branch from being the only biting witness.

### Accounting witness: one row

Separately mutate one canonical row key, duplicate one row, or change one
reported integer count by one while leaving row material unchanged. Exact
recomputation must reject the artifact. Do **not** demand that a 2-point
statistical band notice one row among millions: one row is the smallest
meaningful accounting corruption, not the smallest meaningful scientific
coverage drift. The tolerance doctrine classifies structural/integer checks as
exact ([class 1, revision
`351b1fa`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/50-test-and-oracle-strategy.md#L76-L85)).

## Computational shape

One streaming/grouped pass over the ledger computes `(covered, scored,
resolved, eligible)` counts by `(node, target_date)`; this is `O(n_rows)`.
At the reference shape, three dense `int64` node × date count matrices need
approximately `33,563 * 64 * 3 * 8 = 51.6 MB`. The eight influence series are
then only `8 * 64` floats.

The bootstrap costs `O(B * T * 8)`: at `B=19,999`, `T=64`, this is about 10.2
million scalar gathers/additions. Its index matrix is about 5.1 MB as `int32`;
replicate deviations are about 1.3 MB as `float64`. No replicate materializes
ledger rows, aggregate membership, or a node × replicate output. This makes the
method feasible beside the full-M5 scorer without changing the forecasting
engine.

## Required synthetic validation

Before accepting the implementation, run these result-independent scenarios:

1. **Closed-form iid sanity:** independent Bernoulli dates with many rows per
   date; bootstrap standard errors converge toward the direct date-mean
   calculation as the simulation grows.
2. **Cross-node common shock:** inject one latent date shock shared by every
   node; the selected band must widen relative to the iid-row calculation.
3. **Temporal blocks:** inject 28-day persistent shocks; iid date resampling
   must be narrower than the selected method, demonstrating that blocks bite.
4. **Repeated target / hierarchy:** copy one target actual across several
   origins and construct aggregate rows as bottom sums; row-order and
   node-order permutations must not change bytes of the summary.
5. **Unequal denominators:** vary deterministic eligible masks by node and
   prove the level estimate equals the explicit mean of node rates, not pooled
   level coverage.
6. **Selective deletion:** remove one eligible low-coverage row; the result
   must become `UNDETERMINED` even if the observed estimate moves closer to
   0.90.
7. **Power refusal:** construct an in-band high-dependence case with
   `MDD >= 0.02`; it must be `UNDETERMINED`.
8. **Drift and accounting witnesses:** both ±0.02 statistical witnesses and
   all one-row exact mutations behave as specified above.
9. **Degeneracy:** all-target, all-covered, all-uncovered, no-scored-node, and
   fewer-than-56-date fixtures follow the fixed edge rules without NaN-driven
   passes.

## Machine-readable derivation payload

The later harness should emit, at minimum, these fields; names may follow the
successor's schema conventions, but no semantic field may be dropped:

```yaml
method:
  id: m5-target-date-sb-max-t/v1
  target_date_axis: calendar_date
  expected_block_length: 28
  restart_probability: 0.03571428571428571
  replicates: 19999
  confidence_familywise: 0.95
  quantile_method: higher
  bootstrap_center: replicate_mean
  studentization: sample_sd_ddof_1
  rng:
    library: numpy
    version: 2.5.1
    bit_generator: PCG64DXSM
    seed_material: calibre-m5-gate-c-stationary-bootstrap-v1
    seed_sha256: 3aef0a1f1a9127a7947864dc1dd4e1325b1d86a4d039854063b30cf0d4f0c63e
    seed_hex_128: 3aef0a1f1a9127a7947864dc1dd4e132
    t64_index_matrix_sha256_le_i4: cf11f598ca53f56c74ff72bd57bb2d4a4fb9a6c0230c2d940ab24a62b467cfa2
  no_post_result_tuning: true
acceptance:
  target: 0.90
  smallest_meaningful_drift: 0.02
  desired_power: 0.80
  z_power: 0.8416212335729143
  minimum_target_dates: 56
  family_critical_value: <derived>
  sales_coverage_label: sales-coverage
counts:
  origins: <actual>
  target_dates_resolved: <actual>
  rows_total: <actual>
  rows_resolved: <actual>
  rows_eligible: <actual>
  rows_scored: <actual>
  rows_covered: <actual>
  nodes_declared_by_level: <actual mapping>
  nodes_scored_by_level: <actual mapping>
  early_scored_rows_by_scope: <actual mapping>
  missing_eligible_rows_by_scope: <actual mapping>
criteria:
  population:
    estimate: <derived>
    standard_error: <derived>
    half_width_raw: <derived>
    half_width_applied: <derived>
    band: [<derived>, <derived>]
    mdd_80: <derived>
    completeness_floor: <eligible/resolved>
    completeness_observed: <scored/resolved>
    status: PASS|FAIL|UNDETERMINED
    reasons: []
  levels:
    <level_name>:
      estimate_mean_node_rate: <derived>
      pooled_coverage_diagnostic: <derived>
      standard_error: <derived>
      half_width_raw: <derived>
      half_width_applied: <derived>
      band: [<derived>, <derived>]
      mdd_80: <derived>
      completeness_floor: <eligible/resolved>
      completeness_observed: <scored/resolved>
      eligible_nodes: <actual>
      scored_nodes: <actual>
      status: PASS|FAIL|UNDETERMINED
      reasons: []
verdict:
  status: PASS|FAIL|UNDETERMINED
  exit_nonzero_unless_pass: true
```

The report must additionally carry the declared model, phase, origin window,
conformal method, partition scheme, reconciliation strategy, resolved
configuration, environment-manifest reference, and the sales-coverage label. The protocol requires those declarations and
forbids presenting M5 sales-coverage as demand honesty or conditional coverage
([`[M5-X1]`–`[M5-X5]` and `[M5-R1]`, revision
`351b1fa`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/21-protocol-m5.md#L193-L233)).

## Limitations and unresolved statistical uncertainty

- The 64-origin reference window yields only 64 resolved calendar clusters.
  With expected block length 28, a stationary-bootstrap replicate has only
  about `64/28 = 2.29` expected blocks. Block-bootstrap theory is asymptotic
  under stationarity/weak-dependence conditions; this reference shape is a
  short series, so 95% familywise coverage is an approximation, not a theorem
  for this run. The explicit 80% power refusal exposes rather than conceals
  that limitation ([Politis & Romano 1994](https://doi.org/10.1080/01621459.1994.10476870),
  [Künsch 1989](https://doi.org/10.1214/aos/1176347265)).
- Fixing 28 captures the protocol-mandated overlap scale. Forecast state,
  retail seasonality, or sales shocks can persist longer; this gate does not
  establish robustness to arbitrary long memory or nonstationarity.
- Joint date resampling preserves observed cross-node dependence, including
  aggregate sums, but there is only one realized hierarchy and one calendar
  path. It cannot identify a superpopulation of hierarchies.
- These are marginal population and mean-node **sales-coverage** checks. They
  cannot prove per-node conditional coverage, coherent interval-box coverage,
  simultaneous lattice coverage, or demand/service-level honesty; the protocol
  explicitly makes per-node outliers diagnostic only
  ([`[M5-A5]`](https://github.com/Vzlentin/calibre/blob/351b1faadb91fc418e4363358556292088c36a0e/docs/spec/21-protocol-m5.md#L170-L176)).
- The `MDD` calculation uses a normal approximation around a bootstrap standard
  error. It is a preregistered precision screen, not an exact finite-sample
  power guarantee. If a coarse level cannot meet it, the correct Gate C result
  is `UNDETERMINED`; widening `delta` after seeing that result is forbidden.

## Implementation handoff checklist

- [ ] Freeze method ID, constants, RNG draw schedule, target, and 2-point drift
      in the U15 work order before any full-result coverage indicators are read.
- [ ] Derive and test the result-blind readiness/eligibility mask separately
      from finite-bound and covered predicates.
- [ ] Stream exact `(node, target_date)` counts and independently recompute all
      summary integers from canonical ledger keys.
- [ ] Implement both influence formulas and prove the per-level result equals
      an explicit unweighted mean of node rates under unequal denominators.
- [ ] Generate one joint stationary-bootstrap index matrix and one familywise
      max-*t* critical value; never run per-level bootstrap families.
- [ ] Enforce population-width flooring, the `MDD < 0.02` acceptance condition,
      all degeneracy rules, and `UNDETERMINED` precedence.
- [ ] Add the exact ±0.02 statistical witnesses, the noisy powered witness, and
      separate one-row accounting-corruption witnesses.
- [ ] Add all nine synthetic scenarios and byte-reproducibility checks on the
      pinned environment.
- [ ] Emit every derivation/count/mask field above plus phase, method,
      partition, reconciler, and sales-coverage labels.
- [ ] Fail the process for every non-`PASS` verdict; do not tune or rerun the
      method after seeing the full M5 result.
