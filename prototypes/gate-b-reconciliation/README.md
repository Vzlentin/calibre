# Gate B reconciliation prototype — formulations and dense-memory ceiling

Reaction asset for [Choose Gate B reconciliation formulations and memory
ceiling](https://github.com/Vzlentin/calibre/issues/399), produced under [Plan
Stage 3 Gate B execution](https://github.com/Vzlentin/calibre/issues/397) ahead
of [S3-U12: Reconciliation stage](https://github.com/Vzlentin/calibre/issues/313).

This is planning evidence, not S3-U12 implementation: a throwaway,
fixture-level prototype plus a metadata-only preflight measurement. Nothing
here ships; `newcalibre/` is untouched. Design authority is
`docs/spec/07-reconciliation.md`, `docs/spec/30-performance.md`, and
`docs/spec/50-test-and-oracle-strategy.md`; the frozen
`calibre/reconciliation/` was consulted as behavior provenance only.

## Verdict

**Owner decision (recorded in §7).** The recommended registry ships five
strategies: `none` (no-op), `bottom_up` (synthesis), **wls_struct**
(structural-weights projection), **wls_var** (MinT with the diagonal
residual-covariance model), and **mint_shrink** (MinT with the
Schäfer–Strimmer shrunk covariance — dense-only, gated by the ceiling). Only
**mint_cov** is rejected by name at resolution time per `[REC-10]`. The dense
representation is bounded by a **deterministic 1 GiB (2^30 B) dense-workspace
ceiling**, evaluated from metadata before any allocation: `mint_shrink` runs
where its itemized dense workspace fits (small real-world hierarchies — the
owner's stated reason for keeping it) and the run is rejected above the
ceiling, naming `wls_var` and `wls_struct` as the scalable alternatives.

## 1. Recommended formulations, stated exactly

Every projection candidate is the same weighted projection with a different
weight matrix. Per cross-section (one model name, origin, horizon step —
`[REC-2]`) with base vector `y_hat` over the `n` lattice nodes and summing
matrix `S` (`n × n_bottom`, 0/1, identity block first — `[REC-11]`):

```
beta       = argmin_b (y_hat - S b)' W^-1 (y_hat - S b)
           = (S' W^-1 S)^-1 S' W^-1 y_hat
reconciled = S @ beta                     # coherent by construction
```

### Structural-weights projection — `wls_struct`

- **Matrix formulation.** `W = diag(w)`, `w_i = (S S')_ii` — the bottom-member
  count of node `i` (for a 0/1 summing matrix, the row sum): 1 at bottom rows,
  the member count at aggregate rows, `n_bottom` at the total row.
- **Residual/covariance estimate.** None. The weights are lattice structure,
  read from the hierarchy index — that is exactly why this is the
  residual-free projection entry: no fitted-values sidecar, no estimator, no
  shrinkage. This is the formulation the chapter 30 baseline profiled at full
  M5 (reconcile phase: 163 s over 64 origins on the 32 GB laptop), and the one
  whose realized M5 coverage landed on target (~91% vs bottom-up's ~95%
  over-coverage — the `[REC-23]` provenance).
- **Singular behavior.** `W` is strictly positive by construction (every node
  has at least one member); `S` contains the identity block, so
  `S' W^-1 S` is always positive definite. No degenerate cases exist.

### MinT-family — `wls_var` (MinT, diagonal target, full shrinkage)

- **Matrix formulation.** The identical projection with `W = diag(v)`,
  `v_i = var(e_i, ddof=1)` the sample variance of node `i`'s in-sample
  residuals over `T` aligned periods. This is minimum-trace reconciliation
  under a diagonal residual-covariance model — equivalently, MinT with
  shrinkage toward the diagonal target at full intensity (`lam = 1` in the
  `mint_shrink` parameterization of §2).
- **Covariance estimator / shrinkage target.** Per-node sample variance,
  ddof=1; the "shrinkage target" is the diagonal itself, and the intensity is
  pinned at 1 rather than estimated — no `n²` quantity is ever formed.
- **Fitted-values requirements.** Per origin and model, the `[REC-5]`
  sidecar widened to two complete, timestamp-aligned `(n_nodes, T)` matrices
  (actuals and fitted values at every node, since the projection family
  forecasts every node). `T < 2`, a missing node, or misaligned timestamps
  fails loudly before anything reconciles. The production widening key is
  `(series key, timestamp, model name)`; the landed `FittedValues` domain
  adapter in `newcalibre` already owns that contract.
- **Singular / ill-conditioned behavior.** A node with constant residuals (a
  degenerate perfect fit) has zero variance, which would make `W^-1`
  infinite. Variances are floored at `max(v) * T * eps` — a few rounding
  units of the largest variance, derived, not tunable; a genuinely small
  variance is orders of magnitude above the floor and untouched. The floor is
  part of the formulation. With the floor, `W` stays strictly positive and
  `S' W^-1 S` positive definite.

## 2. The full-covariance MinT members — `mint_shrink` (ships, dense-only) and `mint_cov` (rejected by name)

- **Formulation.** `W = D ((1 - lam) R + lam I) D`, where `R` is the sample
  correlation of the residuals, `D = diag(sd)`, and `lam` is the
  Schäfer–Strimmer shrinkage intensity toward the identity target:

  ```
  lam = sum_{i != j} (T / (T-1)^3) * sum_t (w_ijt - w_bar_ij)^2  /  sum_{i != j} r_ij^2
      (clipped to [0, 1];  w_ijt = z_it * z_jt over standardized residual rows z)
  ```

  `lam = 1` degenerates to `wls_var`; `lam = 0` is `mint_cov`.
- **Why it is dense-only, and what that costs at scale.** Every intermediate
  is `n²` dense: `W` itself, the `W`-solves inside the projection, and the
  `(n_bottom, n_bottom)` normal matrix with its factorization. At full-M5
  metadata the recorded estimate is **33,115,060,840 B ≈ 30.8 GiB** of dense
  workspace (§5) — ~96% of the `[PRF-20]` 32 GiB process budget before the
  rest of the pipeline allocates anything, and ~31× the recommended ceiling.
  Worse at scale, `W` must be re-estimated and re-factored per origin per
  model (the fitted window grows), an `O(n_bottom^3)` solve per origin
  against `[PRF-1]`'s 15-minute budget. No sparse variant exists upstream for
  a reason: nothing in this estimator is sparse. Per the owner decision (§7),
  `mint_shrink` therefore ships **gated by the dense-workspace ceiling**, not
  rejected by name: the registry is not sized purely for full M5, and the
  shrunk-covariance member is the benchmarking reference on smaller
  real-world hierarchies whose itemized dense workspace fits under 1 GiB.
  Above the ceiling the *run* is rejected before allocation, with the message
  naming `wls_var` and `wls_struct` as the scalable alternatives.
- **`mint_cov`** (`lam = 0`, raw sample covariance) is rank-deficient whenever
  `T < n_nodes` — the common case for most of a backtest — and
  ill-conditioned on retail-sized lattices even when `T >= n_nodes`. This is
  `[REC-10]`'s own example of a strategy that must be rejected by name. The
  prototype pins the mechanism: at the fixture (`T = 8 < n = 12`) the raw
  covariance has rank 8 and the projection solve raises, while the shrunk
  estimator stays positive definite on the same residuals.

## 3. Dense vs sparse feasibility, per formulation

A sparse summing matrix makes **`S` and the operator actions through `S`**
cheap. It does nothing for `W`, for the `W`-solve, or for the
`(n_bottom, n_bottom)` normal equations — those are dense or not on their own
merits. Feasibility is therefore per formulation, not per matrix:

| formulation | `W` | dense workspace at M5 (recorded) | sparse path | verdict |
|---|---|---:|---|---|
| `wls_struct` | diagonal (structure) | 23,061,197,064 B ≈ 21.5 GiB | matrix-free operator: `v -> S'((S v) / w)`, `O(nnz(S))` per action | sparse at any scale |
| `wls_var` | diagonal (residual variances) | 24,103,529,592 B ≈ 22.4 GiB | identical operator; plus the `(n,T)` sidecar (~0.97 GiB), representation-independent | sparse at any scale |
| `mint_shrink` | full `n × n` (shrunk) | 33,115,060,840 B ≈ 30.8 GiB | none — `W` and its factorization are the wall, not `S` | dense-only; permitted below the 1 GiB ceiling, run rejected above it |
| `mint_cov` | full `n × n` (raw) | (same shape) + singular for `T < n` | none | rejected by name `[REC-10]` |

Note the first row's shape: even the *diagonal-weight* dense path at M5 is
~21.5 GiB once the normal equations and one factorization temporary are
charged — the 7.62 GiB dense `S` is only the first term. The sparse path for
the same math is **4,476,608 B ≈ 4.3 MB**. The memory pivot is real, and it
is the whole workspace, not `S.nbytes`.

## 4. Memory model and the recommended ceiling

The estimate names every array and its dtype (float64 data, int32 csr
indices), with peak assumptions stated: the csr summing matrix is built
directly from index coordinates (an eager densify-first build would double
the dense term transiently); a dense factorization holds one extra
`n_bottom²` temporary; an iterative solve holds six work vectors;
residual-requiring formulations hold both wide sidecar matrices per origin.

```
S_dense        = n_nodes * n_bottom * 8
S_csr          = nnz * 8 + nnz * 4 + (n_nodes + 1) * 4,   nnz = n_bottom * (A + 2)
W_diagonal     = n_nodes * 8
W_dense        = n_nodes^2 * 8                            # mint_shrink / mint_cov only
normal_eq      = n_bottom^2 * 8                           # dense solve only
factor_temp    = n_bottom^2 * 8                           # dense solve only
iter_vectors   = (2 * n_nodes + 4 * n_bottom) * 8         # sparse solve only
residual_wide  = 2 * n_nodes * T * 8                      # residual formulations only
```

**Recommended ceiling and rejection rule (normative).** A configured
`dense_workspace_ceiling_bytes`, default **2^30 = 1 GiB**, evaluated from
`(n_bottom, n_nodes, n_attributes, T)` metadata *before any allocation*:

1. Sum the formulation's dense components. `<=` ceiling → dense permitted
   (fixtures, small hierarchies, the class-3 closed-form reference).
2. `>` ceiling and the formulation declares a sparse operator path →
   sparse-required; the dense producer is never invoked on its behalf
   (`[REC-15]`).
3. `>` ceiling and no sparse path → the *run* is rejected before allocation
   with the itemized estimate, naming the scalable alternatives (`wls_var`,
   `wls_struct`); the strategy itself stays registered for smaller
   hierarchies. Independently of scale, a name on the `[REC-10]` list
   (`mint_cov`) is rejected at resolution, whatever the metadata.

Derivation of the constant: 1 GiB = 1/32 of Stage 3's 32 GiB process budget
(`[PRF-20]`), so a dense-permitted run's reconciliation allocations are
negligible against the budget by construction. It is deterministic and
machine-independent — unlike the frozen engine's detected-available-memory
comparison, which is the *general* `[REC-16]` preflight and stays; this
ceiling is the strategy/representation gate in front of it, and it is how
`[PRF-21]`'s "configured series-count threshold" should be denominated (bytes
from metadata, not a bare count: at the M5 lattice shape `n_nodes ≈ 1.10 *
n_bottom`, 1 GiB reads as "dense below ≈6.6k bottom series" for the
diagonal-weight formulations and ≈5.6k for full-covariance). Full M5 exceeds
the ceiling 7.6× on the `S` term alone, so the sparse pivot at scale is
unambiguous.

## 5. Recorded evidence

Fixture (hand-checkable: 6 bottom series crossed by `channel` × `region` — a
lattice, not a tree; 12 nodes, nnz 24, structural weights
`[1,1,1,1,1,1,2,2,2,3,3,6]`; `recorded/fixture-result.json`):

- Dense and sparse paths agree within the derived bound: max |dense − sparse|
  = 5.7e-14 (`wls_struct`) and 2.8e-14 (`wls_var`) against derived tolerances
  of 9.6e-7 / 9.1e-7 (bound = `magnitude * max_members * (2 κ (ε + τ) + ε)`,
  shared by the runtime check and the tests per `[REC-12]`; exact fixture
  κ = 4.0 / 3.77; solver τ = 1e-10). Coherence residual is exactly 0.
- `mint_shrink` on the same residuals: λ̂ = 0.3003, κ = 8.24, coherent —
  feasible at fixture scale, infeasible at M5 (§3). `mint_cov` on the same
  residuals: rank 8 < 12, solve raises — the `[REC-10]` mechanism made
  visible.
- Witnesses bite: one corrupted S membership moves the output by O(1) ≫ the
  derived bound; a starved solver (`maxiter=1`) raises with exit-code
  identity; the preflight boundary flips decision on an 8-byte drift; the M5
  preflight itself allocates <1 MB (`tracemalloc`) — O(metadata) proven, not
  asserted.

Full-M5 scale (`recorded/m5-scale-estimate.json`), from checked-in metadata
only — `[M5-H2]` (30,490 bottom, 33,563 nodes), `[M5-H1]`/`[M5-D1]` (five
attribute columns), `[M5-D3]` (T = 1,941); nothing materialized, nothing
downloaded:

- `S_dense` = 8,186,686,960 B = **7.62 GiB** (recomputed product of the
  `[PRF-21]` factors; the "~7.6 GiB" claim holds exactly). nnz = 213,430;
  `S_csr` = 2,695,416 B ≈ **2.6 MiB** — a ~3,000× pivot.
- Decisions: `wls_struct` → sparse-required (dense 21.5 GiB vs sparse
  4.3 MB); `wls_var` → sparse-required (sparse total 1,046,809,136 B ≈ 0.97
  GiB, dominated by the 1,042,332,528 B sidecar — at M5 the diagonal MinT
  entry's cost driver is the fitted-values sidecar, not the solve);
  `mint_shrink` → rejected-at-scale (30.8 GiB over the 1 GiB ceiling — a
  per-run rejection, not a by-name one); `mint_cov` → rejected-by-name.

## Production substrate

S3-U12 does **not** reimplement the nontrivial projection mathematics in this
prototype. It wraps the point-reconciliation methods from a pinned
`hierarchicalforecast` release behind Calibre's reconciler interface:

- `wls_struct` and `wls_var` delegate to
  `hierarchicalforecast.methods.MinTraceSparse` when the sparse representation
  is required, and may use `MinTrace` when the dense representation is
  permitted.
- `mint_shrink` delegates to
  `hierarchicalforecast.methods.MinTrace(method="mint_shrink")` when its
  metadata-only preflight permits the dense representation.
- `none` and the all-members-present `bottom_up` synthesis remain native: they
  express Calibre's identity and completeness contracts and contain no
  nontrivial projection solver to duplicate.

Calibre owns the deep interface around that substrate: strategy declarations
and registry, hierarchy index and deterministic summing-matrix construction,
Nixtla layout conversion, input/cross-section/fitted-value validation,
representation selection, memory preflight, derived coherence and idempotence
checks, and errors carrying cross-section identity. S3-U12 must verify the
exact upstream interface against the version it pins. If that release still
discards the sparse iterative solver's convergence status, a checked adapter
must surface it per `[REC-21]`; if upstream has fixed the behavior, Calibre
must not retain a redundant custom solver implementation. Nixtla's conformal
interval reconciliation remains outside this points-only stage.

The local NumPy/SciPy implementation is intentionally independent class-3
reference evidence: it prevents testing Nixtla against itself, makes the
formula and tolerance derivation inspectable, and provides biting witnesses.
It is not production code.

## 6. Normative product behavior vs prototype machinery

**Normative** (what S3-U12 should bake into `newcalibre`'s reconcile module,
each traceable to spec, reflecting the owner decision of §7):

- The five-strategy registry — `none`, `bottom_up`, `wls_struct`, `wls_var`,
  `mint_shrink` — with the exact `W` estimators of §1–§2, including the
  `max(v) * T * eps` variance floor, the `T >= 2` loud failure, and the
  Schäfer–Strimmer intensity.
- `mint_shrink` registered as **dense-only** (`[REC-8]`c declaration):
  permitted when its itemized dense workspace estimate is within the ceiling,
  the run rejected above it before allocation with `wls_var` and `wls_struct`
  named as scalable alternatives. `mint_cov` alone is rejected by name at
  resolution (`[REC-10]`).
- The 1 GiB deterministic dense-workspace ceiling and the four-way
  dense-permitted / sparse-required / rejected-at-scale / rejected-by-name
  rule, evaluated from metadata before allocation, itemizing the
  summing-matrix term (`[REC-16]`, `[PRF-21]`).
- One shared derived tolerance function consumed by the runtime coherence
  check and the tests (`[REC-12]`); iterative-solver convergence surfaced as
  a first-class error carrying the cross-section identity (`[REC-21]`).
- Representation selection at the strategy's declared-metadata seam
  (`[REC-8]`c): the producer consults the declaration before any allocation,
  and the sparse matrix is built directly from index coordinates — never
  densify-first (`[REC-13]`, `[REC-15]`).

**Prototype/test machinery** (evidence, not product):

- The pure NumPy/SciPy implementations under `gate_b_proto/`, retained only
  as an independent class-3 reference for the Nixtla adapters.
- The 12-node fixture, the recorded JSONs, and the CG solver choice (the
  convergence-guard *contract* is normative; the prototype solver is not).
- Exact condition numbers (fixture-affordable; production substitutes an
  estimate in the same derived tolerance).
- The fixture's exactly-zero coherence residual — a fixture property, not a
  product claim.

Seam note, in design terms: the preflight is one deep module behind a
four-integer interface (metadata in, itemized decision out); the dense closed
form stays behind the test seam as the class-3 reference adapter, while the
production projection adapters delegate to `hierarchicalforecast`. The same
representation ceiling gates both the dense Nixtla producer and the reference
producer rather than shrinking the strategy list. No new glossary terms are
required: "structural weights" and the least-squares/trace-minimization
families are already `[REC-9]` vocabulary; "dense-workspace ceiling" is
proposed here as the configuration name for `[PRF-21]`'s threshold and is
reported, not glossed.

## 7. Owner decision (recorded)

The HITL question this asset posed was the MinT-family disposition. The owner
decided: **ship both `wls_var` and `mint_shrink`.** Rationale: the registry
must not be sized purely for full M5 — dense-only `mint_shrink` is the
useful benchmarking reference on smaller real-world hierarchies — and it is
kept bounded by the metadata-only dense preflight rather than excluded.
Consequences, now reflected throughout this asset:

- Roster: `none`, `bottom_up`, `wls_struct`, `wls_var`, `mint_shrink`.
- `mint_shrink` is dense-only and ceiling-gated, never rejected by name:
  permitted where its itemized dense workspace estimate is `<= 1 GiB`;
  above the ceiling the run is rejected before allocation, naming `wls_var`
  and `wls_struct` as scalable alternatives.
- `mint_cov` remains the only by-name rejection (`[REC-10]`): its raw
  covariance is rank-deficient for `T < n_nodes` and ill-conditioned on
  retail-sized lattices — the common regime.
- The full-M5 result stands: at M5 metadata `mint_shrink`'s ~30.8 GiB
  estimate trips the per-run ceiling; `wls_struct`/`wls_var` remain the
  M5-feasible knobs for the `[REC-23]` coverage lever.

## 8. Commands and results

Worktree from current `origin/main` (`dc6434e`), branch
`prototype/gate-b-reconciliation`; environment `uv sync --frozen --extra dev`:

```
cd prototypes/gate-b-reconciliation
uv run python -m gate_b_proto.record          # wrote recorded/*.json
uv run pytest tests/ -q                       # 16 passed
cd <repo root>
uv run ruff check .                           # All checks passed!
uv run ruff format --check prototypes/        # 8 files already formatted
```

Suite contents: structure/exact-membership and `nnz` identity; dense/sparse
equivalence within the derived κ·eps bound (class 3); coherence per
`[REC-12]`; idempotence per `[REC-22]`; three witnesses (membership
corruption, starved solver, ceiling boundary); the owner-decision biting test
(`mint_shrink` permitted below the ceiling, rejected-at-scale above it, no
allocation either way); variance-floor and degenerate-sidecar behavior;
`mint_cov` rank deficiency vs `mint_shrink` positive-definiteness;
O(metadata) preflight under `tracemalloc`; recorded JSONs pinned against
recomputation.

## Sources

- Spec: `docs/spec/07-reconciliation.md` (`[REC-1]`–`[REC-24]`),
  `docs/spec/30-performance.md` (`[PRF-1]`, `[PRF-2]`, `[PRF-20]`–`[PRF-23]`),
  `docs/spec/50-test-and-oracle-strategy.md` (tolerance classes, witness
  discipline), `docs/spec/21-protocol-m5.md` (`[M5-H1]`–`[M5-H5]`, `[M5-D3]`),
  `docs/spec/03-engine-core.md` (spine placement).
- Frozen-engine provenance (behavior reference only):
  `calibre/reconciliation/summing.py` (dual representation, nnz rule),
  `calibre/reconciliation/nixtla_adapter.py` (sparse roster, the checked
  iterative-solver guard, the `mint_cov` rejection message),
  `calibre/execution/hierarchy_memory.py` (itemized preflight),
  `benchmarks/m5/README.md` (the preflight as deterministic stop).
