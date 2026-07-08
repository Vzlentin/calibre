---
title: "Pipeline authoring — declare, validate, compose"
status: draft
invalidation-tags: []
date: 2026-07-08
---

# 10 — Pipeline authoring

This chapter owns the authoring surface: how a user turns intent into a
running pipeline. It is the user-facing projection of chapters 02–09 — every
concept it exposes is defined there; this chapter adds no engine semantics of
its own. The "easy, clean, fast" vision element lives here and is judged by
one concrete acceptance script (see "Acceptance script"): a new user authors
and validates a runnable backtest without reading engine source. Requirements
carry stable tags (`[AUT-n]`, `[VAL-n]`, `[CMP-n]`) so tests can cite them.

## Declare the pipeline as data

A **pipeline config** is a declarative document — a tree of mappings,
sequences, and scalars (YAML as the reference syntax; any lossless equivalent
is acceptable) — that fully determines a run. Config is data, never code.

- `[AUT-1]` One schema, one validator. The same config document produces the
  same validated object whether it arrives as a file (CLI), an API payload
  (chapter 11), or a programmatic mapping. There is no authoring path that
  bypasses the validator, and no pipeline object the engine consumes that
  cannot be constructed from a validated config.
- `[AUT-2]` The document declares a schema version. The validator rejects any
  version it does not implement; a breaking schema change bumps the version.
- `[AUT-3]` Schemas are closed: an unknown key anywhere is a validation
  error, never silently ignored. The only open namespaces are plugin
  parameter blocks (`[AUT-5]`), and even those are closed against the
  *plugin's* declared schema.
- `[AUT-4]` A full pipeline — dataset adapter, model, reconciler, conformal
  method, cost structure, ordering policy, tuning block — is declarable in
  one document. Authoring a complete backtest is a minutes-scale task, not an
  integration project (the acceptance script makes this testable).
- `[AUT-5]` Plugin parameters (a dataset adapter's options, a model
  configuration's hyperparameters, a conformal method's knobs, a policy's
  parameters) live in the plugin's own block and are validated against the
  schema that plugin declares at registration (chapters 04, 05, 07, 08). A
  parameter no registered plugin declares is an error at validate time — the
  authoring layer never forwards unrecognized keys blindly into constructors.
- `[AUT-12]` **The CLI is a thin front-end.** The command-line surface
  composes the engine's closed verb set (`[DRV-2]`, chapter 03) around
  validated configs; it implements no engine semantics of its own. Layering
  is one-directional, mirroring the API rule (`[API-2]`, chapter 11): the
  CLI depends on the engine; the engine and the API never import CLI
  modules, and the configuration types the CLI accepts are engine-owned
  types, not CLI types.

### Map blocks 1:1 onto domain objects

Each top-level block materializes exactly one chapter 02 domain object or one
pipeline-stage selection. Key names below are illustrative; the schema is
normative.

| Block | Declares | Materializes | Owner |
|---|---|---|---|
| `dataset` | adapter name, source location, adapter options | panel, hierarchy facts, censoring facts | 02, 03 |
| `tasks` | model configuration, horizon, scope (local/global per `[TSK-1]`) | forecast tasks | 02, 04 |
| `origins` | the ordered origin sequence | origins | 02, 03 |
| `reconciliation` | strategy name from the reconciler registry | reconciliation stage | 07 |
| `conformal` | method name, nominal coverage level(s), partition key, method knobs | conformal method + calibration-state granularity | 05, 06 |
| `costs` | underage, overage, holding, shortage | cost structure, attached per dataset or per series `[CST-3]` | 02, 08 |
| `ordering` | policy name + policy parameters | ordering policy | 08 |
| `tuning` | search space, budget, study options | tuning study | 09 |
| `output` | ledger destinations | ledger persistence | 02, 03 |
| `execution` | driver and distribution knobs | engine execution options | 03 |

- `[AUT-6]` The mapping is 1:1 and total: every block maps onto exactly one
  domain object or stage, and every domain object a run needs is expressible
  as a block. No hidden inputs: a run is a pure function of (validated
  config, dataset content, engine version).

An illustrative minimal backtest (three blocks; everything else defaulted):

```yaml
schema: "1"
dataset:
  adapter: csv-panel
  path: ./sales.csv
tasks:
  - model: seasonal-naive
    horizon: 4
origins: {start: 2024-01-01, end: 2024-06-01, freq: W}
```

## Provide sane defaults

- `[AUT-7]` The minimal valid config names a dataset, one task (model
  configuration + horizon), and an origin sequence. Every other block is
  optional with a documented default, and the defaulted run is runnable: it
  produces a ledger.
- `[AUT-8]` An absent stage block means the stage is absent or identity — no
  reconciliation, no calibration, no ordering, local execution — never a
  silently substituted "recommended" method. Choosing a method is always
  explicit; parameterizing a chosen method has defaults.
- `[AUT-9]` Defaults are defined once, in the schema, and are stable within a
  schema version. No call site re-defaults an omitted value.
- `[AUT-10]` When one block's value must equal another's (a shared derived
  quantity, see `[VAL-4]`), exactly one block owns it and the other inherits
  by omission. Inheritance is applied at validation and visible in the
  resolved config `[AUT-11]`; an explicitly authored conflicting value is an
  error, never silently overridden.
- `[AUT-11]` Validation produces a **resolved config**: the input document
  with all defaults expanded and inheritances applied, in a canonical
  serialization. The resolved config is persisted with every run and is the
  reproducibility record — the defining subset of the resolved configuration
  feeds session identity (`[SES-1]` owns which fields are defining), and
  re-running it against the same dataset reproduces the run (chapter 03
  determinism).

## Validate as a first-class verb

`validate` is a top-level verb, peer to `run`, not a debugging afterthought.

- `[VAL-1]` **No config-caused failure after execution begins.** Any run
  failure attributable to the config document alone — a missing key, an
  unresolvable plugin name, an invalid plugin parameter, a violated
  cross-field coupling — is by definition a validation defect: `validate`
  must have rejected that document. This is the chapter's core contract.
- `[VAL-2]` `validate` performs everything checkable without reading the
  dataset: schema and type validation, plugin resolution against the
  registries, plugin-parameter validation `[AUT-5]`, and all cross-field
  couplings `[VAL-4]`. It executes nothing and is safe to run in CI. A deep
  mode additionally loads the dataset and checks the data-contract invariants
  (`[PAN-1]`–`[PAN-4]`, `[SER-2]`, `[HIE-2]`) plus config-against-data
  constraints (e.g. the origin sequence lies within the panel's calendar).
- `[VAL-3]` `run` begins by validating; execution starts only on a document
  that passes `[VAL-2]`'s dataset-free tier.
- `[VAL-4]` **Cross-field couplings are schema obligations.** Whenever two
  blocks must agree — one stage writes a forecast-frame column whose name
  encodes a nominal coverage or quantile level and another stage reads it
  back; one stage emits a bound only at a specific horizon step and another
  reads it at a window it computes from its own parameters — the schema
  either derives one value from the other (`[AUT-10]`) or checks the equality
  at validate time. Illustrative instance: an ordering policy whose decision
  covers a protection window of lead time plus review period must find its
  conformal method emitting a bound at exactly that window, and the window
  must fit within the task horizon — window arithmetic mismatches are
  validate-time errors, never mid-run crashes. Chapters 04–09 must declare
  their couplings to the schema; a coupling discoverable only by running the
  pipeline violates `[VAL-1]`.
- `[VAL-5]` Error quality: every validation error names the offending key
  path, the observed value, and the violated constraint, in terms of the
  config surface (not engine internals). One `validate` invocation reports
  all independent defects — no fix-one-rerun-discover-the-next loops.
- `[VAL-6]` **Full config parity.** Every runtime knob of every registered
  plugin and every engine option is reachable from the authoring surface. An
  automated parity test enumerates each registered plugin's declared
  configuration fields and each engine option and fails on any field not
  expressible in the schema. "Set that one in code" is a spec violation.
- `[VAL-7]` The authoring reference — every block, key, type, default,
  coupling, and each registered plugin's parameters — is generated from the
  schema and the plugin registries, never hand-maintained, so documentation
  cannot drift from the validator.

## Compose configs

### Treat a sweep as a directory of configs

- `[CMP-1]` A **sweep** is a directory of config documents; the sweep verb
  runs each as an independent run. No sweep-specific schema exists — anything
  runnable alone is sweepable, and any config in a sweep runs identically
  alone.
- `[CMP-2]` The sweep verb validates *every* config in the directory before
  executing *any* (fail fast on the whole set); each run's results are keyed
  by its config document, and one run's failure does not abort or corrupt the
  others.

### Treat a tuning run as a config plus a search space

- `[CMP-3]` A tuning run is an ordinary pipeline config plus a `tuning`
  block: a declarative search space (dimension name → type and range or
  choices), a trial budget, and study options (chapter 09). The block is
  validated always and executed only by the tune verb; under `run` it is
  inert.
- `[CMP-4]` Each trial's candidate materializes as a full config — the base
  document with the sampled values applied — and is revalidated through the
  same validator before execution, so a sampled combination that violates a
  cross-field coupling `[VAL-4]` is rejected exactly as an authored one
  would be.
- `[CMP-5]` Search-space hygiene: a dimension must name a tunable parameter.
  Dimensions naming structural identity (model name, scope, horizon,
  frequency, execution backend) or quantities derived from other blocks
  (anything the cost structure determines, e.g. via the critical ratio
  `[CST-2]`, or any value another block owns under `[AUT-10]`) are rejected
  at validate time with an error saying which block owns the value.
- `[CMP-6]` The tuning objective is not authored as a formula; the config
  selects among named objectives, defaulting to the chapter 09 default
  objective (which binds to chapter 08 symbolically).

## Acceptance script

The "easy, clean, fast" commitment is judged by executing this script with a
user who has never seen the engine source, given only the spec chapters 00
and 02, this chapter, and the generated authoring reference `[VAL-7]`:

1. Copy the minimal example; point `dataset` at a provided sample panel;
   pick a model from the authoring reference.
2. Run `validate` — it passes, or every reported error is fixable from the
   message alone `[VAL-5]`.
3. Extend the document with a reconciler, a conformal method, a cost
   structure, and an ordering policy, using only the authoring reference.
4. Run `validate`, then `run`; the run completes and writes a ledger.
5. Wrap the config into a tuning run by adding a `tuning` block; `validate`
   passes.

Pass criteria: no step consults engine source; no failure after step 4's
validation passes `[VAL-1]`; total elapsed time under thirty minutes.

## Conformance

A conforming implementation must demonstrate, by test:

1. An unknown key at any depth, including inside plugin blocks, fails
   validation `[AUT-3]`, `[AUT-5]`.
2. The parity test passes: no registered plugin field or engine option is
   unreachable from the schema `[VAL-6]`.
3. Every declared cross-field coupling has a validate-time test proving a
   violating document is rejected before execution `[VAL-4]`; a fault-
   injection test shows no config-only defect surfaces after execution
   begins `[VAL-1]`.
4. The minimal config runs to a ledger with all defaults `[AUT-7]`; the
   resolved config round-trips — re-running it reproduces the run `[AUT-11]`.
5. Sweep: all-validate-before-any-run, and per-config failure isolation
   `[CMP-2]`.
6. Tuning: a sampled candidate violating a coupling is rejected by the shared
   validator `[CMP-4]`; a forbidden search dimension fails at validate time
   `[CMP-5]`.
7. The acceptance script passes as scripted `[VAL-1]`, `[VAL-5]`, `[VAL-7]`.
8. A dependency check shows no import path from the engine or the API
   package into CLI modules `[AUT-12]`.

## Provenance

For spec authors only; the chapter stands without these. Old repo:
`calibre/cli/config.py` and `calibre/cli/commands.py`. Positive space: a
typed, frozen, unknown-key-rejecting schema with `validate` as a CLI verb;
cross-field validators coupling protection-window arithmetic (lead time +
review period = the conformal protection period ≤ horizon), coverage-level
agreement between the decision-bound writer and the policy reader, and
sweep-as-directory (`run-sweep`). Negative space: parity gaps — runtime
config fields not reachable from YAML were documented as "deferred" in the
schema itself (`[VAL-6]` closes this); couplings were discovered by mid-run
crashes and back-ported into ad-hoc validators one by one (`[VAL-4]` makes
them a declared obligation); cross-block default inheritance was patched on
the raw mapping pre-validation (`[AUT-10]`/`[AUT-11]` make it a resolved-
config feature); dataset adapter options were forwarded unvalidated into
constructors (`[AUT-5]` closes this); and a parse-time mutual exclusion
between two calibration blocks papered over a single-runtime-slot engine
limitation rather than deriving from declared engine capability.
