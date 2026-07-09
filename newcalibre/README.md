# newcalibre

The greenfield successor engine, built to the public architecture spec at
`../docs/spec/` (start at `00-overview.md`). `newcalibre` is a working name:
the repository-wide in-place hard rename happens at cutover, so expect the
package name to change once the successor replaces the frozen engine.

This is an independent uv project: it carries its own lockfile, Python pin
(3.12), Ruff/ty configuration, and pytest tiers. Root-repo tooling never
touches this tree, and nothing under `newcalibre/` may import `calibre` or
`benchmarks` (enforced by a tier-1 test). The frozen engine next door is a
behavior oracle consulted only through promoted captures in tier 3 — never a
design source.

## Start here

Read exactly three spec files, in order, before writing code:

1. `../docs/spec/00-overview.md`
2. `../docs/spec/02-domain-model.md`
3. `../docs/spec/04-forecasting-plugins.md`

Then run the first brick's test:

```bash
uv sync --locked --group dev
uv run --locked pytest tests/tier1/test_seasonal_naive.py
```

Needing anything outside those three chapters on the first-contributor path
is a spec defect to report, never a reader failure (chapter 60's rule).

## Test tiers

| Tier | Content | Cadence |
|---|---|---|
| 0 | lint, types, schema validation | every commit |
| 1 | oracle-property suite on synthetic fixtures — no oracle, no network | every commit |
| 2 | self-consistency (resume, seeded determinism) | every merge |
| 3 | conditional replay vs promoted oracle captures | scheduled/manual |
| 4 | protocol acceptance at scale | scheduled/manual |

Commands (from this directory): `uv run --locked pytest tests/tier1`,
`uv run --locked ruff check .`, `uv run --locked ruff format --check .`,
`uv run --locked ty check src/newcalibre/`.
