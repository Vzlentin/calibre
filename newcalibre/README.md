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
