# Contributing

Read [Architecture](docs/architecture.md) and [Semantics](docs/semantics.md) before you
change code. They define the scope, the array contracts, and the calibration rules.

## Setup

Calibre uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```sh
git clone https://github.com/Vzlentin/calibre.git
cd calibre
uv sync --group dev
uvx pre-commit install  # optional, runs the checks below on each commit
```

## Checks

CI runs these on every pull request. Run them before you push.

```sh
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

## Layout

- `src/calibre` is the library: forecasting, reconciliation, calibration, decisions, and
  metrics over arrays. It has no dataset names, column layouts, or benchmark protocols.
- `bench/` is a folder of scripts for datasets and benchmark comparisons. `calibre` never
  imports from it, and it has no tests.
- Raw data stays out of Git. Loaders take a directory.

## Rules

- A concept is one module or one package. A contract is an `abc.ABC` with only abstract
  methods, and each implementation subclasses it. No `utils`, `common`, or `helpers`
  modules.
- A new calibrator is one file in `src/calibre/conformal/calibrators/` that imports only
  `calibrators.base`, `calibrators.ranks`, and `conformal.losses`. Its state is a nested
  dict of numpy arrays with the node axis kept. See "Write a calibrator" in the README.
- Hot paths work on typed arrays. Validate inputs once at the boundary.
- Numerical tests use real implementations and independent expected values, not mocks.
- Keep files below 1000 lines.

## Pull requests

- Keep each pull request to one change. Put semantic changes and refactors in separate
  pull requests.
- A change to calibration behavior updates `docs/semantics.md` in the same pull request.
- Add tests for new behavior. If you remove a test, cover the behavior it checked.
- Open an issue first for a new public API or a new dependency.
