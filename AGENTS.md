Calibre is an open-source conformal forecasting library. [docs/architecture.md](docs/architecture.md) and [docs/semantics.md](docs/semantics.md) are its contract: architecture for scope, modules, public functions and checks, semantics for calibration rules.

- Milestone work does not change the frozen paths that docs/architecture.md lists: the contract, its checks, the M5 and VN2 loaders, `src/calibre/decision/`, the dependencies and CI. When the contract is wrong, unclear, or blocks the work, stop and ask. Do not work around it, and do not weaken a check.
- `src/calibre` is the only package. `bench/` is a folder of scripts, not a package: dataset loaders, gates, and comparisons. Bench scripts import the installed `calibre` and each other. `calibre` never imports from `bench/`. Benchmarks have no tests.
- `calibre.decision` holds generic ordering: the cost-calibrated level, order-up-to, and lost-sales settlement. Keep run engines, output stores, services, and deployment out of `calibre`. They belong to products built on Calibre. A published protocol, such as the VN2 decision schedule and costs, belongs in `bench/`.
- `backtest.rolling_forecasts` and `backtest.replay` are the only origin loops. The statsforecast engine runs the loop of `rolling_forecasts` inside its workers. Do not add a run store or hidden state. The caller owns the origins.
- `calibre` writes no file, leaves no process behind, and changes no attribute of another package.
- A concept is one module or one package. A contract is an `abc.ABC` with only abstract methods, and each implementation subclasses it. In a module, the contract comes first, then the implementations. In a package, the contract is in `base.py`, each implementation is one file, and shared logic is in a file named for its content. `__init__.py` only re-exports. No `core`, `utils`, `common`, `helpers`, or `misc` modules.
- Settings of plain values are Pydantic models. Data and state are numpy arrays and frozen dataclasses. Packages import only the dependencies listed in docs/architecture.md.
- `calibre` has no dataset names, column layouts, protocol rules, or benchmark code. Those go in `bench/`. Raw data stays out of Git: loaders take a directory.
- The vault and earlier Calibre repositories are history, not authority. Do not restore their specifications or layout.
- No compatibility shims or deprecated aliases. Existing interfaces and tests are not a compatibility contract; cover required behavior before removing its tests.
- A semantic change is a contract PR: the rule in docs/semantics.md, its checks, and its code in one diff, named as such.
- Numerical tests use real implementations and independent expected values, not mocks, stubs, or monkeypatches.
- Hot paths work on typed arrays: no per-cell objects, hashes, or serialization. Validate at input boundaries once.
- Keep files below 1000 lines. Use concise public docstrings; comments explain why.
- Run Python tooling through `uv run`.
