Calibre is an open-source conformal forecasting library. [docs/architecture.md](docs/architecture.md) is canonical for scope and array contracts, [docs/semantics.md](docs/semantics.md) for calibration rules.

- `src/calibre` is the only package: forecasting, reconciliation, calibration, and metrics over arrays. `bench/` is a folder of scripts, not a package: dataset loaders, benchmark simulators, and comparisons. Bench scripts import the installed `calibre` and each other. `calibre` never imports from `bench/`. Benchmarks have no tests.
- `calibre.decision` holds generic ordering: the cost-calibrated level, order-up-to, and lost-sales settlement. Keep run engines, output stores, services, and deployment out of `calibre`. They belong to products built on Calibre. A published protocol, such as the VN2 decision schedule and costs, belongs in `bench/`.
- `backtest.rolling_forecasts` and `backtest.replay` are the only origin loops. `online.step` takes a state and returns the next one. Do not add a kernel, a run store, or hidden state. The caller owns the origins and stores the state.
- A concept is one module or one package. A contract is an `abc.ABC` with only abstract methods, and each implementation subclasses it. In a module, the contract comes first, then the implementations. In a package, the contract is in `base.py`, each implementation is one file, and shared logic is in a file named for its content. `__init__.py` only re-exports. No `core`, `utils`, `common`, `helpers`, or `misc` modules.
- A calibration method is one file in `conformal/calibrators/` that imports only `calibrators.base` and `calibrators.ranks`. `tests/test_layout.py` enforces this and the package dependencies in docs/architecture.md.
- Calibrator state is a nested dict of numpy arrays with the node axis kept. No fitted objects in state: recompute them from the arrays.
- `calibre` has no dataset names, column layouts, protocol rules, or benchmark code. Those go in `bench/`. Raw data stays out of Git: loaders take a directory.
- The vault and earlier Calibre repositories are history, not authority. Do not restore their specifications or layout.
- No compatibility shims or deprecated aliases. Existing interfaces and tests are not a compatibility contract; cover required behavior before removing its tests.
- Name and test semantic changes separately from structural refactoring, and update docs/semantics.md in the same diff. Surface conflicts in the canonical documents instead of choosing silently.
- Numerical tests use real implementations and independent expected values, not mocks, stubs, or monkeypatches.
- Hot paths work on typed arrays: no per-cell objects, hashes, or serialization. Validate at input boundaries once.
- Keep files below 1000 lines. Use concise public docstrings; comments explain why.
- Run Python tooling through `uv run --locked`.
