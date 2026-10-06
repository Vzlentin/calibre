Calibre is an open-source conformal forecasting library. [docs/architecture.md](docs/architecture.md) is canonical for scope and array contracts, [docs/semantics.md](docs/semantics.md) for calibration rules.

- The repository has two packages. `calibre` is the library: forecasting, reconciliation, calibration, and metrics over arrays. `calibre_bench` is the benchmark package: dataset loaders, benchmark simulators, and comparisons. `calibre` never imports `calibre_bench`.
- Keep ordering policies, run engines, output stores, services, and deployment out of both packages. They belong to products built on Calibre. A benchmark simulator that reproduces a published protocol, such as the VN2 inventory rules, belongs in `calibre_bench`.
- `forecast_origins` and `Conformal.replay` are the only origin loops. `Conformal.step` takes a state and returns the next one. Do not add a kernel, a run store, or hidden state. The caller owns the origins and stores the state.
- Calibrator state is a nested dict of numpy arrays with the node axis kept. No fitted objects in state: recompute them from the arrays.
- `calibre` has no dataset names, column layouts, protocol rules, or benchmark code. Those go in `calibre_bench`. Raw data stays out of Git: loaders take a directory.
- The vault and earlier Calibre repositories are history, not authority. Do not restore their specifications or layout.
- No compatibility shims or deprecated aliases. Existing interfaces and tests are not a compatibility contract; cover required behavior before removing its tests.
- Name and test semantic changes separately from structural refactoring, and update docs/semantics.md in the same diff. Surface conflicts in the canonical documents instead of choosing silently.
- Numerical tests use real implementations and independent expected values, not mocks, stubs, or monkeypatches.
- Hot paths work on typed arrays: no per-cell objects, hashes, or serialization. Validate at input boundaries once.
- Keep files below 1000 lines. Use concise public docstrings; comments explain why.
- Run Python tooling through `uv run --locked`.
