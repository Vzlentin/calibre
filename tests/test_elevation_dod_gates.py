"""Guard the four-gate Definition of Done by asserting each gate's owning test exists.

This file is the four-gate acceptance marker — one test per gate that
guards the *existence* of the gate's owning test(s). The heavy assertions (the
``4992.20`` parity reproduction, the live tune, the M5 decision→order→cost run,
the boundary scan) live in the owning tests and run there; this file is the DoD
index that fails loudly if any owning gate-test symbol goes missing.

The mechanism is deliberate: each test imports the owning module via
:func:`importlib.import_module` and asserts ``hasattr`` for the exact function
name. A module-level ``from tests.… import test_*`` is avoided on purpose —
pytest would re-collect the imported ``test_*`` into THIS file and re-run the
heavy gate (a ~22s tune) or error on a fixture not visible here. Importing the
module and probing its symbols collects exactly one node per gate and re-runs
nothing. This is a **symbol-existence** guard, not a collection-time guard.
"""

from __future__ import annotations

import importlib


def test_gate_1_parity_owned_by_vn2_cli_parity() -> None:
    """Assert Gate 1 (parity): ``calibre run`` reproduces VN2 ``4992.20`` via the settle path."""
    mod = importlib.import_module("tests.benchmarks.test_vn2_cli_parity")
    assert hasattr(mod, "test_c5_calibre_run_reproduces_4992_via_settle_path")


def test_gate_2_tuning_owned_by_cli_tune() -> None:
    """Assert Gate 2 (tuning): ``calibre run --tune`` builds the benchmark study; tolerant."""
    mod = importlib.import_module("tests.cli.test_cli_tune")
    assert hasattr(mod, "test_run_tune_matches_benchmark_study_surface")
    assert hasattr(mod, "test_run_tune_small_real_run_completes")


def test_gate_3_generality_owned_by_m5_ordering_run() -> None:
    """Assert Gate 3 (generality): the M5 ordering testbed runs decision→order→cost end-to-end.

    Existence only — the owning test is a default-suite test (NOT ``@regression``),
    so no marker is asserted here.
    """
    mod = importlib.import_module("tests.execution.test_m5_ordering_run")
    assert hasattr(mod, "test_m5_ordering_run_decision_settle_finite_cost")


def test_gate_4_boundary_owned_by_package_layering() -> None:
    """Assert Gate 4 (boundary): no ``benchmarks`` import; decision/tuning live in ``calibre``."""
    mod = importlib.import_module("tests.test_package_layering")
    assert hasattr(mod, "test_calibre_does_not_import_benchmarks")
    assert hasattr(mod, "test_decision_and_tuning_symbols_import_from_calibre")
