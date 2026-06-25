"""Tests that enforce the package import-layering rules."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

# Guard against silent re-introduction of the deleted experimental conformal/
# ordering surface; the live decision path imports its submodules directly.
_DELETED_CONFORMAL_MODULES = (
    "calibre.conformal.split",
    "calibre.conformal.adaptive",
    "calibre.conformal.policies",
)
_DELETED_CONFORMAL_SYMBOLS = (
    "AdaptiveConformalInference",
    "MultiStepAdaptiveConformalInference",
    "CumulativeSplitConformalInference",
    "MultiStepSplitConformalInference",
    "OnlineConformalController",
    "ScaledAbsoluteErrorScore",
    "MultiStepIntervalPrediction",
    "symmetric_intervals",
)
_KEPT_CONFORMAL_SYMBOLS = (
    "IntervalPrediction",
    "AbsoluteErrorScore",
    "scaled_absolute_error",
    "symmetric_interval",
)


def _benchmark_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "benchmarks" or alias.name.startswith("benchmarks."):
                    offenders.append((node.lineno, f"import {alias.name}"))
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "benchmarks" or node.module.startswith("benchmarks."))
        ):
            offenders.append((node.lineno, f"from {node.module}"))
    return offenders


def test_calibre_does_not_import_benchmarks() -> None:
    root = Path(__file__).resolve().parents[1] / "calibre"
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for lineno, stmt in _benchmark_imports(path):
            rel = path.relative_to(root.parent)
            violations.append(f"{rel}:{lineno}: {stmt}")
    assert not violations, "shipped calibre/ must not import benchmarks:\n" + "\n".join(violations)


def test_decision_and_tuning_symbols_import_from_calibre() -> None:
    """Assert Gate 4's positive half: the elevated decision/tuning glue lives in ``calibre``.

    The one-way layering rule has two halves: :func:`test_calibre_does_not_import_benchmarks`
    proves ``calibre`` never reaches into ``benchmarks``; this proves the decision and
    tuning surface the benchmark consumes has a ``calibre`` home — each symbol imports from
    ``calibre`` and is callable/instantiable, so a successful import IS the assertion.
    """
    from calibre.execution.decision_loop import DecisionLoop, RoundResult
    from calibre.ordering.policy_config import (
        RsConfig,
        apply_order_policy,
        build_rs_params,
        derive_warmup_origins,
        orders_from_policy_result,
    )
    from calibre.tuning import (
        CumulativePinball,
        GlobalTuningTask,
        StudyConfig,
        optimize_global_task,
    )

    symbols = (
        RsConfig,
        build_rs_params,
        derive_warmup_origins,
        orders_from_policy_result,
        apply_order_policy,
        DecisionLoop,
        RoundResult,
        CumulativePinball,
        GlobalTuningTask,
        StudyConfig,
        optimize_global_task,
    )
    for symbol in symbols:
        assert callable(symbol), f"{symbol!r} imported from calibre but is not callable"


@pytest.mark.parametrize("module_name", _DELETED_CONFORMAL_MODULES)
def test_deleted_conformal_modules_are_gone(module_name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)


@pytest.mark.parametrize("symbol", _DELETED_CONFORMAL_SYMBOLS)
def test_deleted_conformal_symbols_are_not_re_exported(symbol: str) -> None:
    conformal = importlib.import_module("calibre.conformal")
    assert not hasattr(conformal, symbol)
    assert symbol not in conformal.__all__


def test_cumulative_bound_rule_is_gone_from_ordering() -> None:
    ordering = importlib.import_module("calibre.ordering")
    assert not hasattr(ordering, "CumulativeBoundRule")
    assert "CumulativeBoundRule" not in ordering.__all__


@pytest.mark.parametrize("symbol", _KEPT_CONFORMAL_SYMBOLS)
def test_live_conformal_symbols_still_import(symbol: str) -> None:
    conformal = importlib.import_module("calibre.conformal")
    assert hasattr(conformal, symbol)
    assert symbol in conformal.__all__
