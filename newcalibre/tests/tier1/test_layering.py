"""Enforce static isolation from the frozen engine and benchmark tree."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

FORBIDDEN_IMPORT_ROOTS = frozenset({"benchmarks", "calibre"})
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "newcalibre"
PACKAGE_INIT = PACKAGE_ROOT / "__init__.py"
ORDERING_ROOT = PACKAGE_ROOT / "ordering"
OBJECTIVE_MODULE = PACKAGE_ROOT / "ordering" / "_objective.py"
FORBIDDEN_OBJECTIVE_SYMBOLS = frozenset(
    {
        "InventoryPosition",
        "OrderRow",
        "SettlementRequest",
        "StockoutTransition",
        "lost_sales_transition",
        "settle",
    }
)


def _absolute_imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))

    return imports


def _find_violations(root: Path) -> tuple[list[Path], list[str]]:
    sources = sorted(root.rglob("*.py"))
    violations = [
        f"{path.relative_to(root).as_posix()}:{line}: import {import_root}"
        for path in sources
        for line, module in _absolute_imports(path)
        if (import_root := module.partition(".")[0]) in FORBIDDEN_IMPORT_ROOTS
    ]
    return sources, violations


def _engine_import_violations(root: Path) -> list[str]:
    return [
        f"{path.relative_to(root).as_posix()}:{line}: import {module}"
        for path in sorted(root.rglob("*.py"))
        for line, module in _absolute_imports(path)
        if module == "newcalibre.engine" or module.startswith("newcalibre.engine.")
    ]


def test_successor_package_never_imports_frozen_surfaces() -> None:
    sources, violations = _find_violations(PACKAGE_ROOT)

    assert PACKAGE_INIT in sources, "Successor AST guard did not scan its package initializer."
    assert not violations, "Forbidden frozen-surface imports:\n" + "\n".join(violations)


def test_layering_detector_bites_on_forbidden_import_forms(tmp_path: Path) -> None:
    probe = tmp_path / "src" / "newcalibre" / "probe.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "import calibre.execution\n"
        "import benchmarks.vn2\n"
        "from calibre.core import ForecastTask\n"
        "from benchmarks.vn2 import loader\n"
        "from . import local\n",
        encoding="utf-8",
    )

    sources, violations = _find_violations(tmp_path)

    assert sources == [probe]
    assert violations == [
        "src/newcalibre/probe.py:1: import calibre",
        "src/newcalibre/probe.py:2: import benchmarks",
        "src/newcalibre/probe.py:3: import calibre",
        "src/newcalibre/probe.py:4: import benchmarks",
    ]


def test_ordering_has_no_engine_import_dependency() -> None:
    assert not _engine_import_violations(ORDERING_ROOT)


def test_ordering_engine_import_detector_bites_on_aliases(tmp_path: Path) -> None:
    probe = tmp_path / "ordering" / "objective.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "from newcalibre.engine import settle as reduce\n"
        "import newcalibre.engine.settlement as runtime\n",
        encoding="utf-8",
    )

    assert _engine_import_violations(tmp_path) == [
        "ordering/objective.py:1: import newcalibre.engine",
        "ordering/objective.py:2: import newcalibre.engine.settlement",
    ]


def test_ordering_objective_contains_no_transition_or_booking_vocabulary() -> None:
    tree = ast.parse(OBJECTIVE_MODULE.read_text(encoding="utf-8"), filename=str(OBJECTIVE_MODULE))
    referenced = {
        node.id if isinstance(node, ast.Name) else node.attr
        for node in ast.walk(tree)
        if isinstance(node, (ast.Attribute, ast.Name))
    }

    assert not (referenced & FORBIDDEN_OBJECTIVE_SYMBOLS)
