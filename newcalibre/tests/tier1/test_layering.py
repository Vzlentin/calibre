"""Enforce static isolation from the frozen engine and benchmark tree."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from tests.import_inspection import imported_modules

pytestmark = pytest.mark.tier1

FORBIDDEN_IMPORT_ROOTS = frozenset({"benchmarks", "calibre"})
PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "newcalibre"
PACKAGE_INIT = PACKAGE_ROOT / "__init__.py"
SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts"
VN2_DATA_SCRIPT = SCRIPT_ROOT / "vn2_data.py"
ORDERING_ROOT = PACKAGE_ROOT / "ordering"
CONFORMAL_ROOT = PACKAGE_ROOT / "conformal"
OBJECTIVE_MODULE = PACKAGE_ROOT / "ordering" / "_objective.py"
ENGINE_IMPORT_ROOT = "newcalibre.engine"
SETTLE_PATH_ALLOWED_ATTRIBUTES = frozenset(
    {
        "actuals_semantics",
        "amount",
        "holding",
        "key",
        "period",
        "series_key",
        "session",
        "shortage",
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


def _engine_import_violations(
    root: Path,
    *,
    root_package: str = "newcalibre.ordering",
) -> list[str]:
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        relative_path = path.relative_to(root)
        package = ".".join((root_package, *relative_path.parent.parts))
        for line, module in _engine_imports(path, package=package):
            violations.append(f"{relative_path.as_posix()}:{line}: import {module}")
    return violations


def _engine_imports(path: Path, *, package: str) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    imported_lines: set[int] = set()
    for line, module in imported_modules(path.read_text(encoding="utf-8"), package=package):
        if _is_engine_module(module) and line not in imported_lines:
            imports.append((line, module))
            imported_lines.add(line)
    return imports


def _is_engine_module(module: str) -> bool:
    return module == ENGINE_IMPORT_ROOT or module.startswith(f"{ENGINE_IMPORT_ROOT}.")


def _settle_path_boundary_violations(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    reducers = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "settle_path_cost"
    ]
    relative_name = path.name
    if len(reducers) != 1:
        return [f"{relative_name}: expected exactly one top-level settle_path_cost definition"]

    violations: set[tuple[int, str]] = set()
    for statement in reducers[0].body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Attribute) and node.attr not in SETTLE_PATH_ALLOWED_ATTRIBUTES:
                violations.add(
                    (node.lineno, f"attribute {node.attr!r} is not a booked settlement fact")
                )
            elif isinstance(node, (ast.BinOp, ast.AugAssign)):
                violations.add((node.lineno, "accounting arithmetic is forbidden"))

    return [f"{relative_name}:{line}: {message}" for line, message in sorted(violations)]


def test_successor_package_never_imports_frozen_surfaces() -> None:
    package_sources, package_violations = _find_violations(PACKAGE_ROOT)
    script_sources, script_violations = _find_violations(SCRIPT_ROOT)

    assert PACKAGE_INIT in package_sources, "Successor AST guard did not scan its package."
    assert VN2_DATA_SCRIPT in script_sources, "Successor AST guard did not scan its scripts."
    violations = [*package_violations, *script_violations]
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


def test_layering_detector_bites_for_successor_scripts(tmp_path: Path) -> None:
    probe = tmp_path / "scripts" / "vn2_data.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "from benchmarks.vn2 import download\nfrom calibre.execution import run\n",
        encoding="utf-8",
    )

    sources, violations = _find_violations(probe.parent)

    assert sources == [probe]
    assert violations == [
        "vn2_data.py:1: import benchmarks",
        "vn2_data.py:2: import calibre",
    ]


def test_ordering_has_no_engine_import_dependency() -> None:
    assert not _engine_import_violations(ORDERING_ROOT)


def test_conformal_has_no_engine_import_dependency() -> None:
    assert not _engine_import_violations(
        CONFORMAL_ROOT,
        root_package="newcalibre.conformal",
    )


def test_conformal_engine_import_detector_bites_on_relative_import(tmp_path: Path) -> None:
    probe = tmp_path / "conformal" / "runtime.py"
    probe.parent.mkdir(parents=True)
    probe.write_text("from .. import engine as hidden\n", encoding="utf-8")

    assert _engine_import_violations(
        probe.parent,
        root_package="newcalibre.conformal",
    ) == ["runtime.py:1: import newcalibre.engine"]


def test_ordering_engine_import_detector_bites_on_supported_forms(tmp_path: Path) -> None:
    probe = tmp_path / "ordering" / "objective.py"
    probe.parent.mkdir(parents=True)
    probe.write_text(
        "from newcalibre.engine import settle as reduce\n"
        "import newcalibre.engine.settlement as runtime\n"
        "from newcalibre import engine as hidden\n"
        "from ..engine import settle as relative_reduce\n"
        "from .. import engine as relative_hidden\n"
        "importlib.import_module('newcalibre.engine.settlement')\n"
        "__import__('newcalibre.engine')\n"
        "importlib.import_module('unrelated.engine')\n"
        "__import__('another_runtime')\n",
        encoding="utf-8",
    )

    assert _engine_import_violations(tmp_path, root_package="newcalibre") == [
        "ordering/objective.py:1: import newcalibre.engine",
        "ordering/objective.py:2: import newcalibre.engine.settlement",
        "ordering/objective.py:3: import newcalibre.engine",
        "ordering/objective.py:4: import newcalibre.engine",
        "ordering/objective.py:5: import newcalibre.engine",
        "ordering/objective.py:6: import newcalibre.engine.settlement",
        "ordering/objective.py:7: import newcalibre.engine",
    ]


def test_settle_path_reads_only_booked_settlement_facts() -> None:
    assert not _settle_path_boundary_violations(OBJECTIVE_MODULE)


def test_settle_path_boundary_bites_on_renamed_accounting_arithmetic(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "_objective.py"
    probe.write_text(
        "def settle_path_cost(entries):\n"
        "    for booked in entries:\n"
        "        copied = booked.holding.amount\n"
        "        recalculated = booked.carrying_charge * booked.stock_level\n"
        "    return recalculated\n",
        encoding="utf-8",
    )

    assert _settle_path_boundary_violations(probe) == [
        "_objective.py:4: accounting arithmetic is forbidden",
        "_objective.py:4: attribute 'carrying_charge' is not a booked settlement fact",
        "_objective.py:4: attribute 'stock_level' is not a booked settlement fact",
    ]
