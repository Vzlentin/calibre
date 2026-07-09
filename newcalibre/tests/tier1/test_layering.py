"""Static layering guard (KTD-A9): newcalibre never imports the frozen engine.

Authored directly from the stated prohibition — nothing under ``newcalibre/``
may import ``calibre`` or ``benchmarks`` — with no frozen implementation or
test pattern consulted. This proves static package isolation; provenance of
copied constants/logic is covered separately by the U1 source attestation.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier1

FORBIDDEN_ROOTS = {"calibre", "benchmarks"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_frozen_engine_import_anywhere_under_newcalibre():
    offenders = {}
    for path in sorted(PROJECT_ROOT.rglob("*.py")):
        if ".venv" in path.parts:
            continue
        hits = imported_roots(path) & FORBIDDEN_ROOTS
        if hits:
            offenders[str(path.relative_to(PROJECT_ROOT))] = sorted(hits)
    assert not offenders, f"frozen-engine imports found: {offenders}"


def test_guard_walks_a_nonempty_tree():
    """The guard is non-vacuous: it actually visited the package sources."""
    walked = [p for p in PROJECT_ROOT.rglob("*.py") if ".venv" not in p.parts]
    assert any("seasonal_naive" in p.name for p in walked)
