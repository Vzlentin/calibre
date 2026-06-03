from __future__ import annotations

import ast
from pathlib import Path


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
    assert not violations, "shipped calibre/ must not import benchmarks:\n" + "\n".join(
        violations
    )
