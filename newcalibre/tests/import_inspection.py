"""Resolve static and literal dynamic imports for layering tests."""

from __future__ import annotations

import ast


def imported_modules(source: str, *, package: str) -> list[tuple[int, str]]:
    """Return resolved module imports with their source line numbers."""
    tree = ast.parse(source)
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend(
                (node.lineno, module) for module in _resolved_from_imports(node, package=package)
            )
        elif isinstance(node, ast.Call) and (module := _literal_dynamic_import(node)) is not None:
            imports.append((node.lineno, module))
    return sorted(set(imports))


def _resolved_from_imports(node: ast.ImportFrom, *, package: str) -> list[str]:
    if node.level:
        package_parts = package.split(".")
        retained_parts = package_parts[: max(len(package_parts) - node.level + 1, 0)]
        module_parts = node.module.split(".") if node.module else []
        resolved_module = ".".join((*retained_parts, *module_parts))
    else:
        resolved_module = node.module or ""

    candidates = [resolved_module] if resolved_module else []
    candidates.extend(
        ".".join(part for part in (resolved_module, alias.name) if part)
        for alias in node.names
        if alias.name != "*"
    )
    return candidates


def _literal_dynamic_import(node: ast.Call) -> str | None:
    is_importlib_call = (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
    )
    is_builtin_call = isinstance(node.func, ast.Name) and node.func.id == "__import__"
    if not (is_importlib_call or is_builtin_call):
        return None

    argument = (
        node.args[0]
        if node.args
        else next(
            (keyword.value for keyword in node.keywords if keyword.arg == "name"),
            None,
        )
    )
    return (
        argument.value
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        else None
    )


__all__ = ["imported_modules"]
