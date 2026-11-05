"""A1, A2, A3: what `import calibre` loads, the imports of each module, and the public names."""

import ast
import re
import subprocess
import sys
from pathlib import Path

import calibre

ROOT = Path(calibre.__file__).parent
ENGINE = "calibre.forecast.engines.statsforecast"

PACKAGES = {
    "data": (set(), {"numpy", "pandas", "scipy", "utilsforecast"}),
    "forecast": (
        {"data"},
        {"numpy", "pandas", "scipy", "pydantic", "coreforecast", "conformalforecast", "narwhals"},
    ),
    "conformal": (set(), {"numpy", "pydantic", "conformalforecast"}),
    "backtest": (
        {"data", "forecast", "conformal"},
        {"numpy", "pandas", "coreforecast", "conformalforecast"},
    ),
    "decision": (set(), {"numpy"}),
}
ENGINE_FILE = ("forecast", "engines", "statsforecast.py")
ROLLING_FILE = ("backtest", "rolling.py")
PUBLIC = {
    "Panel",
    "Hierarchy",
    "rolling_forecasts",
    "Forecasts",
    "reconcile",
    "Reconciler",
    "BottomUp",
    "Identity",
    "WlsStruct",
    "Target",
    "Step",
    "Span",
    "Split",
    "replay",
    "Replay",
    "critical_ratio",
    "order_up_to",
    "settle",
    "Settlement",
}
SUPPRESSION = re.compile(
    r"#\s*(noqa|ruff\s*:\s*noqa|type\s*:\s*ignore|ty\s*:\s*ignore|pyright\s*:\s*ignore)",
    re.IGNORECASE,
)
FORBIDDEN_NAMES = {"__import__", "exec", "eval"}
FORBIDDEN_ATTRIBUTES = {("sys", "modules"), ("os", "_exit")}
FORBIDDEN_MODULES = {"importlib", "runpy"}

A1_SCRIPT = f"""
import sys
import numpy as np
import pandas as pd

def loaded(*roots):
    return sorted(m for m in sys.modules if m.split(".")[0] in roots)

import calibre
assert loaded("statsforecast", "hierarchicalforecast") == [], loaded("statsforecast")
assert "{ENGINE}" not in sys.modules

from conformalforecast import AbsoluteResidual
periods = pd.date_range("2024-01-01", periods=30, freq="D")
values = np.arange(60, dtype=np.float32).reshape(2, 30) % 7
panel = calibre.Panel(
    series=np.array(["a", "b"]),
    periods=periods,
    freq="D",
    data=values.ravel(),
    indptr=np.array([0, 30, 60]),
    starts=np.array([0, 0]),
)
hierarchy = calibre.Hierarchy.from_attributes(
    np.array(["a", "b"]), pd.DataFrame({{"g": ["x", "y"]}}, index=["a", "b"])
)
truth = hierarchy.aggregate(panel)
origins = periods[5:25]
points = np.ones((1, 2, 2, len(origins)), np.float32)
forecasts = calibre.Forecasts(
    points=points, models=np.array(["m"]), series=panel.series, origins=origins
)
nodes = calibre.reconcile(forecasts, hierarchy, calibre.BottomUp())
calibre.replay(
    nodes,
    truth,
    target=calibre.Step(),
    method=calibre.Split(score=AbsoluteResidual(), window=5),
    level=0.5,
)
try:
    calibre.rolling_forecasts(panel, [object()], origins, 2)
except TypeError:
    pass
else:
    raise AssertionError("an engine that is not a StatsForecast object did not raise")
assert loaded("statsforecast", "hierarchicalforecast") == [], loaded("statsforecast")
assert "{ENGINE}" not in sys.modules

from statsforecast import StatsForecast
from statsforecast.models import Naive
assert "{ENGINE}" not in sys.modules
calibre.rolling_forecasts(panel, [StatsForecast(models=[Naive()], freq="D")], origins, 2)
assert "{ENGINE}" in sys.modules
assert loaded("hierarchicalforecast") == []
print("ok")
"""


def test_import_loads_no_engine(tmp_path):
    """A1: `import calibre` and every function without an engine load neither statsforecast
    nor hierarchicalforecast. The engine file loads when a StatsForecast object is passed."""
    done = subprocess.run(
        [sys.executable, "-c", A1_SCRIPT],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def modules():
    files = sorted(ROOT.rglob("*.py"))
    assert files, ROOT
    return files


def is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def walk_counted(tree: ast.AST):
    """Every node, except those inside the `if` branch of `if TYPE_CHECKING:`."""
    stack = [tree]
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, ast.If) and is_type_checking(node.test):
            children = [node.test, *node.orelse]
        else:
            children = list(ast.iter_child_nodes(node))
        stack.extend(children)


def imported(node: ast.AST, parts: tuple[str, ...]) -> list[str]:
    """The absolute module names that an import statement loads.

    `from calibre import x` and `from .. import x` name the module `calibre.x`.
    """
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    if node.level == 0:
        assert node.module is not None
        module = node.module
    else:
        package = ["calibre", *parts[:-1]]
        base = package[: len(package) - (node.level - 1)]
        assert base, f"relative import above calibre in {parts}"
        module = ".".join(base + ([node.module] if node.module else []))
    if module == "calibre" or node.module is None:
        return [f"{module}.{alias.name}" for alias in node.names]
    return [module]


def allowed(parts: tuple[str, ...], module: str) -> str | None:
    """None when the module may import `module`, or the reason it may not."""
    root = module.split(".")[0]
    if parts == ("__init__.py",):
        if root == "calibre" or module == "__future__":
            return None
        return "calibre/__init__.py imports only Calibre packages"
    if root in sys.stdlib_module_names:
        return None
    if parts[0] not in PACKAGES:
        return "module outside the packages of the table"
    calibre_packages, others = PACKAGES[parts[0]]
    if root == "calibre":
        names = module.split(".")
        if len(names) == 1:
            return "imports the calibre package"
        if names[1] == parts[0] or names[1] in calibre_packages:
            return None
        return f"package {parts[0]} does not import calibre.{names[1]}"
    if root == "statsforecast" and parts == ENGINE_FILE:
        return None
    if root in others:
        return None
    return f"package {parts[0]} does not import {root}"


def test_imports_follow_the_table():
    """A2: each module imports only the packages of its row, plus the standard library, and
    calibre/__init__.py only Calibre packages. TYPE_CHECKING imports do not count."""
    problems = []
    for path in modules():
        parts = path.relative_to(ROOT).parts
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in walk_counted(tree):
            for module in imported(node, parts):
                reason = allowed(parts, module)
                if reason is not None:
                    problems.append(f"{'/'.join(parts)}:{node.lineno}: {module}: {reason}")
    assert problems == []


def dynamic_uses(tree: ast.AST) -> list[str]:
    """Every use of a forbidden name, attribute or module, as `line: what`."""
    found = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            found.append(f"{line}: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_NAMES:
            found.append(f"{line}: .{node.attr}")
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and (node.value.id, node.attr) in FORBIDDEN_ATTRIBUTES
        ):
            found.append(f"{line}: {node.value.id}.{node.attr}")
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if (node.module, alias.name) in FORBIDDEN_ATTRIBUTES or (
                    node.module == "builtins" and alias.name in FORBIDDEN_NAMES
                ):
                    found.append(f"{line}: from {node.module} import {alias.name}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    found.append(f"{line}: import {alias.name}")
        if (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".")[0] in FORBIDDEN_MODULES
        ):
            names = ", ".join(alias.name for alias in node.names)
            found.append(f"{line}: from {node.module} import {names}")
    return found


def is_dispatch(call: ast.AST) -> bool:
    """A call `importlib.import_module(ENGINE)` or `import_module(ENGINE)`."""
    return (
        isinstance(call, ast.Call)
        and len(call.args) == 1
        and not call.keywords
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == ENGINE
    )


def rolling_dynamic_uses(tree: ast.AST) -> tuple[list[str], int]:
    """The forbidden uses of backtest/rolling.py beyond the engine dispatch, and the number
    of dispatch calls."""
    dispatch_funcs = set()
    dispatches = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not is_dispatch(node):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "import_module":
            if isinstance(func.value, ast.Name) and func.value.id == "importlib":
                dispatch_funcs.add(id(func.value))
                dispatches += 1
        elif isinstance(func, ast.Name) and func.id == "import_module":
            dispatch_funcs.add(id(func))
            dispatches += 1
    allowed_imports = {"import importlib", "from importlib import import_module"}
    found = []
    for item in dynamic_uses(tree):
        if item.split(": ", 1)[1] not in allowed_imports:
            found.append(item)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Name)
            and node.id in {"importlib", "import_module"}
            and id(node) not in dispatch_funcs
        ):
            found.append(f"{node.lineno}: {node.id} outside the engine dispatch")
    return found, dispatches


def test_no_dynamic_code():
    """A2: no module uses importlib, runpy, __import__, exec, eval, sys.modules or os._exit,
    except the engine dispatch of backtest/rolling.py."""
    problems = []
    for path in modules():
        parts = path.relative_to(ROOT).parts
        tree = ast.parse(path.read_text(), filename=str(path))
        if parts == ROLLING_FILE:
            found, _ = rolling_dynamic_uses(tree)
        else:
            found = dynamic_uses(tree)
        problems += [f"{'/'.join(parts)}:{item}" for item in found]
    assert problems == []


def test_engine_dispatch_loads_the_engine_file():
    """A2: backtest/rolling.py loads calibre.forecast.engines.statsforecast with
    `importlib.import_module`, and uses importlib for nothing else."""
    tree = ast.parse(ROOT.joinpath(*ROLLING_FILE).read_text())
    found, dispatches = rolling_dynamic_uses(tree)
    assert found == []
    assert dispatches >= 1


def test_no_suppression_comments():
    """A2: no line of src/calibre holds a noqa, type: ignore, ty: ignore or pyright: ignore."""
    problems = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        lines = path.read_text(errors="replace").splitlines()
        for number, line in enumerate(lines, start=1):
            if SUPPRESSION.search(line):
                problems.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()}")
    assert problems == []


def test_public_names():
    """A3: `calibre.__all__` is exactly the public names, and each one is defined."""
    assert sorted(calibre.__all__) == sorted(PUBLIC)
    assert len(calibre.__all__) == len(PUBLIC)
    for name in PUBLIC:
        assert getattr(calibre, name) is not None
    assert not hasattr(calibre, "Window")
