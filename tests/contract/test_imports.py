"""A1, A2, A3: what `import calibre` loads, the imports of each module, and the public names."""

import ast
import inspect
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from conformalforecast.nonconformity import AbsoluteResidual
from statsforecast import StatsForecast
from statsforecast.models import Naive

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
    "Window",
    "Forecaster",
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


def _sig_spec(target):
    pk = inspect.Parameter.POSITIONAL_OR_KEYWORD
    ko = inspect.Parameter.KEYWORD_ONLY
    empty = inspect.Parameter.empty
    return [
        (
            param.name,
            "pk" if param.kind == pk else ("ko" if param.kind == ko else str(param.kind)),
            None if param.default is empty else ("default", param.default),
        )
        for param in inspect.signature(target).parameters.values()
    ]


def test_public_signatures():
    """A3: public functions, methods and constructors keep their documented parameter names,
    positional-or-keyword vs keyword-only kinds, and defaults, and accept keyword arguments."""
    expected = {
        calibre.Panel: [
            ("series", "pk", None),
            ("periods", "pk", None),
            ("freq", "pk", None),
            ("data", "pk", None),
            ("indptr", "pk", None),
            ("starts", "pk", None),
            ("censored", "pk", ("default", None)),
        ],
        calibre.Panel.from_frame: [
            ("frame", "pk", None),
            ("freq", "ko", None),
            ("id_col", "ko", ("default", "unique_id")),
            ("time_col", "ko", ("default", "ds")),
            ("target_col", "ko", ("default", "y")),
            ("censored_col", "ko", ("default", None)),
        ],
        calibre.Panel.take: [("self", "pk", None), ("index", "pk", None)],
        calibre.Hierarchy: [
            ("nodes", "pk", None),
            ("summing", "pk", None),
            ("level", "pk", None),
        ],
        calibre.Hierarchy.from_attributes: [
            ("series", "pk", None),
            ("attributes", "pk", None),
            ("levels", "pk", ("default", None)),
        ],
        calibre.Hierarchy.flat: [("series", "pk", None)],
        calibre.Hierarchy.aggregate: [("self", "pk", None), ("panel", "pk", None)],
        calibre.Window: [
            ("origin", "pk", None),
            ("start", "pk", None),
            ("stop", "pk", None),
        ],
        calibre.Window.at: [
            ("panel", "pk", None),
            ("origin", "pk", None),
            ("lookback", "pk", ("default", None)),
        ],
        calibre.Forecaster.forecast: [
            ("self", "pk", None),
            ("panel", "pk", None),
            ("windows", "pk", None),
            ("horizon", "pk", None),
        ],
        calibre.Forecasts: [
            ("points", "pk", None),
            ("models", "pk", None),
            ("series", "pk", None),
            ("origins", "pk", None),
        ],
        calibre.Forecasts.from_cv: [
            ("cv", "pk", None),
            ("series", "ko", None),
            ("horizon", "ko", None),
            ("models", "ko", ("default", None)),
            ("id_col", "ko", ("default", "unique_id")),
            ("time_col", "ko", ("default", "ds")),
            ("target_col", "ko", ("default", "y")),
        ],
        calibre.Forecasts.select: [("self", "pk", None), ("model", "pk", None)],
        calibre.rolling_forecasts: [
            ("panel", "pk", None),
            ("engines", "pk", None),
            ("origins", "pk", None),
            ("horizon", "pk", None),
            ("lookback", "ko", ("default", None)),
        ],
        calibre.reconcile: [
            ("forecasts", "pk", None),
            ("hierarchy", "pk", None),
            ("method", "pk", None),
        ],
        calibre.Reconciler.project: [
            ("self", "pk", None),
            ("points", "pk", None),
            ("hierarchy", "pk", None),
        ],
        calibre.Target.cover: [
            ("self", "pk", None),
            ("n_nodes", "pk", None),
            ("horizon", "pk", None),
        ],
        calibre.Step: [],
        calibre.Span: [("first", "pk", None), ("last", "pk", None)],
        calibre.replay: [
            ("forecasts", "pk", None),
            ("truth", "pk", None),
            ("target", "ko", None),
            ("method", "ko", None),
            ("level", "ko", None),
        ],
        calibre.Replay: [
            ("origins", "pk", None),
            ("point", "pk", None),
            ("truth", "pk", None),
            ("score", "pk", None),
            ("threshold", "pk", None),
            ("lower", "pk", None),
            ("upper", "pk", None),
            ("censored", "pk", None),
        ],
        calibre.critical_ratio: [("holding", "pk", None), ("shortage", "pk", None)],
        calibre.order_up_to: [
            ("bound", "pk", None),
            ("on_hand", "pk", None),
            ("in_transit", "pk", None),
        ],
        calibre.settle: [
            ("on_hand", "pk", None),
            ("in_transit", "pk", None),
            ("demand", "pk", None),
            ("order", "pk", None),
            ("holding", "pk", None),
            ("shortage", "pk", None),
        ],
    }
    for target, spec in expected.items():
        assert _sig_spec(target) == spec, target

    periods = pd.date_range("2024-01-01", periods=10, freq="D")
    frame = pd.DataFrame(
        {
            "unique_id": np.repeat(["a", "b"], 10),
            "ds": np.tile(periods, 2),
            "y": np.arange(20, dtype=np.float32),
        }
    )
    panel = calibre.Panel.from_frame(frame=frame, freq="D")
    assert len(panel.take(index=np.array([0])).series) == 1
    hierarchy = calibre.Hierarchy.from_attributes(
        series=panel.series,
        attributes=pd.DataFrame({"g": ["x", "y"]}, index=pd.Index(["a", "b"])),
        levels=["g"],
    )
    assert calibre.Hierarchy.flat(series=panel.series).n_bottom == 2
    truth = hierarchy.aggregate(panel=panel)
    win = calibre.Window.at(panel=panel, origin=4, lookback=3)
    assert win.origin == 4
    sf = StatsForecast(models=[Naive()], freq="D")
    fc = calibre.rolling_forecasts(
        panel=panel, engines=[sf], origins=periods[3:7], horizon=2, lookback=3
    )
    selected = fc.select(model="Naive")
    rec = calibre.reconcile(forecasts=selected, hierarchy=hierarchy, method=calibre.BottomUp())
    rep = calibre.replay(
        forecasts=rec,
        truth=truth,
        target=calibre.Span(first=1, last=2),
        method=calibre.Split(score=AbsoluteResidual(), window=2),
        level=0.5,
    )
    assert rep.origins.equals(periods[3:7])
    cv = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b"],
            "ds": [periods[4], periods[5], periods[4], periods[5]],
            "cutoff": [periods[3]] * 4,
            "y": [1.0, 2.0, 3.0, 4.0],
            "Naive": [1.0, 1.0, 3.0, 3.0],
        }
    )
    from_cv = calibre.Forecasts.from_cv(cv=cv, series=panel.series, horizon=2)
    assert list(from_cv.models) == ["Naive"]
    tau = calibre.critical_ratio(holding=0.2, shortage=1.0)
    orders = calibre.order_up_to(
        bound=np.array([5.0], np.float32),
        on_hand=np.array([1.0], np.float32),
        in_transit=np.array([[1.0]], np.float32),
    )
    calibre.settle(
        on_hand=np.array([1.0], np.float32),
        in_transit=np.array([[1.0]], np.float32),
        demand=np.array([2.0], np.float32),
        order=orders,
        holding=0.2,
        shortage=float(tau),
    )
