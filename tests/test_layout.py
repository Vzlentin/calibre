import ast
import inspect
from pathlib import Path

import pytest

from calibre.conformal import Calibrator, Score, Target
from calibre.forecast import Fitted, Forecaster, Reconciler

SOURCE = Path(__file__).parent.parent / "src" / "calibre"

# Each top-level package may import only itself and these packages.
ALLOWED = {
    "data": set(),
    "forecast": {"data"},
    "conformal": set(),
    "online": {"conformal"},
    "backtest": {"data", "forecast", "conformal", "online"},
    "metrics": set(),
    "decision": set(),
}
# A calibration method sees only the calibrator contract and the rank rules.
METHOD_IMPORTS = {"calibre.conformal.calibrators.base", "calibre.conformal.calibrators.ranks"}


def calibre_imports(path: Path) -> set[str]:
    imports = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        imports |= {name for name in names if name.split(".")[0] == "calibre"}
    return imports


MODULES = sorted(
    path for path in SOURCE.rglob("*.py") if path.parent != SOURCE or path.stem != "__init__"
)


@pytest.mark.parametrize("path", MODULES, ids=lambda path: str(path.relative_to(SOURCE)))
def test_packages_import_only_their_allowed_dependencies(path):
    package = path.relative_to(SOURCE).parts[0].removesuffix(".py")
    allowed = ALLOWED[package] | {package}
    for name in calibre_imports(path):
        assert name.split(".")[1] in allowed, f"{package} imports {name}"


def test_calibration_methods_import_only_the_contract_and_the_ranks():
    methods = SOURCE / "conformal" / "calibrators"
    for path in methods.glob("*.py"):
        if path.stem in ("__init__", "base", "ranks"):
            continue
        assert calibre_imports(path) <= METHOD_IMPORTS, path.name


def test_no_module_has_a_name_without_content():
    vague = {"core", "utils", "common", "helpers", "misc"}
    assert not [path for path in SOURCE.rglob("*.py") if path.stem in vague]


@pytest.mark.parametrize("contract", [Target, Score, Calibrator, Forecaster, Fitted, Reconciler])
def test_contracts_hold_only_abstract_methods(contract):
    methods = {name for name, value in vars(contract).items() if inspect.isfunction(value)}
    assert methods == set(contract.__abstractmethods__)
