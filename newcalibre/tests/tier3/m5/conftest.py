"""Provision exact real-M5 and frozen-scorer prerequisites for Tier 3."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import newcalibre.protocols.m5.runner as runner
from newcalibre.engine import LedgerReader
from newcalibre.protocols.m5 import M5Diagnostics
from newcalibre.protocols.m5.inventory import (
    M5InputError,
    load_m5_inventory,
    read_verified_m5_input,
)

REPOSITORY_ROOT = Path(__file__).parents[4]
PROJECT_ROOT = REPOSITORY_ROOT / "newcalibre"
CONFIG = PROJECT_ROOT / "tests" / "fixtures" / "m5" / "reduced-real.yaml"
DATA = PROJECT_ROOT / "data" / "m5"
INVENTORY = PROJECT_ROOT / "benchmarks" / "m5" / "m5-inputs.json"
FROZEN_WORKTREE_ENV = "CALIBRE_FROZEN_ORACLE_WORKTREE"
FROZEN_TAG = "oracle-freeze-2026-07-06"
FROZEN_COMMIT = "686a1b284a4f4879123b4095d306f07b88d2ddc3"


@pytest.fixture(scope="session")
def exact_m5_project_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Validate required real-M5 bytes and expose one exact isolated input view."""
    try:
        inventory = load_m5_inventory(INVENTORY)
    except M5InputError as error:
        pytest.fail(f"tier 3 M5 inventory is invalid: {error}")
    missing = [entry.name for entry in inventory.files if not (DATA / entry.name).exists()]
    if missing:
        pytest.skip(f"tier 3 M5 parity skipped: missing real-M5 inputs {missing}")
    try:
        for entry in inventory.files:
            read_verified_m5_input(DATA, entry.name, inventory)
    except M5InputError as error:
        pytest.fail(f"tier 3 M5 input validation failed: {error}")

    root = tmp_path_factory.mktemp("m5-parity-project")
    target = root / "data" / "m5"
    target.mkdir(parents=True)
    for entry in inventory.files:
        source = DATA / entry.name
        destination = target / entry.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    inventory_target = root / "benchmarks" / "m5" / "m5-inputs.json"
    inventory_target.parent.mkdir(parents=True)
    shutil.copy2(INVENTORY, inventory_target)
    return root


@pytest.fixture(scope="session")
def frozen_oracle_worktree() -> Path:
    """Require an explicitly declared clean detached worktree at the frozen tag."""
    declaration = os.environ.get(FROZEN_WORKTREE_ENV)
    if declaration is None:
        pytest.skip(f"tier 3 M5 parity skipped: {FROZEN_WORKTREE_ENV} is not declared")
    worktree = Path(declaration)
    if not worktree.is_absolute():
        pytest.fail(f"{FROZEN_WORKTREE_ENV} must be an absolute path")
    if worktree.is_symlink() or not worktree.is_dir():
        pytest.fail(f"{FROZEN_WORKTREE_ENV} must name a real directory")
    head = _git(worktree, "rev-parse", "HEAD")
    if head != FROZEN_COMMIT:
        pytest.fail(f"frozen M5 worktree must be {FROZEN_TAG} at {FROZEN_COMMIT}, found {head}")
    tag_commit = _git(worktree, "rev-parse", f"{FROZEN_TAG}^{{commit}}")
    if tag_commit != FROZEN_COMMIT:
        pytest.fail(f"frozen M5 worktree cannot verify exact tag {FROZEN_TAG}")
    symbolic = subprocess.run(
        ("git", "-C", str(worktree), "symbolic-ref", "-q", "HEAD"),
        check=False,
        capture_output=True,
        text=True,
    )
    if symbolic.returncode != 1:
        pytest.fail("frozen M5 worktree must have a detached HEAD")
    if _git(worktree, "status", "--porcelain", "--untracked-files=no"):
        pytest.fail("frozen M5 worktree must have no tracked changes")
    if not (worktree / "uv.lock").is_file():
        pytest.fail("frozen M5 worktree is missing its locked environment declaration")
    if not (worktree / ".venv" / "bin" / "calibre").is_file():
        pytest.fail("frozen M5 worktree locked scorer environment is not provisioned")
    return worktree


@pytest.fixture(scope="session")
def m5_parity_run(
    exact_m5_project_root: Path,
) -> tuple[M5Diagnostics, LedgerReader]:
    """Run reduced real M5 once and retain its closed reader only in this process."""
    captured: list[LedgerReader] = []
    original_root = runner._PROJECT_ROOT
    original_score = runner.score_m5

    def retaining_score(config, ledger, *, output_dir: Path) -> M5Diagnostics:
        if captured:
            pytest.fail("tier 3 M5 parity intercepted more than one scorer call")
        captured.append(ledger)
        return original_score(config, ledger, output_dir=output_dir)

    runner._PROJECT_ROOT = exact_m5_project_root
    runner.score_m5 = retaining_score
    try:
        result = runner.run_m5(CONFIG)
    finally:
        runner.score_m5 = original_score
        runner._PROJECT_ROOT = original_root
    if len(captured) != 1:
        pytest.fail("tier 3 M5 parity did not retain exactly one closed ledger reader")
    return result.diagnostics, captured[0]


def _git(worktree: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(worktree), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(completed.stdout + completed.stderr)
    return completed.stdout.strip()
