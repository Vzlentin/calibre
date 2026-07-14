"""Decide the /go execution mode and provision an isolated worktree.

``decide`` reads git state in the current checkout and prints the execution
mode: ``direct`` when the checkout is on ``main`` with a clean tree, else
``worktree``. Clean/dirty comes from ``git status --porcelain`` run through
Python's subprocess, which bypasses the MSYS wrapper that can print a literal
``ok`` on a clean tree.

``provision <type>/<slug>`` cuts ``.worktrees/<slug>`` on a fresh
``<type>/<slug>`` branch from ``origin/main``, runs the ``setup-worktree-unix``
steps read dynamically from ``.cursor/worktrees.json`` (substituting
``$ROOT_WORKTREE_PATH`` with the main checkout path), aborts on the first
failed step, then gates on ``uv run python -c "import calibre"`` — plus a
``data/m5`` presence check with ``--require-m5``. It never mutates the caller
checkout and refuses an existing worktree path or branch.

Exit codes: 0 success, 1 failure.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

WORKTREES_DIR = ".worktrees"
SETUP_KEY = "setup-worktree-unix"


def _bash_executable() -> str:
    """Resolve the bash used for setup steps, preferring Git Bash on Windows.

    A bare ``bash`` on Windows PATH commonly resolves to the System32 WSL
    launcher, which cannot run the MSYS-flavored setup steps (Windows-style
    ``$ROOT_WORKTREE_PATH`` substitutions, ``uv`` on the Windows PATH). Derive
    Git for Windows' own bash from the ``git`` executable's install root.
    """
    if sys.platform == "win32":
        git = shutil.which("git")
        if git is not None:
            candidate = Path(git).parent.parent / "bin" / "bash.exe"
            if candidate.is_file():
                return str(candidate)
    return "bash"


def _run(args: list[str], cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing text output without raising on failure."""
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)


def decide_mode(branch: str, porcelain: str) -> str:
    """Decide the execution mode from branch name and porcelain status.

    Args:
        branch: Output of ``git branch --show-current`` (empty on detached HEAD).
        porcelain: Output of ``git status --porcelain``.

    Returns:
        ``"direct"`` on a clean ``main`` checkout, else ``"worktree"``.
    """
    clean = porcelain.strip() == ""
    return "direct" if branch.strip() == "main" and clean else "worktree"


def read_setup_steps(config_text: str, main_path: str) -> list[str]:
    """Parse worktrees.json setup steps, substituting the main checkout path.

    Args:
        config_text: Raw JSON text of ``.cursor/worktrees.json``.
        main_path: Absolute path of the main checkout, substituted for every
            ``$ROOT_WORKTREE_PATH`` occurrence.

    Returns:
        The setup commands in declaration order.
    """
    config = json.loads(config_text)
    steps = config[SETUP_KEY]
    if not isinstance(steps, list) or not all(isinstance(s, str) for s in steps):
        raise SystemExit(f"{SETUP_KEY} in worktrees.json is not a list of strings")
    return [step.replace("$ROOT_WORKTREE_PATH", main_path) for step in steps]


def _run_step(step: str, cwd: Path) -> int:
    """Run one setup step through bash, streaming its output; return the exit code."""
    return subprocess.run([_bash_executable(), "-c", step], cwd=cwd, check=False).returncode


def _main_checkout() -> Path:
    """Resolve the main checkout root (the parent of the shared git common dir)."""
    common = _run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"])
    if common.returncode != 0:
        raise SystemExit(f"not a git repository: {common.stderr.strip()}")
    return Path(common.stdout.strip()).parent


def cmd_decide() -> int:
    """Print the execution mode and the main checkout path."""
    branch = _run(["git", "branch", "--show-current"])
    status = _run(["git", "status", "--porcelain"])
    if branch.returncode != 0 or status.returncode != 0:
        print(f"git state read failed: {(branch.stderr + status.stderr).strip()}", file=sys.stderr)
        return 1
    mode = decide_mode(branch.stdout, status.stdout)
    print(json.dumps({"mode": mode, "main": str(_main_checkout())}, indent=2))
    return 0


def cmd_provision(branch: str, require_m5: bool) -> int:
    """Provision an isolated worktree on a fresh branch cut from origin/main."""
    if "/" not in branch:
        print(f"branch must be <type>/<slug>, got {branch!r}", file=sys.stderr)
        return 1
    slug = branch.split("/", 1)[1]
    main = _main_checkout()
    worktree = main / WORKTREES_DIR / slug

    if worktree.exists():
        print(f"refusing: worktree path already exists: {worktree}", file=sys.stderr)
        return 1
    if _run(["git", "rev-parse", "--verify", f"refs/heads/{branch}"], cwd=main).returncode == 0:
        print(f"refusing: branch already exists: {branch}", file=sys.stderr)
        return 1

    # Read the setup config before creating anything: a malformed config must
    # not strand a half-provisioned worktree.
    config_path = main / ".cursor" / "worktrees.json"
    try:
        steps = read_setup_steps(config_path.read_text(encoding="utf-8"), main.as_posix())
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"cannot read setup steps from {config_path}: {exc!r}", file=sys.stderr)
        return 1

    fetch = _run(["git", "fetch", "origin", "main"], cwd=main)
    if fetch.returncode != 0:
        print(f"git fetch failed: {fetch.stderr.strip()}", file=sys.stderr)
        return 1
    add = _run(["git", "worktree", "add", str(worktree), "-b", branch, "origin/main"], cwd=main)
    if add.returncode != 0:
        print(f"git worktree add failed: {add.stderr.strip()}", file=sys.stderr)
        return 1

    def _fail(message: str) -> int:
        """Report a post-add failure with the recovery commands for the debris."""
        print(message, file=sys.stderr)
        print(
            "the worktree and branch are preserved for debugging; to retry, remove them:\n"
            f"  git worktree remove --force {WORKTREES_DIR}/{slug}\n"
            f"  git branch -D {branch}",
            file=sys.stderr,
        )
        return 1

    for step in steps:
        print(f"  -> {step}")
        if _run_step(step, cwd=worktree) != 0:
            return _fail(f"FAILED setup step: {step}")

    gate = _run(["uv", "run", "python", "-c", "import calibre"], cwd=worktree)
    if gate.returncode != 0:
        return _fail(f"venv gate failed (import calibre): {gate.stderr.strip()}")
    if require_m5 and not (worktree / "data" / "m5").is_dir():
        return _fail(f"data gate failed: no data/m5 in {worktree}")

    print(json.dumps({"workdir": str(worktree), "branch": branch}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to decide or provision."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("decide", help="print execution mode (direct/worktree) and main path")

    p_prov = sub.add_parser("provision", help="provision .worktrees/<slug> on <type>/<slug>")
    p_prov.add_argument("branch", help="<type>/<slug> branch name")
    p_prov.add_argument("--require-m5", action="store_true", help="also gate on data/m5 presence")

    args = parser.parse_args(argv)
    if args.command == "decide":
        return cmd_decide()
    return cmd_provision(args.branch, args.require_m5)


if __name__ == "__main__":
    sys.exit(main())
