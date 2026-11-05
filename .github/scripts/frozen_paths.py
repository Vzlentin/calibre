"""Fail when a milestone PR changes a frozen path or adds a file that the contract bans.

    python3 .github/scripts/frozen_paths.py BASE HEAD

The rules are in the Checks chapter of docs/architecture.md.
"""

import subprocess
import sys
from fnmatch import fnmatch
from pathlib import PurePosixPath

FROZEN = [
    "docs/architecture.md",
    "docs/semantics.md",
    "AGENTS.md",
    "tests/__init__.py",
    "tests/contract/*",
    "bench/m5.py",
    "bench/vn2.py",
    "bench/gate.py",
    "bench/gate_m5.py",
    "bench/gate_m5_run.py",
    "bench/gate_vn2.py",
    "bench/gate_vn2_run.py",
    "bench/baselines/*",
    "src/calibre/decision/*",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".github/*",
]
BANNED_NAMES = {
    "conftest.py",
    "sitecustomize.py",
    "usercustomize.py",
    "pyproject.toml",
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "tox.ini",
    "setup.cfg",
    "ruff.toml",
    ".ruff.toml",
    "ty.toml",
}


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def problems(base: str, head: str) -> list[str]:
    merge_base = git("merge-base", base, head).strip()
    top = set(git("ls-tree", "--name-only", merge_base).split())
    found = []
    for line in git("diff", "--name-status", "--no-renames", merge_base, head).splitlines():
        status, path = line.split("\t", 1)
        parts = PurePosixPath(path).parts
        if any(fnmatch(path, pattern) for pattern in FROZEN):
            found.append(f"{path}: frozen path, status {status}")
        if status != "A":
            continue
        name = parts[-1]
        if len(parts) == 1 or parts[0] not in top:
            found.append(f"{path}: new file or directory at the root")
        elif parts[0] == "bench":
            found.append(f"{path}: new file in bench/")
        elif parts[0] == "src" and parts[1:2] != ("calibre",):
            found.append(f"{path}: new file in src/ outside src/calibre/")
        if name in BANNED_NAMES or name.endswith(".pth"):
            found.append(f"{path}: banned file name")
    return found


if __name__ == "__main__":
    found = problems(sys.argv[1], sys.argv[2])
    for problem in found:
        print(problem)
    sys.exit(1 if found else 0)
