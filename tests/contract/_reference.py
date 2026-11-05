"""Run `references.py` in an isolated interpreter that does not import calibre."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).with_name("references.py")


def run(task: str, arrays: dict, directory: Path) -> dict:
    """Compute one reference task. Arrays go in and out through .npz files in `directory`."""
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / f"{task}-in.npz"
    target = directory / f"{task}-out.npz"
    np.savez(source, **arrays)
    done = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), task, str(source), str(target)],
        capture_output=True,
        text=True,
        cwd=directory,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    with np.load(target, allow_pickle=False) as loaded:
        out = {key: loaded[key] for key in loaded.files}
    assert not bool(out.pop("calibre_loaded")), "the reference process imported calibre"
    return out


def text(value) -> np.ndarray:
    """A JSON value as a 0-d str array, for an .npz file."""
    return np.array(json.dumps(value))
