"""Run `references.py` in an isolated interpreter that does not import calibre."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ._panels import Dense, to_frame

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


def cv_frames(
    task: str, jobs: list, panels: dict[str, Dense], directory: Path
) -> list[pd.DataFrame]:
    """The cross-validation frame of each job of a `cv` or `mlcv` task. Each job reads the
    long frame of `panels[job["frame"]]`."""
    arrays = {"jobs": text(jobs)}
    for name, dense in panels.items():
        frame = to_frame(dense)
        arrays[f"{name}/unique_id"] = frame["unique_id"].to_numpy(dtype=str)
        arrays[f"{name}/ds"] = frame["ds"].to_numpy(dtype="datetime64[ns]")
        arrays[f"{name}/y"] = frame["y"].to_numpy()
    out = run(task, arrays, directory)
    frames = []
    for k in range(len(jobs)):
        columns = [str(column) for column in out[f"{k}/columns"]]
        frames.append(pd.DataFrame({column: out[f"{k}/col/{column}"] for column in columns}))
    return frames
