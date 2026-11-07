"""Harness of the gates. It imports no `calibre`.

`run` starts a gate script in a child process, in a new session, with a new temporary
directory as its `HOME`, `TMPDIR` and `XDG_CACHE_HOME`. The child prints `start` before it
reads the data and `end` after its last array, then writes its outputs. The parent times
the two lines with its own clock and samples the resident memory of the run every 50 ms.
`check` runs the checker of a gate in another process that does not import `calibre`.
`hierarchy` builds the nodes of `Hierarchy.from_attributes` again, without `calibre`.
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
GRACE = 5.0
DEADLINE = 120.0
INTERVAL = 0.05


class GateError(Exception):
    """A run or a check broke a rule of the gate."""


@dataclass(frozen=True)
class Run:
    wall: float
    peak: int


def run(script: str, data: Path, out: Path) -> Run:
    """Run `bench/<script>` on `data`, write its outputs to `out`, and measure it."""
    home = Path(tempfile.mkdtemp(prefix="calibre-gate-"))
    env = dict(
        os.environ,
        HOME=str(home),
        TMPDIR=str(home),
        XDG_CACHE_HOME=str(home),
        PYTHONDONTWRITEBYTECODE="1",
    )
    before = _repository_state(data)
    child = subprocess.Popen(
        [sys.executable, str(ROOT / "bench" / script), str(data), str(out)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    session = child.pid
    peak = [0]
    marks: list[float | None] = [None, None]
    done = threading.Event()
    sampler = threading.Thread(target=_sample, args=(session, peak, done), daemon=True)
    reader = threading.Thread(target=_read_marks, args=(child.stdout, marks), daemon=True)
    sampler.start()
    reader.start()
    try:
        try:
            code = child.wait(timeout=DEADLINE)
        except subprocess.TimeoutExpired as error:
            raise GateError(f"{script} timed out after {DEADLINE} s") from error
        done.set()
        sampler.join()
        reader.join(timeout=1.0)
        if code != 0:
            raise GateError(f"{script} exited with code {code}")
        if marks[0] is None or marks[1] is None:
            raise GateError(f"{script} did not print start and end")
        time.sleep(GRACE)
        left = _session(session)
        if left:
            raise GateError(f"{script} left processes behind: {sorted(left)}")
        if any(home.iterdir()):
            raise GateError(f"{script} left files in its temporary directory {home}")
        if _repository_state(data) != before:
            raise GateError(f"{script} wrote in the repository")
        return Run(wall=marks[1] - marks[0], peak=peak[0])
    finally:
        done.set()
        _kill_session(session)
        if child.stdout is not None:
            child.stdout.close()
        sampler.join(timeout=1.0)
        reader.join(timeout=1.0)
        shutil.rmtree(home, ignore_errors=True)


def check(script: str, *args: str) -> None:
    """Run the checker mode of `bench/<script>` in a process that does not import calibre."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "bench" / script), "--check", *args], cwd=ROOT
    )
    if result.returncode != 0:
        raise GateError(f"the checker of {script} failed")


def hierarchy(
    series: np.ndarray, attributes: pd.DataFrame, levels: list
) -> tuple[np.ndarray, sp.csr_array]:
    """Node labels and the summing matrix `[N, B]`, by the rule H1 of the contract."""
    table = attributes.loc[series]
    n = len(series)
    seen = {(index,) for index in range(n)} | {tuple(range(n))}
    labels, groups = [], []
    for names in levels:
        names = [names] if isinstance(names, str) else list(names)
        for key, members in table.groupby(names, sort=True).indices.items():
            members = tuple(np.sort(members).tolist())
            if members in seen:
                continue
            seen.add(members)
            groups.append(members)
            values = key if isinstance(key, tuple) else (key,)
            labels.append(
                "/".join(f"{name}={value}" for name, value in zip(names, values, strict=True))
            )
    rows = [[index] for index in range(n)] + [list(group) for group in groups] + [list(range(n))]
    sizes = [len(row) for row in rows]
    summing = sp.csr_array(
        (
            np.ones(sum(sizes)),
            (np.repeat(np.arange(len(rows)), sizes), np.concatenate(rows)),
        ),
        shape=(len(rows), n),
    )
    nodes = np.concatenate([np.asarray(series, dtype=str), labels, ["total"]])
    return nodes, summing


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def _read_marks(stream, marks: list[float | None]) -> None:
    if stream is None:
        return
    try:
        for line in stream:
            text = line.strip()
            if text == "start" and marks[0] is None:
                marks[0] = time.perf_counter()
            elif text == "end" and marks[0] is not None and marks[1] is None:
                marks[1] = time.perf_counter()
                return
    except ValueError:
        return


def _sample(session: int, peak: list[int], done: threading.Event) -> None:
    while not done.is_set():
        peak[0] = max(peak[0], _rss(_session(session)))
        done.wait(INTERVAL)


def _session(session: int) -> set[int]:
    """The processes in the session of the run, or under its first process."""
    pids = set()
    try:
        pids |= {child.pid for child in psutil.Process(session).children(recursive=True)}
        pids.add(session)
    except psutil.NoSuchProcess:
        pass
    for pid in psutil.pids():
        try:
            if os.getsid(pid) == session:
                pids.add(pid)
        except (ProcessLookupError, PermissionError):
            continue
    return pids


def _kill_session(session: int) -> None:
    for pid in _session(session):
        try:
            psutil.Process(pid).kill()
        except psutil.Error:
            continue


def _rss(pids: set[int]) -> int:
    total = 0
    for pid in pids:
        try:
            total += psutil.Process(pid).memory_info().rss
        except psutil.Error:
            continue
    return total


def _snapshot_tree(base: Path, stat_only: set[str]) -> dict[str, tuple]:
    entries: dict[str, tuple] = {}
    root = base.resolve()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root)
        top = rel_dir.parts[0] if rel_dir.parts else ""
        st_dir = os.lstat(dirpath)
        entries[str(rel_dir)] = ("d", st_dir.st_mode, st_dir.st_mtime_ns)
        for name in sorted(dirnames):
            path = Path(dirpath) / name
            if path.is_symlink():
                st = os.lstat(path)
                entries[str(path.relative_to(root))] = ("l", st.st_mode, os.readlink(path))
        for name in sorted(filenames):
            path = Path(dirpath) / name
            rel = str(path.relative_to(root))
            st = os.lstat(path)
            if path.is_symlink():
                entries[rel] = ("l", st.st_mode, os.readlink(path))
            elif top in stat_only or st.st_size > 16 * 1024 * 1024:
                entries[rel] = ("s", st.st_mode, st.st_size, st.st_mtime_ns)
            else:
                digest = hashlib.blake2b(path.read_bytes(), digest_size=16).hexdigest()
                entries[rel] = ("f", st.st_mode, st.st_size, st.st_mtime_ns, digest)
    return entries


def _repository_state(data: Path) -> dict[str, dict[str, tuple]]:
    return {
        "root": _snapshot_tree(ROOT, {".git", ".venv"}),
        "data": _snapshot_tree(data, set()),
    }
