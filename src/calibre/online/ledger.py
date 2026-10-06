"""The ledger: issued forecasts that wait for their targets, and the order they become known.

An h-step forecast is known h periods after its origin. The ledger keeps the issued
points and thresholds until then, and releases a column only when the last step it
covers is known. It scores the columns it releases, and knows no calibrator.

Origins wait in a ring of slots. An origin waits until its last covered step, `reach`
periods later, so at most `reach` origins wait, and slot `origin % reach` is free again
when reused. Only the covered steps are kept. Origin -1 marks an empty slot.
"""

import numpy as np

from calibre.conformal.calibrators.base import Feedback, State
from calibre.conformal.scores import Score


def last_step(cover: np.ndarray) -> np.ndarray:
    """Last step that each column of `cover` `[C, H]` covers, counted from 1."""
    return (cover * np.arange(1, cover.shape[1] + 1)).max(axis=1)


def reach(cover: np.ndarray) -> int:
    """Number of steps up to the last step that any column covers."""
    return int(last_step(cover).max())


def initial_state(n_nodes: int, cover: np.ndarray) -> State:
    """Empty ledger for `n_nodes` nodes and the columns of `cover` `[C, H]`."""
    slots = reach(cover)
    return {
        "last": np.array(-1, dtype=np.int64),
        "origin": np.full(slots, -1, dtype=np.int64),
        "point": np.zeros((slots, n_nodes, slots), dtype=np.float32),
        "actual": np.full((slots, n_nodes, slots), np.nan, dtype=np.float32),
        "censored": np.zeros((slots, n_nodes, slots), dtype=bool),
        "issued": np.zeros((slots, n_nodes, len(cover)), dtype=np.float32),
        "released": np.zeros((slots, len(cover)), dtype=bool),
    }


def observe(
    state: State,
    cover: np.ndarray,
    score: Score,
    origin: int,
    actuals: np.ndarray,
    censored: np.ndarray,
) -> tuple[State, Feedback | None]:
    """Record the periods that end at `origin`, and release the columns now known.

    `actuals` is `[N, k]`, the node values of the k periods after the previous origin.
    On the first call, k can be any length.
    """
    last = int(state["last"])
    if last >= 0 and (origin <= last or actuals.shape[1] != origin - last):
        raise ValueError(f"origin {origin} after {last} needs {origin - last} new periods")
    cover = cover[:, : state["point"].shape[2]]
    _record(state, origin - actuals.shape[1] + 1, actuals, censored)
    rows, columns = _due(state, cover, origin)
    feedback = _feedback(state, cover, score, rows, columns) if len(rows) else None
    state["released"][rows, columns] = True
    state["origin"][state["released"].all(axis=1)] = -1
    state["last"] = np.array(origin, dtype=np.int64)
    return state, feedback


def issue(state: State, origin: int, points: np.ndarray, threshold: np.ndarray) -> State:
    """Put the points `[N, H]` and threshold `[N, C]` issued at `origin` in its slot."""
    slot = origin % len(state["origin"])
    if state["origin"][slot] >= 0:
        raise ValueError(
            f"origin {origin} reuses the slot of waiting origin {state['origin'][slot]}"
        )
    state["origin"][slot] = origin
    state["point"][slot] = points[:, : state["point"].shape[2]]
    state["actual"][slot] = np.nan
    state["censored"][slot] = False
    state["issued"][slot] = threshold
    state["released"][slot] = False
    return state


def _record(state: State, first: int, actuals: np.ndarray, censored: np.ndarray) -> None:
    """Write the periods from `first` into the targets of the waiting origins."""
    steps = np.arange(1, state["point"].shape[2] + 1)
    periods = state["origin"][:, None] + steps - first
    waiting = (state["origin"] >= 0)[:, None]
    rows, columns = np.nonzero(waiting & (periods >= 0) & (periods < actuals.shape[1]))
    periods = periods[rows, columns]
    state["actual"][rows, :, columns] = actuals[:, periods].T
    state["censored"][rows, :, columns] = censored[:, periods].T


def _due(state: State, cover: np.ndarray, origin: int) -> tuple[np.ndarray, np.ndarray]:
    """Slots and columns whose last covered step is now known, in origin order."""
    waiting = (state["origin"] >= 0)[:, None]
    ended = state["origin"][:, None] + last_step(cover) <= origin
    rows, columns = np.nonzero(waiting & ended & ~state["released"])
    order = np.lexsort((columns, state["origin"][rows]))
    return rows[order], columns[order]


def _feedback(
    state: State, cover: np.ndarray, score: Score, rows: np.ndarray, columns: np.ndarray
) -> Feedback:
    """Scores of the column sums of the given slots, with their issued thresholds."""
    covered = cover[columns][:, None, :]
    target = np.where(covered, state["actual"][rows], 0).sum(axis=-1)
    point = np.where(covered, state["point"][rows], 0).sum(axis=-1)
    return Feedback(
        origin=state["origin"][rows],
        column=columns,
        scores=score.score(target, point).astype(np.float32),
        issued=state["issued"][rows, :, columns],
        censored=(covered & state["censored"][rows]).any(axis=-1),
    )
