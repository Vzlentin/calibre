"""The ledger: issued forecasts that wait for their targets, and the order they mature in.

An h-step forecast is known h periods after its origin. The ledger keeps the issued
points and thresholds until then, and releases a column only when the last step it
covers is known. It knows the target cover, not scores or calibrators.

Origins wait in a ring of slots. An origin waits until its last covered step, `reach`
periods later, so at most `reach` origins wait, and slot `origin % reach` is free again
when reused. Only the covered steps are kept. Origin -1 marks an empty slot.
"""

from dataclasses import dataclass

import numpy as np

from calibre.conformal.calibrators.base import State


@dataclass(frozen=True)
class Matured:
    """Columns whose targets became known at one origin, in origin order.

    Each row `[K]` is one column of one issuing origin. `target`, `point`, `issued`,
    and `censored` are `[K, N]`: the column sums and the threshold issued for them.
    """

    origin: np.ndarray
    column: np.ndarray
    target: np.ndarray
    point: np.ndarray
    issued: np.ndarray
    censored: np.ndarray


def reach(cover: np.ndarray) -> int:
    """Number of steps up to the last step that any column covers."""
    return int(np.flatnonzero(cover.any(axis=0))[-1]) + 1


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
        "known": np.zeros((slots, len(cover)), dtype=bool),
    }


def observe(
    state: State, cover: np.ndarray, origin: int, actuals: np.ndarray, censored: np.ndarray
) -> tuple[State, Matured | None]:
    """Record the periods that end at `origin`, and release the columns now known.

    `actuals` is `[N, k]`, the node values of the k periods after the previous origin.
    On the first call, k can be any length.
    """
    last = int(state["last"])
    n_new = actuals.shape[1]
    if last >= 0 and (origin <= last or n_new != origin - last):
        raise ValueError(f"origin {origin} after {last} needs {origin - last} new periods")
    _record(state, actuals, censored, origin - n_new, origin)
    matured = _release(state, cover[:, : reach(cover)], origin)
    state["last"] = np.array(origin, dtype=np.int64)
    return state, matured


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
    state["known"][slot] = False
    return state


def _record(
    state: State, actuals: np.ndarray, censored: np.ndarray, last: int, origin: int
) -> None:
    """Write the newly observed periods into the targets of waiting origins, in place."""
    slots = state["point"].shape[2]
    targets = state["origin"][:, None] + np.arange(1, slots + 1)
    waiting = (state["origin"] >= 0)[:, None]
    rows, steps = np.nonzero(waiting & (targets > last) & (targets <= origin))
    periods = targets[rows, steps] - last - 1
    state["actual"][rows, :, steps] = actuals[:, periods].T
    state["censored"][rows, :, steps] = censored[:, periods].T


def _release(state: State, cover: np.ndarray, origin: int) -> Matured | None:
    """Sum the columns whose last step is now known, and free finished slots."""
    last_step = cover.shape[1] - np.argmax(cover[:, ::-1], axis=1)
    waiting = state["origin"] >= 0
    known = waiting[:, None] & (state["origin"][:, None] + last_step <= origin)
    new = known & ~state["known"]
    rows, columns = np.nonzero(new)
    order = np.lexsort((columns, state["origin"][rows]))
    rows, columns = rows[order], columns[order]
    matured = None
    if len(rows):
        n_nodes = state["point"].shape[1]
        target = np.empty((len(rows), n_nodes), dtype=np.float32)
        point = np.empty((len(rows), n_nodes), dtype=np.float32)
        flags = np.empty((len(rows), n_nodes), dtype=bool)
        for column in np.unique(columns):
            these = np.flatnonzero(columns == column)
            steps = np.flatnonzero(cover[column])
            target[these] = state["actual"][rows[these]][:, :, steps].sum(axis=-1)
            point[these] = state["point"][rows[these]][:, :, steps].sum(axis=-1)
            flags[these] = state["censored"][rows[these]][:, :, steps].any(axis=-1)
        matured = Matured(
            origin=state["origin"][rows],
            column=columns,
            target=target,
            point=point,
            issued=state["issued"][rows, :, columns],
            censored=flags,
        )
    state["known"] |= new
    state["origin"][waiting & state["known"].all(axis=1)] = -1
    return matured
