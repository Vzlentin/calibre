from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# logging.Logger.makeRecord raises KeyError for extra keys that collide with
# built-in LogRecord attributes. The span finally runs while a body exception
# may be propagating, so a colliding attribute must never raise there — rename
# instead.
_UNSAFE_EXTRA_KEYS = frozenset(logging.makeLogRecord({}).__dict__) | {"asctime", "message"}


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    """Time a sub-step and log its duration on exit.

    Emits a single ``"completed span"`` JSON record carrying the span ``name``,
    a ``duration_ms``, and any pass-through ``attributes`` as flat fields
    (attributes colliding with reserved logging keys are prefixed ``attr_``).
    Timing lives in ``finally`` so a raising body is still attributed; the
    exception propagates unchanged. No metric, nesting state, or IDs — the
    smallest contract that closes the dark-time gap.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        extra: dict[str, object] = {
            (f"attr_{key}" if key in _UNSAFE_EXTRA_KEYS else key): value
            for key, value in attributes.items()
        }
        extra["span"] = name
        extra["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        logger.info("completed span", extra=extra)
