from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    """Time a sub-step and log its duration on exit.

    Emits a single ``"completed span"`` JSON record carrying the span ``name``,
    a ``duration_ms``, and any pass-through ``attributes`` as flat fields. Timing
    lives in ``finally`` so a raising body is still attributed; the exception
    propagates unchanged. No metric, nesting state, or IDs — the smallest
    contract that closes the dark-time gap.
    """
    started = time.perf_counter()
    try:
        yield
    finally:
        logger.info(
            "completed span",
            extra={
                "span": name,
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                **attributes,
            },
        )
