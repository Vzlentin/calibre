from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def span(name: str, **attributes: object) -> Iterator[None]:
    del name, attributes
    yield
