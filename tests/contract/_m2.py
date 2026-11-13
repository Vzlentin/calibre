"""The M2 names, read when a test uses them, so that the tests collect on code without them.

Use them as attributes of this module, such as `_m2.combine(...)`.
"""

from typing import Any

import calibre


def __getattr__(name: str) -> Any:
    if name == "combine":
        return calibre.combine  # ty: ignore[unresolved-attribute]
    if name == "concat":
        return calibre.Forecasts.concat  # ty: ignore[unresolved-attribute]
    if name == "WindowAverage":
        return calibre.WindowAverage  # ty: ignore[unresolved-attribute]
    if name == "SeasonalNaive":
        return calibre.SeasonalNaive  # ty: ignore[unresolved-attribute]
    raise AttributeError(name)
