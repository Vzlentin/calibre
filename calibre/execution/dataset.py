from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from calibre.core.order_types import CostStruct


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    history: pd.DataFrame
    future_x: pd.DataFrame | None
    costs: CostStruct | dict[str, CostStruct]
    hierarchy: pd.DataFrame | None
    censoring: pd.DataFrame | None


class DatasetAdapter(Protocol):
    def load(self, path: str | Path, **kwargs) -> DatasetBundle: ...

    def name(self) -> str: ...
