"""SQL-backed sales ingestion (roadmap P1.8, issue #61).

``SqlSalesAdapter`` implements the ``SalesAdapter`` seam against the project's
own Postgres ``sales`` table, mirroring ``SnapshotSalesAdapter`` (parquet) so
``/fit`` and ``/tune`` can resolve history from a ``sql://`` URI instead of an
inline JSON body. Point-in-time reads honour ``as_of <= origin`` (NULL ``as_of``
rows are always visible), then keep the latest revision per ``(unique_id, ds)``.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y
from calibre.storage.models import Sales
from calibre.storage.postgres import session_scope


class SqlSalesAdapter:
    """``SalesAdapter`` over the project's own ``sales`` table."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._factory = session_factory

    def load_sales(self, sku_set: list[str], as_of: pd.Timestamp | None) -> pd.DataFrame:
        empty = pd.DataFrame(columns=[UNIQUE_ID, DS, Y])
        if not sku_set:
            return empty
        with session_scope(self._factory) as session:
            stmt = select(Sales).where(Sales.unique_id.in_([str(s) for s in sku_set]))
            if as_of is not None:
                cutoff = pd.Timestamp(as_of).to_pydatetime()
                stmt = stmt.where((Sales.as_of.is_(None)) | (Sales.as_of <= cutoff))
            rows = session.scalars(stmt).all()
            frame = pd.DataFrame(
                [
                    {
                        UNIQUE_ID: row.unique_id,
                        DS: pd.Timestamp(row.ds),
                        Y: float(row.y),
                        "as_of": row.as_of,
                    }
                    for row in rows
                ]
            )
        if frame.empty:
            return empty
        frame = frame.sort_values("as_of").drop_duplicates([UNIQUE_ID, DS], keep="last")
        frame = frame.drop(columns="as_of")
        frame[DS] = pd.to_datetime(frame[DS])
        return frame.reset_index(drop=True)
