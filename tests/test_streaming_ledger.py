from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine


class _StubAdapter:
    def fit(self, task: ForecastTask) -> None:
        self._task = task

    def predict(self, task: ForecastTask) -> pd.DataFrame:
        last = float(task.history[Y].iloc[-1])
        return pd.DataFrame(
            {
                UNIQUE_ID: [task.unique_id] * task.horizon,
                DS: pd.date_range(
                    task.forecast_origin + pd.Timedelta(weeks=1), periods=task.horizon, freq="W"
                ),
                Y_HAT: [last + h for h in range(1, task.horizon + 1)],
                H: list(range(1, task.horizon + 1)),
            }
        )


def test_streaming_output_matches_in_memory_ledger(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2024-01-07", periods=8, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]})
    task = ForecastTask(
        history=actuals,
        horizon=2,
        model_config={"backend": "stub", "model": "stub_model"},
    )
    origins = [dates[3], dates[4]]

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    expected = BackendEngine(freq="W").execute([task], actuals, origins).ledger.to_df()
    path = tmp_path / "ledger.parquet"
    streaming_result = BackendEngine(freq="W", streaming_output=str(path)).execute(
        [task],
        actuals,
        origins,
    )

    actual = streaming_result.ledger.to_df()
    pd.testing.assert_frame_equal(actual, expected)
    assert streaming_result.ledger._frames == []
    assert path.exists()
    assert (tmp_path / "ledger.resolved.parquet").exists()


def test_streaming_output_accepts_fsspec_uri(monkeypatch) -> None:
    dates = pd.date_range("2024-01-07", periods=8, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(8)]})
    task = ForecastTask(
        history=actuals,
        horizon=1,
        model_config={"backend": "stub", "model": "stub_model"},
    )
    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    result = BackendEngine(
        freq="W",
        streaming_output="memory://calibre-tests/streaming/ledger.parquet",
    ).execute([task], actuals, [dates[3]])

    assert len(result.ledger.to_df()) == 1
    assert pd.read_parquet("memory://calibre-tests/streaming/ledger.parquet").shape[0] == 1
