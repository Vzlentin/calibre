from __future__ import annotations

import pandas as pd

from calibre.core.forecast_frame import DS, UNIQUE_ID, Y_HAT, H, Y
from calibre.core.forecast_task import ForecastTask
from calibre.execution.backend import BackendEngine
from calibre.execution.ledger import ForecastLedger


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
    assert not hasattr(streaming_result.ledger, "_stream_current")
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


def test_streaming_resolution_keeps_only_pending_rows(monkeypatch, tmp_path) -> None:
    dates = pd.date_range("2024-01-07", periods=24, freq="W")
    actuals = pd.DataFrame({UNIQUE_ID: "A", DS: dates, Y: [float(i) for i in range(24)]})
    task = ForecastTask(
        history=actuals,
        horizon=4,
        model_config={"backend": "stub", "model": "stub_model"},
    )
    origins = list(dates[5:17])

    monkeypatch.setattr("calibre.execution.backend.resolve_adapter", lambda _: _StubAdapter())

    expected = BackendEngine(freq="W").execute([task], actuals, origins).ledger.to_df()
    path = tmp_path / "bounded-ledger.parquet"
    result = BackendEngine(freq="W", streaming_output=str(path)).execute(
        [task],
        actuals,
        origins,
    )

    actual = result.ledger.to_df()
    pd.testing.assert_frame_equal(actual, expected)
    assert len(result.ledger._pending) <= 10
    assert len(pd.read_parquet(path)) == len(origins) * task.horizon


def test_partitioned_streaming_output_writes_hive_partitions(tmp_path) -> None:
    ledger = ForecastLedger()
    path = tmp_path / "partitioned-ledger"
    first = pd.DataFrame(
        {
            UNIQUE_ID: ["A", "B"],
            DS: pd.date_range("2024-01-07", periods=2, freq="W"),
            Y: [float("nan"), float("nan")],
            Y_HAT: [1.0, 2.0],
            H: [1, 1],
            "forecast_origin": [pd.Timestamp("2024-01-01")] * 2,
            "model_name": ["stub", "stub"],
        }
    )
    second = first.assign(**{Y_HAT: [3.0, 4.0]})

    ledger.stream_to(path, partition_cols=[UNIQUE_ID])
    ledger.append(first)
    ledger.append(second)
    ledger.close()

    written = pd.read_parquet(path).sort_values([UNIQUE_ID, Y_HAT]).reset_index(drop=True)
    expected = pd.concat([first, second], ignore_index=True).sort_values([UNIQUE_ID, Y_HAT])
    expected = expected.reset_index(drop=True)
    written[UNIQUE_ID] = written[UNIQUE_ID].astype(str)
    pd.testing.assert_frame_equal(written[expected.columns], expected, check_dtype=False)
    assert (path / "unique_id=A" / "part-0.parquet").exists()
    assert (path / "unique_id=B" / "part-0.parquet").exists()
    assert ledger._frames == []


def test_partitioned_streaming_requires_partition_columns(tmp_path) -> None:
    ledger = ForecastLedger()
    ledger.stream_to(tmp_path / "partitioned-ledger", partition_cols=["missing"])
    frame = pd.DataFrame(
        {
            UNIQUE_ID: ["A"],
            DS: [pd.Timestamp("2024-01-07")],
            Y: [float("nan")],
            Y_HAT: [1.0],
            H: [1],
            "forecast_origin": [pd.Timestamp("2024-01-01")],
            "model_name": ["stub"],
        }
    )

    try:
        ledger.append(frame)
    except ValueError as exc:
        assert "Missing partition columns" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected missing partition columns to fail")
    finally:
        ledger.close()
