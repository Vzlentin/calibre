from __future__ import annotations

import os

import boto3
import fsspec
import pandas as pd
import pytest

from calibre.cli.commands import run
from calibre.execution.data_loading import load_period
from calibre.storage.objstore import write_ledger_shard


def _write_fixture(root) -> None:
    (root / "week_0_sales.csv").write_text(
        "Store,Product,2024-01-01,2024-01-08,2024-01-15,2024-01-22\n1,10,2,3,4,5\n2,20,4,5,6,7\n",
        encoding="utf-8",
    )


def _write_config(path, *, dataset_path: str, ledger_path: str) -> None:
    path.write_text(
        f"""
config_schema: "1.0"
dataset:
  adapter: vn2
  path: {dataset_path}
  period: 0
tasks:
  - model: SeasonalNaive
    horizon: 1
    config:
      backend: statsforecast
      season_length: 1
origins:
  start: 2024-01-22
  end: 2024-01-22
  freq: W-MON
output:
  ledger_path: {ledger_path}
  streaming: false
execution:
  engine: null
  seed: 123
""",
        encoding="utf-8",
    )


def _clear_fsspec_cache() -> None:
    fsspec.AbstractFileSystem.clear_instance_cache()


def test_cli_run_with_s3_config_matches_local_fixture(monkeypatch, tmp_path) -> None:
    pytest.importorskip("s3fs")
    moto_server = pytest.importorskip("moto.server")

    _write_fixture(tmp_path)
    expected_period = load_period(tmp_path, 0)
    local_config = tmp_path / "local.yaml"
    local_ledger = tmp_path / "local-ledger.parquet"
    _write_config(
        local_config,
        dataset_path=tmp_path.as_posix(),
        ledger_path=local_ledger.as_posix(),
    )
    expected_result = run(local_config)
    expected_ledger = expected_result.ledger.to_df().reset_index(drop=True)

    server = moto_server.ThreadedMotoServer(port=0)
    server.start()
    _, port = server.get_host_and_port()
    endpoint = f"http://127.0.0.1:{port}"

    try:
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
        monkeypatch.setenv("AWS_ENDPOINT_URL", endpoint)
        os.environ.pop("AWS_SESSION_TOKEN", None)
        _clear_fsspec_cache()

        s3 = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint)
        s3.create_bucket(Bucket="calibre-test-bucket")
        s3.put_object(
            Bucket="calibre-test-bucket",
            Key="vn2/week_0_sales.csv",
            Body=(tmp_path / "week_0_sales.csv").read_bytes(),
        )
        actual_period = load_period("s3://calibre-test-bucket/vn2", 0)
        s3_config = tmp_path / "s3.yaml"
        _write_config(
            s3_config,
            dataset_path="s3://calibre-test-bucket/vn2",
            ledger_path="s3://calibre-test-bucket/output/ledger.parquet",
        )
        s3.put_object(
            Bucket="calibre-test-bucket",
            Key="configs/s3.yaml",
            Body=s3_config.read_bytes(),
        )

        actual_result = run("s3://calibre-test-bucket/configs/s3.yaml")
        actual = actual_result.ledger.to_df().reset_index(drop=True)
        written = pd.read_parquet("s3://calibre-test-bucket/output/ledger.parquet")
        pointer = write_ledger_shard(
            expected_ledger,
            "s3://calibre-test-bucket/artifacts/shard.parquet",
        )
        shard = pd.read_parquet(str(pointer["uri"]))
    finally:
        server.stop()

    pd.testing.assert_frame_equal(actual_period, expected_period)
    pd.testing.assert_frame_equal(actual, expected_ledger)
    pd.testing.assert_frame_equal(written.reset_index(drop=True), expected_ledger)
    pd.testing.assert_frame_equal(shard.reset_index(drop=True), expected_ledger)
    assert pointer["byte_size"] > 0
