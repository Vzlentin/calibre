from __future__ import annotations

import os

import boto3
import pandas as pd
import pytest

from calibre.execution.data_loading import load_period


def _write_fixture(root) -> None:
    (root / "week_0_sales.csv").write_text(
        "Store,Product,2024-01-01,2024-01-08\n1,10,2,3\n2,20,4,5\n",
        encoding="utf-8",
    )


def test_s3_ingestion_matches_local_fixture(monkeypatch, tmp_path) -> None:
    pytest.importorskip("s3fs")
    moto_server = pytest.importorskip("moto.server")

    _write_fixture(tmp_path)
    expected = load_period(tmp_path, 0)

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

        s3 = boto3.client("s3", region_name="us-east-1", endpoint_url=endpoint)
        s3.create_bucket(Bucket="calibre-test-bucket")
        s3.put_object(
            Bucket="calibre-test-bucket",
            Key="vn2/week_0_sales.csv",
            Body=(tmp_path / "week_0_sales.csv").read_bytes(),
        )

        actual = load_period("s3://calibre-test-bucket/vn2", 0)
    finally:
        server.stop()

    pd.testing.assert_frame_equal(actual, expected)
