"""Tests for the dataset adapter registry."""

from __future__ import annotations

import sys

import pytest

from calibre.execution.dataset_registry import (
    available_dataset_adapters,
    get_dataset_adapter_cls,
    resolve_dataset_adapter,
)


def test_builtin_adapters_registered() -> None:
    adapters = available_dataset_adapters()
    assert adapters == sorted(adapters)
    assert set(adapters) >= {"m5", "vn2"}
    assert get_dataset_adapter_cls("m5").__name__ == "M5DatasetAdapter"
    assert get_dataset_adapter_cls("vn2").__name__ == "VN2DatasetAdapter"


def test_resolve_without_benchmarks_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "benchmarks", raising=False)
    vn2 = resolve_dataset_adapter("vn2")
    m5 = resolve_dataset_adapter("m5")
    assert vn2.name() == "vn2"
    assert m5.name() == "m5"


def test_get_dataset_adapter_cls_returns_types() -> None:
    assert get_dataset_adapter_cls("vn2").__name__ == "VN2DatasetAdapter"
    assert get_dataset_adapter_cls("m5").__name__ == "M5DatasetAdapter"
