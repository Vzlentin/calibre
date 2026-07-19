"""Exercise atomic actual acceptance and immutable observe state."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from newcalibre.domain import SERIES_KEY, CensoringAssertion, HierarchyIndex
from newcalibre.observe import (
    ActualRecord,
    ActualsSubmission,
    ObservedActual,
    ObserveError,
    ObserveLoop,
)

pytestmark = pytest.mark.tier1
_TIMESTAMP = pd.Timestamp("2026-01-01")


def _hierarchy() -> HierarchyIndex:
    return HierarchyIndex.from_facts(
        pd.DataFrame(
            {
                SERIES_KEY: ["sku-a", "sku-b"],
                "category": ["tops", "tops"],
            }
        ),
        bottom_series=("sku-a", "sku-b"),
    )


def _record(
    series_key: str = "sku-a",
    *,
    value: int | float = 3,
    assertion: CensoringAssertion | None = CensoringAssertion.UNCENSORED,
    availability_bound: int | float | None = None,
    timestamp: pd.Timestamp = _TIMESTAMP,
) -> ActualRecord:
    return ActualRecord(
        series_key=series_key,
        timestamp=timestamp,
        recorded_value=value,
        censoring_assertion=assertion,
        availability_bound=availability_bound,
    )


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("series_key", "", "series key"),
        ("series_key", "\ud800", "UTF-8"),
        ("timestamp", pd.NaT, "timestamp"),
        ("timestamp", pd.Timestamp("2026-01-01", tz="UTC"), "timezone-naive"),
        ("recorded_value", True, "finite real"),
        ("recorded_value", np.nan, "finite real"),
        ("recorded_value", np.inf, "finite real"),
        ("censoring_assertion", "censored", "CensoringAssertion"),
        ("availability_bound", True, "finite real"),
        ("availability_bound", np.nan, "finite real"),
    ],
)
def test_actual_record_rejects_malformed_facts(
    field: str,
    invalid: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "series_key": "sku-a",
        "timestamp": _TIMESTAMP,
        "recorded_value": 3.0,
        "censoring_assertion": CensoringAssertion.UNCENSORED,
        "availability_bound": 4.0,
    }
    values[field] = invalid

    with pytest.raises(ObserveError, match=message):
        ActualRecord(**values)  # type: ignore[arg-type]


def test_actual_record_normalizes_python_and_numpy_reals_without_losing_integers() -> None:
    integer = _record(value=np.int64(3), availability_bound=np.float32(4.5))
    floating = _record(value=np.float64(-0.0), assertion=None)

    assert integer.recorded_value == 3
    assert type(integer.recorded_value) is int
    assert integer.availability_bound == 4.5
    assert type(integer.availability_bound) is float
    assert floating.recorded_value == 0.0
    assert floating.censoring_assertion is None


def test_submission_snapshots_records_and_rejects_duplicate_keys() -> None:
    source = [_record()]
    submission = ActualsSubmission(source)
    source.append(_record("sku-b"))

    assert submission.records == (_record(),)
    with pytest.raises(FrozenInstanceError):
        submission.records = ()  # type: ignore[misc]
    with pytest.raises(ObserveError, match="duplicate"):
        ActualsSubmission((_record(), _record(value=9)))


def test_accept_rejects_unknown_and_derived_labels_atomically() -> None:
    hierarchy = _hierarchy()
    aggregate = next(
        node.label for node in hierarchy.nodes if node.label.startswith("__aggregate__")
    )
    loop = ObserveLoop(hierarchy=hierarchy)

    for key in ("missing", aggregate, "__total__"):
        with pytest.raises(ObserveError, match="bottom series"):
            loop.accept(ActualsSubmission((_record(), _record(key))))
        assert loop.staged_history == ()


def test_accept_is_all_or_nothing_for_mixed_conflicting_submission() -> None:
    loop = ObserveLoop(hierarchy=_hierarchy())
    first = loop.accept(ActualsSubmission((_record(),)))
    before = loop.staged_history

    assert first.history_appends == (ObservedActual.from_record(_record()),)
    with pytest.raises(ObserveError, match="sku-a.*2026-01-01"):
        loop.accept(
            ActualsSubmission(
                (
                    _record("sku-b"),
                    _record(value=4),
                )
            )
        )

    assert loop.staged_history == before


def test_identical_reposts_are_idempotent_across_committed_and_staged_history() -> None:
    committed = ObservedActual.from_record(_record())
    loop = ObserveLoop(hierarchy=_hierarchy(), observed_history=(committed,))

    committed_noop = loop.accept(ActualsSubmission((_record(),)))
    accepted = loop.accept(ActualsSubmission((_record("sku-b", value=7),)))
    staged_noop = loop.accept(ActualsSubmission((_record("sku-b", value=7),)))

    assert committed_noop.history_appends == ()
    assert committed_noop.idempotent_keys == (committed.key,)
    assert accepted.history_appends == (ObservedActual.from_record(_record("sku-b", value=7)),)
    assert staged_noop.history_appends == ()
    assert staged_noop.idempotent_keys == ((_record("sku-b").series_key, _TIMESTAMP),)
    assert loop.staged_history == accepted.history_appends


@pytest.mark.parametrize(
    "conflict",
    [
        _record(value=4),
        _record(assertion=CensoringAssertion.CENSORED),
        _record(availability_bound=3.0),
    ],
)
def test_value_status_and_bound_conflicts_name_the_key_without_state_drift(
    conflict: ActualRecord,
) -> None:
    loop = ObserveLoop(
        hierarchy=_hierarchy(),
        observed_history=(ObservedActual.from_record(_record()),),
    )

    with pytest.raises(ObserveError, match="sku-a.*2026-01-01"):
        loop.accept(ActualsSubmission((conflict,)))

    assert loop.staged_history == ()


def test_acceptance_and_committed_history_are_defensive_snapshots() -> None:
    committed_source = [ObservedActual.from_record(_record())]
    loop = ObserveLoop(hierarchy=_hierarchy(), observed_history=committed_source)
    committed_source.clear()
    records = [_record("sku-b")]

    acceptance = loop.accept(ActualsSubmission(records))
    records.clear()
    cycle = loop.cycle(pd.Timestamp("2026-01-02"))

    assert acceptance.history_appends == (ObservedActual.from_record(_record("sku-b")),)
    assert cycle.history_appends == acceptance.history_appends
    assert loop.committed_history == (ObservedActual.from_record(_record()),)
