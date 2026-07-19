"""Prove the ACI replay rejects a one-ULP closed-threshold crossing."""

from __future__ import annotations

import math

import pytest

from tier4.reference.aci.test_aci_reference import (
    _assert_case_matches,
    _case_scores,
    _load_reference_trace,
    _reference_case,
)

pytestmark = pytest.mark.tier4


@pytest.mark.oracle_witness("aci-reference-parity")
def test_aci_reference_rejects_one_ulp_equality_threshold_crossing() -> None:
    document = _load_reference_trace()
    case = _reference_case(document, "shared-adaptive-eta-0.125")
    _assert_case_matches(case)
    scores = _case_scores(case)
    drifted = list(scores)
    drifted[8] = math.nextafter(drifted[8], math.inf)

    with pytest.raises(AssertionError) as raised:
        _assert_case_matches(case, scores_override=tuple(drifted))

    assert str(raised.value) == (
        "ACI reference mismatch: step=8 quantity=covered expected=1 actual=0"
    )
