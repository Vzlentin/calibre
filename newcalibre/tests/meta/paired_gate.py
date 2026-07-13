"""Collection-only fixture with a valid two-node numeric gate pair."""

import pytest


@pytest.mark.tier3
@pytest.mark.oracle_gate("meta-paired")
def test_meta_paired_gate() -> None:
    pass


@pytest.mark.tier3
@pytest.mark.oracle_witness("meta-paired")
def test_meta_paired_witness() -> None:
    pass
