"""Collection-only fixture whose gate incorrectly owns its own witness."""

import pytest


@pytest.mark.tier3
@pytest.mark.oracle_gate("meta-self-owned")
@pytest.mark.oracle_witness("meta-self-owned")
def test_meta_self_owned_gate() -> None:
    pass
