"""Collection-only fixture with a deliberately missing numeric witness."""

import pytest


@pytest.mark.tier3
@pytest.mark.oracle_gate("meta-orphan")
def test_meta_orphan_gate() -> None:
    pass
