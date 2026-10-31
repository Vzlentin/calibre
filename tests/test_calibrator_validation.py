import numpy as np
import pytest

from calibre.conformal.calibrators.base import check_ahead, check_integer


@pytest.mark.parametrize("value", [1, np.int32(1), np.int64(1), np.uint64(1)])
def test_check_integer_returns_a_python_integer(value):
    result = check_integer("window", value, 1)
    assert type(result) is int
    assert result == 1


@pytest.mark.parametrize("value", [True, np.bool_(False), 1.0, 1.5, None, "1", np.array(1)])
def test_check_integer_rejects_noninteger_values(value):
    with pytest.raises(ValueError, match="window"):
        check_integer("window", value, 1)


def test_check_integer_accepts_inclusive_bounds_and_unbounded_integers():
    assert check_integer("random_state", 0, 0, 2**32 - 1) == 0
    assert check_integer("random_state", np.uint32(2**32 - 1), 0, 2**32 - 1) == 2**32 - 1
    assert check_integer("offset", -1) == -1
    assert check_integer("offset", 0) == 0
    assert check_integer("offset", 1) == 1


@pytest.mark.parametrize(
    ("value", "limits"),
    [
        (0, {"minimum": 1}),
        (-1, {"minimum": 0}),
        (2**32, {"minimum": 0, "maximum": 2**32 - 1}),
    ],
)
def test_check_integer_rejects_out_of_range_values(value, limits):
    with pytest.raises(ValueError, match="setting"):
        check_integer("setting", value, **limits)


def test_check_ahead_preserves_scalar_shape_and_copies_int64_horizons():
    scalar = check_ahead(np.int32(3))
    assert scalar.shape == ()
    assert scalar.dtype == np.int64
    assert scalar == 3
    original = np.array([1, 3, np.iinfo(np.int64).max], dtype=np.uint64)
    result = check_ahead(original)
    np.testing.assert_array_equal(result, original)
    assert result.dtype == np.int64
    result[0] = 10
    assert original[0] == 1


@pytest.mark.parametrize(
    "ahead",
    [
        0,
        -1,
        True,
        [True],
        1.5,
        np.nan,
        np.inf,
        [],
        [[1]],
        [1, 0],
        "1",
        None,
        np.array([2**63], dtype=np.uint64),
    ],
)
def test_check_ahead_rejects_invalid_horizons(ahead):
    with pytest.raises(ValueError, match="ahead"):
        check_ahead(ahead)
