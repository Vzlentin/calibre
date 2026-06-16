"""Tests for the Spread seam and the AnalyticRadius adapter."""

from __future__ import annotations

import numpy as np

from calibre.conformal.spread import AnalyticRadius


def test_analytic_radius_happy_path_is_center_minus_plus_radius():
    spread = AnalyticRadius()
    centers = np.array([10.0, 20.0, 30.0])
    radii = np.array([1.0, 2.5, 4.0])
    issue = np.array([True, True, True])

    lower, upper = spread.to_interval(centers, radii, issue)

    # Exact equality: this is the byte-identity anchor. A mutated adapter
    # (e.g. ``centers + radii + 1.0`` or a sign flip) must fail here.
    np.testing.assert_array_equal(lower, centers - radii)
    np.testing.assert_array_equal(upper, centers + radii)


def test_analytic_radius_not_issued_rows_are_nan():
    spread = AnalyticRadius()
    centers = np.array([10.0, 20.0, 30.0])
    radii = np.array([1.0, 2.0, 3.0])
    issue = np.array([True, False, True])

    lower, upper = spread.to_interval(centers, radii, issue)

    assert lower[0] == 9.0
    assert upper[0] == 11.0
    assert np.isnan(lower[1])
    assert np.isnan(upper[1])
    assert lower[2] == 27.0
    assert upper[2] == 33.0


def test_analytic_radius_non_finite_radius_propagates_to_nan():
    spread = AnalyticRadius()
    centers = np.array([10.0, 20.0])
    radii = np.array([np.inf, np.nan])
    # The runtime gates ``issue`` on ``np.isfinite(radii)`` upstream, so a
    # non-finite radius arrives as ``issue=False`` and yields NaN bounds.
    issue = np.array([False, False])

    lower, upper = spread.to_interval(centers, radii, issue)

    assert np.isnan(lower).all()
    assert np.isnan(upper).all()


def test_analytic_radius_length_one_arrays_match_cumulative_shape():
    spread = AnalyticRadius()
    centers = np.array([42.0])
    radii = np.array([2.5])
    issue = np.array([True])

    lower, upper = spread.to_interval(centers, radii, issue)

    np.testing.assert_array_equal(lower, np.array([39.5]))
    np.testing.assert_array_equal(upper, np.array([44.5]))


def test_analytic_radius_is_stateless():
    spread = AnalyticRadius()
    centers = np.array([5.0])
    radii = np.array([1.0])
    issue = np.array([True])

    first = spread.to_interval(centers, radii, issue)
    second = spread.to_interval(centers, radii, issue)

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
