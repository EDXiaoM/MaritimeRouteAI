"""Tests of the spherical geometry helpers. 球面几何测试。

Covers ``maritime_route.routing.geodesy``: great-circle (haversine) distance,
initial bearing, great-circle sampling, densification, Douglas-Peucker
simplification and bounding boxes. These functions are used by the cost map,
the planner (leg lengths, bearings, baseline comparison) and the exporters,
so their correctness is checked against known values and geometric
invariants. No trained model is needed.
测试球面距离、方位角、大圆采样、加密、简化与外包框；不依赖模型。
"""
from __future__ import annotations

import math

import pytest

from maritime_route.routing.geodesy import (
    bounding_box, densify, douglas_peucker, great_circle_points,
    haversine_km, initial_bearing_deg, path_length_km,
)


def test_haversine_known_distance():
    """Haversine distance Kiel -> Tallinn matches an external reference (1045-1060 km)."""
    # Kiel -> Tallinn, reference value from an independent great-circle calculator.
    d = float(haversine_km(54.3233, 10.1228, 59.4370, 24.7536))
    assert 1045 < d < 1060


def test_haversine_zero_and_antipodal():
    """Distance is 0 for identical points and half the Earth circumference for antipodes on the equator."""
    assert float(haversine_km(10, 20, 10, 20)) == pytest.approx(0.0, abs=1e-9)
    half_circumference = math.pi * 6371.0088
    assert float(haversine_km(0, 0, 0, 180)) == pytest.approx(half_circumference, rel=1e-6)


def test_bearing_cardinal_directions():
    """Initial bearing is 0, 90 and 180 degrees for due north, east and south."""
    assert initial_bearing_deg(0, 0, 10, 0) == pytest.approx(0.0, abs=1e-6)
    assert initial_bearing_deg(0, 0, 0, 10) == pytest.approx(90.0, abs=1e-6)
    assert initial_bearing_deg(0, 0, -10, 0) == pytest.approx(180.0, abs=1e-6)


def test_great_circle_points_endpoints_and_length():
    """Sampled great circle has the requested count, exact end points and the direct length."""
    start, end = (54.0, 10.0), (59.0, 25.0)
    line = great_circle_points(start, end, 50)
    assert len(line) == 50
    assert line[0] == pytest.approx(start, abs=1e-6)
    assert line[-1] == pytest.approx(end, abs=1e-6)
    # A sampled great circle is the shortest path: its length equals the direct one.
    assert path_length_km(line) == pytest.approx(
        float(haversine_km(*start, *end)), rel=1e-3)


def test_densify_limits_segment_length():
    """Densified poly-line has no segment much longer than the requested step (50 km, 10 % tolerance)."""
    dense = densify([(54.0, 10.0), (59.0, 25.0)], step_km=50.0)
    segments = [float(haversine_km(a[0], a[1], b[0], b[1]))
                for a, b in zip(dense[:-1], dense[1:])]
    assert max(segments) <= 55.0


def test_douglas_peucker_preserves_endpoints_and_reduces():
    """Douglas-Peucker keeps both end points and removes vertices."""
    line = great_circle_points((50.0, 0.0), (55.0, 20.0), 200)
    simple = douglas_peucker(line, tolerance_km=5.0)
    assert simple[0] == line[0] and simple[-1] == line[-1]
    assert len(simple) < len(line)


def test_bounding_box_with_margin():
    """Bounding box of two points is widened by the margin on every side."""
    lat_min, lat_max, lon_min, lon_max = bounding_box([(10, 20), (30, 40)], margin_deg=2)
    assert (lat_min, lat_max, lon_min, lon_max) == (8, 32, 18, 42)
