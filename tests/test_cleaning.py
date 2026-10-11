"""Tests of the data-cleaning rule and of the sea-region assignment.

清洗规则与海域划分的测试。
"""
import numpy as np
import pandas as pd

from maritime_route.data.cleaning import (
    OTHER_REGION,
    assign_regions,
    conflict_mask,
    measurement_conflicts,
    remove_label_conflicts,
)


def _frame(rows):
    """Small corpus from (class_code, noaa_depth, elevation) tuples."""
    return pd.DataFrame(
        [{"latitude": 55.0, "longitude": 15.0, "class_code": c,
          "noaa_depth": d, "elevation": e} for c, d, e in rows]
    )


def test_sea_point_above_sea_level_is_removed():
    frame = _frame([("COASTAL_SEA", 12.0, np.nan), ("OPEN_SEA", -300.0, np.nan)])
    assert conflict_mask(frame).tolist() == [True, False]


def test_land_point_below_sea_level_is_removed():
    frame = _frame([("COASTLINE", -40.0, np.nan), ("NEAR_COAST", 3.0, np.nan)])
    assert conflict_mask(frame).tolist() == [True, False]


def test_tolerance_keeps_small_deviations():
    # +4 m on a sea point and -4 m on a land point are within the 5 m tolerance.
    frame = _frame([("COASTAL_SEA", 4.0, 4.0), ("NEAR_COAST", -4.0, -4.0)])
    assert not conflict_mask(frame).any()
    assert conflict_mask(frame, tolerance_m=0.0).all()


def test_elevation_alone_can_reveal_a_conflict():
    # The relief model agrees with the class, the collector's elevation does not.
    frame = _frame([("OPEN_SEA", -20.0, 35.0), ("COASTLINE", 80.0, -60.0)])
    assert not measurement_conflicts(frame, "noaa_depth").any()
    assert measurement_conflicts(frame, "elevation").all()
    assert conflict_mask(frame).all()


def test_missing_measurements_are_never_flagged():
    frame = _frame([("COASTAL_SEA", np.nan, np.nan), ("COASTLINE", np.nan, np.nan)])
    assert not conflict_mask(frame).any()
    # A frame without an elevation column is handled as "not measured".
    assert not measurement_conflicts(frame.drop(columns="elevation"), "elevation").any()


def test_report_counts_by_measurement():
    frame = _frame([
        ("OPEN_SEA", 10.0, np.nan),     # depth only
        ("OPEN_SEA", -10.0, 10.0),      # elevation only
        ("COASTLINE", -10.0, -10.0),    # both
        ("COASTLINE", 10.0, 10.0),      # consistent
    ])
    cleaned, report = remove_label_conflicts(frame)
    assert len(cleaned) == 1 and report.n_removed == 3
    assert report.removed_by_measure == {"noaa_depth_only": 1, "elevation_only": 1, "both": 1}
    assert report.removed_by_class["OPEN_SEA"] == 2
    assert report.to_dict()["n_after"] == 1


def test_aegean_points_get_their_own_region():
    lat = np.array([37.9, 40.5, 43.0, 54.0, 0.0])
    lon = np.array([23.6, 22.9, 34.0, 18.0, 0.0])
    regions = assign_regions(lat, lon).tolist()
    assert regions[0] == "Aegean Sea"            # Piraeus
    assert regions[1] == "Aegean Sea"            # Thessaloniki
    assert regions[2] == "Black Sea and Eastern Mediterranean"
    assert regions[3] == "North Europe (Baltic and North Sea)"
    assert regions[4] == OTHER_REGION
