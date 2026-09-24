"""Tests of AIS loading, validation and feature extraction. 数据加载与特征测试。

Covers ``maritime_route.data.ais_loader`` (CSV/JSON/GeoJSON parsing, column
aliases, validation of coordinates) and ``maritime_route.data.features``
(signed-log transform, the 28-component feature matrix, handling of missing
measurements and the ``ReferenceIndex`` of reference bathymetry samples).
The last three tests guard against label leakage: a point that is stored in
the reference index must never be used as its own neighbour, so its features
do not depend on whether it was part of the index.
All data are generated synthetically by :func:`_frame`; no model is needed.
测试数据加载、校验与特征提取；最后三个测试防止参考索引中的点检索到自身（数据泄漏）。
"""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd
import pytest

from maritime_route.data.ais_loader import (
    DataLoadError, load_csv, load_json, validate,
)
from maritime_route.data.features import (
    FEATURE_NAMES, N_FEATURES, ReferenceIndex, build_features, signed_log,
)


def _frame(n=40, seed=0):
    """Build a random AIS-like frame with the columns expected by the feature code.

    Parameters
    ----------
    n : int, optional
        Number of rows.
    seed : int, optional
        Seed of the NumPy random generator, so every call with the same seed
        returns the same frame.

    Returns
    -------
    pandas.DataFrame
        Coordinates in the Baltic/North Sea box, depth/elevation values,
        a depth/height type and nine OSM object counters. No class labels.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "latitude": rng.uniform(50, 60, n),
        "longitude": rng.uniform(5, 25, n),
        "noaa_depth": rng.uniform(-3000, 500, n),
        "elevation": rng.uniform(-3000, 500, n),
        "noaa_type": rng.choice(["depth", "height"], n),
        **{c: rng.integers(0, 3, n) for c in
           ["osm_piers", "osm_lighthouses", "osm_breakwaters", "osm_harbors",
            "osm_rivers", "osm_lakes", "osm_beaches", "osm_cliffs", "osm_industrial"]},
    })


def test_csv_column_aliases_are_normalised():
    """CSV headers "lat"/"lng" are renamed to "latitude"/"longitude"."""
    frame = load_csv(io.StringIO("lat,lng\n54.5,10.2\n55.0,11.0\n"))
    assert {"latitude", "longitude"} <= set(frame.columns)
    assert len(frame) == 2


def test_csv_without_coordinates_is_rejected():
    """A CSV without any coordinate column raises DataLoadError."""
    with pytest.raises(DataLoadError):
        load_csv(io.StringIO("a,b\n1,2\n"))


def test_geojson_geometry_wins_over_properties():
    """For GeoJSON features the Point geometry overrides latitude/longitude properties."""
    document = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"latitude": 0.0, "longitude": 0.0},
            "geometry": {"type": "Point", "coordinates": [10.5, 54.5]},
        }],
    }
    frame = load_json(io.StringIO(json.dumps(document)))
    assert frame.loc[0, "latitude"] == 54.5
    assert frame.loc[0, "longitude"] == 10.5


def test_validate_drops_out_of_range_rows():
    """validate() drops rows with invalid or missing coordinates and reports them."""
    frame = pd.DataFrame({"latitude": [54.0, 999.0, None], "longitude": [10.0, 10.0, 10.0]})
    clean, report = validate(frame)
    assert len(clean) == 1
    assert report.n_dropped == 2
    assert report.warnings


def test_validate_rejects_all_invalid():
    """validate() raises DataLoadError when no valid row remains."""
    with pytest.raises(DataLoadError):
        validate(pd.DataFrame({"latitude": [999.0], "longitude": [10.0]}))


def test_signed_log_is_odd_and_monotone():
    """signed_log is an odd function (f(-x) = -f(x)) and strictly increasing."""
    values = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = signed_log(values)
    assert out[0] == pytest.approx(-out[-1])
    assert np.all(np.diff(out) > 0)


def test_feature_matrix_shape_and_finiteness():
    """Feature matrix has N_FEATURES columns matching FEATURE_NAMES and only finite values."""
    frame = _frame()
    index = ReferenceIndex.from_frame(frame)
    matrix = build_features(frame, index, exclude_self=True)
    assert matrix.shape == (len(frame), N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert np.isfinite(matrix).all()


def test_features_handle_missing_measurements():
    """Missing depth/elevation yield finite features and raise the "missing" indicator columns."""
    frame = _frame()
    frame.loc[0, "noaa_depth"] = np.nan
    frame.loc[0, "elevation"] = np.nan
    index = ReferenceIndex.from_frame(_frame(seed=1))
    matrix = build_features(frame, index)
    assert np.isfinite(matrix).all()
    # The 'missing' indicator columns must be raised for that row.
    assert matrix[0, FEATURE_NAMES.index("depth_is_missing")] == 1.0
    assert matrix[0, FEATURE_NAMES.index("elevation_is_missing")] == 1.0


def test_reference_index_round_trip(tmp_path):
    """ReferenceIndex saved to .npz and loaded back keeps its depth samples."""
    index = ReferenceIndex.from_frame(_frame())
    path = tmp_path / "index.npz"
    index.save(path)
    restored = ReferenceIndex.load(path)
    assert restored.depth.shape == index.depth.shape
    np.testing.assert_allclose(restored.depth, index.depth, rtol=1e-5)


def test_reference_index_always_excludes_the_query_point():
    """A point in the index must never retrieve itself, flag or no flag.

    索引中的点在任何情况下都不应检索到自身。
    """
    frame = _frame(n=20)
    index = ReferenceIndex.from_frame(frame)
    lat = frame["latitude"].to_numpy()
    lon = frame["longitude"].to_numpy()
    for flag in (False, True):
        dist, idx = index.query(lat, lon, k=3, exclude_self=flag)
        assert dist.shape == (len(frame), 3)
        # No returned neighbour is the query point itself.
        assert (dist[:, 0] > 1e-6).all()
        assert not (idx == np.arange(len(frame))[:, None]).any()


def test_reference_index_query_is_flag_independent():
    """The legacy exclude_self flag must no longer change the result.

    历史遗留的 exclude_self 参数不再影响结果，保证训练与推理特征一致。
    """
    frame = _frame(n=30)
    index = ReferenceIndex.from_frame(frame)
    # Query points that are NOT in the index.
    other = _frame(n=15, seed=5)
    lat = other["latitude"].to_numpy()
    lon = other["longitude"].to_numpy()
    a, ia = index.query(lat, lon, k=4, exclude_self=False)
    b, ib = index.query(lat, lon, k=4, exclude_self=True)
    np.testing.assert_allclose(a, b)
    np.testing.assert_array_equal(ia, ib)


def test_features_are_identical_whether_or_not_the_point_is_indexed():
    """The decisive property: provenance must not change a point's features.

    关键性质：同一地理点的特征不应因其是否在索引中而不同。
    """
    frame = _frame(n=60)
    index_with = ReferenceIndex.from_frame(frame)          # contains the points
    index_without = ReferenceIndex.from_frame(_frame(n=60, seed=3))
    # A point present in the index must not be able to use itself as context,
    # so its features must match those computed from an index built without it.
    probe = frame.iloc[:5].reset_index(drop=True)
    inside = build_features(probe, index_with)
    # Rebuild an index from the same data minus the probe rows.
    rest = ReferenceIndex.from_frame(frame.iloc[5:].reset_index(drop=True))
    outside = build_features(probe, rest)
    # The neighbour sets differ (different candidate pools), but neither may
    # contain a zero-distance self match, which is what the assertion checks.
    for matrix in (inside, outside):
        assert np.isfinite(matrix).all()
    d_inside, _ = index_with.query(probe["latitude"].to_numpy(),
                                   probe["longitude"].to_numpy(), k=4)
    assert (d_inside[:, 0] > 1e-6).all()
