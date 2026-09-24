"""Spherical geodesy helpers used by the feature extractor and the planner.

球面几何工具：大圆距离、方位角、球面插值。
All functions are vectorised with NumPy where it matters for performance.

Role in the pipeline
--------------------
Every distance in the application is a great-circle distance on a sphere of
radius :data:`~maritime_route.config.EARTH_RADIUS_KM` (6371.0088 km, the IUGG
mean Earth radius). The ellipsoidal error of this model is below 0.5 %, which
is far smaller than the size of one lattice cell. The module is used by

* :mod:`~maritime_route.routing.astar` / ``dijkstra`` / ``genetic`` - edge
  lengths of the lattice graph (through :func:`haversine_km`);
* :mod:`~maritime_route.routing.planner` - route length, the great-circle
  reference route, bearings of the waypoint table, and optional route
  simplification (:func:`douglas_peucker`);
* :mod:`~maritime_route.export.exporters` - the great-circle line written to
  GeoJSON.

Conventions
-----------
* A coordinate is a ``(latitude, longitude)`` tuple in decimal degrees
  (WGS 84 values treated as spherical). Latitude comes first everywhere in
  this module; only the GeoJSON exporter swaps the order to lon/lat.
* Distances are in kilometres, bearings in degrees clockwise from true north.

约定：坐标为 (纬度, 经度)，单位为度；距离单位为千米；方位角以正北为 0 度顺时针计。
"""
from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from ..config import EARTH_RADIUS_KM

Coordinate = Tuple[float, float]  # (latitude, longitude) in degrees


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres. Accepts scalars or arrays.

    Uses the haversine formula, which is numerically stable for the short
    distances (a few kilometres between neighbouring lattice cells) that
    dominate the search::

        a = sin^2(dlat / 2) + cos(lat1) * cos(lat2) * sin^2(dlon / 2)
        d = 2 * R * asin(sqrt(a))

    The law-of-cosines form ``acos(...)`` loses precision for such small
    angles, which is why it is not used here.
    半正矢公式：对小距离数值稳定，适合相邻网格单元之间的距离计算。

    Parameters
    ----------
    lat1, lon1 : float or array_like
        Latitude and longitude of the first point(s), degrees.
    lat2, lon2 : float or array_like
        Latitude and longitude of the second point(s), degrees. Arrays are
        broadcast against the first point(s) by NumPy rules.

    Returns
    -------
    float or numpy.ndarray
        Distance in kilometres; a NumPy scalar for scalar input, an array of
        the broadcast shape otherwise.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    # Rounding can push ``a`` marginally outside [0, 1] for identical or
    # antipodal points; clipping keeps sqrt/arcsin defined.
    # 浮点误差可能使 a 略超出 [0, 1]，裁剪后再开方与求反正弦。
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2, degrees [0, 360).

    The initial bearing (forward azimuth) is the course a ship must steer at
    point 1 to follow the great circle to point 2. On a great circle the
    course changes continuously, so this is the heading only at the start of
    the segment. Formula::

        theta = atan2(sin(dlon) * cos(lat2),
                      cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon))

    初始方位角：从点 1 沿大圆驶向点 2 时在点 1 处的航向。

    Parameters
    ----------
    lat1, lon1 : float
        Departure point, degrees.
    lat2, lon2 : float
        Target point, degrees.

    Returns
    -------
    float
        Bearing in degrees clockwise from true north, normalised to [0, 360).
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    # atan2 returns (-180, 180]; "+360 then mod 360" maps it to [0, 360).
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def to_unit_vector(lat, lon):
    """Convert degrees to 3-D unit vectors on the sphere (n x 3).

    Earth-centred Cartesian coordinates of a unit sphere:
    ``x = cos(lat) cos(lon)``, ``y = cos(lat) sin(lon)``, ``z = sin(lat)``.
    Working with vectors avoids the singularities of latitude/longitude at
    the poles and at the antimeridian during interpolation.
    经纬度转换为单位球面上的三维向量（地心直角坐标）。

    Parameters
    ----------
    lat, lon : float or array_like
        Latitude and longitude in degrees.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(..., 3)``; shape ``(3,)`` for scalar input.
    """
    p, l = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(p) * np.cos(l), np.cos(p) * np.sin(l), np.sin(p)], axis=-1)


def great_circle_points(start: Coordinate, end: Coordinate, n: int = 64) -> List[Coordinate]:
    """Sample ``n`` points along the great circle using spherical interpolation.

    球面线性插值(slerp)，用于生成"最短距离"参考航线。

    Algorithm (slerp): with unit vectors ``v1``, ``v2`` of the end points and
    the central angle ``omega = acos(v1 . v2)``, the point at fraction ``t``
    of the arc is::

        v(t) = sin((1 - t) * omega) / sin(omega) * v1 + sin(t * omega) / sin(omega) * v2

    The ``n`` samples are equally spaced in arc length (``t = i / (n - 1)``)
    and include both end points. The result is converted back to
    latitude/longitude with ``lat = asin(z)``, ``lon = atan2(y, x)``.

    Parameters
    ----------
    start, end : Coordinate
        End points ``(lat, lon)`` in degrees.
    n : int, default 64
        Number of samples including both ends; values below 2 are raised to 2.

    Returns
    -------
    list of Coordinate
        ``n`` points from ``start`` to ``end``. If the two points coincide
        (``omega`` ~ 0) only ``[start, end]`` is returned. Longitudes are in
        (-180, 180], so a line crossing the antimeridian jumps from +180 to
        -180 between two samples.

    Notes
    -----
    For exactly antipodal points the great circle is not unique
    (``sin(omega) = 0``); this does not occur for the regional voyages the
    application plans.
    """
    n = max(2, int(n))
    v1 = to_unit_vector(start[0], start[1])
    v2 = to_unit_vector(end[0], end[1])
    # Clip the dot product so that rounding cannot make acos undefined.
    dot = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    omega = math.acos(dot)                  # central angle between the points, radians
    if omega < 1e-12:
        # Coincident points: interpolation would divide by sin(0) = 0.
        # 起终点重合：避免除以零。
        return [start, end]
    sin_omega = math.sin(omega)
    out: List[Coordinate] = []
    for i in range(n):
        t = i / (n - 1)                     # fraction of the arc, 0 .. 1
        a = math.sin((1.0 - t) * omega) / sin_omega
        b = math.sin(t * omega) / sin_omega
        v = a * v1 + b * v2
        # Slerp keeps |v| = 1 analytically; normalising removes rounding drift.
        v /= np.linalg.norm(v)
        lat = math.degrees(math.asin(float(v[2])))
        lon = math.degrees(math.atan2(float(v[1]), float(v[0])))
        out.append((lat, lon))
    return out


def path_length_km(path: Sequence[Coordinate]) -> float:
    """Total great-circle length of a poly-line given as (lat, lon) pairs.

    Each segment between consecutive vertices is treated as a great-circle
    arc; the lengths are summed in one vectorised :func:`haversine_km` call.

    Parameters
    ----------
    path : sequence of Coordinate
        Vertices ``(lat, lon)`` in degrees.

    Returns
    -------
    float
        Length in kilometres; ``0.0`` for fewer than two vertices.
    """
    if len(path) < 2:
        return 0.0
    arr = np.asarray(path, dtype=float)
    # arr[:-1] are the segment starts and arr[1:] the segment ends.
    return float(np.sum(haversine_km(arr[:-1, 0], arr[:-1, 1], arr[1:, 0], arr[1:, 1])))


def densify(path: Sequence[Coordinate], step_km: float = 25.0) -> List[Coordinate]:
    """Insert intermediate points so that no segment is longer than ``step_km``.

    Every segment is replaced by points sampled on its great circle
    (:func:`great_circle_points`), so the densified line follows the sphere
    rather than a straight line in the latitude/longitude plane. This matters
    when a route is drawn on a Mercator map or sampled against the lattice.
    航线加密：按大圆插值插入中间点，使每段长度不超过 step_km。

    Parameters
    ----------
    path : sequence of Coordinate
        Poly-line vertices ``(lat, lon)`` in degrees.
    step_km : float, default 25.0
        Maximum segment length after densification, kilometres.

    Returns
    -------
    list of Coordinate
        Densified poly-line; the original vertices are kept, and a path with
        fewer than two vertices is returned unchanged (as a list).
    """
    if len(path) < 2:
        return list(path)
    dense: List[Coordinate] = [tuple(path[0])]
    for a, b in zip(path[:-1], path[1:]):
        seg = float(haversine_km(a[0], a[1], b[0], b[1]))
        # Number of samples including both ends; the max(..., 1e-6) guards
        # against division by zero for a non-positive step.
        n = max(2, int(seg / max(step_km, 1e-6)) + 1)
        # [1:] skips the first sample, which equals the previous segment's end.
        for p in great_circle_points(tuple(a), tuple(b), n)[1:]:
            dense.append(p)
    return dense


def douglas_peucker(path: Sequence[Coordinate], tolerance_km: float = 3.0) -> List[Coordinate]:
    """Simplify a poly-line, keeping its shape within ``tolerance_km``.

    道格拉斯-普克算法，用于压缩网格搜索产生的锯齿状路径。

    Algorithm (Ramer-Douglas-Peucker): keep the two end points; find the
    interior vertex farthest from the chord between them; if that distance
    exceeds the tolerance, keep the vertex and repeat on the two halves,
    otherwise drop every interior vertex of the span. The recursion is
    implemented with an explicit stack so that long lattice paths (thousands
    of cells) cannot hit Python's recursion limit. Complexity is O(n log n)
    on average and O(n^2) in the worst case.

    Parameters
    ----------
    path : sequence of Coordinate
        Poly-line ``(lat, lon)`` in degrees, typically the staircase path of
        the lattice search.
    tolerance_km : float, default 3.0
        Maximum allowed perpendicular deviation of a removed vertex from the
        simplified line, kilometres.

    Returns
    -------
    list of Coordinate
        Subset of the input vertices in their original order; the first and
        the last vertex are always kept.

    Notes
    -----
    The simplification checks only geometry, not the cost map: a shortcut
    within ``tolerance_km`` may cut a corner of a land cell. The planner
    therefore applies it only when the user asks for it (``simplify_km > 0``).
    """
    pts = [tuple(p) for p in path]
    if len(pts) < 3:
        return pts

    def perpendicular_km(pt: Coordinate, a: Coordinate, b: Coordinate) -> float:
        """Distance from ``pt`` to the segment ``a``-``b`` in kilometres.

        The three points are projected onto a local plane (equirectangular
        projection around the mean latitude of the segment): one degree of
        longitude is ``111.320 * cos(lat0)`` km and one degree of latitude
        110.574 km (WGS 84 values at the equator). The point is then projected
        onto the segment with the parameter ``t`` clamped to [0, 1], so the
        distance is to the *segment*, not to the infinite line.
        局部平面近似下点到线段的距离（千米）。

        Parameters
        ----------
        pt : Coordinate
            The vertex being tested.
        a, b : Coordinate
            End points of the chord.

        Returns
        -------
        float
            Distance in kilometres.
        """
        # Local planar approximation is accurate enough at the tolerances used.
        lat0 = math.radians((a[0] + b[0]) / 2.0)
        kx = 111.320 * math.cos(lat0)       # km per degree of longitude at lat0
        ky = 110.574                        # km per degree of latitude
        # Vectors a->pt and a->b in kilometres (x = east, y = north).
        px, py = (pt[1] - a[1]) * kx, (pt[0] - a[0]) * ky
        bx, by = (b[1] - a[1]) * kx, (b[0] - a[0]) * ky
        den = bx * bx + by * by
        if den < 1e-12:
            # Degenerate chord (a == b): distance to the point a.
            return math.hypot(px, py)
        # Projection parameter of pt on a->b, clamped to the segment.
        t = max(0.0, min(1.0, (px * bx + py * by) / den))
        return math.hypot(px - t * bx, py - t * by)

    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    # Stack of index spans (i, j) still to be examined. 待处理的区间栈。
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue                        # no interior vertex in this span
        best, best_d = -1, 0.0
        for k in range(i + 1, j):
            d = perpendicular_km(pts[k], pts[i], pts[j])
            if d > best_d:
                best, best_d = k, d
        if best_d > tolerance_km and best > 0:
            # The farthest vertex is significant: keep it and split the span.
            keep[best] = True
            stack.append((i, best))
            stack.append((best, j))
        # Otherwise every interior vertex of (i, j) stays keep=False (dropped).
    return [p for p, k in zip(pts, keep) if k]


def bounding_box(points: Iterable[Coordinate], margin_deg: float = 0.0):
    """Axis aligned bounding box (lat_min, lat_max, lon_min, lon_max).

    The box is enlarged by ``margin_deg`` on every side and clamped to
    latitude [-89.5, 89.5] (away from the poles, where a longitude lattice
    degenerates) and longitude [-180, 180]. Boxes crossing the antimeridian
    are not supported: the clamp simply cuts them at +/-180.
    外包矩形：四周扩展 margin_deg 度，并裁剪到合法经纬度范围。

    Parameters
    ----------
    points : iterable of Coordinate
        Points ``(lat, lon)`` in degrees; must contain at least one point.
    margin_deg : float, default 0.0
        Margin added on each side, degrees.

    Returns
    -------
    tuple of float
        ``(lat_min, lat_max, lon_min, lon_max)`` in degrees.

    Raises
    ------
    IndexError, ValueError
        If ``points`` is empty (NumPy cannot index or reduce an empty array).
    """
    arr = np.asarray(list(points), dtype=float)
    lat_min = float(arr[:, 0].min()) - margin_deg
    lat_max = float(arr[:, 0].max()) + margin_deg
    lon_min = float(arr[:, 1].min()) - margin_deg
    lon_max = float(arr[:, 1].max()) + margin_deg
    return (max(-89.5, lat_min), min(89.5, lat_max), max(-180.0, lon_min), min(180.0, lon_max))
