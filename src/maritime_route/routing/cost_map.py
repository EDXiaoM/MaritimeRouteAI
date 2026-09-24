"""Construction of the navigation cost map from neural-network classification.

通行代价地图构建模块。
The planner works on a regular geographic lattice. Every cell centre is
classified by the neural network; the predicted zone is converted into a
traversal cost, and cells adjacent to land receive an additional safety
penalty so that the optimiser keeps a clearance from the shoreline.

Role in the pipeline
--------------------
This module is the bridge between the classifier and the graph searches::

    GridSpec (make_grid_spec)            - where the lattice lies and how fine it is
      -> classifier.predict_coordinates  - zone probabilities per lattice node
      -> CostMap (build_cost_map)        - class, confidence, traversal cost per node
      -> astar / dijkstra / genetic      - search on the 8-connected lattice graph

Lattice
-------
The lattice is a regular grid in latitude/longitude with the same step
``resolution_deg`` in both directions. Node ``(row, col)`` lies at::

    lat = lat_min + row * resolution_deg
    lon = lon_min + col * resolution_deg

so ``row`` grows northwards and ``col`` eastwards, and the first and the last
row/column lie on the box boundary. A node is called a "cell" throughout the
code; it stands for the area of one resolution step around it. In kilometres a
cell is ``111.2 * res`` tall and ``111.2 * res * cos(lat)`` wide, so cells
become narrower towards the poles; the searches account for this through
latitude-dependent step lengths (see ``astar._cell_distance_table``).

Cost model
----------
Each zone class has a weight ``w`` from :data:`~maritime_route.config.ZONE_COST`
(open sea 1.0, coastal sea 1.8, near-coast and coastline infinite). Infinite
weights are stored as the finite sentinel
:data:`~maritime_route.config.IMPASSABLE_COST` (1e6) so that the array stays a
plain ``float32`` array; every cell with ``cost >= IMPASSABLE_COST`` is land.
Water cells close to land are multiplied by ``proximity_penalty`` (the safety
margin). The searches then charge ``d(u, v) * (w(u) + w(v)) / 2`` for a move
between neighbouring cells ``u`` and ``v``.

网格节点 (row, col) 对应纬度 lat_min + row*res、经度 lon_min + col*res；
代价 >= IMPASSABLE_COST 的单元为陆地（不可通行）。
"""
from __future__ import annotations

import math

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    CLASS_NAMES,
    IMPASSABLE_COST,
    INDEX_TO_CLASS,
    NAVIGABLE_CLASSES,
    ROUTING,
    ZONE_COST,
    RoutingConfig,
)

LOGGER = logging.getLogger(__name__)

#: Optional callback ``progress(stage, fraction, details)`` used to report the
#: build progress to the web client (``fraction`` in [0, 1]).
#: 进度回调：progress(阶段名, 完成比例, 详情字典)。
ProgressCallback = Optional[Callable[[str, float, Dict], None]]


@dataclass
class GridSpec:
    """Geographic lattice definition. 规则经纬网格定义。

    A regular latitude/longitude lattice covering the box
    ``[lat_min, lat_max] x [lon_min, lon_max]`` with step ``resolution_deg``.
    The class only stores the five numbers; the shape and all coordinate
    conversions are derived from them.

    Attributes
    ----------
    lat_min, lat_max : float
        Southern and northern edge of the box, degrees.
    lon_min, lon_max : float
        Western and eastern edge of the box, degrees. The box must not cross
        the antimeridian (``lon_min < lon_max``).
    resolution_deg : float
        Distance between neighbouring nodes in both latitude and longitude,
        degrees (0.25 deg ~ 27.8 km in latitude).
    """

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    resolution_deg: float

    @property
    def n_rows(self) -> int:
        """Number of lattice rows (latitude direction).

        ``round(span / resolution) + 1`` because both boundaries are nodes
        (fence-post count); at least 2 so that the lattice is never degenerate.

        Returns
        -------
        int
            Row count, >= 2.
        """
        return max(2, int(round((self.lat_max - self.lat_min) / self.resolution_deg)) + 1)

    @property
    def n_cols(self) -> int:
        """Number of lattice columns (longitude direction), >= 2.

        Computed like :attr:`n_rows` from the longitude span.

        Returns
        -------
        int
            Column count, >= 2.
        """
        return max(2, int(round((self.lon_max - self.lon_min) / self.resolution_deg)) + 1)

    @property
    def n_cells(self) -> int:
        """Total number of lattice nodes, ``n_rows * n_cols``.

        This is the quantity limited by ``RoutingConfig.max_grid_cells``,
        because it determines both classifier time and search memory.

        Returns
        -------
        int
            Number of cells.
        """
        return self.n_rows * self.n_cols

    def cell_to_coord(self, row: int, col: int) -> Tuple[float, float]:
        """Geographic coordinate of a lattice node.

        Parameters
        ----------
        row, col : int
            Lattice indices; not range-checked.

        Returns
        -------
        tuple of float
            ``(lat, lon)`` in degrees: ``lat_min + row * res``,
            ``lon_min + col * res``.
        """
        return (self.lat_min + row * self.resolution_deg,
                self.lon_min + col * self.resolution_deg)

    def coord_to_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Nearest lattice node of a geographic coordinate.

        Inverse of :meth:`cell_to_coord`: the offset from the south-west
        corner is divided by the step and rounded to the nearest integer, so
        a point is assigned to the node within half a step of it. Points
        outside the box are clamped onto the border row/column instead of
        raising an error, which lets callers sample lines that leave the box.
        坐标转网格：四舍五入到最近节点，并裁剪到网格范围内。

        Parameters
        ----------
        lat, lon : float
            Coordinate in degrees.

        Returns
        -------
        tuple of int
            ``(row, col)`` with ``0 <= row < n_rows`` and ``0 <= col < n_cols``.
        """
        row = int(round((lat - self.lat_min) / self.resolution_deg))
        col = int(round((lon - self.lon_min) / self.resolution_deg))
        return (min(max(row, 0), self.n_rows - 1), min(max(col, 0), self.n_cols - 1))

    def mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Latitude and longitude of every node as two 2-D arrays.

        ``indexing="ij"`` makes the first axis the row (latitude) and the
        second the column (longitude), matching ``class_index[row, col]``.

        Returns
        -------
        tuple of numpy.ndarray
            ``(grid_lat, grid_lon)``, each of shape ``(n_rows, n_cols)``,
            degrees.
        """
        lats = self.lat_min + np.arange(self.n_rows) * self.resolution_deg
        lons = self.lon_min + np.arange(self.n_cols) * self.resolution_deg
        grid_lat, grid_lon = np.meshgrid(lats, lons, indexing="ij")
        return grid_lat, grid_lon

    def to_dict(self) -> Dict:
        """JSON-serialisable description of the lattice.

        Used in API responses, in :class:`~maritime_route.routing.planner.RoutePlan`
        and in the benchmark files. Box edges are rounded to 5 decimals
        (about 1 m).

        Returns
        -------
        dict
            Box edges, resolution and the derived row, column and cell counts.
        """
        return {
            "lat_min": round(self.lat_min, 5), "lat_max": round(self.lat_max, 5),
            "lon_min": round(self.lon_min, 5), "lon_max": round(self.lon_max, 5),
            "resolution_deg": self.resolution_deg,
            "n_rows": self.n_rows, "n_cols": self.n_cols, "n_cells": self.n_cells,
        }


@dataclass
class CostMap:
    """Classified lattice plus the derived traversal cost. 代价地图。

    All three arrays have shape ``(spec.n_rows, spec.n_cols)`` and are indexed
    ``[row, col]`` exactly like the lattice.

    Attributes
    ----------
    spec : GridSpec
        Lattice geometry.
    class_index : numpy.ndarray of int8
        Predicted zone of every node as an index into
        :data:`~maritime_route.config.CLASS_NAMES`
        (0 OPEN_SEA, 1 COASTAL_SEA, 2 NEAR_COAST, 3 COASTLINE).
    confidence : numpy.ndarray of float32
        Probability of the predicted class (max of the softmax output), 0..1.
    cost : numpy.ndarray of float32
        Traversal weight ``w`` of every node after the safety margin;
        ``IMPASSABLE_COST`` marks land.
    build_seconds : float
        Wall-clock time spent building the map (classification included).
    """

    spec: GridSpec
    class_index: np.ndarray   # (rows, cols) int8
    confidence: np.ndarray    # (rows, cols) float32
    cost: np.ndarray          # (rows, cols) float32, IMPASSABLE_COST on land
    build_seconds: float = 0.0

    @property
    def passable(self) -> np.ndarray:
        """Boolean mask of navigable cells.

        A cell is navigable when its cost is below the land sentinel. Water
        cells penalised by the safety margin stay navigable because their
        cost is capped at ``IMPASSABLE_COST - 1``.

        Returns
        -------
        numpy.ndarray of bool
            Shape ``(n_rows, n_cols)``; recomputed on every access.
        """
        return self.cost < IMPASSABLE_COST

    def zone_of(self, row: int, col: int) -> str:
        """Zone class name of one cell.

        Parameters
        ----------
        row, col : int
            Lattice indices.

        Returns
        -------
        str
            One of :data:`~maritime_route.config.CLASS_NAMES`.
        """
        return INDEX_TO_CLASS[int(self.class_index[row, col])]

    def statistics(self) -> Dict:
        """Summary of the map for the UI, the reports and the benchmarks.

        Returns
        -------
        dict
            ``grid`` (see :meth:`GridSpec.to_dict`), total and navigable cell
            counts, the navigable share, the count and share of every zone,
            the mean classifier confidence and the build time in seconds.
        """
        total = int(self.class_index.size)
        counts = {
            name: int((self.class_index == i).sum()) for i, name in enumerate(CLASS_NAMES)
        }
        return {
            "grid": self.spec.to_dict(),
            "cells_total": total,
            "cells_navigable": int(self.passable.sum()),
            "navigable_share": round(float(self.passable.mean()), 4),
            "zone_counts": counts,
            # max(total, 1) avoids a division by zero for an empty map.
            "zone_shares": {k: round(v / max(total, 1), 4) for k, v in counts.items()},
            "mean_confidence": round(float(self.confidence.mean()), 4),
            "build_seconds": round(self.build_seconds, 3),
        }

    def to_overlay(self, max_cells: int = 20000) -> Dict:
        """Down-sampled cell list for rendering the classification on the map.

        为前端地图生成降采样的分类图层。返回的 ``cell_size_deg`` 已按降采样
        步长放大，使前端绘制的矩形彼此相接而不留缝隙。
        Each returned cell represents a ``step x step`` block, whose class is the
        majority vote of the block so that thin land features are not lost.

        The step is ``ceil(sqrt(n_cells / max_cells))``: taking every
        ``step``-th row and column divides the cell count by about
        ``step^2``, which brings it down to ``max_cells`` or fewer. A browser
        can draw about 20 000 rectangles without noticeable delay.

        Parameters
        ----------
        max_cells : int, default 20000
            Upper bound on the number of cells sent to the client.

        Returns
        -------
        dict
            ``cells`` - list of ``{"lat", "lon", "c", "p"}`` where ``lat``/``lon``
            is the centre of the block (degrees, 4 decimals), ``c`` the class
            index and ``p`` the mean confidence of the block;
            ``step`` - the down-sampling factor;
            ``cell_size_deg`` - side of one drawn block in degrees.
        """
        step = max(1, int(np.ceil(np.sqrt(self.spec.n_cells / max(max_cells, 1)))))
        cells: List[Dict] = []
        for row in range(0, self.spec.n_rows, step):
            for col in range(0, self.spec.n_cols, step):
                # Blocks on the northern/eastern border may be smaller than
                # step x step; NumPy slicing truncates them automatically.
                block = self.class_index[row:row + step, col:col + step]
                counts = np.bincount(block.ravel(), minlength=len(CLASS_NAMES))
                # Majority vote of the block. On a tie argmax returns the lowest
                # class index, i.e. the water class.
                # (An earlier note said "land dominates visually: keep a block
                # as land if any cell is land"; the code uses the majority.)
                # 块内多数投票；平票时取索引较小的类别。
                winner = int(counts.argmax())
                lat, lon = self.spec.cell_to_coord(row, col)
                cells.append({
                    # Shift from the block's first node to the block centre:
                    # the block spans (step - 1) steps, half of it is added.
                    "lat": round(lat + (step - 1) * self.spec.resolution_deg / 2.0, 4),
                    "lon": round(lon + (step - 1) * self.spec.resolution_deg / 2.0, 4),
                    "c": winner,
                    "p": round(float(self.confidence[row:row + step, col:col + step].mean()), 3),
                })
        return {
            "cells": cells,
            "step": step,
            "cell_size_deg": round(self.spec.resolution_deg * step, 5),
        }


def make_grid_spec(
    waypoints: List[Tuple[float, float]],
    resolution_deg: float = ROUTING.grid_resolution_deg,
    margin_deg: float = ROUTING.margin_deg,
    max_cells: int = ROUTING.max_grid_cells,
) -> GridSpec:
    """Bounding box around the waypoints, coarsened until it fits the budget.

    自动确定网格范围与分辨率：单元数超过上限时自动降低分辨率。

    The box is the bounding box of the waypoints (normally departure and
    destination) enlarged by ``margin_deg`` on each side, so that the route
    can leave the straight corridor between the ports to go round a
    peninsula. Latitude is clamped to +/-85 deg (the limit of the Web
    Mercator map in the client) and longitude to +/-180 deg.

    If the lattice would exceed ``max_cells`` nodes, the step is multiplied by
    1.5 repeatedly (each step reduces the cell count by about 2.25x) until
    it fits or the step reaches 4 deg. This bounds both classifier time and
    search memory for ocean-scale voyages.

    Parameters
    ----------
    waypoints : list of (lat, lon)
        Points that must lie inside the lattice, degrees.
    resolution_deg : float
        Requested lattice step, degrees.
    margin_deg : float
        Margin around the bounding box, degrees.
    max_cells : int
        Upper limit for ``n_rows * n_cols``.

    Returns
    -------
    GridSpec
        Lattice whose resolution is ``resolution_deg`` or a coarser value.
    """
    arr = np.asarray(waypoints, dtype=float)
    lat_min = max(-85.0, float(arr[:, 0].min()) - margin_deg)
    lat_max = min(85.0, float(arr[:, 0].max()) + margin_deg)
    lon_min = max(-180.0, float(arr[:, 1].min()) - margin_deg)
    lon_max = min(180.0, float(arr[:, 1].max()) + margin_deg)

    spec = GridSpec(lat_min, lat_max, lon_min, lon_max, resolution_deg)
    # Coarsen by a factor of 1.5 until the cell budget is met (upper limit 4 deg).
    # 每次将分辨率放大 1.5 倍，直到单元数不超过上限（分辨率上限 4 度）。
    while spec.n_cells > max_cells and spec.resolution_deg < 4.0:
        spec = GridSpec(lat_min, lat_max, lon_min, lon_max, round(spec.resolution_deg * 1.5, 4))
        LOGGER.info("Grid too large, coarsening to %.3f deg", spec.resolution_deg)
    return spec


def build_cost_map(
    spec: GridSpec,
    classifier,
    config: RoutingConfig = ROUTING,
    progress: ProgressCallback = None,
    batch_size: int = 16384,
) -> CostMap:
    """Classify every lattice cell and convert the zones into traversal costs.

    对网格每个单元进行神经网络分类并转换为通行代价。

    Steps:

    1. Flatten the node coordinates and classify them in batches of
       ``batch_size`` (bounds the memory of the feature matrix and lets the
       client see progress). The predicted class is the argmax of the
       probabilities; the confidence is the maximum probability.
    2. Map each class to its weight from ``ZONE_COST``; classes with an
       infinite weight keep the ``IMPASSABLE_COST`` sentinel.
    3. Apply the safety margin (:func:`_apply_safety_margin`) to water cells
       next to land.

    Parameters
    ----------
    spec : GridSpec
        Lattice to classify.
    classifier
        Object with ``predict_coordinates(lat, lon) -> ndarray (n, 4)`` of
        class probabilities, normally
        :class:`~maritime_route.model.inference.ZoneClassifierService`.
    config : RoutingConfig
        Supplies ``safety_margin_cells`` and ``proximity_penalty``.
    progress : callable, optional
        Receives ``("classify", fraction, {...})`` after every batch and
        ``("cost_map", 1.0, statistics)`` at the end.
    batch_size : int, default 16384
        Number of nodes classified per call.

    Returns
    -------
    CostMap
        The classified lattice with traversal costs and its build time.
    """
    t0 = time.perf_counter()
    grid_lat, grid_lon = spec.mesh()
    # Row-major flattening: flat index = row * n_cols + col; reshaped back below.
    flat_lat = grid_lat.ravel()
    flat_lon = grid_lon.ravel()
    n = flat_lat.size

    class_flat = np.zeros(n, dtype=np.int8)
    conf_flat = np.zeros(n, dtype=np.float32)

    # At least ~20 batches, so that the web client receives regular progress
    # frames even for a small lattice; never fewer than 1024 cells per batch.
    # 至少分成约 20 批，使小网格也能向前端连续推送进度；每批不少于 1024 个格子。
    batch_size = min(batch_size, max(1024, math.ceil(n / 20)))
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        probs = classifier.predict_coordinates(flat_lat[start:stop], flat_lon[start:stop])
        # Predicted class = most probable zone; confidence = its probability.
        class_flat[start:stop] = probs.argmax(axis=1).astype(np.int8)
        conf_flat[start:stop] = probs.max(axis=1).astype(np.float32)
        if progress is not None:
            progress("classify", stop / n, {"cells_done": int(stop), "cells_total": int(n)})

    class_index = class_flat.reshape(spec.n_rows, spec.n_cols)
    confidence = conf_flat.reshape(spec.n_rows, spec.n_cols)

    # Start with every cell impassable, then write the finite zone weights.
    # 先全部设为不可通行，再为有限权重的类别写入代价。
    cost = np.full(class_index.shape, IMPASSABLE_COST, dtype=np.float32)
    for i, name in enumerate(CLASS_NAMES):
        weight = ZONE_COST[name]
        if np.isfinite(weight):
            cost[class_index == i] = float(weight)

    # Land mask = every class that is not navigable (NEAR_COAST, COASTLINE).
    land = ~np.isin(class_index, [CLASS_NAMES.index(c) for c in NAVIGABLE_CLASSES])
    cost = _apply_safety_margin(cost, land, config)

    cost_map = CostMap(spec=spec, class_index=class_index, confidence=confidence,
                       cost=cost, build_seconds=time.perf_counter() - t0)
    if progress is not None:
        progress("cost_map", 1.0, cost_map.statistics())
    LOGGER.info("Cost map built: %s", cost_map.statistics()["zone_shares"])
    return cost_map


def _apply_safety_margin(cost: np.ndarray, land: np.ndarray, config: RoutingConfig) -> np.ndarray:
    """Penalise water cells within ``safety_margin_cells`` of land.

    安全余量：靠近陆地的水域单元代价乘以惩罚系数，使航线远离海岸。

    The land mask is dilated ``safety_margin_cells`` times with a
    4-neighbourhood (cross-shaped) structuring element: each pass marks every
    cell whose north, south, west or east neighbour is already marked. After
    ``m`` passes the mask covers every cell within Manhattan distance ``m`` of
    land. The water cells in that band have their weight multiplied by
    ``proximity_penalty`` (default 1.6, so open sea 1.0 -> 1.6 and coastal sea
    1.8 -> 2.88). The result is capped at ``IMPASSABLE_COST - 1`` so that a
    penalised cell never turns into land: narrow straits stay passable, only
    more expensive. The searches therefore prefer a route one or more cells
    away from the shore whenever such a route exists.

    Parameters
    ----------
    cost : numpy.ndarray
        Cell weights before the margin; not modified.
    land : numpy.ndarray of bool
        True for non-navigable cells.
    config : RoutingConfig
        ``safety_margin_cells`` (band width in cells) and
        ``proximity_penalty`` (multiplier).

    Returns
    -------
    numpy.ndarray
        A new cost array, or the input array itself when the margin is
        disabled (width 0 or penalty <= 1).
    """
    margin = max(0, int(config.safety_margin_cells))
    if margin == 0 or config.proximity_penalty <= 1.0:
        return cost
    influence = land.astype(np.float32)
    padded = influence
    for _ in range(margin):
        # One dilation pass: copy the mask shifted by one cell in each of the
        # four axis directions, without wrapping around the array edges.
        # 一次膨胀：将掩膜沿上下左右各平移一格后取最大值。
        shifted = np.zeros_like(padded)
        shifted[1:, :] = np.maximum(shifted[1:, :], padded[:-1, :])    # from the south neighbour
        shifted[:-1, :] = np.maximum(shifted[:-1, :], padded[1:, :])   # from the north neighbour
        shifted[:, 1:] = np.maximum(shifted[:, 1:], padded[:, :-1])    # from the west neighbour
        shifted[:, :-1] = np.maximum(shifted[:, :-1], padded[:, 1:])   # from the east neighbour
        padded = np.maximum(padded, shifted)
    # Water cells inside the dilated band (land itself is excluded).
    near_land = (padded > 0) & (~land)
    result = cost.copy()
    result[near_land] = np.minimum(
        result[near_land] * float(config.proximity_penalty), IMPASSABLE_COST - 1.0
    )
    return result


def nearest_navigable(cost_map: CostMap, row: int, col: int, max_radius: int = 40
                      ) -> Optional[Tuple[int, int]]:
    """Snap a cell onto the closest navigable cell (ring search).

    将起点/终点吸附到最近的可航行单元。

    Ports lie on the shore, so their lattice cell is usually classified as
    land. The search examines square rings of growing Chebyshev radius
    ``r = 1, 2, ...`` around the cell; only the perimeter of each ring is
    visited (the full top and bottom rows, and the two side cells of every
    other row). The first ring that contains a navigable cell ends the search,
    and within that ring the cell with the smallest Euclidean distance
    ``dr^2 + dc^2`` (in cell units) is returned.

    The result is the nearest cell up to the ring granularity: a corner cell
    of ring ``r`` (distance ``r * sqrt(2)``) can be chosen although an axis
    cell of ring ``r + 1`` would be slightly closer. Distances are counted in
    cells, not kilometres. Both effects are below one cell and irrelevant for
    snapping a port.
    按方环逐圈向外搜索，返回第一个含可航行单元的环中欧氏距离最小的单元。

    Parameters
    ----------
    cost_map : CostMap
        Map providing the ``passable`` mask.
    row, col : int
        Cell to snap.
    max_radius : int, default 40
        Largest ring radius examined, in cells. The planner derives it from
        :attr:`RoutePlanner.SNAP_RADIUS_KM` so that a point far inland is not
        moved into open sea.

    Returns
    -------
    tuple of int or None
        ``(row, col)`` of a navigable cell (the input cell itself if it is
        navigable), or ``None`` if none lies within ``max_radius``.
    """
    if cost_map.passable[row, col]:
        return (row, col)
    rows, cols = cost_map.class_index.shape
    for radius in range(1, max_radius + 1):
        best: Optional[Tuple[int, int]] = None
        best_d = float("inf")
        for dr in range(-radius, radius + 1):
            # Top/bottom row of the ring (|dr| == radius): every column.
            # Other rows: only the left and right edge (dc = -radius, +radius).
            for dc in (-radius, radius) if abs(dr) != radius else range(-radius, radius + 1):
                r, c = row + dr, col + dc
                if 0 <= r < rows and 0 <= c < cols and cost_map.passable[r, c]:
                    d = dr * dr + dc * dc       # squared Euclidean distance, cells^2
                    if d < best_d:
                        best, best_d = (r, c), d
        if best is not None:
            return best
    return None
