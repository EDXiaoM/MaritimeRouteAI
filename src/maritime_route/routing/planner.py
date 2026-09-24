"""High level route planning service.

航线规划服务：串联"分类 -> 代价地图 -> 路径搜索 -> 与大圆航线对比"全过程。

Role in the pipeline
--------------------
:class:`RoutePlanner` is the single entry point that the web application
(:mod:`maritime_route.web.app`) and the scripts call to obtain a route. For one
request it performs:

1. **Lattice** - :func:`~.cost_map.make_grid_spec` around the two ports.
2. **Classification** - :func:`~.cost_map.build_cost_map` asks the neural
   classifier for the zone of every lattice cell and derives the costs
   (skipped when a cost map is passed in for reuse).
3. **Snapping** - the ports, which normally lie on a shore cell, are moved to
   the nearest navigable cell within :attr:`RoutePlanner.SNAP_RADIUS_KM`.
4. **Search** - A*, Dijkstra, the genetic algorithm, or the great-circle
   reference, all on the same cost map.
5. **Refinement** - if no route exists on the lattice (a strait narrower than
   a cell, or a passage outside the box), the lattice is made finer and wider
   and the whole procedure is repeated (up to :attr:`RoutePlanner.MAX_REFINEMENTS`
   times).
6. **Post-processing** - the path is anchored on the exact port coordinates,
   optionally simplified (Douglas-Peucker), turned into a waypoint table
   (:class:`RouteLeg`) and compared with the great-circle route.

The outcome is a :class:`RoutePlan`, which :mod:`maritime_route.storage`
persists and :mod:`maritime_route.export` writes to GeoJSON, CSV and PDF.
:func:`benchmark` runs every method on one shared cost map for the
comparison in the thesis.

Units: distances in km, coordinates in degrees, speed in knots
(1 kn = 1.852 km/h), passage time in hours, cost in weighted km
(see :mod:`.astar`).
单位：距离千米，坐标度，航速节（1 节 = 1.852 千米/小时），航行时间小时。
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
from .astar import SearchResult, astar
from .cost_map import CostMap, GridSpec, build_cost_map, make_grid_spec, nearest_navigable
from .dijkstra import dijkstra
from .dynamic_programming import dynamic_programming
from .genetic import genetic
from .geodesy import douglas_peucker, great_circle_points, haversine_km, path_length_km

LOGGER = logging.getLogger(__name__)

#: ``(latitude, longitude)`` in degrees. 坐标 (纬度, 经度)。
Coordinate = Tuple[float, float]
#: Optional callback ``progress(stage, fraction, details)`` for the web client.
ProgressCallback = Optional[Callable[[str, float, Dict], None]]


@dataclass
class RouteLeg:
    """One sampled point of the final route. 航线上的一个航路点。

    One row of the waypoint table shown in the UI, stored in the ``waypoint``
    database table and written to the CSV/PDF exports.

    Attributes
    ----------
    latitude, longitude : float
        Position, degrees.
    class_code : str
        Zone of the lattice cell nearest to the point.
    zone_cost : float
        Nominal weight of that zone from ``ZONE_COST`` (without the safety
        margin); ``IMPASSABLE_COST`` for a land zone.
    cumulative_km : float
        Distance along the route from the departure to this point, km.
    bearing_deg : float
        Initial great-circle course to the next point, degrees from true
        north; the last point repeats the previous course.
    """

    latitude: float
    longitude: float
    class_code: str
    zone_cost: float
    cumulative_km: float
    bearing_deg: float

    def to_dict(self) -> Dict:
        """JSON-ready dictionary with display rounding.

        Coordinates keep 6 decimals (about 0.1 m), distances 2 decimals,
        bearings 1 decimal.

        Returns
        -------
        dict
            The six fields, rounded.
        """
        return {
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "class_code": self.class_code,
            "zone_cost": round(self.zone_cost, 3),
            "cumulative_km": round(self.cumulative_km, 2),
            "bearing_deg": round(self.bearing_deg, 1),
        }


@dataclass
class RoutePlan:
    """Complete planning outcome, ready for storage, export and rendering.

    完整规划结果，可直接用于数据库存储、文件导出与前端渲染。

    Attributes
    ----------
    route_id : str
        16 hexadecimal characters taken from a random UUID4 (64 bits);
        stored as ``route.route_uid`` and used in URLs and file names.
    algorithm : str
        Method actually used: ``astar``, ``dijkstra``, ``genetic`` or
        ``great_circle``.
    start, end : Coordinate
        Departure and destination exactly as requested, degrees.
    waypoints : list of Coordinate
        Route poly-line from ``start`` to ``end``; empty if infeasible.
    legs : list of RouteLeg
        Waypoint table, one entry per vertex of ``waypoints``.
    total_cost : float
        Search cost in weighted km (``inf`` if infeasible).
    distance_km : float
        Geometric route length, km.
    great_circle_km : float
        Shortest possible distance between the two ports, km.
    detour_ratio : float
        ``distance_km / great_circle_km`` (>= 1; ``nan`` if infeasible).
    estimated_hours : float
        Passage time at ``speed_knots``, hours.
    nodes_expanded : int
        Search effort (see :class:`~.astar.SearchResult`).
    runtime_s : float
        Wall-clock planning time, seconds (includes the cost-map build unless
        a cost map was reused).
    grid : dict
        Lattice description (:meth:`~.cost_map.GridSpec.to_dict`) of the
        lattice finally used.
    zone_profile : dict
        Number of route waypoints in each zone.
    baseline : dict
        Great-circle comparison (:meth:`RoutePlanner.compare_with_great_circle`).
    cost_map_stats : dict
        :meth:`~.cost_map.CostMap.statistics` of the map used.
    created_utc : str
        ISO 8601 UTC timestamp.
    speed_knots : float
        Planned speed, knots.
    interior_land_waypoints : int
        See the field comment below; a validity check that must be 0.
    feasible : bool
        False when no navigable route was found.
    message : str
        Outcome text for the UI.
    """

    route_id: str
    algorithm: str
    start: Coordinate
    end: Coordinate
    waypoints: List[Coordinate]
    legs: List[RouteLeg]
    total_cost: float
    distance_km: float
    great_circle_km: float
    detour_ratio: float
    estimated_hours: float
    nodes_expanded: int
    runtime_s: float
    grid: Dict
    zone_profile: Dict[str, int]
    baseline: Dict
    cost_map_stats: Dict
    created_utc: str
    speed_knots: float
    #: Non-navigable waypoints strictly between departure and destination.
    #: A voyage begins and ends at a quay, so the two end points legitimately
    #: fall into a shore zone; every *intermediate* waypoint must be navigable.
    #: 航程始于泊位、终于泊位，故首尾点位于岸线区属正常；
    #: 中间航路点必须全部可航行，该计数应恒为 0。
    interior_land_waypoints: int = 0
    feasible: bool = True
    message: str = ""

    def to_dict(self) -> Dict:
        """JSON-ready dictionary of the whole plan.

        Nested dataclasses are converted by :func:`dataclasses.asdict`; the
        legs are then replaced by their rounded form, tuples become lists
        (``[lat, lon]``), and non-finite numbers (``inf``/``nan`` of an
        infeasible plan) become ``None`` because JSON cannot represent them.

        Returns
        -------
        dict
            Payload sent to the web client and stored in the job results.
        """
        payload = asdict(self)
        payload["legs"] = [leg.to_dict() for leg in self.legs]
        payload["waypoints"] = [[round(a, 6), round(b, 6)] for a, b in self.waypoints]
        payload["start"] = [round(self.start[0], 6), round(self.start[1], 6)]
        payload["end"] = [round(self.end[0], 6), round(self.end[1], 6)]
        for key in ("total_cost", "distance_km", "great_circle_km", "detour_ratio",
                    "estimated_hours", "runtime_s"):
            value = payload[key]
            payload[key] = None if value is None or not math.isfinite(value) else round(value, 4)
        return payload


class RoutePlanner:
    """Builds cost maps and searches optimal maritime routes.

    One instance is created per classifier (the web app keeps one for the
    process lifetime). The planner itself holds no per-request state, so
    :meth:`plan` can be called repeatedly.
    航线规划器：负责构建代价地图并调用搜索算法。

    Parameters
    ----------
    classifier
        Zone classifier with ``predict_coordinates(lat, lon)``, normally
        :class:`~maritime_route.model.inference.ZoneClassifierService`.
    config : RoutingConfig
        Default lattice and search parameters.
    """

    #: Largest distance over which a departure or destination is moved onto water.
    SNAP_RADIUS_KM: float = 60.0

    def __init__(self, classifier, config: RoutingConfig = ROUTING) -> None:
        """Store the classifier and the default routing configuration.

        Parameters
        ----------
        classifier
            Object providing ``predict_coordinates``.
        config : RoutingConfig
            Defaults used when :meth:`plan` is not given overrides.
        """
        self.classifier = classifier
        self.config = config

    # ------------------------------------------------------------------ #
    def plan(
        self,
        start: Coordinate,
        end: Coordinate,
        algorithm: str = "astar",
        resolution_deg: Optional[float] = None,
        speed_knots: float = 14.0,
        simplify_km: float = 0.0,
        progress: ProgressCallback = None,
        reuse_cost_map: Optional[CostMap] = None,
        auto_refine: bool = True,
        _attempt: int = 0,
        _margin_deg: Optional[float] = None,
    ) -> Tuple[RoutePlan, CostMap]:
        """Plan a route between two coordinates. 规划从起点到终点的航线。

        When the search fails - because a strait is narrower than one grid cell,
        or because the only passage lies outside the rectangle around the two
        ports - the method automatically retries on a finer *and* wider lattice
        (``auto_refine``).
        当海峡窄于一个网格单元、或唯一通道位于起终点外包矩形之外而导致搜索失败时，
        自动加密并扩大网格重试。

        Parameters
        ----------
        start, end : Coordinate
            Departure and destination ``(lat, lon)``, degrees.
        algorithm : str, default "astar"
            ``"astar"``, ``"dijkstra"``, ``"genetic"``, ``"dynamic"`` or ``"great_circle"``
            (case-insensitive); any other value falls back to A*.
        resolution_deg : float, optional
            Lattice step; ``None`` uses ``config.grid_resolution_deg``.
            May be coarsened by :func:`~.cost_map.make_grid_spec` to respect
            ``max_grid_cells``.
        speed_knots : float, default 14.0
            Planned speed for the passage-time estimate.
        simplify_km : float, default 0.0
            Douglas-Peucker tolerance in km; 0 keeps every lattice vertex.
        progress : callable, optional
            Progress callback forwarded to every stage.
        reuse_cost_map : CostMap, optional
            Existing cost map to search on (no classification, no
            refinement). Used by :func:`benchmark` so that all methods share
            one map.
        auto_refine : bool, default True
            Allow the finer/wider retry on failure.
        _attempt : int
            Internal: refinement depth of this call.
        _margin_deg : float, optional
            Internal: margin override set by :meth:`_refine`.

        Returns
        -------
        tuple
            ``(RoutePlan, CostMap)`` - the plan (``feasible=False`` with a
            message if no route was found) and the cost map it was computed on.
        """
        t0 = time.perf_counter()
        algorithm = algorithm.lower()
        config = self.config
        # Per-call overrides (resolution from the user, margin from _refine)
        # produce a new config; the other parameters are copied unchanged.
        # 若指定了分辨率或边距，则基于默认配置生成一份新的配置。
        if resolution_deg is not None or _margin_deg is not None:
            config = RoutingConfig(
                grid_resolution_deg=float(resolution_deg if resolution_deg is not None
                                          else self.config.grid_resolution_deg),
                max_grid_cells=self.config.max_grid_cells,
                margin_deg=float(_margin_deg if _margin_deg is not None
                                 else self.config.margin_deg),
                diagonal_moves=self.config.diagonal_moves,
                heuristic_weight=self.config.heuristic_weight,
                safety_margin_cells=self.config.safety_margin_cells,
                proximity_penalty=self.config.proximity_penalty,
            )

        # ---- 1-2. lattice and cost map --------------------------------- #
        if progress:
            progress("grid", 0.0, {"stage": "Building geographic lattice"})
        cost_map = reuse_cost_map
        if cost_map is None:
            spec = make_grid_spec([start, end], config.grid_resolution_deg,
                                  config.margin_deg, config.max_grid_cells)
            LOGGER.info("Grid %dx%d (%d cells) at %.3f deg",
                        spec.n_rows, spec.n_cols, spec.n_cells, spec.resolution_deg)
            cost_map = build_cost_map(spec, self.classifier, config, progress=progress)
        spec = cost_map.spec

        # ---- 3. snapping the ports onto water -------------------------- #
        # A port usually lies in a shore cell, so the ends are snapped to the
        # nearest navigable cell - but never further than SNAP_RADIUS_KM, so
        # that a point far inland is reported instead of being moved to open sea.
        # 起终点吸附到最近的可航行格子，但距离不超过 SNAP_RADIUS_KM，避免把内陆点移到远海。
        # Radius in cells = km / (degrees per cell * 111.2 km per degree),
        # rounded up, at least 3 cells. 111.2 km is one degree of latitude; a
        # degree of longitude is shorter, so the east-west reach is smaller.
        snap_cells = max(3, math.ceil(self.SNAP_RADIUS_KM / (spec.resolution_deg * 111.2)))
        start_cell = nearest_navigable(cost_map, *spec.coord_to_cell(*start), max_radius=snap_cells)
        goal_cell = nearest_navigable(cost_map, *spec.coord_to_cell(*end), max_radius=snap_cells)
        if start_cell is not None and start_cell == goal_cell:
            # Both ends snapped onto the same cell: the lattice is too coarse.
            # 起终点吸附到同一格子，说明网格过粗。
            start_cell = goal_cell = None
        great_circle_km = float(haversine_km(start[0], start[1], end[0], end[1]))

        if start_cell is None or goal_cell is None:
            # No water near a port: try a finer lattice, else report infeasible.
            retry = self._refine(start, end, algorithm, config, speed_knots, simplify_km,
                                 progress, auto_refine, _attempt, reuse_cost_map)
            if retry is not None:
                return retry
            return (self._infeasible(start, end, algorithm, great_circle_km, cost_map,
                                     speed_knots, time.perf_counter() - t0,
                                     "Start or destination lies on land and no navigable "
                                     "cell was found nearby"), cost_map)

        # ---- 4. search -------------------------------------------------- #
        if progress:
            progress("search", 0.0, {"stage": f"Running {algorithm.upper()} search"})

        if algorithm == "dijkstra":
            result = dijkstra(cost_map, start_cell, goal_cell, config, progress=progress)
        elif algorithm == "genetic":
            # The GA uses its own GeneticConfig defaults (GENETIC); the routing
            # config does not apply to it.
            result = genetic(cost_map, start_cell, goal_cell, progress=progress)
        elif algorithm == "dynamic":
            # Dynamic programming: value iteration on the Bellman equation.
            # 动态规划：在贝尔曼方程上做价值迭代。
            result = dynamic_programming(cost_map, start_cell, goal_cell, progress=progress)
        elif algorithm == "great_circle":
            result = self._great_circle_result(cost_map, start, end)
        else:
            algorithm = "astar"
            result = astar(cost_map, start_cell, goal_cell, config, progress=progress)

        # ---- 5. refinement on failure ----------------------------------- #
        if not result.found:
            # A failed genetic run does not prove that no path exists: the lattice
            # is refined only when the exact search confirms that it is blocked.
            # 遗传算法失败不代表无路可走：仅当 A* 也确认不连通时才细化网格。
            lattice_blocked = True
            if algorithm == "genetic":
                lattice_blocked = not astar(cost_map, start_cell, goal_cell, config).found
            retry = None
            if lattice_blocked:
                retry = self._refine(start, end, algorithm, config, speed_knots, simplify_km,
                                     progress, auto_refine, _attempt, reuse_cost_map)
            if retry is not None:
                return retry
            return (self._infeasible(start, end, algorithm, great_circle_km, cost_map,
                                     speed_knots, time.perf_counter() - t0,
                                     result.message + " at the finest lattice tried "
                                     f"({spec.resolution_deg}°)"), cost_map)

        # ---- 6. post-processing ----------------------------------------- #
        coords = list(result.path_coords)
        # Anchor the poly-line on the exact user coordinates.
        # The search starts and ends at the snapped cell centres; replacing the
        # first and last vertex draws the route from/to the actual quay.
        # 用用户输入的精确坐标替换首尾网格点。
        coords[0], coords[-1] = start, end
        if simplify_km > 0:
            coords = douglas_peucker(coords, simplify_km)

        legs = self._build_legs(coords, cost_map)
        distance_km = path_length_km(coords)
        zone_profile = {name: 0 for name in CLASS_NAMES}
        for leg in legs:
            zone_profile[leg.class_code] = zone_profile.get(leg.class_code, 0) + 1
        # Land waypoints other than the two ports - a self-check, expected 0.
        interior_land = sum(1 for leg in legs[1:-1]
                            if leg.class_code not in NAVIGABLE_CLASSES)

        baseline = self.compare_with_great_circle(start, end, cost_map, distance_km,
                                                  result.total_cost)
        # Knots to km/h: 1 nautical mile = 1.852 km. The floor of 0.1 kn avoids
        # a division by zero for a zero speed. 节换算为千米/小时。
        speed_kmh = max(speed_knots, 0.1) * 1.852

        if progress:
            progress("done", 1.0, {"stage": "Route ready"})

        plan = RoutePlan(
            route_id=uuid.uuid4().hex[:16],
            algorithm=algorithm,
            start=start,
            end=end,
            waypoints=coords,
            legs=legs,
            total_cost=float(result.total_cost),
            distance_km=distance_km,
            great_circle_km=great_circle_km,
            detour_ratio=distance_km / great_circle_km if great_circle_km > 0 else 1.0,
            estimated_hours=distance_km / speed_kmh,
            nodes_expanded=result.nodes_expanded,
            runtime_s=time.perf_counter() - t0,
            grid=spec.to_dict(),
            zone_profile=zone_profile,
            interior_land_waypoints=interior_land,
            baseline=baseline,
            cost_map_stats=cost_map.statistics(),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            speed_knots=float(speed_knots),
            feasible=True,
            message=result.message,
        )
        return plan, cost_map

    # ------------------------------------------------------------------ #
    def compare_with_great_circle(
        self,
        start: Coordinate,
        end: Coordinate,
        cost_map: CostMap,
        optimal_km: float,
        optimal_cost: float,
        samples: int = 160,
    ) -> Dict:
        """Evaluate the naive shortest-distance route on the same cost map.

        在同一代价地图上评估"最短距离"大圆航线，用于安全性对比。
        Returns the zone profile of the great-circle line, the number of samples
        that fall on land, and the relative gain of the optimised route.

        The great circle is sampled at ``samples`` equally spaced points; each
        sample is assigned the zone of its nearest lattice cell. The weighted
        cost is approximated as ``sum(segment_km * w(end of segment))`` (a
        right-endpoint rule, coarser than the trapezoidal edge cost of the
        searches, which is adequate for a reference value).

        Parameters
        ----------
        start, end : Coordinate
            Ports, degrees.
        cost_map : CostMap
            Map on which the optimised route was computed.
        optimal_km : float
            Length of the optimised route, km.
        optimal_cost : float
            Search cost of the optimised route, weighted km.
        samples : int, default 160
            Number of points on the great circle.

        Returns
        -------
        dict
            ``distance_km``, ``weighted_cost`` (None if the line crosses
            land), ``zone_profile``, ``samples``, ``land_samples``,
            ``interior_land_samples`` (ports excluded), ``land_share``,
            ``is_navigable``, ``extra_distance_km`` and ``extra_distance_pct``
            of the optimised route, ``cost_saving_pct`` (None if not
            comparable) and a textual ``verdict``.
        """
        line = great_circle_points(start, end, samples)
        spec = cost_map.spec
        profile = {name: 0 for name in CLASS_NAMES}
        unsafe = 0
        interior_unsafe = 0
        total_cost = 0.0
        previous: Optional[Coordinate] = None
        last_index = len(line) - 1
        for i, (lat, lon) in enumerate(line):
            row, col = spec.coord_to_cell(lat, lon)
            code = INDEX_TO_CLASS[int(cost_map.class_index[row, col])]
            profile[code] += 1
            if code not in NAVIGABLE_CLASSES:
                unsafe += 1
                # Exclude the two berths so that this figure is directly
                # comparable with interior_land_waypoints of the optimised route.
                # 排除首尾泊位，使该指标与优化航线的中间航路点统计口径一致。
                if 0 < i < last_index:
                    interior_unsafe += 1
            weight = float(cost_map.cost[row, col])
            if previous is not None:
                seg = float(haversine_km(previous[0], previous[1], lat, lon))
                total_cost += seg * min(weight, IMPASSABLE_COST)
            previous = (lat, lon)

        gc_km = path_length_km(line)
        navigable = unsafe == 0
        # A route that crosses land has no meaningful finite cost; report None
        # so that the UI and the PDF show "infeasible" instead of a huge number.
        # 穿越陆地的航线不存在有限代价，此处返回 None，前端显示"不可行"。
        return {
            "distance_km": round(gc_km, 3),
            "weighted_cost": round(total_cost, 3) if navigable else None,
            "zone_profile": profile,
            "samples": samples,
            "land_samples": unsafe,
            "interior_land_samples": interior_unsafe,
            "land_share": round(unsafe / max(samples, 1), 4),
            "is_navigable": navigable,
            # Extra length the optimised route accepts for safety, km and %.
            "extra_distance_km": round(optimal_km - gc_km, 3),
            "extra_distance_pct": round(100.0 * (optimal_km - gc_km) / gc_km, 2) if gc_km else 0.0,
            # Relative cost reduction versus the great circle (only when the
            # great circle is navigable and therefore has a finite cost).
            "cost_saving_pct": (round(100.0 * (total_cost - optimal_cost) / total_cost, 2)
                                if navigable and total_cost > 0 else None),
            "verdict": ("Great-circle route crosses land and is not navigable"
                        if unsafe else "Great-circle route is navigable but ignores zone risk"),
        }

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_legs(coords: Sequence[Coordinate], cost_map: CostMap) -> List[RouteLeg]:
        """Turn the route poly-line into the waypoint table.

        For every vertex: the cumulative great-circle distance from the
        departure, the zone of its nearest lattice cell with the nominal zone
        weight, and the initial bearing towards the next vertex. The last
        vertex has no successor and repeats the previous bearing.
        生成航路点表：累计距离、所在区域及其权重、驶向下一点的方位角。

        Parameters
        ----------
        coords : sequence of Coordinate
            Route vertices, degrees.
        cost_map : CostMap
            Map used to look up the zone of each vertex.

        Returns
        -------
        list of RouteLeg
            One entry per vertex, in route order.
        """
        from .geodesy import initial_bearing_deg

        spec = cost_map.spec
        legs: List[RouteLeg] = []
        cumulative = 0.0
        for i, (lat, lon) in enumerate(coords):
            if i > 0:
                cumulative += float(haversine_km(coords[i - 1][0], coords[i - 1][1], lat, lon))
            row, col = spec.coord_to_cell(lat, lon)
            code = INDEX_TO_CLASS[int(cost_map.class_index[row, col])]
            # Nominal zone weight (the safety-margin penalty is not included).
            weight = ZONE_COST[code]
            bearing = (initial_bearing_deg(lat, lon, coords[i + 1][0], coords[i + 1][1])
                       if i + 1 < len(coords) else (legs[-1].bearing_deg if legs else 0.0))
            legs.append(RouteLeg(
                latitude=float(lat), longitude=float(lon), class_code=code,
                # inf (land) is stored as the finite sentinel for numeric columns.
                zone_cost=float(weight) if math.isfinite(weight) else IMPASSABLE_COST,
                cumulative_km=cumulative, bearing_deg=bearing,
            ))
        return legs

    def _great_circle_result(self, cost_map: CostMap, start: Coordinate,
                             end: Coordinate) -> SearchResult:
        """Wrap the naive baseline in a :class:`SearchResult` for uniformity.

        The great circle is sampled at 160 points and priced like in
        :meth:`compare_with_great_circle` (segment length times the weight of
        the cell at the segment end; land at ``IMPASSABLE_COST``). ``found``
        is always True because the line always exists; whether it crosses
        land is reported through ``RoutePlan.baseline["is_navigable"]``.
        ``nodes_expanded`` is set to the number of samples.

        Parameters
        ----------
        cost_map : CostMap
            Map used for pricing.
        start, end : Coordinate
            Ports, degrees.

        Returns
        -------
        SearchResult
            ``algorithm="great_circle"`` with an empty ``path_cells``.
        """
        t0 = time.perf_counter()
        coords = great_circle_points(start, end, 160)
        spec = cost_map.spec
        total = 0.0
        for a, b in zip(coords[:-1], coords[1:]):
            row, col = spec.coord_to_cell(*b)
            seg = float(haversine_km(a[0], a[1], b[0], b[1]))
            total += seg * min(float(cost_map.cost[row, col]), IMPASSABLE_COST)
        return SearchResult(
            found=True, path_cells=[], path_coords=coords, total_cost=total,
            distance_km=path_length_km(coords), nodes_expanded=len(coords),
            nodes_generated=len(coords), runtime_s=time.perf_counter() - t0,
            algorithm="great_circle", message="Great-circle reference route",
        )

    #: Maximum number of automatic finer/wider retries. 最大重试次数。
    MAX_REFINEMENTS: int = 3
    #: Finest lattice step the refinement may reach, degrees (~4.4 km).
    MIN_RESOLUTION_DEG: float = 0.04

    def _refine(self, start, end, algorithm, config, speed_knots, simplify_km,
                progress, auto_refine, attempt, reuse_cost_map):
        """Retry on a finer and wider lattice; ``None`` when refinement is over.

        Every retry multiplies the cell size by 0.55 (to open straits narrower
        than a cell) and widens the margin around the two ports by 2 degrees
        (so that a passage lying outside the first rectangle - for example the
        Skagerrak on a voyage from the North Sea into the Baltic - can be found).

        网格重试：每次分辨率乘以 0.55、外扩边距增加 2 度，最多重试 MAX_REFINEMENTS 次。

        With the default 0.25 deg the sequence is 0.1375 -> 0.0756 -> 0.0416
        deg. Refinement is not done when a cost map was supplied for reuse
        (the benchmark must stay on one map) or when ``auto_refine`` is off.

        Parameters
        ----------
        start, end : Coordinate
            Ports, degrees.
        algorithm : str
            Method to rerun.
        config : RoutingConfig
            Configuration of the failed attempt (its resolution and margin
            are the basis of the next one).
        speed_knots, simplify_km, progress
            Passed through to :meth:`plan`.
        auto_refine : bool
            Whether refinement is allowed.
        attempt : int
            Number of refinements already made.
        reuse_cost_map : CostMap or None
            Cost map supplied by the caller, if any.

        Returns
        -------
        tuple or None
            The ``(RoutePlan, CostMap)`` of the retry, or ``None`` if no
            further refinement is allowed.
        """
        if (not auto_refine or reuse_cost_map is not None
                or attempt >= self.MAX_REFINEMENTS):
            return None
        finer = round(config.grid_resolution_deg * 0.55, 4)
        if finer < self.MIN_RESOLUTION_DEG:
            return None
        LOGGER.info("No path at %.3f deg - refining to %.3f deg (attempt %d)",
                    config.grid_resolution_deg, finer, attempt + 1)
        if progress:
            progress("refine", 0.0,
                     {"stage": f"No passage at {config.grid_resolution_deg}° - "
                               f"refining the lattice to {finer}°"})
        # Recursive call: builds a new cost map on the finer, wider lattice.
        return self.plan(start, end, algorithm, finer, speed_knots, simplify_km,
                         progress=progress, auto_refine=True, _attempt=attempt + 1,
                         _margin_deg=config.margin_deg + 2.0)

    def _infeasible(self, start, end, algorithm, gc_km, cost_map, speed_knots,
                    runtime, message) -> RoutePlan:
        """Build a :class:`RoutePlan` that reports failure.

        The plan has no waypoints, ``feasible=False``, infinite cost and
        ``nan`` ratios (converted to ``None`` by :meth:`RoutePlan.to_dict`),
        so the client can show the reason instead of a route.
        构造"不可行"结果：无航路点，代价为无穷大。

        Parameters
        ----------
        start, end : Coordinate
            Requested ports.
        algorithm : str
            Method that was attempted.
        gc_km : float
            Great-circle distance between the ports, km.
        cost_map : CostMap
            Last cost map tried (its grid and statistics are reported).
        speed_knots : float
            Requested speed.
        runtime : float
            Elapsed time, seconds.
        message : str
            Reason shown to the user.

        Returns
        -------
        RoutePlan
            Infeasible plan.
        """
        return RoutePlan(
            route_id=uuid.uuid4().hex[:16], algorithm=algorithm, start=start, end=end,
            waypoints=[], legs=[], total_cost=float("inf"), distance_km=0.0,
            great_circle_km=gc_km, detour_ratio=float("nan"), estimated_hours=float("nan"),
            nodes_expanded=0, runtime_s=runtime, grid=cost_map.spec.to_dict(),
            zone_profile={name: 0 for name in CLASS_NAMES},
            baseline={}, cost_map_stats=cost_map.statistics(),
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            speed_knots=float(speed_knots), feasible=False, message=message,
        )


def benchmark(planner: RoutePlanner, start: Coordinate, end: Coordinate,
              resolution_deg: Optional[float] = None) -> Dict:
    """Run all algorithms on one shared cost map and compare them.

    在同一代价地图上运行三种算法并对比耗时、扩展节点数与总代价。
    The cost map is built once *before* timing so that the reported runtimes
    measure the search only and are therefore directly comparable.
    先构建一次代价地图再计时，使三种算法的耗时可直接比较。

    Timing: each ``runtime_s`` is the ``time.perf_counter`` interval of one
    :meth:`RoutePlanner.plan` call with ``reuse_cost_map``, so it excludes
    classification and cost-map construction; it includes the search plus
    the same small post-processing for every method (snapping, waypoint
    table, great-circle comparison). Four methods are run: A*, Dijkstra, the
    genetic algorithm and the great-circle reference.

    Parameters
    ----------
    planner : RoutePlanner
        Planner with a loaded classifier.
    start, end : Coordinate
        Ports, degrees.
    resolution_deg : float, optional
        Initial lattice step; the warm-up run may refine it.

    Returns
    -------
    dict
        ``grid``, ``cost_map_build_s``, a ``results`` row per method
        (feasibility, distance, cost, nodes, runtime, detour ratio, note),
        ``astar_node_reduction`` (Dijkstra nodes / A* nodes),
        ``astar_time_ratio`` (Dijkstra time / A* time),
        ``optimality_gap_pct`` (|A* - Dijkstra| cost, expected 0),
        ``genetic_gap_pct`` (GA cost above the A* optimum) and the
        cost-map statistics.
    """
    # Build (and, if a strait is too narrow, refine) the shared cost map once.
    warmup, cost_map = planner.plan(start, end, "astar", resolution_deg)
    resolved = cost_map.spec.resolution_deg

    rows = []
    plans = {}
    for algorithm in ("astar", "dijkstra", "genetic", "dynamic", "great_circle"):
        plan, _ = planner.plan(start, end, algorithm, resolved,
                               reuse_cost_map=cost_map, auto_refine=False)
        plans[algorithm] = plan
        # The great-circle "route" always exists; it counts as infeasible
        # when any of its samples lies on land.
        crosses_land = (algorithm == "great_circle"
                        and not (plan.baseline or {}).get("is_navigable", True))
        rows.append({
            "algorithm": algorithm,
            "feasible": bool(plan.feasible and not crosses_land),
            "distance_km": round(plan.distance_km, 2),
            "total_cost": (None if crosses_land or not math.isfinite(plan.total_cost)
                           else round(plan.total_cost, 2)),
            "nodes_expanded": plan.nodes_expanded,
            "runtime_s": round(plan.runtime_s, 4),
            "detour_ratio": (None if not math.isfinite(plan.detour_ratio)
                             else round(plan.detour_ratio, 4)),
            "note": ("crosses land - not navigable" if crosses_land else plan.message),
        })

    a, d = plans["astar"], plans["dijkstra"]
    # How many times fewer nodes / less time A* needs than Dijkstra.
    speedup = round(d.nodes_expanded / a.nodes_expanded, 2) if a.nodes_expanded else None
    time_ratio = round(d.runtime_s / a.runtime_s, 2) if a.runtime_s > 0 else None
    # Both searches are optimal, so their costs must agree within rounding.
    optimality_gap = None
    if math.isfinite(a.total_cost) and math.isfinite(d.total_cost) and d.total_cost > 0:
        optimality_gap = round(100.0 * abs(a.total_cost - d.total_cost) / d.total_cost, 6)

    # The genetic algorithm is not exact: its distance from the optimum found by
    # A* is reported as a percentage. 遗传算法不保证最优，报告其与 A* 最优代价的差距。
    g = plans["genetic"]
    genetic_gap = None
    if g.feasible and math.isfinite(g.total_cost) and math.isfinite(a.total_cost) and a.total_cost > 0:
        # max(0, ...) removes a "-0.0" left by floating-point summation order.
        genetic_gap = max(0.0, round(100.0 * (g.total_cost - a.total_cost) / a.total_cost, 3))

    # Dynamic programming is exact as well: its cost must equal the A* optimum.
    # 动态规划同样是精确算法，其代价应与 A* 最优值相同。
    dp = plans["dynamic"]
    dynamic_gap = None
    if dp.feasible and math.isfinite(dp.total_cost) and math.isfinite(a.total_cost) and a.total_cost > 0:
        dynamic_gap = round(100.0 * abs(dp.total_cost - a.total_cost) / a.total_cost, 6)

    return {
        "start": list(start), "end": list(end),
        "grid": cost_map.spec.to_dict(),
        "cost_map_build_s": round(cost_map.build_seconds, 4),
        "results": rows,
        "genetic_gap_pct": genetic_gap,
        "dynamic_gap_pct": dynamic_gap,
        "astar_node_reduction": speedup,
        "astar_time_ratio": time_ratio,
        "optimality_gap_pct": optimality_gap,
        "cost_map": cost_map.statistics(),
    }
