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
6. **Coastline check** - the route is sampled every kilometre against an
   independent 1 km land mask (:mod:`.coast_check`). If it crosses land, the
   offending cells are made impassable and the search is repeated; if that
   does not help, the lattice is refined; a route that still crosses land is
   returned with a warning.
7. **Post-processing** - the path is anchored on the exact port coordinates,
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
from . import coast_check as coastline
from .astar import SearchResult, astar
from .control import (STATUS_CANCELLED, STATUS_FOUND, STATUS_NO_PATH, STATUS_STEP_LIMIT,
                      STATUS_TEXT, RunControl, SearchStopped)
from .parameters import SearchParameters
from .cost_map import (CostMap, GridSpec, build_cost_map, corridor_mask, fuse_land_mask,
                       make_grid_spec, nearest_navigable)
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
    coast_check : dict
        Result of the independent coastline check
        (:meth:`~.coast_check.CoastCheck.to_dict`): verdict, longest land
        stretch, number of automatic corrections and the land points.
    warning : str
        Non-empty when the route still crosses land after the automatic
        corrections; shown in the UI and in the PDF report.
    status : str
        ``found``, ``no_path``, ``step_limit``, ``cancelled`` or
        ``time_limit`` (:mod:`.control`).
    parameters : dict
        Parameters actually used (common ones and those of the method).
    nodes_generated : int
        Queue insertions (A*, Dijkstra), sweeps (dynamic programming) or
        generations (genetic algorithm) of the final search.
    attempts : list of dict
        One entry per lattice tried (resolution, cells, outcome, steps,
        seconds) - shows how the planner reached the result on a complex course.
    via : list of Coordinate
        Intermediate points the route had to pass through (multi-leg route).
    result_file : str
        Absolute path of the JSON result file written for this search.
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
    #: Independent coastline check (see :mod:`.coast_check`). 独立海岸线检查结果。
    coast_check: Dict = field(default_factory=dict)
    #: Warning for the navigator; empty when the route passed the check.
    warning: str = ""
    #: Outcome of the search (see :mod:`.control`). 搜索状态。
    status: str = STATUS_FOUND
    #: Parameters used (common + method). 实际使用的参数。
    parameters: Dict = field(default_factory=dict)
    nodes_generated: int = 0
    #: Lattices tried, in order. 依次尝试过的网格。
    attempts: List[Dict] = field(default_factory=list)
    #: Intermediate points of a multi-leg route. 途经点。
    via: List[Coordinate] = field(default_factory=list)
    #: Absolute path of the JSON result file (set by the web server). 结果文件路径。
    result_file: str = ""

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
        payload["via"] = [[round(a, 6), round(b, 6)] for a, b in self.via]
        payload["status_text"] = STATUS_TEXT.get(self.status, self.status)
        for key in ("total_cost", "distance_km", "great_circle_km", "detour_ratio",
                    "estimated_hours", "runtime_s"):
            value = payload[key]
            payload[key] = None if value is None or not math.isfinite(value) else round(value, 4)
        return payload


def _better_plan(a, b):
    """The better of two ``(RoutePlan, CostMap)`` results (``None`` = no result).

    A feasible route beats an infeasible one; between feasible routes, the one
    that passes the coastline check, then the one with the shorter land
    stretch, then the cheaper one.
    """
    candidates = [r for r in (a, b) if r is not None and r[0].feasible]
    if not candidates:
        return a or b

    def key(r):
        c = r[0].coast_check or {}
        return (c.get("verdict") == "crossing", c.get("max_land_run_km", 0.0) or 0.0,
                r[0].total_cost)
    return min(candidates, key=key)


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
        coast_guard: bool = True,
        params: Optional[SearchParameters] = None,
        control: Optional[RunControl] = None,
        _attempt: int = 0,
        _margin_deg: Optional[float] = None,
        _coast_initial: Optional[Dict] = None,
        _ctx: Optional[Dict] = None,
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
            Allow the finer/wider retry on failure (also switched by
            ``params.auto_refine``).
        coast_guard : bool, default True
            Check the route against the independent land mask and correct it
            automatically when it crosses land (:meth:`_coast_guard`). With
            ``reuse_cost_map`` the route is only checked, not corrected, so
            that the benchmark compares all methods on one unchanged map.
            Also switched by ``params.coast_guard``.
        params : SearchParameters, optional
            Parameters set by the user in the interface (lattice budget,
            margin, step limits, genetic-algorithm settings, time limit...).
            ``None`` keeps the planner's configuration and the defaults.
        control : RunControl, optional
            Stop flag and time limit; when absent one is created from
            ``params.time_limit_s``. When the user stops the run, or the time
            limit passes, the method returns a plan with ``feasible=False``
            and ``status`` ``"cancelled"`` / ``"time_limit"``.
        _attempt : int
            Internal: refinement depth of this call.
        _margin_deg : float, optional
            Internal: margin override set by :meth:`_refine`.
        _coast_initial : dict, optional
            Internal: coastline check of the first route when this call is a
            refinement made because of the coastline check.
        _ctx : dict, optional
            Internal: state shared by the refinement calls of one request
            (lattices tried, last cost map).

        Returns
        -------
        tuple
            ``(RoutePlan, CostMap)`` - the plan (``feasible=False`` with a
            message if no route was found) and the cost map it was computed on
            (``None`` only when the run was stopped before the first map was
            built).
        """
        top = _ctx is None
        if top:
            if params is None:
                params = SearchParameters.from_config(self.config)
            if control is None:
                control = RunControl(params.time_limit_s)
            _ctx = {"t0": time.perf_counter(), "attempts": [], "cost_map": None}
            auto_refine = auto_refine and params.auto_refine
            coast_guard = coast_guard and params.coast_guard
        if not top:
            return self._plan(start, end, algorithm, resolution_deg, speed_knots, simplify_km,
                              progress, reuse_cost_map, auto_refine, coast_guard, params,
                              control, _attempt, _margin_deg, _coast_initial, _ctx)
        try:
            plan, cost_map = self._plan(start, end, algorithm, resolution_deg, speed_knots,
                                        simplify_km, progress, reuse_cost_map, auto_refine,
                                        coast_guard, params, control, _attempt, _margin_deg,
                                        _coast_initial, _ctx)
        except SearchStopped as stop:
            # Stop button or time limit: an infeasible plan that says so.
            # 用户停止或超时：返回带状态的不可行结果。
            LOGGER.info("Search stopped: %s", stop)
            cost_map = _ctx["cost_map"] or reuse_cost_map
            if _ctx["attempts"] and _ctx["attempts"][-1].get("status") == "running":
                _ctx["attempts"][-1].update(status=stop.reason)
            plan = self._infeasible(
                start, end, algorithm, float(haversine_km(start[0], start[1], end[0], end[1])),
                cost_map, speed_knots, time.perf_counter() - _ctx["t0"],
                f"{STATUS_TEXT[stop.reason]} after {stop.elapsed_s:.1f} s"
                + (f" (time limit {control.time_limit_s:g} s)" if stop.reason != STATUS_CANCELLED else ""),
                status=stop.reason)
            if progress:
                progress("stopped", 1.0, {"stage": plan.message})
        plan.algorithm = plan.algorithm if plan.algorithm in (
            "astar", "dijkstra", "genetic", "dynamic", "great_circle") else "astar"
        plan.parameters = params.for_algorithm(plan.algorithm)
        plan.attempts = _ctx["attempts"]
        plan.runtime_s = time.perf_counter() - _ctx["t0"]
        return plan, cost_map

    def _plan(self, start, end, algorithm, resolution_deg, speed_knots, simplify_km,
              progress, reuse_cost_map, auto_refine, coast_guard, params, control,
              _attempt, _margin_deg, _coast_initial, _ctx) -> Tuple[RoutePlan, CostMap]:
        """One attempt of :meth:`plan` on one lattice (see there for the arguments)."""
        t0 = time.perf_counter()
        algorithm = algorithm.lower()
        if algorithm not in ("dijkstra", "genetic", "dynamic", "great_circle"):
            algorithm = "astar"
        # The configuration of this attempt: the user's parameters with the
        # resolution of the request (or of the refinement) and the margin.
        # 本次尝试的配置：用户参数 + 本次的分辨率与边距。
        config = params.routing_config(
            resolution_deg if resolution_deg is not None else self.config.grid_resolution_deg,
            _margin_deg)
        attempt_log = {"attempt": len(_ctx["attempts"]) + 1, "resolution_deg": config.grid_resolution_deg,
                       "margin_deg": config.margin_deg, "status": "running"}
        _ctx["attempts"].append(attempt_log)

        # ---- 1-2. lattice and cost map --------------------------------- #
        if progress:
            progress("grid", 0.0, {"stage": "Building geographic lattice"})
        cost_map = reuse_cost_map
        if cost_map is None and _ctx.get("prebuilt_map") is not None:
            # Corridor map of the two-level search, built by _refine_for_coast.
            cost_map = _ctx.pop("prebuilt_map")
            attempt_log["corridor_km"] = params.corridor_km
        elif cost_map is None:
            spec = make_grid_spec([start, end], config.grid_resolution_deg,
                                  config.margin_deg, config.max_grid_cells)
            LOGGER.info("Grid %dx%d (%d cells) at %.3f deg",
                        spec.n_rows, spec.n_cols, spec.n_cells, spec.resolution_deg)
            cost_map = build_cost_map(
                spec, self.classifier, config, progress=progress, control=control,
                land_mask_share=self._mask_share(params, always=True),
                mask_exempt=(start, end))
        _ctx["cost_map"] = cost_map
        spec = cost_map.spec
        attempt_log.update(resolution_deg=spec.resolution_deg, cells=spec.n_cells,
                           rows=spec.n_rows, cols=spec.n_cols,
                           mask_land_cells=cost_map.mask_land_cells,
                           classify_s=round(cost_map.build_seconds, 3) if reuse_cost_map is None else 0.0)

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
            attempt_log.update(status=STATUS_NO_PATH, seconds=round(time.perf_counter() - t0, 3),
                               message="no navigable cell near a point")
            retry = self._refine(start, end, algorithm, config, speed_knots, simplify_km,
                                 progress, auto_refine, _attempt, reuse_cost_map,
                                 coast_guard, _coast_initial, params, control, _ctx)
            if retry is not None:
                return retry
            return (self._infeasible(start, end, algorithm, great_circle_km, cost_map,
                                     speed_knots, time.perf_counter() - t0,
                                     "Start or destination lies on land and no navigable "
                                     "cell was found nearby"), cost_map)

        # ---- 4. search -------------------------------------------------- #
        if progress:
            progress("search", 0.0, {"stage": f"Running {algorithm.upper()} search"})

        if algorithm == "great_circle":
            result = self._great_circle_result(cost_map, start, end)
        else:
            result = self._search(algorithm, cost_map, start_cell, goal_cell, config, progress,
                                  params, control)
        attempt_log.update(status=result.status, steps=result.nodes_expanded,
                           seconds=round(time.perf_counter() - t0, 3), message=result.message)

        # ---- 5. refinement on failure ----------------------------------- #
        if not result.found:
            if result.status == STATUS_STEP_LIMIT:
                # The step limit was reached: a finer lattice would need even
                # more steps, so the user is told to raise the limit instead.
                # 达到步数上限：不再细化网格，提示用户提高上限。
                plan = self._infeasible(start, end, algorithm, great_circle_km, cost_map,
                                        speed_knots, time.perf_counter() - t0, result.message,
                                        status=STATUS_STEP_LIMIT)
                plan.nodes_expanded = result.nodes_expanded
                plan.nodes_generated = result.nodes_generated
                return plan, cost_map
            # A failed genetic run does not prove that no path exists: the lattice
            # is refined only when the exact search confirms that it is blocked.
            # 遗传算法失败不代表无路可走：仅当 A* 也确认不连通时才细化网格。
            lattice_blocked = True
            if algorithm == "genetic":
                lattice_blocked = not astar(cost_map, start_cell, goal_cell, config,
                                            control=control).found
            retry = None
            # A corridor map that has no route does not lead to a refinement
            # of its own: the caller falls back to the whole area.
            if lattice_blocked and "corridor_km" not in attempt_log:
                # Where is the obstacle? If the water around a port reaches the
                # edge of the lattice, the passage may lie outside the box.
                widen = (params.adaptive_area
                         and _ctx.get("widenings", 0) < self.MAX_WIDENINGS
                         and self._open_to_border(cost_map, start_cell, goal_cell, config))
                retry = self._refine(start, end, algorithm, config, speed_knots, simplify_km,
                                     progress, auto_refine, _attempt, reuse_cost_map,
                                     coast_guard, _coast_initial, params, control, _ctx,
                                     widen=widen)
            if retry is not None:
                return retry
            plan = self._infeasible(start, end, algorithm, great_circle_km, cost_map,
                                    speed_knots, time.perf_counter() - t0,
                                    result.message + " at the finest lattice tried "
                                    f"({spec.resolution_deg}°)", status=result.status)
            plan.nodes_expanded = result.nodes_expanded
            plan.nodes_generated = result.nodes_generated
            return plan, cost_map

        # ---- 6. independent coastline check ----------------------------- #
        # The route is optimal on the classified lattice; the land mask, which
        # plays no part in the classification, checks it against reality.
        # 用与分类器无关的陆地掩膜检查航线，穿陆时自动修正。
        search_map = cost_map
        coast = coastline.check_route(self._anchored(result, start, end), approach=True)
        corrections, blocked = 0, 0
        initial = _coast_initial or coast.to_dict()
        if (coast.verdict == "crossing" and coast_guard and reuse_cost_map is None
                and algorithm != "great_circle" and params.use_land_mask
                and cost_map.mask_land_cells == 0):
            # Correction 1 (version 1.3): fuse the land mask into this map - the
            # zones are kept, no new classification - and search again.
            # 修正一：在当前地图上融合陆地掩膜（不重新分类）后重新搜索。
            fused = fuse_land_mask(cost_map, config, params.land_mask_share, (start, end))
            attempt_log["mask_land_cells"] = fused.mask_land_cells
            if fused.mask_land_cells:
                if progress:
                    progress("coast", 0.0, {"stage": (
                        f"Route crosses land for {coast.max_land_run_km:.0f} km - fusing the "
                        f"1 km land mask ({fused.mask_land_cells:,} cells) and searching again")})
                corrections += 1
                candidate = self._search(algorithm, fused, start_cell, goal_cell, config,
                                         progress, params, control)
                if candidate.found:
                    check = coastline.check_route(self._anchored(candidate, start, end),
                                                  approach=True)
                    if check.verdict != "crossing" or check.max_land_run_km < coast.max_land_run_km:
                        result, coast = candidate, check
                        cost_map = search_map = fused
                        attempt_log["mask_fused"] = True
        if (coast.verdict == "crossing" and coast_guard and reuse_cost_map is None
                and algorithm != "great_circle"):
            result, search_map, coast, more, blocked = self._coast_guard(
                algorithm, result, cost_map, start_cell, goal_cell, config, start, end,
                coast, progress, params, control)
            corrections += more
            if coast.verdict == "crossing" and auto_refine:
                # Blocking cells did not help (or closed the passage): plan the
                # voyage again on a finer lattice, which checks itself again.
                # 封锁网格无效时，在更细的网格上重新规划（新规划同样会自检）。
                retry = self._refine_for_coast(start, end, algorithm, config, speed_knots,
                                               simplify_km, progress, _attempt, initial,
                                               params, control, _ctx, cost_map,
                                               self._anchored(result, start, end))
                if retry is not None and retry[0].feasible and (
                        retry[0].coast_check.get("verdict") != "crossing"
                        or retry[0].coast_check.get("max_land_run_km", 1e9)
                        < coast.max_land_run_km):
                    return retry
        attempt_log["coast_verdict"] = coast.verdict

        # ---- 7. post-processing ----------------------------------------- #
        coords = self._anchored(result, start, end)
        if simplify_km > 0:
            coords = douglas_peucker(coords, simplify_km)
            # The simplified line may cut a corner the lattice path went round.
            coast = coastline.check_route(coords, approach=True)

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
            status=STATUS_FOUND,
            nodes_generated=result.nodes_generated,
        )
        refined = _coast_initial is not None
        plan.coast_check = coast.to_dict(corrections=corrections, refined=refined,
                                         cells_blocked=blocked,
                                         initial=initial.get("verdict"))
        plan.coast_check["initial_max_land_run_km"] = initial.get("max_land_run_km", 0.0)
        plan.message, plan.warning = self._coast_messages(plan, algorithm)
        # The map returned is the one the route was found on (with the cells
        # blocked by the coastline check), so that the overlay matches the route.
        return plan, search_map

    # ------------------------------------------------------------------ #
    @staticmethod
    def _anchored(result: SearchResult, start: Coordinate, end: Coordinate) -> List[Coordinate]:
        """Route poly-line of a search result drawn from/to the exact ports.

        The search starts and ends at the snapped cell centres; replacing the
        first and last vertex draws the route from/to the actual quay.
        用用户输入的精确坐标替换首尾网格点。
        """
        coords = list(result.path_coords)
        coords[0], coords[-1] = start, end
        return coords

    @staticmethod
    def _search(algorithm: str, cost_map: CostMap, start_cell, goal_cell,
                config: RoutingConfig, progress: ProgressCallback,
                params: Optional[SearchParameters] = None,
                control: Optional[RunControl] = None) -> SearchResult:
        """Run one lattice search method on ``cost_map``.

        Parameters
        ----------
        algorithm : str
            ``astar``, ``dijkstra``, ``genetic`` or ``dynamic``.
        cost_map : CostMap
            Map to search on.
        start_cell, goal_cell : (row, col)
            Snapped ends.
        config : RoutingConfig
            Lattice and search parameters (A* and Dijkstra).
        progress : callable, optional
            Progress callback.
        params : SearchParameters, optional
            Step limits and genetic-algorithm settings set by the user.
        control : RunControl, optional
            Stop button and time limit.

        Returns
        -------
        SearchResult
            Result of the method.
        """
        params = params or SearchParameters()
        if algorithm == "dijkstra":
            return dijkstra(cost_map, start_cell, goal_cell, config, progress=progress,
                            max_steps=params.dijkstra_max_steps, control=control)
        if algorithm == "genetic":
            # The GA has its own configuration (population, generations, ...)
            # built from the user's parameters; the routing config does not apply.
            return genetic(cost_map, start_cell, goal_cell, config=params.genetic_config(),
                           progress=progress, control=control)
        if algorithm == "dynamic":
            # Dynamic programming: value iteration on the Bellman equation.
            # 动态规划：在贝尔曼方程上做价值迭代。
            return dynamic_programming(cost_map, start_cell, goal_cell, progress=progress,
                                       max_sweeps=params.dp_max_sweeps, control=control)
        return astar(cost_map, start_cell, goal_cell, config, progress=progress,
                     max_steps=params.astar_max_steps, control=control)

    #: Largest number of A* re-searches with blocked cells per lattice (phase 1
    #: of :meth:`_coast_guard`); A* takes a fraction of a second per round.
    #: 每个网格上"封锁后用 A* 重新搜索"的最大次数。
    MAX_COAST_CORRECTIONS: int = 8
    #: Extra rounds for the genetic algorithm on its own route (phase 2).
    MAX_METHOD_CORRECTIONS: int = 3
    #: Coarsest lattice used when the coastline check asks for refinement, deg.
    COAST_REFINE_DEG: float = 0.1

    def _coast_guard(self, algorithm, result, cost_map, start_cell, goal_cell, config,
                     start, end, coast, progress, params=None, control=None):
        """Correct a route that crosses land by blocking cells and searching again.

        封锁穿陆格子并重新搜索，直到航线不再穿越陆地。

        Phase 1 - finding the land. Every round makes impassable, on a copy of
        the map, the cells chosen by :func:`~.coast_check.cells_to_block` (the
        cells under the land samples and the vertices of the offending legs)
        and searches again with A*, which is exact and fast; the new route is
        checked, and its land is blocked in the next round. The rounds stop
        when the route no longer crosses land, when there is nothing left to
        block, when the blocked map has no route (the blocked cells closed the
        passage) or after :attr:`MAX_COAST_CORRECTIONS` rounds.

        Phase 2 - the chosen method. Dijkstra, dynamic programming and the
        genetic algorithm are then run once on the corrected map, so that the
        route shown is still the one of the method the user chose. The genetic
        algorithm is heuristic: if its route still crosses land, its own land
        is blocked for up to :attr:`MAX_METHOD_CORRECTIONS` more rounds.
        For A* the route of phase 1 is already the result.

        Parameters
        ----------
        algorithm : str
            Method chosen by the user.
        result : SearchResult
            Route that failed the check.
        cost_map : CostMap
            Map it was found on.
        start_cell, goal_cell : (row, col)
            Snapped ends (never blocked).
        config : RoutingConfig
            Search parameters.
        start, end : Coordinate
            Exact ports.
        coast : CoastCheck
            Check of ``result``.
        progress : callable, optional
            Progress callback.
        params, control : optional
            User parameters and run control, passed to every re-search.

        Returns
        -------
        tuple
            ``(result, cost_map, coast, corrections, cells_blocked)`` - the best
            route of the chosen method, the map it was found on, its check,
            the number of re-searches and the number of blocked cells.
        """
        protected = {tuple(start_cell), tuple(goal_cell)}
        state = {"map": cost_map, "corrections": 0, "blocked": 0}
        best = (result, coast, cost_map)

        def better(check):
            return (check.verdict != "crossing"
                    or check.max_land_run_km < best[1].max_land_run_km)

        def block_and_search(route, check, method):
            """Block the land of ``route`` and search again; None if impossible."""
            cells = coastline.cells_to_block(check, state["map"], route.path_cells, protected)
            if not cells:
                return None
            state["corrections"] += 1
            if progress:
                progress("coast", 0.0, {"stage": (
                    f"Route crosses land for {check.max_land_run_km:.0f} km on the coastline "
                    f"check - blocking {len(cells)} cells and searching again")})
            LOGGER.info("Coastline check: land run %.1f km, blocking %d cells (round %d, %s)",
                        check.max_land_run_km, len(cells), state["corrections"], method)
            candidate_map = coastline.block_cells(state["map"], cells)
            if method == "astar":
                # The probes are not limited by the user's step limit (they are
                # part of the correction, not of the chosen method).
                candidate = astar(candidate_map, start_cell, goal_cell, config, control=control)
            else:
                candidate = self._search(method, candidate_map, start_cell, goal_cell, config,
                                         progress, params, control)
            if not candidate.found:
                # The blocked cells closed the only passage on this lattice.
                return None
            state["map"], state["blocked"] = candidate_map, state["blocked"] + len(cells)
            return candidate, coastline.check_route(self._anchored(candidate, start, end),
                                                    approach=True)

        # ---- phase 1: A* probes find the land to block ----------------- #
        probe, probe_check = result, coast
        for _ in range(self.MAX_COAST_CORRECTIONS):
            step = block_and_search(probe, probe_check, "astar")
            if step is None:
                break
            probe, probe_check = step
            if algorithm == "astar" and better(probe_check):
                best = (probe, probe_check, state["map"])
            if probe_check.verdict != "crossing":
                break

        # ---- phase 2: the chosen method on the corrected map ----------- #
        if algorithm != "astar" and state["map"] is not cost_map:
            state["corrections"] += 1
            final = self._search(algorithm, state["map"], start_cell, goal_cell, config, progress,
                                 params, control)
            if final.found:
                check = coastline.check_route(self._anchored(final, start, end), approach=True)
                if better(check):
                    best = (final, check, state["map"])
                for _ in range(self.MAX_METHOD_CORRECTIONS):
                    if check.verdict != "crossing":
                        break
                    step = block_and_search(final, check, algorithm)
                    if step is None:
                        break
                    final, check = step
                    if better(check):
                        best = (final, check, state["map"])
        result, coast, best_map = best
        return result, best_map, coast, state["corrections"], state["blocked"]

    #: Finest step of the corridor lattice of the two-level search, degrees (~2.2 km).
    CORRIDOR_MIN_DEG: float = 0.02
    #: The corridor step is the current step divided by this factor.
    CORRIDOR_FACTOR: float = 2.5
    #: Largest full lattice (all cells, classified or not) of a corridor map.
    CORRIDOR_MAX_LATTICE: int = 8_000_000

    def _refine_for_coast(self, start, end, algorithm, config, speed_knots, simplify_km,
                          progress, attempt, initial, params=None, control=None, ctx=None,
                          cost_map: Optional[CostMap] = None, route=None):
        """Plan again on a finer lattice because the route crosses land.

        因海岸线检查不通过而加密网格重新规划。

        Two-level search (version 1.3, ``params.corridor_search``): the box
        of the current lattice is covered with a lattice
        :attr:`CORRIDOR_FACTOR` times finer (at most :attr:`CORRIDOR_MIN_DEG`),
        but only the cells within ``params.corridor_km`` of the current route
        are classified (:func:`~.cost_map.corridor_mask`); the others are
        treated as land. The fine search therefore costs about as much as the
        corridor, not as the whole box - on a long voyage a few per cent of
        the cells. Without the corridor (or when it is switched off), the
        whole box is refined: the new step is :attr:`COAST_REFINE_DEG`
        (0.1 deg) or 0.55 of the current step, not finer than
        :attr:`MIN_RESOLUTION_DEG`. The margin is kept. Returns ``None`` when
        no further refinement is allowed.
        """
        if attempt >= self._max_refinements(params):
            return None
        corridor = None
        if (params is not None and params.corridor_search and cost_map is not None and route
                and not (cost_map.region is not None and ctx is not None
                         and ctx.get("corridor_failed"))):
            corridor = self._corridor_search(start, end, algorithm, config, speed_knots,
                                             simplify_km, progress, attempt, initial, params,
                                             control, ctx, cost_map, route)
            if (corridor is not None and corridor[0].feasible
                    and corridor[0].coast_check.get("verdict") != "crossing"):
                return corridor
            if ctx is not None:
                # The corridor did not remove the crossing: refine the whole area
                # (as in version 1.2) and keep the better of the two results.
                ctx["corridor_failed"] = True
                ctx["attempts"][-1].setdefault("next", "")
        if cost_map is not None and cost_map.region is not None:
            return corridor                    # a corridor map is not refined as a whole
        whole = self._refine_whole_for_coast(start, end, algorithm, config, speed_knots,
                                             simplify_km, progress, attempt, initial, params,
                                             control, ctx)
        return _better_plan(corridor, whole)

    def _refine_whole_for_coast(self, start, end, algorithm, config, speed_knots, simplify_km,
                                progress, attempt, initial, params, control, ctx):
        """Refinement of the whole area because the route crosses land (version 1.2)."""
        current = config.grid_resolution_deg
        finer = self.COAST_REFINE_DEG if current > self.COAST_REFINE_DEG + 1e-9 else round(current * 0.55, 4)
        if finer < self.MIN_RESOLUTION_DEG:
            return None
        if progress:
            progress("coast", 0.0, {"stage": (
                f"Route still crosses land at {current}° - planning again on a {finer}° lattice")})
        LOGGER.info("Coastline check: refining %.3f -> %.3f deg", current, finer)
        if ctx is not None:
            ctx["last_action"] = "coast_refine"
            ctx["attempts"][-1]["next"] = f"lattice refined to {finer}° (coastline check)"
        return self.plan(start, end, algorithm, finer, speed_knots, simplify_km,
                         progress=progress, auto_refine=True, params=params, control=control,
                         _attempt=attempt + 1, _margin_deg=config.margin_deg,
                         _coast_initial=initial, _ctx=ctx)

    def _corridor_search(self, start, end, algorithm, config, speed_knots, simplify_km,
                         progress, attempt, initial, params, control, ctx, cost_map, route):
        """Second level of the two-level search: a fine lattice in a corridor."""
        spec = cost_map.spec
        finer = max(self.CORRIDOR_MIN_DEG, round(spec.resolution_deg / self.CORRIDOR_FACTOR, 4))
        if finer >= spec.resolution_deg - 1e-9:
            return None
        fine = GridSpec(spec.lat_min, spec.lat_max, spec.lon_min, spec.lon_max, finer)
        region = None
        while True:
            if fine.n_cells <= self.CORRIDOR_MAX_LATTICE:
                region = corridor_mask(fine, route, params.corridor_km)
                if int(region.sum()) <= params.max_grid_cells:
                    break
            # Too many cells even in the corridor: a slightly coarser step.
            finer = round(finer * 1.25, 4)
            if finer >= spec.resolution_deg - 1e-9:
                return None
            fine = GridSpec(spec.lat_min, spec.lat_max, spec.lon_min, spec.lon_max, finer)
        n_cells = int(region.sum())
        message = (f"Route still crosses land at {spec.resolution_deg}° - two-level search: "
                   f"{finer}° lattice in a ±{params.corridor_km:g} km corridor "
                   f"({n_cells:,} of {fine.n_cells:,} cells)")
        if progress:
            progress("coast", 0.0, {"stage": message})
        LOGGER.info(message)
        fine_config = params.routing_config(finer, config.margin_deg)
        corridor = build_cost_map(
            fine, self.classifier, fine_config, progress=progress, control=control,
            land_mask_share=self._mask_share(params),
            region=region, mask_exempt=(start, end))
        if ctx is not None:
            ctx["last_action"] = "corridor"
            ctx["attempts"][-1]["next"] = (f"two-level search: {finer}° lattice in a "
                                           f"±{params.corridor_km:g} km corridor")
            ctx["prebuilt_map"] = corridor
        return self.plan(start, end, algorithm, finer, speed_knots, simplify_km,
                         progress=progress, auto_refine=True, params=params, control=control,
                         _attempt=attempt + 1, _margin_deg=config.margin_deg,
                         _coast_initial=initial, _ctx=ctx)

    @staticmethod
    def _coast_messages(plan: RoutePlan, algorithm: str) -> Tuple[str, str]:
        """Outcome message and warning text derived from the coastline check."""
        c = plan.coast_check
        message = plan.message
        verdict = c.get("verdict")
        if verdict == "unchecked":
            return message, ""
        changed = c.get("initial_verdict") == "crossing" and verdict != "crossing"
        if changed:
            steps = []
            if c.get("corrections"):
                n = c["corrections"]
                steps.append(f"{n} re-search{'es' if n > 1 else ''} with "
                             f"{c['cells_blocked']} cells blocked")
            if c.get("refined"):
                steps.append(f"a finer lattice ({plan.grid.get('resolution_deg')}°)")
            message += (f"; coastline check: the first route crossed land for "
                        f"{c.get('initial_max_land_run_km', 0):.0f} km and was corrected "
                        f"automatically ({' and '.join(steps)})")
        elif verdict in ("clear", "grazing"):
            message += "; coastline check passed"
        if verdict != "crossing" or algorithm == "great_circle":
            return message, ""
        warning = (f"The route crosses land for {c['max_land_run_km']:.0f} km according to the "
                   f"independent 1 km coastline (GLOBE land mask), and the automatic correction "
                   f"could not remove it. Do not follow this course without checking the "
                   f"passage on an official nautical chart.")
        return message, warning

    # ------------------------------------------------------------------ #
    def plan_via(
        self,
        points: Sequence[Coordinate],
        algorithm: str = "astar",
        resolution_deg: Optional[float] = None,
        speed_knots: float = 14.0,
        simplify_km: float = 0.0,
        progress: ProgressCallback = None,
        params: Optional[SearchParameters] = None,
        control: Optional[RunControl] = None,
    ) -> Tuple[RoutePlan, Optional[CostMap]]:
        """Plan a route through intermediate points (a multi-leg course).

        经途经点规划多段航线：逐段规划后拼接。

        A complex course - round a peninsula, through a chosen strait, past a
        chosen side of an island group - is often planned more reliably and
        much faster as a chain of simpler legs than as one search on a huge
        lattice. Each leg ``points[i] -> points[i+1]`` is planned by
        :meth:`plan` with the same method and parameters on its own lattice
        (so every leg gets the lattice it needs); the legs are then joined.
        The Stop button and the time limit act on the whole route.

        Parameters
        ----------
        points : sequence of Coordinate
            Departure, intermediate points and destination (at least two).
        algorithm, resolution_deg, speed_knots, simplify_km, progress, params
            As in :meth:`plan`.
        control : RunControl, optional
            One control for all legs.

        Returns
        -------
        tuple
            ``(RoutePlan, CostMap)``: the joined plan (``via`` holds the
            intermediate points, ``attempts`` the lattices of all legs) and
            the cost map of the last leg. If a leg fails, the plan is
            infeasible and its message names the leg.
        """
        points = [tuple(map(float, p)) for p in points]
        if len(points) < 2:
            raise ValueError("A route needs at least two points")
        if len(points) == 2:
            return self.plan(points[0], points[1], algorithm, resolution_deg, speed_knots,
                             simplify_km, progress=progress, params=params, control=control)
        params = params or SearchParameters.from_config(self.config)
        control = control or RunControl(params.time_limit_s)
        t0 = time.perf_counter()
        n_legs = len(points) - 1
        plans: List[RoutePlan] = []
        cost_map = None
        for i, (a, b) in enumerate(zip(points[:-1], points[1:]), 1):
            def leg_progress(stage, fraction, detail, _i=i):
                if progress is None:
                    return
                detail = dict(detail or {})
                if "stage" in detail:
                    detail["stage"] = f"Leg {_i}/{n_legs}: {detail['stage']}"
                detail["leg"], detail["legs"] = _i, n_legs
                # The bar shows the progress of the whole route.
                progress(stage if stage != "done" else "leg", ((_i - 1) + fraction) / n_legs, detail)
            plan, leg_map = self.plan(a, b, algorithm, resolution_deg, speed_knots, simplify_km,
                                      progress=leg_progress, params=params, control=control)
            cost_map = leg_map or cost_map
            for att in plan.attempts:
                att["leg"] = i
            plans.append(plan)
            if not plan.feasible:
                break
        joined = self._join_legs(points, plans, algorithm, speed_knots, n_legs)
        joined.parameters = params.for_algorithm(joined.algorithm)
        joined.runtime_s = time.perf_counter() - t0
        if progress:
            progress("done", 1.0, {"stage": "Route ready" if joined.feasible else joined.message})
        return joined, cost_map

    def _join_legs(self, points, plans: List[RoutePlan], algorithm: str, speed_knots: float,
                   n_legs: int) -> RoutePlan:
        """Join the plans of the legs of :meth:`plan_via` into one plan."""
        start, end, via = points[0], points[-1], list(points[1:-1])
        gc_km = float(haversine_km(start[0], start[1], end[0], end[1]))
        attempts = [att for p in plans for att in p.attempts]
        last = plans[-1]
        if not last.feasible:
            k = len(plans)
            a, b = points[k - 1], points[k]
            plan = self._infeasible(start, end, last.algorithm, gc_km, None, speed_knots, 0.0,
                                    f"Leg {k} of {n_legs} ({a[0]:.3f}, {a[1]:.3f} -> "
                                    f"{b[0]:.3f}, {b[1]:.3f}): {last.message}",
                                    status=last.status)
            plan.grid = last.grid
            plan.nodes_expanded = sum(p.nodes_expanded for p in plans)
            plan.attempts, plan.via = attempts, via
            return plan
        waypoints: List[Coordinate] = []
        legs: List[RouteLeg] = []
        offset = 0.0
        for k, p in enumerate(plans):
            wp, lg = list(p.waypoints), list(p.legs)
            if k > 0:                                 # the junction point is shared
                wp, lg = wp[1:], lg[1:]
            for leg in lg:
                legs.append(RouteLeg(leg.latitude, leg.longitude, leg.class_code, leg.zone_cost,
                                     leg.cumulative_km + offset, leg.bearing_deg))
            waypoints += wp
            offset += p.distance_km
        if len(plans) > 1:
            # Bearing at each junction: towards the first point of the next leg.
            from .geodesy import initial_bearing_deg
            for i in range(len(legs) - 1):
                a, b = legs[i], legs[i + 1]
                legs[i].bearing_deg = float(initial_bearing_deg(a.latitude, a.longitude,
                                                                b.latitude, b.longitude))
        zone_profile = {name: 0 for name in CLASS_NAMES}
        for leg in legs:
            zone_profile[leg.class_code] = zone_profile.get(leg.class_code, 0) + 1
        distance = sum(p.distance_km for p in plans)
        speed_kmh = max(speed_knots, 0.1) * 1.852
        checks = [p.coast_check for p in plans if p.coast_check]
        order = {"unchecked": -1, "clear": 0, "grazing": 1, "crossing": 2}
        coast: Dict = {}
        if checks:
            worst = max(checks, key=lambda c: order.get(c.get("verdict"), -1))
            coast = dict(worst)
            for key in ("land_km", "samples", "samples_on_land", "corrections", "cells_blocked",
                        "approach_land_km"):
                coast[key] = round(sum(c.get(key, 0) or 0 for c in checks), 1) \
                    if isinstance(worst.get(key), float) else sum(c.get(key, 0) or 0 for c in checks)
            coast["max_land_run_km"] = max(c.get("max_land_run_km", 0) for c in checks)
            coast["initial_max_land_run_km"] = max(c.get("initial_max_land_run_km", 0) for c in checks)
            coast["refined"] = any(c.get("refined") for c in checks)
            coast["initial_verdict"] = max((c.get("initial_verdict") for c in checks),
                                           key=lambda v: order.get(v, -1))
            coast["land_points"] = [pt for c in checks for pt in c.get("land_points", [])]
        baseline = {
            "distance_km": round(gc_km, 3),
            "extra_distance_pct": round(100.0 * (distance - gc_km) / gc_km, 3) if gc_km else 0.0,
            "samples": sum((p.baseline or {}).get("samples", 0) for p in plans),
            "land_samples": sum((p.baseline or {}).get("land_samples", 0) for p in plans),
            "is_navigable": all((p.baseline or {}).get("is_navigable", True) for p in plans),
            "verdict": "Great circles of the individual legs: " + ", ".join(
                "navigable" if (p.baseline or {}).get("is_navigable", True) else "crosses land"
                for p in plans),
        }
        baseline["land_share"] = (baseline["land_samples"] / baseline["samples"]
                                  if baseline["samples"] else 0.0)
        warnings = [f"Leg {k + 1}: {p.warning}" for k, p in enumerate(plans) if p.warning]
        plan = RoutePlan(
            route_id=uuid.uuid4().hex[:16], algorithm=last.algorithm, start=start, end=end,
            waypoints=waypoints, legs=legs,
            total_cost=float(sum(p.total_cost for p in plans)),
            distance_km=distance, great_circle_km=gc_km,
            detour_ratio=distance / gc_km if gc_km > 0 else 1.0,
            estimated_hours=distance / speed_kmh,
            nodes_expanded=sum(p.nodes_expanded for p in plans),
            runtime_s=sum(p.runtime_s for p in plans),
            grid={**plans[0].grid, "legs": [p.grid for p in plans]},
            zone_profile=zone_profile,
            baseline=baseline,
            cost_map_stats=plans[0].cost_map_stats,
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            speed_knots=float(speed_knots),
            interior_land_waypoints=sum(p.interior_land_waypoints for p in plans),
            feasible=True,
            message=f"Route through {len(via)} intermediate point{'s' if len(via) > 1 else ''} "
                    f"({n_legs} legs): " + "; ".join(f"leg {k + 1}: {p.message}"
                                                    for k, p in enumerate(plans)),
            coast_check=coast,
            warning=" ".join(warnings),
            status=STATUS_FOUND,
            nodes_generated=sum(p.nodes_generated for p in plans),
            attempts=attempts,
            via=via,
        )
        return plan

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
    #: Maximum number of search-area widenings (adaptive area, version 1.3);
    #: they do not count against the refinements. 最多扩大搜索区域的次数。
    MAX_WIDENINGS: int = 2
    #: Finest lattice step the refinement may reach, degrees (~4.4 km).
    MIN_RESOLUTION_DEG: float = 0.04

    def _max_refinements(self, params: Optional[SearchParameters]) -> int:
        """Refinement limit: the user's value, else :attr:`MAX_REFINEMENTS`."""
        return self.MAX_REFINEMENTS if params is None else int(params.max_refinements)

    @staticmethod
    def _mask_share(params: SearchParameters, always: bool = False) -> float:
        """Land share used for the mask fusion of a new map (0 = no fusion).

        ``always=True`` asks for the first lattice of a request, which uses the
        mask only with ``land_mask_always``; corridor maps (a correction) use
        it whenever ``use_land_mask`` is on.
        """
        if not params.use_land_mask or (always and not params.land_mask_always):
            return 0.0
        return float(params.land_mask_share)

    @staticmethod
    def _open_to_border(cost_map: CostMap, start_cell, goal_cell, config: RoutingConfig) -> bool:
        """True if the water bodies of both the start and the goal reach the lattice edge.

        起点与终点所在的连通水域是否都延伸到网格边界（若是，连接它们的通道可能位于网格之外）。

        The navigable cells are split into connected components (8- or
        4-connectivity like the search). When the search failed, the start
        and the goal lie in different components. Only if *both* touch the
        border of the lattice can a connecting passage lie outside the
        rectangle - then the search area is widened. If one of them is
        enclosed inside the rectangle (a gulf whose mouth is closed on this
        lattice), a wider area cannot help and only a finer lattice can open
        the passage (a strait narrower than a cell). Computed with
        :func:`scipy.ndimage.label` in O(V).
        """
        from scipy import ndimage

        structure = np.ones((3, 3), dtype=bool) if config.diagonal_moves else None
        labels, _ = ndimage.label(cost_map.passable, structure=structure)
        border = set(np.unique(np.concatenate([labels[0, :], labels[-1, :],
                                               labels[:, 0], labels[:, -1]]))) - {0}
        return bool(labels[start_cell] in border and labels[goal_cell] in border)

    def _refine(self, start, end, algorithm, config, speed_knots, simplify_km,
                progress, auto_refine, attempt, reuse_cost_map, coast_guard=True,
                coast_initial=None, params=None, control=None, ctx=None, widen=False):
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
        coast_guard : bool
            Passed through to :meth:`plan`.
        coast_initial : dict, optional
            Passed through to :meth:`plan` (refinement inside a refinement
            started by the coastline check).
        params, control, ctx : optional
            User parameters, run control and request state, passed through.
        widen : bool, default False
            Adaptive search area (version 1.3): keep the cell size and widen
            the margin by half the larger side of the box of the two points
            (at least 2 degrees) - used when the water around a point reaches
            the edge of the lattice (:meth:`_open_to_border`).

        Returns
        -------
        tuple or None
            The ``(RoutePlan, CostMap)`` of the retry, or ``None`` if no
            further refinement is allowed.
        """
        if (not auto_refine or reuse_cost_map is not None
                or attempt >= self._max_refinements(params)):
            return None
        if widen:
            span = max(abs(start[0] - end[0]), abs(start[1] - end[1]))
            margin = round(config.margin_deg + max(2.0, 0.5 * span), 2)
            LOGGER.info("No path, water reaches the lattice edge - widening the margin "
                        "%.1f -> %.1f deg (attempt %d)", config.margin_deg, margin, attempt + 1)
            if progress:
                progress("refine", 0.0, {"stage": (
                    f"No passage inside the search area - widening it "
                    f"(margin {config.margin_deg:g}° -> {margin:g}°)")})
            if ctx is not None:
                ctx["last_action"] = "widen"
                ctx["widenings"] = ctx.get("widenings", 0) + 1
                ctx["attempts"][-1]["next"] = f"search area widened to a {margin:g}° margin"
            # A widening keeps the cell size and is not counted as a refinement.
            return self.plan(start, end, algorithm, config.grid_resolution_deg, speed_knots,
                             simplify_km, progress=progress, auto_refine=True,
                             coast_guard=coast_guard, params=params, control=control,
                             _attempt=attempt, _margin_deg=margin,
                             _coast_initial=coast_initial, _ctx=ctx)
        finer = round(config.grid_resolution_deg * 0.55, 4)
        if finer < self.MIN_RESOLUTION_DEG:
            return None
        LOGGER.info("No path at %.3f deg - refining to %.3f deg (attempt %d)",
                    config.grid_resolution_deg, finer, attempt + 1)
        if progress:
            progress("refine", 0.0,
                     {"stage": f"No passage at {config.grid_resolution_deg}° - "
                               f"refining the lattice to {finer}°"})
        if ctx is not None:
            ctx["last_action"] = "refine"
            ctx["attempts"][-1]["next"] = f"lattice refined to {finer}°"
        # Recursive call: builds a new cost map on the finer, wider lattice.
        return self.plan(start, end, algorithm, finer, speed_knots, simplify_km,
                         progress=progress, auto_refine=True, coast_guard=coast_guard,
                         params=params, control=control,
                         _attempt=attempt + 1, _margin_deg=config.margin_deg + 2.0,
                         _coast_initial=coast_initial, _ctx=ctx)

    def _infeasible(self, start, end, algorithm, gc_km, cost_map, speed_knots,
                    runtime, message, status: str = STATUS_NO_PATH) -> RoutePlan:
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
        status : str, default "no_path"
            Outcome code (``no_path``, ``step_limit``, ``cancelled``, ``time_limit``).

        Returns
        -------
        RoutePlan
            Infeasible plan.
        """
        return RoutePlan(
            route_id=uuid.uuid4().hex[:16], algorithm=algorithm, start=start, end=end,
            waypoints=[], legs=[], total_cost=float("inf"), distance_km=0.0,
            great_circle_km=gc_km, detour_ratio=float("nan"), estimated_hours=float("nan"),
            nodes_expanded=0, runtime_s=runtime,
            grid=cost_map.spec.to_dict() if cost_map is not None else {},
            zone_profile={name: 0 for name in CLASS_NAMES},
            baseline={}, cost_map_stats=cost_map.statistics() if cost_map is not None else {},
            created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            speed_knots=float(speed_knots), feasible=False, message=message, status=status,
        )


#: Theoretical complexity of every method, shown next to the measured work.
#: V = navigable cells, E = edges (<= 8V), S = sweeps, G = generations,
#: P = population, L = cells along a route. 各算法的理论复杂度。
COMPLEXITY = {
    "astar": {"time": "O((V + E) log V)", "memory": "O(V)",
              "work": "cells expanded (each relaxes <= 8 edges)",
              "note": "same worst case as Dijkstra; the heuristic expands only the cells "
                      "whose f = g + h is below the optimum"},
    "dijkstra": {"time": "O((V + E) log V)", "memory": "O(V)",
                 "work": "cells expanded (each relaxes <= 8 edges)",
                 "note": "expands every cell closer to the start than the goal"},
    "dynamic": {"time": "O(S * E), S <= V", "memory": "O(V)",
                "work": "Bellman updates = sweeps x cells",
                "note": "S = moves of the longest shortest path; every sweep updates all cells"},
    "genetic": {"time": "O(G * P * L)", "memory": "O(P * K)",
                "work": "fitness evaluations = generations x population",
                "note": "each evaluation rasterises a route of L cells; not exact"},
    "great_circle": {"time": "O(n)", "memory": "O(n)", "work": "samples along the line",
                     "note": "reference only: ignores land and zones"},
}

#: Methods of the comparison, in display order. 对比的方法。
BENCHMARK_METHODS = ("astar", "dijkstra", "dynamic", "genetic", "great_circle")


def _theoretical_bound(algorithm: str, cost_map: CostMap, row: Dict, params: SearchParameters) -> Optional[int]:
    """Worst-case number of elementary operations from the complexity formula."""
    v = int(cost_map.passable.sum())
    if v <= 1:
        return None
    e = 8 * v
    if algorithm in ("astar", "dijkstra"):
        return int((v + e) * math.log2(v))
    if algorithm == "dynamic":
        # S sweeps of E updates each, with S <= V (a shortest path has at most V moves).
        return int(v * e)
    if algorithm == "genetic":
        # (G + 1) populations of P routes (the initial one included) and the final
        # evaluation, each of about L cells.
        cells = max(row.get("route_cells", 1), 1)
        return int(((params.ga_generations + 1) * params.ga_population + 1) * cells)
    return None


def benchmark(planner: RoutePlanner, start: Coordinate, end: Coordinate,
              resolution_deg: Optional[float] = None,
              params: Optional[SearchParameters] = None,
              control: Optional[RunControl] = None,
              progress: ProgressCallback = None,
              methods: Sequence[str] = BENCHMARK_METHODS) -> Dict:
    """Run all algorithms on one shared cost map and compare them.

    在同一代价地图上运行全部算法并对比耗时、搜索步数、总代价与理论复杂度。
    The cost map is built once *before* timing so that the reported runtimes
    measure the search only and are therefore directly comparable.
    先构建一次代价地图再计时，使各算法的耗时可直接比较。

    Timing: each ``runtime_s`` is the ``time.perf_counter`` interval of one
    :meth:`RoutePlanner.plan` call with ``reuse_cost_map``, so it excludes
    classification and cost-map construction; it includes the search plus
    the same small post-processing for every method (snapping, waypoint
    table, great-circle comparison).

    Parameters
    ----------
    planner : RoutePlanner
        Planner with a loaded classifier.
    start, end : Coordinate
        Ports, degrees.
    resolution_deg : float, optional
        Initial lattice step; the warm-up run may refine it.
    params : SearchParameters, optional
        The user's parameters (step limits, genetic-algorithm settings...);
        ``params.time_limit_s`` applies to *each* method separately.
    control : RunControl, optional
        Stop button of the whole comparison: the method running when Stop is
        pressed ends with ``status="cancelled"``, the remaining ones are skipped.
    progress : callable, optional
        Receives ``("benchmark", fraction, {"stage", "method", ...})`` before
        every method and the progress frames of the cost-map build.
    methods : sequence of str
        Methods to compare.

    Returns
    -------
    dict
        ``grid``, ``cost_map_build_s``, a ``results`` row per method
        (status, feasibility, distance, cost, steps, runtime, detour ratio,
        note, theoretical complexity, worst-case bound, waypoints),
        ``astar_node_reduction`` (Dijkstra steps / A* steps),
        ``astar_time_ratio`` (Dijkstra time / A* time),
        ``optimality_gap_pct`` (|A* - Dijkstra| cost, expected 0),
        ``genetic_gap_pct`` (GA cost above the optimum),
        ``dynamic_gap_pct``, ``parameters``, ``total_seconds`` and the
        cost-map statistics.
    """
    t_total = time.perf_counter()
    params = params or SearchParameters.from_config(planner.config)
    control = control or RunControl()
    # Build (and, if a strait is too narrow, refine) the shared cost map once.
    # The warm-up is an ordinary A* request without a step limit: the map it
    # returns is the one its route was found on, including the corrections of
    # the coastline check (blocked cells, land mask), so that all methods are
    # compared on the corrected map, as in version 1.2.
    warm = SearchParameters.from_dict({**params.to_dict(), "astar_max_steps": 0,
                                       "time_limit_s": 0})
    if progress:
        progress("benchmark", 0.0, {"stage": "Building the shared cost map"})
    warmup, cost_map = planner.plan(start, end, "astar", resolution_deg, params=warm,
                                    control=RunControl(parent=control), progress=progress)
    rows: List[Dict] = []
    plans: Dict[str, RoutePlan] = {}
    if cost_map is None or control.cancelled:
        return {"start": list(start), "end": list(end), "grid": {}, "results": [],
                "status": STATUS_CANCELLED, "parameters": params.to_dict(),
                "total_seconds": round(time.perf_counter() - t_total, 3)}
    resolved = cost_map.spec.resolution_deg

    for k, algorithm in enumerate(methods):
        if control.cancelled:
            rows.append({"algorithm": algorithm, "status": STATUS_CANCELLED, "feasible": False,
                         "note": "skipped - the comparison was stopped",
                         "complexity": COMPLEXITY[algorithm]})
            continue
        if progress:
            progress("benchmark", k / len(methods),
                     {"stage": f"Comparison: running {algorithm} ({k + 1}/{len(methods)})",
                      "method": algorithm})
        own = RunControl(params.time_limit_s, parent=control)
        plan, _ = planner.plan(start, end, algorithm, resolved, reuse_cost_map=cost_map,
                               auto_refine=False, params=params, control=own)
        plans[algorithm] = plan
        # The great-circle "route" always exists; it counts as infeasible
        # when any of its samples lies on land.
        crosses_land = (algorithm == "great_circle"
                        and not (plan.baseline or {}).get("is_navigable", True))
        row = {
            "algorithm": algorithm,
            "status": plan.status,
            "feasible": bool(plan.feasible and not crosses_land),
            "distance_km": round(plan.distance_km, 2),
            "total_cost": (None if crosses_land or not math.isfinite(plan.total_cost)
                           else round(plan.total_cost, 2)),
            "nodes_expanded": plan.nodes_expanded,
            "nodes_generated": plan.nodes_generated,
            "step_limit": params.max_steps(algorithm) or None,
            "runtime_s": round(plan.runtime_s, 4),
            "detour_ratio": (None if not math.isfinite(plan.detour_ratio)
                             else round(plan.detour_ratio, 4)),
            "note": ("crosses land - not navigable" if crosses_land else plan.message),
            "coast_verdict": (plan.coast_check or {}).get("verdict"),
            "complexity": COMPLEXITY[algorithm],
            "route_cells": max(len(plan.waypoints), 1),
            "waypoints": [[round(a, 5), round(b, 5)] for a, b in plan.waypoints],
        }
        row["theoretical_bound_ops"] = _theoretical_bound(algorithm, cost_map, row, params)
        # Elementary operations actually made, in the unit of the bound: moves
        # relaxed (8 per expansion or Bellman update) or cells of the routes evaluated.
        row["measured_ops"] = (plan.nodes_expanded * row["route_cells"] if algorithm == "genetic"
                               else plan.nodes_expanded * 8 if algorithm != "great_circle"
                               else plan.nodes_expanded)
        rows.append(row)

    def get(name):
        return plans.get(name)

    a, d = get("astar"), get("dijkstra")
    speedup = time_ratio = optimality_gap = None
    if a is not None and d is not None:
        # How many times fewer steps / less time A* needs than Dijkstra.
        if a.feasible and d.feasible:
            speedup = round(d.nodes_expanded / a.nodes_expanded, 2) if a.nodes_expanded else None
            time_ratio = round(d.runtime_s / a.runtime_s, 2) if a.runtime_s > 0 else None
        # Both searches are optimal, so their costs must agree within rounding.
        if math.isfinite(a.total_cost) and math.isfinite(d.total_cost) and d.total_cost > 0:
            optimality_gap = round(100.0 * abs(a.total_cost - d.total_cost) / d.total_cost, 6)

    # Reference optimum: the cheapest exact result (A*, Dijkstra or DP).
    exact = [p.total_cost for name, p in plans.items()
             if name in ("astar", "dijkstra", "dynamic") and p.feasible and math.isfinite(p.total_cost)]
    optimum = min(exact) if exact else None
    # The genetic algorithm is not exact: its distance from the optimum is
    # reported as a percentage. 遗传算法不保证最优，报告其与最优代价的差距。
    g = get("genetic")
    genetic_gap = None
    if g is not None and g.feasible and math.isfinite(g.total_cost) and optimum:
        # max(0, ...) removes a "-0.0" left by floating-point summation order.
        genetic_gap = max(0.0, round(100.0 * (g.total_cost - optimum) / optimum, 3))
    # Dynamic programming is exact as well: its cost must equal the A* optimum.
    # 动态规划同样是精确算法，其代价应与 A* 最优值相同。
    dp = get("dynamic")
    dynamic_gap = None
    if (dp is not None and a is not None and dp.feasible and a.feasible
            and math.isfinite(dp.total_cost) and a.total_cost > 0):
        dynamic_gap = round(100.0 * abs(dp.total_cost - a.total_cost) / a.total_cost, 6)
    for row in rows:
        plan = plans.get(row["algorithm"])
        cost = plan.total_cost if plan is not None and row.get("total_cost") is not None else None
        row["gap_to_optimum_pct"] = (max(0.0, round(100.0 * (cost - optimum) / optimum, 3))
                                     if cost is not None and optimum else None)
    if progress:
        progress("benchmark", 1.0, {"stage": "Comparison finished"})

    return {
        "start": list(start), "end": list(end),
        "grid": cost_map.spec.to_dict(),
        "navigable_cells": int(cost_map.passable.sum()),
        "cost_map_build_s": round(cost_map.build_seconds, 4),
        "results": rows,
        "status": STATUS_CANCELLED if control.cancelled else STATUS_FOUND,
        "genetic_gap_pct": genetic_gap,
        "dynamic_gap_pct": dynamic_gap,
        "astar_node_reduction": speedup,
        "astar_time_ratio": time_ratio,
        "optimality_gap_pct": optimality_gap,
        "parameters": params.to_dict(),
        "total_seconds": round(time.perf_counter() - t_total, 3),
        "cost_map": cost_map.statistics(),
    }
