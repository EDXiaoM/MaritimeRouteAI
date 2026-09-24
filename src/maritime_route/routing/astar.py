"""A* search over the navigation cost map.

A* 寻路算法实现。
Cost model: moving from cell *u* to neighbour *v* costs

    d(u, v) * (w(u) + w(v)) / 2

where ``d`` is the great-circle distance in kilometres and ``w`` the zone
weight produced by the neural classifier. The heuristic is the great-circle
distance to the goal multiplied by the minimum zone weight, which keeps it
admissible (never over-estimates) and therefore keeps A* optimal.

Role in the pipeline
--------------------
A* is the default search of :class:`~maritime_route.routing.planner.RoutePlanner`.
It receives a :class:`~maritime_route.routing.cost_map.CostMap` and two
navigable lattice cells and returns a :class:`SearchResult`. The module also
holds the pieces shared with :mod:`.dijkstra` and :mod:`.genetic`: the
neighbourhood offsets, the step-length table, path reconstruction and the
path-length helper.

Graph
-----
Nodes are the lattice cells; each cell is connected to its 8 neighbours
(4 when ``RoutingConfig.diagonal_moves`` is off). Land cells
(``cost >= IMPASSABLE_COST``) are not entered. The edge cost is the length of
the move multiplied by the mean weight of its two end cells - the trapezoidal
rule for the integral of the weight along the segment. The unit of cost is
therefore "weighted kilometres": a route entirely in open sea (w = 1) costs
exactly its length in km.

Admissibility of the heuristic
------------------------------
Let ``w_min`` be the smallest weight of any navigable cell. Every edge costs
at least ``d(u, v) * w_min``, and the sum of the edge lengths of any path from
``n`` to the goal is at least the great-circle distance ``d_gc(n, goal)``
(triangle inequality on the sphere). Hence

    h(n) = w_min * d_gc(n, goal)

never exceeds the true remaining cost, i.e. it is admissible. It is also
consistent (``h(u) <= c(u, v) + h(v)``), so a cell never has to be re-opened
after it has been closed, and the first time the goal is popped its cost is
optimal. ``RoutingConfig.heuristic_weight`` > 1 would turn this into weighted
A*, which is faster but no longer guaranteed optimal; the default is 1.

启发函数 h(n) = 最小权重 × 到终点的大圆距离，既可采纳又一致，保证 A* 得到最优解。
"""
from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import EARTH_RADIUS_KM, IMPASSABLE_COST, ROUTING, RoutingConfig
from .cost_map import CostMap
from .geodesy import haversine_km

#: A lattice cell as ``(row, col)``. 网格单元 (行, 列)。
Cell = Tuple[int, int]
#: Optional callback ``progress(stage, fraction, details)`` for the web client.
ProgressCallback = Optional[Callable[[str, float, Dict], None]]

#: 8-connected neighbourhood offsets. 八邻域偏移量。
#: ``(drow, dcol)``: the four axis moves (N, S, W, E) first, then the four
#: diagonals. A diagonal move is about sqrt(2) times longer than an axis move
#: (exactly so only where one degree of longitude equals one of latitude).
OFFSETS_8: Tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1))
#: 4-connected neighbourhood (axis moves only), used when diagonal moves are off.
#: 四邻域偏移量（不允许斜向移动时使用）。
OFFSETS_4: Tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass
class SearchResult:
    """Outcome of a graph search. 搜索结果。

    Returned by :func:`astar`, :func:`~maritime_route.routing.dijkstra.dijkstra`,
    :func:`~maritime_route.routing.genetic.genetic` and the planner's
    great-circle wrapper, so that the planner treats every method alike.

    Attributes
    ----------
    found : bool
        True if a navigable path was found.
    path_cells : list of Cell
        Lattice cells from start to goal (empty if not found; empty for the
        great-circle reference, which is not a lattice path).
    path_coords : list of (lat, lon)
        Coordinates of the path vertices, degrees.
    total_cost : float
        Sum of edge costs, weighted km; ``inf`` when no path was found.
    distance_km : float
        Geometric length of ``path_coords``, km.
    nodes_expanded : int
        Cells removed from the priority queue and expanded (A*, Dijkstra) or
        number of fitness evaluations (genetic algorithm).
    nodes_generated : int
        Queue insertions (A*, Dijkstra) or generations run (genetic algorithm).
    runtime_s : float
        Wall-clock time of the search, seconds.
    algorithm : str
        ``"astar"``, ``"dijkstra"``, ``"genetic"`` or ``"great_circle"``.
    message : str
        Human-readable outcome shown in the UI.
    history : list of float
        Best fitness after each generation (genetic algorithm only).
    """

    found: bool
    path_cells: List[Cell] = field(default_factory=list)
    path_coords: List[Tuple[float, float]] = field(default_factory=list)
    total_cost: float = float("inf")
    distance_km: float = 0.0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    runtime_s: float = 0.0
    algorithm: str = "astar"
    message: str = ""
    #: Best fitness per generation (genetic algorithm only). 每代最优适应度（仅遗传算法）。
    history: List[float] = field(default_factory=list)
    #: Value function V (dynamic programming only), shape (n_rows, n_cols).
    #: 价值函数（仅动态规划）。
    value: Optional[np.ndarray] = None

    def to_dict(self) -> Dict:
        """JSON-serialisable summary without the path itself.

        An infinite ``total_cost`` is written as ``None`` because JSON has no
        representation for infinity.

        Returns
        -------
        dict
            Scalar fields rounded for display plus ``n_waypoints``.
        """
        return {
            "found": self.found,
            "algorithm": self.algorithm,
            "total_cost": None if not math.isfinite(self.total_cost) else round(self.total_cost, 3),
            "distance_km": round(self.distance_km, 3),
            "nodes_expanded": self.nodes_expanded,
            "nodes_generated": self.nodes_generated,
            "runtime_s": round(self.runtime_s, 4),
            "n_waypoints": len(self.path_coords),
            "message": self.message,
        }


def _cell_distance_table(cost_map: CostMap) -> Dict[Cell, np.ndarray]:
    """Pre-compute per-row step lengths for each neighbour offset.

    预计算每一行、每个方向的步长(km)，避免在搜索内部反复做三角函数运算。

    On a regular latitude/longitude lattice the length of a move depends only
    on the latitude of the starting row and on the direction, not on the
    column: an east-west move shrinks with ``cos(lat)``, a north-south move is
    constant. The table therefore stores, for every offset ``(dr, dc)``, a
    vector of ``n_rows`` lengths computed once with :func:`haversine_km`
    (longitude difference ``dc * res`` measured from longitude 0, which is
    equivalent because only the difference matters).

    Parameters
    ----------
    cost_map : CostMap
        Supplies the lattice geometry.

    Returns
    -------
    dict
        ``{(dr, dc): ndarray of shape (n_rows,)}`` - length in km of the move
        ``(dr, dc)`` starting from each row. All 8 offsets are included even
        when only 4 are used.
    """
    spec = cost_map.spec
    table: Dict[Cell, np.ndarray] = {}
    rows = np.arange(spec.n_rows)
    lat = spec.lat_min + rows * spec.resolution_deg
    for dr, dc in OFFSETS_8:
        # Latitude of the target row; the clip keeps it valid for moves that
        # would leave the lattice (they are rejected by the bounds check anyway).
        lat2 = np.clip(lat + dr * spec.resolution_deg, -90.0, 90.0)
        table[(dr, dc)] = haversine_km(lat, 0.0, lat2, dc * spec.resolution_deg).astype(np.float64)
    return table


def _reconstruct(came_from: Dict[Cell, Cell], goal: Cell) -> List[Cell]:
    """Rebuild the path by following predecessor links back from the goal.

    ``came_from`` maps each reached cell to the cell it was reached from on
    the best known path. The start cell has no entry, which ends the walk.
    沿前驱指针从终点回溯到起点，再反转得到路径。

    Parameters
    ----------
    came_from : dict
        Predecessor of every reached cell.
    goal : Cell
        Last cell of the path.

    Returns
    -------
    list of Cell
        Cells in order from start to goal.
    """
    path = [goal]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def astar(
    cost_map: CostMap,
    start: Cell,
    goal: Cell,
    config: RoutingConfig = ROUTING,
    progress: ProgressCallback = None,
    progress_every: int = 20000,
) -> SearchResult:
    """A* shortest-cost path between two lattice cells.

    Classical A* with a binary heap and lazy deletion:

    * The open list is a heap of ``(f, g, cell)`` with ``f = g + h``. Python's
      ``heapq`` has no decrease-key operation, so when a cheaper path to a
      cell is found a *new* entry is pushed and the old one stays in the heap.
    * When an entry is popped for a cell that is already in the closed set
      it is a stale duplicate and is skipped ("lazy deletion").
    * Because the heuristic is consistent, the first pop of a cell is its
      final, optimal cost; the cell is then closed.
    * The search stops when the goal is popped.

    With ``h = 0`` this would be Dijkstra's algorithm; the heuristic only
    changes the order of expansion, steering the search towards the goal and
    expanding far fewer cells (compared in the benchmark).

    Parameters
    ----------
    cost_map : CostMap
        Lattice with traversal weights.
    start, goal : Cell
        ``(row, col)`` of departure and destination; both must be navigable
        (the planner snaps them beforehand).
    config : RoutingConfig
        Uses ``diagonal_moves`` (8- or 4-neighbourhood) and
        ``heuristic_weight`` (1.0 = admissible A*).
    progress : callable, optional
        Called every ``progress_every`` expansions with
        ``("astar", fraction, {"expanded", "frontier"})``.
    progress_every : int, default 20000
        Expansion interval between progress reports.

    Returns
    -------
    SearchResult
        ``found=True`` with the optimal path, cost and counters, or
        ``found=False`` with a message when start/goal is on land or the goal
        is unreachable.
    """
    t0 = time.perf_counter()
    spec = cost_map.spec
    weights = cost_map.cost
    offsets = OFFSETS_8 if config.diagonal_moves else OFFSETS_4
    steps = _cell_distance_table(cost_map)
    n_rows, n_cols = spec.n_rows, spec.n_cols

    if not cost_map.passable[start] or not cost_map.passable[goal]:
        return SearchResult(found=False, algorithm="astar",
                            runtime_s=time.perf_counter() - t0,
                            message="Start or goal cell is not navigable")

    # Smallest weight of any navigable cell: the factor that keeps h admissible.
    # 可航行单元的最小权重，用于保证启发函数可采纳。
    min_weight = float(np.min(weights[cost_map.passable])) if cost_map.passable.any() else 1.0
    goal_lat, goal_lon = spec.cell_to_coord(*goal)
    h_weight = float(config.heuristic_weight)

    # The heuristic is evaluated once per generated node, so it is written with
    # the math module and pre-computed trigonometry: a NumPy call per node costs
    # roughly an order of magnitude more and dominates the whole search.
    # 启发函数每生成一个节点都会调用一次，这里改用 math 并预计算三角函数，
    # 否则 NumPy 的标量调用开销会成为搜索的瓶颈。
    # h = h_weight * w_min * 2R * asin(sqrt(a)) - the haversine formula with
    # the constant factor folded into `scale`.
    scale = h_weight * min_weight * 2.0 * EARTH_RADIUS_KM
    goal_lat_rad = math.radians(goal_lat)
    goal_lon_rad = math.radians(goal_lon)
    cos_goal_lat = math.cos(goal_lat_rad)
    lat0, res_rad = math.radians(spec.lat_min), math.radians(spec.resolution_deg)
    lon0 = math.radians(spec.lon_min)
    # Per-row latitude trigonometry is identical for every column.
    # row_sin[r] = sin^2(dlat / 2) and row_cos[r] = cos(lat_r) * cos(lat_goal):
    # the two latitude-only terms of the haversine formula.
    row_lat_rad = [lat0 + r * res_rad for r in range(n_rows)]
    row_sin = [math.sin((lat - goal_lat_rad) * 0.5) ** 2 for lat in row_lat_rad]
    row_cos = [math.cos(lat) * cos_goal_lat for lat in row_lat_rad]

    def heuristic(cell: Cell) -> float:
        """Admissible estimate of the remaining cost from ``cell`` to the goal.

        ``w_min * great-circle distance(cell, goal)`` (times
        ``heuristic_weight``), evaluated with the pre-computed per-row terms
        so that only one ``sin`` depends on the column.

        Parameters
        ----------
        cell : Cell
            ``(row, col)`` of the cell.

        Returns
        -------
        float
            Lower bound of the remaining cost, weighted km (0 at the goal).
        """
        row, col = cell
        dlon_half = (lon0 + col * res_rad - goal_lon_rad) * 0.5
        a = row_sin[row] + row_cos[row] * math.sin(dlon_half) ** 2
        if a <= 0.0:
            return 0.0
        # Clamp a <= 1 against rounding before asin(sqrt(a)).
        return scale * math.asin(math.sqrt(a) if a < 1.0 else 1.0)

    # g_score: best known cost from the start; came_from: predecessor links;
    # closed: cells whose optimal cost is final. 已知最优代价 / 前驱 / 关闭集。
    g_score: Dict[Cell, float] = {start: 0.0}
    came_from: Dict[Cell, Cell] = {}
    closed: set = set()
    # Heap entries are (f, g, cell); ties on f are broken by the smaller g.
    heap: List[Tuple[float, float, Cell]] = [(heuristic(start), 0.0, start)]
    expanded = 0
    generated = 1
    # Admissible heuristic value at the start - used for the progress estimate.
    h0 = max(heuristic(start), 1e-6)

    while heap:
        f_score, g_current, current = heapq.heappop(heap)
        if current in closed:
            # Stale duplicate left by a later, cheaper push (lazy deletion).
            # 过期的重复条目（惰性删除），直接跳过。
            continue
        closed.add(current)
        expanded += 1

        if current == goal:
            cells = _reconstruct(came_from, goal)
            coords = [spec.cell_to_coord(r, c) for r, c in cells]
            return SearchResult(
                found=True, path_cells=cells, path_coords=coords,
                total_cost=g_current,
                distance_km=_path_km(coords),
                nodes_expanded=expanded, nodes_generated=generated,
                runtime_s=time.perf_counter() - t0, algorithm="astar",
                message="Optimal path found",
            )

        if progress is not None and expanded % progress_every == 0:
            # Progress estimate: share of the initial heuristic distance
            # already covered; capped below 1 until the goal is reached.
            remaining = heuristic(current)
            done = float(np.clip(1.0 - remaining / h0, 0.0, 0.995))
            progress("astar", done, {"expanded": expanded, "frontier": len(heap)})

        row, col = current
        w_current = float(weights[row, col])
        for dr, dc in offsets:
            r, c = row + dr, col + dc
            if not (0 <= r < n_rows and 0 <= c < n_cols):
                continue                    # outside the lattice
            neighbour = (r, c)
            if neighbour in closed:
                continue                    # already final
            w_next = float(weights[r, c])
            if w_next >= IMPASSABLE_COST:
                continue                    # land cell - never entered
            # Edge cost d(u, v) * (w(u) + w(v)) / 2, d looked up by source row.
            # 边代价 = 步长 × 两端权重的平均值。
            step_km = float(steps[(dr, dc)][row])
            tentative = g_current + step_km * 0.5 * (w_current + w_next)
            if tentative < g_score.get(neighbour, float("inf")):
                # Cheaper path to the neighbour: record it and push a new entry
                # (the old entry, if any, becomes stale).
                g_score[neighbour] = tentative
                came_from[neighbour] = current
                heapq.heappush(heap, (tentative + heuristic(neighbour), tentative, neighbour))
                generated += 1

    # Heap exhausted without reaching the goal: the two cells lie in
    # different connected water bodies of this lattice.
    return SearchResult(found=False, algorithm="astar", nodes_expanded=expanded,
                        nodes_generated=generated, runtime_s=time.perf_counter() - t0,
                        message="No navigable path exists between the given points")


def _path_km(coords: List[Tuple[float, float]]) -> float:
    """Great-circle length of a coordinate path in kilometres.

    Same computation as :func:`~maritime_route.routing.geodesy.path_length_km`;
    kept here as the shared helper imported by the search modules.

    Parameters
    ----------
    coords : list of (lat, lon)
        Path vertices, degrees.

    Returns
    -------
    float
        Length in km; 0.0 for fewer than two vertices.
    """
    if len(coords) < 2:
        return 0.0
    arr = np.asarray(coords, dtype=float)
    return float(np.sum(haversine_km(arr[:-1, 0], arr[:-1, 1], arr[1:, 0], arr[1:, 1])))
