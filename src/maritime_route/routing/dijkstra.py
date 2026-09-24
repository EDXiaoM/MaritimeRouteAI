"""Dijkstra search over the navigation cost map.

迪杰斯特拉算法实现（即启发函数恒为 0 的 A*），用于与 A* 做性能对比。
Implemented separately rather than as ``astar(h=0)`` so that the thesis can
present two genuinely independent implementations and compare their node
expansion counts on identical cost maps.

Role in the pipeline
--------------------
Selected in :class:`~maritime_route.routing.planner.RoutePlanner` with
``algorithm="dijkstra"`` and run next to A* by
:func:`~maritime_route.routing.planner.benchmark`. It uses exactly the same
graph (8- or 4-neighbourhood on the lattice), the same edge cost
``d(u, v) * (w(u) + w(v)) / 2`` and the same step-length table as
:mod:`.astar`, so both searches must return the same optimal cost; the
benchmark reports their difference as ``optimality_gap_pct`` (expected 0).

Algorithm
---------
Uniform-cost search: cells are expanded in order of increasing distance from
the start, using a binary heap with lazy deletion (see :func:`dijkstra`).
Without a heuristic the search grows as a roughly circular "wave front"
around the start and typically expands several times more cells than A*.
Time complexity is O((V + E) log V) for V cells and E = 8V edges.
"""
from __future__ import annotations

import heapq
import time
from typing import Dict, List, Tuple

import numpy as np

from ..config import IMPASSABLE_COST, ROUTING, RoutingConfig
from .astar import OFFSETS_4, OFFSETS_8, Cell, ProgressCallback, SearchResult, _cell_distance_table, _path_km, _reconstruct
from .cost_map import CostMap


def dijkstra(
    cost_map: CostMap,
    start: Cell,
    goal: Cell,
    config: RoutingConfig = ROUTING,
    progress: ProgressCallback = None,
    progress_every: int = 20000,
) -> SearchResult:
    """Uniform-cost search - guarantees the optimum without a heuristic.

    The priority queue holds ``(distance, cell)``. As in :func:`astar`,
    ``heapq`` has no decrease-key, so an improved distance is pushed as a new
    entry and outdated entries are skipped when popped for an already visited
    cell (lazy deletion). Because all edge costs are positive, the first time
    a cell is popped its distance is final; the search ends when the goal is
    popped.
    一致代价搜索：按到起点的距离从小到大扩展节点，采用惰性删除的二叉堆。

    Parameters
    ----------
    cost_map : CostMap
        Lattice with traversal weights.
    start, goal : Cell
        ``(row, col)`` of departure and destination; both must be navigable.
    config : RoutingConfig
        Uses ``diagonal_moves`` to choose the 8- or 4-neighbourhood.
    progress : callable, optional
        Called every ``progress_every`` expansions with
        ``("dijkstra", fraction, {"expanded", "frontier"})``.
    progress_every : int, default 20000
        Expansion interval between progress reports.

    Returns
    -------
    SearchResult
        ``found=True`` with the optimal path, or ``found=False`` with a
        message when start/goal is on land or no path exists.
    """
    t0 = time.perf_counter()
    spec = cost_map.spec
    weights = cost_map.cost
    offsets = OFFSETS_8 if config.diagonal_moves else OFFSETS_4
    # Step length (km) of every move, per source row - shared with A*.
    steps = _cell_distance_table(cost_map)
    n_rows, n_cols = spec.n_rows, spec.n_cols

    if not cost_map.passable[start] or not cost_map.passable[goal]:
        return SearchResult(found=False, algorithm="dijkstra",
                            runtime_s=time.perf_counter() - t0,
                            message="Start or goal cell is not navigable")

    # distance: best known cost from the start; visited: cells with final cost.
    # 距离表与已确定集合。
    distance: Dict[Cell, float] = {start: 0.0}
    came_from: Dict[Cell, Cell] = {}
    visited: set = set()
    heap: List[Tuple[float, Cell]] = [(0.0, start)]
    expanded, generated = 0, 1
    # Without a heuristic there is no distance-to-goal estimate, so progress is
    # reported as the share of all lattice cells expanded so far (an upper bound
    # of the work). 无启发信息，进度按已扩展单元占总单元数的比例估计。
    budget = float(n_rows * n_cols)

    while heap:
        d_current, current = heapq.heappop(heap)
        if current in visited:
            continue                        # stale heap entry (lazy deletion)
        visited.add(current)
        expanded += 1

        if current == goal:
            cells = _reconstruct(came_from, goal)
            coords = [spec.cell_to_coord(r, c) for r, c in cells]
            return SearchResult(
                found=True, path_cells=cells, path_coords=coords,
                total_cost=d_current, distance_km=_path_km(coords),
                nodes_expanded=expanded, nodes_generated=generated,
                runtime_s=time.perf_counter() - t0, algorithm="dijkstra",
                message="Optimal path found",
            )

        if progress is not None and expanded % progress_every == 0:
            progress("dijkstra", min(expanded / budget, 0.995),
                     {"expanded": expanded, "frontier": len(heap)})

        row, col = current
        w_current = float(weights[row, col])
        for dr, dc in offsets:
            r, c = row + dr, col + dc
            if not (0 <= r < n_rows and 0 <= c < n_cols):
                continue                    # outside the lattice
            neighbour = (r, c)
            if neighbour in visited:
                continue
            w_next = float(weights[r, c])
            if w_next >= IMPASSABLE_COST:
                continue                    # land cell
            # Edge relaxation with cost d(u, v) * (w(u) + w(v)) / 2.
            # 松弛操作：边代价 = 步长 × 两端权重平均值。
            step_km = float(steps[(dr, dc)][row])
            tentative = d_current + step_km * 0.5 * (w_current + w_next)
            if tentative < distance.get(neighbour, float("inf")):
                distance[neighbour] = tentative
                came_from[neighbour] = current
                heapq.heappush(heap, (tentative, neighbour))
                generated += 1

    # Every reachable cell was visited and the goal was not among them.
    return SearchResult(found=False, algorithm="dijkstra", nodes_expanded=expanded,
                        nodes_generated=generated, runtime_s=time.perf_counter() - t0,
                        message="No navigable path exists between the given points")
