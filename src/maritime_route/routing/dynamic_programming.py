"""Dynamic-programming route search: value iteration on the Bellman equation.

This module is the fourth search method of the application. It follows the
suggestion of the project supervisor to solve the routing problem as a problem
of optimal control by dynamic programming (R. Bellman), and to compare the
evaluation function it is built on - the *Bellman function*, also called the
*value function* - with the *fitness function* of the genetic algorithm.

The discrete step and its cost
------------------------------
The state of the vessel is a lattice cell ``v``; a control is one of the eight
moves to a neighbouring cell ``u``. A move costs exactly what it costs in A*,
Dijkstra and the genetic algorithm::

    c(v, u) = d_gc(v, u) * (w(v) + w(u)) / 2

where ``d_gc`` is the great-circle length of the step (km) and ``w`` the zone
weight predicted by the neural network. Moves into land are not allowed.

The Bellman function
--------------------
The value ``V(v)`` of a cell is the minimum cost of reaching the goal from it.
It satisfies the Bellman equation::

    V(goal) = 0
    V(v)    = min over neighbours u of  [ c(v, u) + V(u) ]

The equation is solved by *value iteration*: ``V`` starts at 0 in the goal and
at +infinity everywhere else, and every sweep replaces each cell's value by the
right-hand side of the equation, computed for all cells at once with NumPy
array shifts. After ``k`` sweeps ``V(v)`` is the cheapest cost over routes of at
most ``k`` moves, so the iteration stops as soon as a sweep changes nothing;
the values are then exact.

The optimal route is read off the value function by the *greedy policy*: from
the departure, repeatedly move to the neighbour ``u`` that minimises
``c(v, u) + V(u)``, until the goal is reached. Its cost equals ``V(start)`` and
is the same optimum that A* and Dijkstra find - which the tests assert.

Unlike A*, the method computes the cost-to-go of *every* cell, not only of the
cells near the optimal route. That costs more work for one query, but once
``V`` is known the optimal route to the same destination from any other
departure is obtained immediately by the greedy policy.

动态规划寻路：在栅格上用价值迭代求解贝尔曼方程 V(v) = min_u [c(v,u) + V(u)]，
V(goal) = 0；再从起点按贪心策略沿价值函数下降即得最优航线。离散步代价与 A*、
Dijkstra、遗传算法完全相同，因此最优代价应与 A* 一致。
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import IMPASSABLE_COST
from .astar import OFFSETS_8, Cell, ProgressCallback, SearchResult, _cell_distance_table, _path_km
from .cost_map import CostMap


def _shift(values: np.ndarray, dr: int, dc: int, fill: float) -> np.ndarray:
    """Array whose element ``[r, c]`` is ``values[r + dr, c + dc]``.

    Positions whose neighbour lies outside the lattice receive ``fill``.
    Used to read, for every cell at once, the value of its neighbour in the
    direction ``(dr, dc)``.

    Parameters
    ----------
    values : numpy.ndarray
        2-D array of shape ``(n_rows, n_cols)``.
    dr, dc : int
        Row and column offset of the neighbour, each in ``{-1, 0, 1}``.
    fill : float
        Value used where the neighbour does not exist.

    Returns
    -------
    numpy.ndarray
        Array of the same shape as ``values``.

    数组平移：结果的 [r, c] 元素为 values[r+dr, c+dc]，越界处填 fill。
    """
    out = np.full_like(values, fill)
    n_rows, n_cols = values.shape
    r0, r1 = max(0, -dr), n_rows - max(0, dr)
    c0, c1 = max(0, -dc), n_cols - max(0, dc)
    out[r0:r1, c0:c1] = values[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


def step_costs(cost_map: CostMap) -> Dict[Cell, np.ndarray]:
    """Cost of every discrete step, for all cells and all eight directions.

    For each offset ``(dr, dc)`` the returned array holds, in ``[r, c]``, the
    cost ``c(v, u)`` of the move from cell ``v = (r, c)`` to ``u = (r + dr,
    c + dc)``: the great-circle length of the step multiplied by the mean of
    the two zone weights. The cost is ``+inf`` where the move leaves the
    lattice or where either cell is impassable. This is the "evaluation
    function of the discrete step" on which the Bellman equation is built.

    Parameters
    ----------
    cost_map : CostMap
        Classified lattice with zone weights.

    Returns
    -------
    dict
        ``{(dr, dc): ndarray (n_rows, n_cols) of float64}``.

    离散步代价：对每个方向给出从每个格子出发走一步的代价，越界或陆地为 +inf。
    """
    weights = cost_map.cost.astype(np.float64)
    blocked = weights >= IMPASSABLE_COST
    weights = np.where(blocked, np.inf, weights)
    lengths = _cell_distance_table(cost_map)          # km per row and direction
    n_rows, n_cols = weights.shape
    costs: Dict[Cell, np.ndarray] = {}
    for dr, dc in OFFSETS_8:
        w_next = _shift(weights, dr, dc, np.inf)       # weight of the target cell
        step_km = lengths[(dr, dc)][:, None]           # (n_rows, 1), broadcast over columns
        costs[(dr, dc)] = step_km * 0.5 * (weights + w_next)
    return costs


def value_iteration(
    cost_map: CostMap,
    goal: Cell,
    progress: ProgressCallback = None,
    max_sweeps: Optional[int] = None,
) -> Tuple[np.ndarray, int, Dict[Cell, np.ndarray]]:
    """Solve the Bellman equation for the cost-to-go of every cell.

    Parameters
    ----------
    cost_map : CostMap
        Classified lattice with zone weights.
    goal : tuple of int
        Destination cell (row, col); its value is fixed at zero.
    progress : callable, optional
        Receives ``("dynamic", fraction, {"sweep": k, "updated": m})`` after
        every tenth sweep; ``m`` is the number of cells whose value changed.
    max_sweeps : int, optional
        Safety limit; by default the number of cells, which is never reached
        because a shortest path cannot have more moves than there are cells.

    Returns
    -------
    (numpy.ndarray, int, dict)
        The value function ``V`` (``+inf`` for cells from which the goal cannot
        be reached), the number of sweeps performed, and the step costs used.

    价值迭代：V 初值在终点为 0、其余为 +inf；每次扫描用贝尔曼方程右端更新所有格子，
    直到一次扫描不再改变任何值为止，此时 V 即精确的剩余代价。
    """
    costs = step_costs(cost_map)
    n_rows, n_cols = cost_map.cost.shape
    value = np.full((n_rows, n_cols), np.inf)
    value[goal] = 0.0
    limit = max_sweeps or n_rows * n_cols
    sweeps = 0
    while sweeps < limit:
        sweeps += 1
        # Right-hand side of the Bellman equation for all cells at once:
        # min over the eight moves of (cost of the step + value of the target).
        # 贝尔曼方程右端：对八个方向取 (步代价 + 目标格价值) 的最小值。
        best = value.copy()
        for (dr, dc), step in costs.items():
            np.minimum(best, step + _shift(value, dr, dc, np.inf), out=best)
        best[goal] = 0.0
        updated = int(np.count_nonzero(best < value - 1e-9))
        value = best
        if progress is not None and sweeps % 10 == 0:
            # The number of sweeps is not known in advance; the fraction of the
            # lattice already reached is used as the progress estimate.
            reached = float(np.isfinite(value).mean())
            progress("dynamic", min(reached, 0.99), {"sweep": sweeps, "updated": updated})
        if updated == 0:
            break
    return value, sweeps, costs


def greedy_policy(value: np.ndarray, costs: Dict[Cell, np.ndarray],
                  start: Cell, goal: Cell) -> List[Cell]:
    """Follow the value function from ``start`` down to ``goal``.

    At every cell the move that minimises ``c(v, u) + V(u)`` is taken. With an
    exact value function this move always leads to a cell of strictly lower
    value, so the walk ends in the goal after as many moves as the optimal
    route has.

    Parameters
    ----------
    value : numpy.ndarray
        Converged value function from :func:`value_iteration`.
    costs : dict
        Step costs from :func:`step_costs`.
    start, goal : tuple of int
        Departure and destination cells.

    Returns
    -------
    list of tuple
        The cells of the optimal route, from ``start`` to ``goal`` inclusive;
        empty if the goal cannot be reached from ``start``.

    贪心策略：每一步选择使 c(v,u) + V(u) 最小的相邻格子，直到到达终点。
    """
    if not np.isfinite(value[start]):
        return []
    n_rows, n_cols = value.shape
    path = [start]
    current = start
    for _ in range(n_rows * n_cols):                   # guard against a cycle
        if current == goal:
            return path
        r, c = current
        best_cell, best_total = None, np.inf
        for (dr, dc), step in costs.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < n_rows and 0 <= nc < n_cols:
                total = step[r, c] + value[nr, nc]
                if total < best_total:
                    best_cell, best_total = (nr, nc), total
        if best_cell is None or not np.isfinite(best_total):
            return []
        path.append(best_cell)
        current = best_cell
    return []


def dynamic_programming(
    cost_map: CostMap,
    start: Cell,
    goal: Cell,
    progress: ProgressCallback = None,
) -> SearchResult:
    """Optimal route by dynamic programming (value iteration + greedy policy).

    Parameters
    ----------
    cost_map : CostMap
        Classified lattice with traversal costs (shared with the other methods).
    start, goal : tuple of int
        Navigable departure and destination cells (row, col).
    progress : callable, optional
        Receives progress frames during value iteration, see
        :func:`value_iteration`.

    Returns
    -------
    SearchResult
        ``total_cost`` is ``V(start)``; ``nodes_expanded`` is the number of
        Bellman updates performed (sweeps × navigable cells), so that the work
        can be compared with the node counts of A* and Dijkstra;
        ``nodes_generated`` is the number of sweeps; ``value`` holds the value
        function for plotting.

    动态规划求最优航线：先价值迭代求 V，再按贪心策略提取路径。
    """
    t0 = time.perf_counter()
    spec = cost_map.spec
    passable = cost_map.passable
    if not passable[start] or not passable[goal]:
        return SearchResult(found=False, algorithm="dynamic",
                            runtime_s=time.perf_counter() - t0,
                            message="Start or goal cell is not navigable")

    value, sweeps, costs = value_iteration(cost_map, goal, progress=progress)
    cells = greedy_policy(value, costs, start, goal)
    updates = sweeps * int(passable.sum())
    if not cells:
        return SearchResult(found=False, algorithm="dynamic", nodes_expanded=updates,
                            nodes_generated=sweeps, runtime_s=time.perf_counter() - t0,
                            value=value,
                            message="No navigable path exists between the given points")
    coords = [spec.cell_to_coord(r, c) for r, c in cells]
    return SearchResult(
        found=True, path_cells=cells, path_coords=coords,
        total_cost=float(value[start]), distance_km=_path_km(coords),
        nodes_expanded=updates, nodes_generated=sweeps,
        runtime_s=time.perf_counter() - t0, algorithm="dynamic", value=value,
        message=f"Bellman equation solved in {sweeps} sweeps",
    )
