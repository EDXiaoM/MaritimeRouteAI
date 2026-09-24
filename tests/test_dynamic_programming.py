"""Tests of the dynamic-programming route search (value iteration).

Dynamic programming is exact, so its route must cost exactly what the A*
optimum costs; the value function must satisfy the Bellman equation in every
reachable cell; and cells cut off from the goal must keep an infinite value.

动态规划测试：代价与 A* 最优值一致；价值函数在每个可达格子满足贝尔曼方程；
与终点不连通的格子价值保持为无穷大。
"""
from __future__ import annotations

import numpy as np
import pytest

from maritime_route.config import IMPASSABLE_COST
from maritime_route.routing.astar import astar
from maritime_route.routing.dynamic_programming import (
    dynamic_programming, step_costs, value_iteration,
)

from test_routing import synthetic_map


def test_dynamic_programming_matches_astar_on_open_water():
    """The DP route costs exactly the A* optimum on an open lattice."""
    cost_map = synthetic_map(barrier=False)
    exact = astar(cost_map, (2, 2), (17, 17))
    dp = dynamic_programming(cost_map, (2, 2), (17, 17))
    assert dp.found
    assert dp.total_cost == pytest.approx(exact.total_cost, rel=1e-9)


def test_dynamic_programming_goes_through_the_gap():
    """With a land wall the DP route matches A* and never touches land."""
    cost_map = synthetic_map(barrier=True)
    exact = astar(cost_map, (2, 2), (17, 17))
    dp = dynamic_programming(cost_map, (2, 2), (17, 17))
    assert dp.total_cost == pytest.approx(exact.total_cost, rel=1e-9)
    assert all(cost_map.passable[cell] for cell in dp.path_cells)
    assert dp.path_cells[0] == (2, 2) and dp.path_cells[-1] == (17, 17)


def test_value_function_satisfies_the_bellman_equation():
    """V(v) = min_u [c(v, u) + V(u)] holds in every reachable cell but the goal."""
    cost_map = synthetic_map(barrier=True)
    goal = (17, 17)
    value, sweeps, costs = value_iteration(cost_map, goal)
    assert value[goal] == 0.0 and sweeps > 1
    n_rows, n_cols = value.shape
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) == goal or not np.isfinite(value[r, c]):
                continue
            rhs = min(costs[(dr, dc)][r, c] + value[r + dr, c + dc]
                      for (dr, dc) in costs
                      if 0 <= r + dr < n_rows and 0 <= c + dc < n_cols)
            assert value[r, c] == pytest.approx(rhs, rel=1e-12)


def test_land_cells_have_infinite_step_cost():
    """No step may enter or leave a land cell."""
    cost_map = synthetic_map(barrier=True)
    mid = cost_map.spec.n_cols // 2
    costs = step_costs(cost_map)
    assert np.isinf(costs[(0, 1)][0, mid - 1])      # into the wall
    assert np.isinf(costs[(0, 1)][0, mid])          # out of the wall


def test_unreachable_goal_is_reported():
    """A closed wall leaves the far side with an infinite value and no route."""
    cost_map = synthetic_map(barrier=True)
    mid = cost_map.spec.n_cols // 2
    cost_map.cost[:, mid] = IMPASSABLE_COST          # close the gap
    dp = dynamic_programming(cost_map, (2, 2), (17, 17))
    assert not dp.found
    assert np.isinf(dp.value[2, 2])
