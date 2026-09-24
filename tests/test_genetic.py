"""Tests of the genetic-algorithm route search.

The genetic algorithm is not exact, so the tests check the properties that it
must always have: the route it returns never touches land, it is never cheaper
than the proven optimum of A*, it is reproducible for a fixed seed, and it finds
the single gap in a wall of land ("the maze").

遗传算法测试：不穿越陆地、代价不低于 A* 最优解、固定种子可复现、能找到陆地墙上的唯一缺口。
"""
from __future__ import annotations

import pytest

from maritime_route.config import GeneticConfig
from maritime_route.routing.astar import astar
from maritime_route.routing.genetic import _Evaluator, genetic

# Reuse the 20x20 synthetic map (open sea with an optional land wall) from
# test_routing; pytest puts tests/ on sys.path, so a plain import works.
# 复用 test_routing 中的合成地图。
from test_routing import synthetic_map

#: Small population and generation budget so the tests run quickly.
#: 较小的种群与代数，保证测试快速完成。
FAST = GeneticConfig(population_size=40, generations=120, stagnation_limit=40)


def test_rasterised_segment_is_eight_connected():
    """Rasterised segment starts/ends at the given cells and every step moves to an 8-neighbour."""
    rows, cols = _Evaluator.rasterise((0, 0), (7, 3))
    assert (rows[0], cols[0]) == (0, 0) and (rows[-1], cols[-1]) == (7, 3)
    for r0, c0, r1, c1 in zip(rows[:-1], cols[:-1], rows[1:], cols[1:]):
        assert max(abs(r1 - r0), abs(c1 - c0)) == 1


def test_genetic_finds_the_gap_in_the_wall():
    """GA finds a route through the single gap in a land wall and every cell of it is navigable."""
    cost_map = synthetic_map(barrier=True)
    result = genetic(cost_map, (2, 2), (17, 17), config=FAST)
    assert result.found
    assert all(cost_map.passable[cell] for cell in result.path_cells)


def test_genetic_is_never_better_than_the_astar_optimum():
    """GA route cost is never below the exact A* optimum on the same map."""
    cost_map = synthetic_map(barrier=True)
    exact = astar(cost_map, (2, 2), (17, 17))
    evolved = genetic(cost_map, (2, 2), (17, 17), config=FAST)
    assert evolved.total_cost >= exact.total_cost - 1e-6


def test_genetic_reaches_the_optimum_on_open_water():
    """On open water the GA cost is within 2 % of the A* optimum."""
    cost_map = synthetic_map(barrier=False)
    exact = astar(cost_map, (2, 2), (17, 17))
    evolved = genetic(cost_map, (2, 2), (17, 17), config=FAST)
    assert evolved.total_cost == pytest.approx(exact.total_cost, rel=0.02)


def test_genetic_is_reproducible_with_a_fixed_seed():
    """Two GA runs with the same seed return identical paths and costs."""
    cost_map = synthetic_map(barrier=True)
    a = genetic(cost_map, (2, 2), (17, 17), config=FAST, seed=7)
    b = genetic(cost_map, (2, 2), (17, 17), config=FAST, seed=7)
    assert a.path_cells == b.path_cells and a.total_cost == b.total_cost


def test_genetic_records_a_non_increasing_best_fitness():
    """Best cost recorded per generation never increases (elitism)."""
    cost_map = synthetic_map(barrier=True)
    result = genetic(cost_map, (2, 2), (17, 17), config=FAST)
    history = result.history
    assert all(later <= earlier + 1e-9 for earlier, later in zip(history, history[1:]))


def test_genetic_refuses_a_start_on_land():
    """GA reports no route when the start cell lies on land."""
    cost_map = synthetic_map(barrier=True)
    mid = cost_map.spec.n_cols // 2
    result = genetic(cost_map, (0, mid), (17, 17), config=FAST)
    assert not result.found
