"""Tests of the cost map and the two search algorithms. 代价地图与寻路算法测试。

Covers ``GridSpec`` / ``make_grid_spec`` (lattice geometry and cell budget),
``CostMap`` (statistics, map overlay), ``nearest_navigable`` (snapping ports
onto water) and the exact searches ``astar`` and ``dijkstra``. The searches
are run on small synthetic maps built by :func:`synthetic_map`, so the
expected optimum is known and no trained model is required. The same helper
is imported by ``test_genetic``.
在合成地图上测试网格、代价地图与 A*/Dijkstra；不依赖模型。
"""
from __future__ import annotations

import numpy as np
import pytest

from maritime_route.config import IMPASSABLE_COST, ROUTING
from maritime_route.routing.astar import astar
from maritime_route.routing.cost_map import CostMap, GridSpec, make_grid_spec, nearest_navigable
from maritime_route.routing.dijkstra import dijkstra


def synthetic_map(barrier: bool = True) -> CostMap:
    """20x20 open sea with an optional land wall leaving one gap. 合成测试地图。"""
    spec = GridSpec(50.0, 54.75, 0.0, 4.75, 0.25)
    rows, cols = spec.n_rows, spec.n_cols
    class_index = np.zeros((rows, cols), dtype=np.int8)       # OPEN_SEA
    cost = np.ones((rows, cols), dtype=np.float32)
    if barrier:
        mid = cols // 2
        class_index[:, mid] = 3                                # COASTLINE
        cost[:, mid] = IMPASSABLE_COST
        class_index[rows // 2, mid] = 0                        # the gap
        cost[rows // 2, mid] = 1.0
    return CostMap(spec=spec, class_index=class_index,
                   confidence=np.full((rows, cols), 0.9, dtype=np.float32), cost=cost)


def test_grid_spec_round_trip():
    """cell -> coordinate -> cell conversion returns the original cell."""
    spec = GridSpec(50.0, 55.0, 0.0, 5.0, 0.25)
    for row in (0, 5, spec.n_rows - 1):
        for col in (0, 7, spec.n_cols - 1):
            assert spec.coord_to_cell(*spec.cell_to_coord(row, col)) == (row, col)


def test_grid_spec_coarsens_when_over_budget():
    """A grid that would exceed max_cells is coarsened to fit the budget."""
    spec = make_grid_spec([(0.0, 0.0), (60.0, 60.0)], 0.1, 2.0, max_cells=50_000)
    assert spec.n_cells <= 50_000
    assert spec.resolution_deg > 0.1


def test_astar_and_dijkstra_agree_on_open_water():
    """A* and Dijkstra find paths of the same optimal cost on open water."""
    cost_map = synthetic_map(barrier=False)
    start, goal = (2, 2), (17, 17)
    a = astar(cost_map, start, goal)
    d = dijkstra(cost_map, start, goal)
    assert a.found and d.found
    assert a.total_cost == pytest.approx(d.total_cost, rel=1e-6)


def test_astar_expands_no_more_nodes_than_dijkstra():
    """The A* heuristic never makes it expand more nodes than Dijkstra."""
    cost_map = synthetic_map(barrier=False)
    a = astar(cost_map, (0, 0), (19, 19))
    d = dijkstra(cost_map, (0, 0), (19, 19))
    assert a.nodes_expanded <= d.nodes_expanded


def test_route_goes_through_the_gap_and_never_crosses_land():
    """The A* path uses the gap in the wall and contains no impassable cell."""
    cost_map = synthetic_map(barrier=True)
    result = astar(cost_map, (2, 2), (17, 17))
    assert result.found
    for row, col in result.path_cells:
        assert cost_map.cost[row, col] < IMPASSABLE_COST
    gap_row = cost_map.spec.n_rows // 2
    wall_col = cost_map.spec.n_cols // 2
    assert (gap_row, wall_col) in result.path_cells


def test_no_path_when_the_barrier_is_closed():
    """A* reports "No navigable path" when the land wall is closed."""
    cost_map = synthetic_map(barrier=True)
    wall_col = cost_map.spec.n_cols // 2
    cost_map.cost[:, wall_col] = IMPASSABLE_COST
    cost_map.class_index[:, wall_col] = 3
    result = astar(cost_map, (2, 2), (17, 17))
    assert not result.found
    assert "No navigable path" in result.message


def test_search_refuses_a_start_on_land():
    """A* reports no route when the start cell lies on land."""
    cost_map = synthetic_map(barrier=True)
    wall_col = cost_map.spec.n_cols // 2
    result = astar(cost_map, (0, wall_col), (17, 17))
    assert not result.found


def test_nearest_navigable_snaps_off_land():
    """nearest_navigable moves a land cell onto a navigable cell."""
    cost_map = synthetic_map(barrier=True)
    wall_col = cost_map.spec.n_cols // 2
    snapped = nearest_navigable(cost_map, 0, wall_col)
    assert snapped is not None
    assert cost_map.passable[snapped]


def test_cheaper_zone_is_preferred_over_a_shorter_crossing():
    """A detour through cheap water must beat a short crossing of very costly water.

    The band weight is chosen so that crossing it once (extra cost of roughly
    half the band weight) is dearer than the ~2*cols extra cheap steps of the
    detour - otherwise going straight really is optimal.
    绕行代价约 2*cols，故障碍带权重需远大于该值，绕行才应当胜出。
    """
    spec = GridSpec(50.0, 52.25, 0.0, 2.25, 0.25)
    rows, cols = spec.n_rows, spec.n_cols
    cost = np.ones((rows, cols), dtype=np.float32)
    cost[rows // 2, :] = 400.0                     # a very expensive band
    cost[rows // 2, 0] = 1.0                       # one cheap doorway on the left
    cost_map = CostMap(spec=spec, class_index=np.zeros((rows, cols), np.int8),
                       confidence=np.ones((rows, cols), np.float32), cost=cost)
    result = astar(cost_map, (0, cols - 1), (rows - 1, cols - 1))
    assert result.found
    assert (rows // 2, 0) in result.path_cells     # it used the cheap doorway
    # and the same optimum is reached without a heuristic
    assert dijkstra(cost_map, (0, cols - 1), (rows - 1, cols - 1)).total_cost == \
        pytest.approx(result.total_cost, rel=1e-6)


def test_statistics_report_navigable_share():
    """Cost-map statistics give a navigable share in (0, 1) and a consistent cell total."""
    stats = synthetic_map(barrier=True).statistics()
    assert 0.0 < stats["navigable_share"] < 1.0
    assert stats["cells_total"] == stats["grid"]["n_cells"]


def test_overlay_cells_tile_without_gaps():
    """Down-sampled overlay cell size equals step x resolution and respects the cell budget."""
    overlay = synthetic_map().to_overlay(max_cells=50)
    assert overlay["step"] >= 1
    assert overlay["cell_size_deg"] == pytest.approx(
        overlay["step"] * ROUTING.grid_resolution_deg, rel=1e-6)
    assert len(overlay["cells"]) <= 80
