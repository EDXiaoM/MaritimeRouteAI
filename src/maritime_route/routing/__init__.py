"""Routing package: cost map construction and graph search algorithms.

航线规划层：代价地图与图搜索算法。

This package turns two coordinates (departure, destination) into a navigable
route. It sits after the neural classifier in the pipeline::

    classifier (model.inference) -> cost_map -> astar | dijkstra | genetic -> planner -> storage / export / web

Modules
-------
geodesy
    Spherical geometry: haversine distance, bearing, great-circle
    interpolation, Douglas-Peucker simplification.
cost_map
    :class:`GridSpec` (regular latitude/longitude lattice) and
    :class:`CostMap` (zone class, confidence and traversal cost of every cell),
    built by :func:`build_cost_map` from the classifier output.
astar
    :func:`astar` - optimal search with an admissible great-circle heuristic;
    also defines :class:`SearchResult` and the shared helpers.
dijkstra
    :func:`dijkstra` - uniform-cost search, the reference exact method.
genetic
    :func:`~maritime_route.routing.genetic.genetic` - evolutionary search on
    the same cost model (not re-exported here; imported by the planner).
planner
    :class:`RoutePlanner` - the service used by the web application: builds
    the lattice, snaps the ports onto water, runs the chosen algorithm,
    refines the lattice on failure and compares the result with the great
    circle. :func:`benchmark` runs all methods on one shared cost map.

The names below are the public API of the package.
以下名称为本包对外公开的接口。
"""
from .astar import SearchResult, astar
from .cost_map import CostMap, GridSpec, build_cost_map, make_grid_spec
from .dijkstra import dijkstra
from .planner import RouteLeg, RoutePlan, RoutePlanner, benchmark

# Public names exported by ``from maritime_route.routing import *``.
__all__ = [
    "SearchResult", "astar", "dijkstra", "CostMap", "GridSpec", "build_cost_map",
    "make_grid_spec", "RouteLeg", "RoutePlan", "RoutePlanner", "benchmark",
]
