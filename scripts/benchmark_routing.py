#!/usr/bin/env python3
"""Routing benchmark used for chapter 3 of the explanatory note.

Six voyages in European waters are planned with every method of the
application - A*, Dijkstra, the genetic algorithm and the great circle - on
one shared cost map per voyage, and the node counts, run times and costs are
compared.

路径规划基准测试：在多条真实航线上比较 A*、Dijkstra、遗传算法与大圆航线。
Results are written to docs/figures/routing_benchmark.json

Method
------
For every voyage :func:`maritime_route.routing.planner.benchmark` first builds
(and if necessary refines) one cost map, then plans the voyage with each
method on that same map. The reported run times are ``time.perf_counter``
intervals of the planning call *excluding* classification and cost-map
construction, so they compare the searches themselves. Each case is run once;
run times therefore vary by a few per cent between executions, while node
counts and costs are deterministic (the GA uses a fixed seed).

Output
------
``routing_benchmark.json`` with one record per case (grid, per-method
distance in km, cost in weighted km, nodes expanded, run time in s,
``node_reduction`` = Dijkstra nodes / A* nodes, ``time_ratio`` = Dijkstra time
/ A* time, ``optimality_gap_pct`` between A* and Dijkstra, ``genetic_gap_pct``
of the GA above the A* optimum) and an ``aggregate`` block. The file is read
by ``scripts/make_figures.py`` to draw ``fig_routing_benchmark.png``.

Usage::

    python scripts/benchmark_routing.py [--output PATH]

The trained model must exist (``scripts/train_model.py``).
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

# Make the package importable without installation (adds <project>/src).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maritime_route.config import FIGURE_DIR
from maritime_route.model.inference import ZoneClassifierService
from maritime_route.routing.planner import RoutePlanner, benchmark

#: Representative voyages of increasing length. 代表性航线测试用例。
#: Each entry: (name, departure (lat, lon), destination (lat, lon), lattice
#: step in degrees). Longer ocean voyages use a coarser step to stay within
#: the cell budget of ``RoutingConfig.max_grid_cells``.
CASES = [
    ("Kiel - Tallinn (Baltic)",        (54.3233, 10.1228), (59.4370, 24.7536), 0.25),
    ("Helsinki - Copenhagen",          (60.1699, 24.9384), (55.6761, 12.5683), 0.25),
    ("Gdansk - Stockholm",             (54.3520, 18.6466), (59.3293, 18.0686), 0.25),
    ("Rotterdam - Lisbon (North Sea)", (51.9500,  4.1400), (38.7223, -9.1393), 0.40),
    ("Bergen - Reykjavik (Atlantic)",  (60.3913,  5.3221), (64.1466, -21.9426), 0.50),
    ("Brest - Gibraltar",              (48.3904, -4.4861), (36.1408, -5.3536), 0.40),
]


def main() -> int:
    """Run all benchmark cases, print a line per case and save the JSON.

    Returns
    -------
    int
        Process exit code (0).
    """
    parser = argparse.ArgumentParser(description="Benchmark the routing algorithms")
    parser.add_argument("--output", default=str(FIGURE_DIR / "routing_benchmark.json"))
    args = parser.parse_args()
    # WARNING level hides the per-grid INFO messages of the planner.
    logging.basicConfig(level=logging.WARNING)

    # One classifier instance (singleton) is shared by all cases.
    planner = RoutePlanner(ZoneClassifierService.instance())
    records = []
    for name, start, end, resolution in CASES:
        print(f"-> {name}", flush=True)
        data = benchmark(planner, start, end, resolution)
        # Index the per-method result rows by algorithm name.
        by_alg = {r["algorithm"]: r for r in data["results"]}
        records.append({
            "case": name,
            "start": list(start), "end": list(end),
            "grid": data["grid"],
            "cost_map_build_s": data["cost_map_build_s"],
            "astar": by_alg["astar"],
            "dijkstra": by_alg["dijkstra"],
            "genetic": by_alg["genetic"],
            "dynamic": by_alg["dynamic"],
            "dynamic_gap_pct": data["dynamic_gap_pct"],
            "genetic_gap_pct": data["genetic_gap_pct"],
            "great_circle": by_alg["great_circle"],
            "node_reduction": data["astar_node_reduction"],
            "time_ratio": data["astar_time_ratio"],
            "optimality_gap_pct": data["optimality_gap_pct"],
            "navigable_share": data["cost_map"]["navigable_share"],
        })
        a, d, g = by_alg["astar"], by_alg["dijkstra"], by_alg["great_circle"]
        print(f"   A*: {a['distance_km']:.0f} km, {a['nodes_expanded']:,} nodes, "
              f"{a['runtime_s']:.3f} s | Dijkstra: {d['nodes_expanded']:,} nodes, "
              f"{d['runtime_s']:.3f} s | GA gap: {data['genetic_gap_pct']} % | "
              f"GC: {g['distance_km']:.0f} km "
              f"({'navigable' if g['feasible'] else 'crosses land'})")

    # Aggregates over all cases. Ratios that are None (e.g. A* expanded no
    # node) are filtered out before averaging. mean_detour_pct is the extra
    # length of the A* route relative to the great-circle distance.
    # 汇总统计：对各测试用例求平均/最大值。
    summary = {
        "cases": records,
        "aggregate": {
            "mean_node_reduction": round(statistics.mean(
                [r["node_reduction"] for r in records if r["node_reduction"]]), 3),
            "mean_time_ratio": round(statistics.mean(
                [r["time_ratio"] for r in records if r["time_ratio"]]), 3),
            "max_optimality_gap_pct": max(
                [r["optimality_gap_pct"] or 0.0 for r in records]),
            "mean_detour_pct": round(statistics.mean(
                [100.0 * (r["astar"]["distance_km"] / r["great_circle"]["distance_km"] - 1.0)
                 for r in records]), 2),
            "genetic_feasible_cases": sum(1 for r in records if r["genetic"]["feasible"]),
            "max_dynamic_gap_pct": max(r["dynamic_gap_pct"] or 0.0 for r in records),
            "mean_genetic_gap_pct": round(statistics.mean(
                [r["genetic_gap_pct"] for r in records if r["genetic_gap_pct"] is not None]), 3),
            "great_circle_infeasible_cases": sum(
                1 for r in records if not r["great_circle"]["feasible"]),
            "n_cases": len(records),
        },
    }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nAggregate:", json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
