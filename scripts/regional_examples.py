#!/usr/bin/env python3
"""Example voyages in each of the ten sea regions of the corpus.

The labelled corpus covers ten separate sea regions (``config.REGIONS``). This
script plans one realistic port-to-port voyage inside every region and runs all
five methods of the application on the same cost map:

* A* - the default exact search,
* Dijkstra - the exact search without a heuristic,
* the genetic algorithm - the evolutionary search (fitness function),
* dynamic programming - value iteration on the Bellman equation (value function),
* the great circle - the naive shortest-distance reference.

It writes

* ``docs/figures/regional_routes.json`` - the numbers used in the thesis
  (distance, cost, nodes or fitness evaluations, run time, gap of the GA to the
  optimum, whether the great circle crosses land);
* ``docs/figures/fig_regional_routes.png`` - ten map panels, each showing the
  classified lattice with the A*, genetic and great-circle routes.

Usage::

    python scripts/regional_examples.py

在十个海域各规划一条真实港口到港口的航线，在同一张代价图上运行 A*、Dijkstra、
遗传算法与大圆航线四种方法，输出数值结果（JSON）与十幅地图（PNG）。

Before planning, the script asserts that both ports of every voyage fall into
the declared region (using the same region boxes as the data-cleaning step),
so the table in the thesis cannot silently mix regions.
"""
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

# Make the package importable without installation (adds <project>/src).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# Non-interactive backend: renders to files, works without a display.
# 使用无界面后端，仅输出图片文件。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from maritime_route.config import CLASS_COLOR, CLASS_NAMES, FIGURE_DIR
from maritime_route.data.cleaning import assign_regions
from maritime_route.model.inference import ZoneClassifierService
from maritime_route.routing.geodesy import great_circle_points
from maritime_route.routing.planner import RoutePlanner

#: One voyage per region: (region, voyage name, departure, destination).
#: Coordinates are the approaches of real ports. 每个海域一条航线（真实港口外锚地坐标）。
VOYAGES = [
    ("North Europe (Baltic and North Sea)", "Rotterdam - Gdansk",
     (51.9500, 4.1400), (54.4000, 18.6600)),
    ("Black Sea and Eastern Mediterranean", "Constanta - Batumi",
     (44.1700, 28.6600), (41.6500, 41.6300)),
    ("Red Sea and Gulf of Aden", "Suez - Djibouti",
     (29.9300, 32.5500), (11.6000, 43.1400)),
    ("Persian Gulf and Arabian Sea", "Jebel Ali - Karachi",
     (25.0100, 55.0600), (24.8000, 66.9700)),
    ("Bay of Bengal", "Chennai - Chittagong",
     (13.1000, 80.3000), (22.2500, 91.8000)),
    ("Strait of Malacca and Andaman Sea", "Singapore - Phuket",
     (1.2600, 103.8200), (7.8800, 98.4000)),
    ("South China Sea", "Vung Tau - Hong Kong",
     (10.3500, 107.0800), (22.2800, 114.1700)),
    ("Japan and North-West Pacific", "Kochi - Yokohama",
     (33.4500, 133.6000), (35.3000, 139.7500)),
    ("Southern Africa (Cape of Good Hope)", "Cape Town - East London",
     (-33.9100, 18.4300), (-33.0300, 27.9200)),
    ("Central America (Caribbean coast)", "Colon - Puerto Cabezas",
     (9.3800, -79.9200), (14.0300, -83.3700)),
]

#: Methods run on every voyage, in this order (A* first builds the cost map).
METHODS = ("astar", "dijkstra", "genetic", "dynamic", "great_circle")


def run_voyage(planner: RoutePlanner, start, end):
    """Plan one voyage with all five methods on one shared cost map.

    The lattice is built (and refined if a strait is too narrow) once by A*;
    the other methods then reuse exactly the same cost map, so their costs and
    run times are directly comparable.
    先用 A* 建立（必要时细化）网格，其余方法复用同一张代价图，保证结果可比。

    Parameters
    ----------
    planner : RoutePlanner
        Planner with the trained classifier.
    start, end : tuple of float
        Departure and destination ``(lat, lon)``, degrees.

    Returns
    -------
    tuple
        ``(plans, cost_map)`` - a dict method name -> RoutePlan, and the
        shared CostMap.
    """
    _, cost_map = planner.plan(start, end, "astar")
    # Resolution actually used after any refinement; passed on so that the
    # per-method plans report the same lattice.
    resolution = cost_map.spec.resolution_deg
    plans = {}
    for method in METHODS:
        plan, _ = planner.plan(start, end, method, resolution,
                               reuse_cost_map=cost_map, auto_refine=False)
        plans[method] = plan
    return plans, cost_map


def summarise(plans) -> dict:
    """Numbers reported in the thesis for one voyage. 论文中报告的单条航线数据。

    Parameters
    ----------
    plans : dict
        Method name -> RoutePlan, as returned by :func:`run_voyage`.

    Returns
    -------
    dict
        Per method: ``feasible``, ``distance_km``, ``total_cost`` (weighted
        km, None when infeasible or crossing land), ``nodes_expanded`` (for
        the GA: fitness evaluations) and ``runtime_s``; plus
        ``genetic_gap_pct`` (GA cost above the A* optimum, %),
        ``great_circle_land_share`` (share of great-circle samples on land)
        and ``detour_pct`` (A* length above the great-circle distance, %).
    """
    out = {}
    for method, plan in plans.items():
        # The great circle always "exists"; it is infeasible when it crosses land.
        crosses = method == "great_circle" and not (plan.baseline or {}).get("is_navigable", True)
        out[method] = {
            "feasible": bool(plan.feasible and not crosses),
            "distance_km": round(plan.distance_km, 1),
            "total_cost": (round(plan.total_cost, 1)
                           if plan.feasible and not crosses and math.isfinite(plan.total_cost)
                           else None),
            "nodes_expanded": int(plan.nodes_expanded),
            "runtime_s": round(plan.runtime_s, 3),
        }
    a, g = plans["astar"], plans["genetic"]
    # max(0, ...) removes a -0.00 caused by floating-point summation order.
    out["genetic_gap_pct"] = (max(0.0, round(100.0 * (g.total_cost - a.total_cost) / a.total_cost, 2))
                              if g.feasible and a.feasible and a.total_cost > 0 else None)
    gc = plans["great_circle"].baseline or {}
    out["great_circle_land_share"] = gc.get("land_share")
    out["detour_pct"] = round(100.0 * (a.distance_km / a.great_circle_km - 1.0), 2)
    return out


def draw(ax, cost_map, plans, title: str) -> None:
    """One map panel: classified lattice plus the three routes.

    The zone classes are drawn as an image in plain longitude/latitude axes
    (equirectangular, not a map projection). Dijkstra is not drawn: it has
    the same optimal cost as A* and would overlap its line.
    单幅子图：分类网格 + A*、遗传算法与大圆三条航线。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target panel.
    cost_map : CostMap
        Shared cost map of the voyage.
    plans : dict
        Method name -> RoutePlan.
    title : str
        Panel title.
    """
    spec = cost_map.spec
    # One colour per class index 0..3; vmin/vmax = -0.5/3.5 centre each
    # integer class on its colour.
    cmap = ListedColormap([CLASS_COLOR[c] for c in CLASS_NAMES])
    # Lattice nodes are cell centres, so the image extends half a step beyond
    # the first and last node on every side.
    extent = (spec.lon_min - spec.resolution_deg / 2, spec.lon_max + spec.resolution_deg / 2,
              spec.lat_min - spec.resolution_deg / 2, spec.lat_max + spec.resolution_deg / 2)
    # origin="lower": row 0 (southernmost latitude) at the bottom.
    ax.imshow(cost_map.class_index, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5,
              extent=extent, interpolation="nearest", aspect="auto")
    a, g = plans["astar"], plans["genetic"]
    gc = np.array(great_circle_points(a.start, a.end, 120))
    if not a.feasible:                      # nothing to draw but the lattice
        ax.set_title(title + " (no route)", fontsize=8)
        return
    # Plot x = longitude (column 1), y = latitude (column 0).
    ax.plot(gc[:, 1], gc[:, 0], ":", color="#d62728", lw=1.4, label="Great circle")
    if g.feasible:
        pts = np.array(g.waypoints)
        ax.plot(pts[:, 1], pts[:, 0], "--", color="#ff9f1c", lw=1.6, label="Genetic algorithm")
    pts = np.array(a.waypoints)
    ax.plot(pts[:, 1], pts[:, 0], "-", color="#111111", lw=1.6, label="A* (optimal)")
    # [::-1] turns (lat, lon) into (lon, lat) = (x, y). Green circle =
    # departure, red square = destination.
    ax.plot(*a.start[::-1], "o", color="#2ca02c", ms=5)
    ax.plot(*a.end[::-1], "s", color="#d62728", ms=5)
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])


def main() -> int:
    """Run all voyages, save the table and the figure.

    Returns
    -------
    int
        Process exit code (0).

    Raises
    ------
    AssertionError
        If a port does not lie in the region its voyage is declared for.
    """
    logging.basicConfig(level=logging.WARNING)
    planner = RoutePlanner(ZoneClassifierService.instance())
    records = []
    # 5 x 2 panels on an A4-sized figure (inches). 5 行 2 列子图。
    fig, axes = plt.subplots(5, 2, figsize=(8.3, 11.0))
    for i, (region, name, start, end) in enumerate(VOYAGES):
        # Sanity check: both ends of the voyage lie inside the declared region.
        # 检查起终点确实位于所声明的海域内。
        assigned = assign_regions(np.array([start[0], end[0]]), np.array([start[1], end[1]]))
        assert all(r == region for r in assigned), (name, assigned)
        print(f"-> {region}: {name}", flush=True)
        plans, cost_map = run_voyage(planner, start, end)
        row = {"region": region, "voyage": name, "start": list(start), "end": list(end),
               "grid": cost_map.spec.to_dict(), **summarise(plans)}
        records.append(row)
        a, g = row["astar"], row["genetic"]
        print(f"   A* {a['distance_km']:.0f} km cost {a['total_cost']} | GA gap "
              f"{row['genetic_gap_pct']} % | GC land {row['great_circle_land_share']}")
        draw(axes.flat[i], cost_map, plans, f"{i + 1}. {name}")
    # The legend is built explicitly, because a panel may lack one of the lines.
    # 图例单独构建，因为某些子图可能缺少某条线（例如遗传算法未找到路线）。
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color="#111111", lw=1.6, label="A* (optimal)"),
               Line2D([], [], color="#ff9f1c", lw=1.6, ls="--", label="Genetic algorithm"),
               Line2D([], [], color="#d62728", lw=1.4, ls=":", label="Great circle")]
    handles += [Patch(color=CLASS_COLOR[c], label=c) for c in CLASS_NAMES]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)
    # Reserve the bottom 4.5 % of the figure for the shared legend.
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / "fig_regional_routes.png", dpi=200)

    gaps = [r["genetic_gap_pct"] for r in records if r["genetic_gap_pct"] is not None]
    # Aggregates over the ten voyages. astar_dijkstra_equal_cost checks that
    # the two exact methods agree on every voyage (they must: both are optimal).
    summary = {
        "voyages": records,
        "aggregate": {
            "n_voyages": len(records),
            "astar_feasible": sum(r["astar"]["feasible"] for r in records),
            "genetic_feasible": sum(r["genetic"]["feasible"] for r in records),
            "great_circle_crossing_land": sum(not r["great_circle"]["feasible"] for r in records),
            "genetic_gap_mean_pct": round(float(np.mean(gaps)), 2) if gaps else None,
            "genetic_gap_max_pct": round(float(np.max(gaps)), 2) if gaps else None,
            "dynamic_equal_cost": all(
                r["astar"]["total_cost"] == r["dynamic"]["total_cost"] for r in records),
            "astar_dijkstra_equal_cost": all(
                r["astar"]["total_cost"] == r["dijkstra"]["total_cost"] for r in records),
            "mean_detour_pct": round(float(np.mean([r["detour_pct"] for r in records])), 2),
        },
    }
    (FIGURE_DIR / "regional_routes.json").write_text(json.dumps(summary, indent=2),
                                                      encoding="utf-8")
    print("\n", json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
