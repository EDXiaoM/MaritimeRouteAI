"""Experiment of version 1.3: complex courses, step limits, parameters, Stop.

实验（1.3 版）：复杂航线、步数上限、算法参数与停止功能。

The supervisor asked that the application find routes for complex point
arrangements, that every algorithm have its own parameters set in the
interface (e.g. the maximum number of steps of Dijkstra), that the search can
be stopped, and that the algorithms be compared on one course. This script
measures all of it:

E1  Seven complex courses planned with A* twice: with the planner of
    version 1.2 (no adaptive search area, no land-mask correction, no
    two-level search) and with version 1.3. Outcome, length, coastline-check
    verdict, longest land stretch, lattices tried and time.
E2  The same courses: all five methods on one shared cost map
    (:func:`~maritime_route.routing.planner.benchmark`) - status, length,
    cost, gap to the optimum, steps, time, theoretical worst-case bound.
E3  Step limits on one course (Gdansk -> Bergen): A* and Dijkstra with a
    maximum of 100, 1,000, 10,000, 100,000 and 1,000,000 steps.
E4  Parameters of the genetic algorithm on the same course: population x
    generations and the maximum number of waypoints.
E5  Stop latency: a time limit of 1 s on a long search; how long after the
    limit the search actually ends.
E7  The 100 random voyages of ``scripts/route_audit.py`` planned again with
    the defaults of version 1.3 and judged by the same land-mask test.
E6  Via points: the genetic algorithm (and A* for reference) on two courses
    that the genetic algorithm cannot solve in one leg, without and with
    intermediate points.

Outputs: ``docs/figures/complex_routes.json`` (every number) and the figures
``fig_complex_routes.png`` (routes of all methods per course),
``fig_step_limits.png`` (outcome and time against the step limit) and
``fig_ga_parameters.png``.

Usage::

    python scripts/complex_routes.py            # about 10-20 minutes on a CPU
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from maritime_route.config import FIGURE_DIR  # noqa: E402
from maritime_route.model.inference import ZoneClassifierService  # noqa: E402
from maritime_route.routing.control import RunControl  # noqa: E402
from maritime_route.routing.parameters import SearchParameters  # noqa: E402
from maritime_route.routing.planner import RoutePlanner, benchmark  # noqa: E402

LOGGER = logging.getLogger("complex_routes")

#: Complex courses: name -> (departure, destination, what makes it complex).
COURSES = {
    "Gdańsk → Bergen": ((54.40, 18.70), (60.39, 5.30), "round Denmark: Danish straits, Kattegat, Skagerrak"),
    "Helsinki → Copenhagen": ((60.1699, 24.9384), (55.6761, 12.5683), "Gulf of Finland, Baltic, Øresund"),
    "Rotterdam → Gdańsk": ((51.95, 4.14), (54.40, 18.70), "North Sea, Skagerrak, Danish straits, Baltic"),
    "Piraeus → Rhodes": ((37.94, 23.62), (36.45, 28.23), "between the Aegean islands"),
    "Jeddah → Dubai": ((21.48, 39.15), (25.27, 55.28), "Red Sea, Bab-el-Mandeb, Arabian Sea, Hormuz"),
    "Singapore → Hong Kong": ((1.26, 103.85), (22.28, 114.16), "Singapore Strait, South China Sea"),
    "Kapellskär → Oulu": ((59.72, 19.07), (65.01, 25.40), "Gulf of Bothnia, the Kvarken"),
}
#: Settings that reproduce the planner of version 1.2.
V12 = {"adaptive_area": False, "use_land_mask": False, "corridor_search": False}
STEP_LIMITS = (100, 1_000, 10_000, 100_000, 1_000_000)
GA_GRID = [(20, 100), (40, 200), (80, 300), (160, 600), (320, 1200)]
GA_WAYPOINTS = (6, 14, 30)
METHOD_COLOURS = {"astar": "#1f77b4", "dijkstra": "#9467bd", "dynamic": "#2a9d8f",
                  "genetic": "#ff9f1c", "great_circle": "#d62728"}
METHOD_NAMES = {"astar": "A*", "dijkstra": "Dijkstra", "dynamic": "Dynamic programming",
                "genetic": "Genetic algorithm", "great_circle": "Great circle"}


def summary(plan) -> dict:
    """The numbers of one plan that the experiment reports."""
    c = plan.coast_check or {}
    return {
        "status": plan.status, "feasible": plan.feasible,
        "distance_km": round(plan.distance_km, 1),
        "total_cost": None if not np.isfinite(plan.total_cost) else round(plan.total_cost, 1),
        "seconds": round(plan.runtime_s, 2), "steps": plan.nodes_expanded,
        "coast_verdict": c.get("verdict"), "initial_verdict": c.get("initial_verdict"),
        "max_land_run_km": c.get("max_land_run_km"),
        "initial_max_land_run_km": c.get("initial_max_land_run_km"),
        "resolution_deg": plan.grid.get("resolution_deg"),
        "lattices": len(plan.attempts),
        "attempts": [{k: a.get(k) for k in ("resolution_deg", "cells", "status", "steps", "seconds",
                                            "corridor_km", "mask_fused", "next")}
                     for a in plan.attempts],
        "message": plan.message,
    }


def e1_planner_versions(planner) -> dict:
    out = {}
    for name, (start, end, why) in COURSES.items():
        row = {"start": start, "end": end, "complexity": why}
        for label, values in (("v1.2", V12), ("v1.3", {})):
            plan, _ = planner.plan(start, end, "astar", params=SearchParameters.from_dict(values))
            row[label] = summary(plan)
            LOGGER.info("E1 %-24s %s: %s %s km, %s, %.1f s", name, label, plan.status,
                        row[label]["distance_km"], row[label]["coast_verdict"], plan.runtime_s)
        out[name] = row
    return out


def e2_methods(planner) -> dict:
    out = {}
    for name, (start, end, _) in COURSES.items():
        data = benchmark(planner, start, end)
        out[name] = data
        LOGGER.info("E2 %-24s %s", name, ", ".join(
            f"{r['algorithm']}={r['status']}/{r['runtime_s']:.2f}s" for r in data["results"]))
    return out


def e3_step_limits(planner, course="Gdańsk → Bergen", resolution=0.1) -> dict:
    start, end, _ = COURSES[course]
    warm, cost_map = planner.plan(start, end, "astar", resolution,
                                  params=SearchParameters.from_dict({"coast_guard": False}))
    rows = []
    for algorithm in ("astar", "dijkstra"):
        for limit in STEP_LIMITS:
            params = SearchParameters.from_dict({f"{algorithm}_max_steps": limit})
            plan, _ = planner.plan(start, end, algorithm, cost_map.spec.resolution_deg,
                                   reuse_cost_map=cost_map, auto_refine=False, params=params)
            rows.append({"algorithm": algorithm, "max_steps": limit, "status": plan.status,
                         "steps": plan.nodes_expanded, "seconds": round(plan.runtime_s, 4),
                         "distance_km": round(plan.distance_km, 1)})
            LOGGER.info("E3 %s limit %d: %s (%d steps)", algorithm, limit, plan.status,
                        plan.nodes_expanded)
    return {"course": course, "grid": cost_map.spec.to_dict(),
            "navigable_cells": int(cost_map.passable.sum()), "rows": rows}


#: Course of the parameter study of the genetic algorithm (it solves it in one leg).
GA_COURSE = ("Helsinki → Copenhagen", (60.1699, 24.9384), (55.6761, 12.5683))
#: Courses with intermediate points for E6.
VIA_COURSES = {
    "Gdańsk → Bergen": [(54.40, 18.70), (57.95, 11.0), (60.39, 5.30)],
    "Jeddah → Dubai": [(21.48, 39.15), (12.4, 43.6), (13.0, 51.5), (22.7, 60.2), (25.27, 55.28)],
}


def e4_genetic(planner, course=GA_COURSE[0], resolution=0.1) -> dict:
    start, end = GA_COURSE[1], GA_COURSE[2]
    _, cost_map = planner.plan(start, end, "astar", resolution,
                               params=SearchParameters.from_dict({"coast_guard": False}))
    res = cost_map.spec.resolution_deg
    optimum, _ = planner.plan(start, end, "astar", res, reuse_cost_map=cost_map, auto_refine=False)
    rows = []
    for waypoints in GA_WAYPOINTS:
        for population, generations in GA_GRID:
            costs, times, found = [], [], 0
            for seed in (1, 2, 3):
                params = SearchParameters.from_dict({
                    "ga_population": population, "ga_generations": generations,
                    "ga_stagnation": max(60, generations // 5), "ga_max_waypoints": waypoints,
                    "ga_seed": seed})
                plan, _ = planner.plan(start, end, "genetic", res, reuse_cost_map=cost_map,
                                       auto_refine=False, params=params)
                times.append(plan.runtime_s)
                if plan.feasible and np.isfinite(plan.total_cost):
                    found += 1
                    costs.append(plan.total_cost)
            gap = (100.0 * (np.mean(costs) - optimum.total_cost) / optimum.total_cost
                   if costs else None)
            rows.append({"population": population, "generations": generations,
                         "max_waypoints": waypoints, "runs": 3, "found": found,
                         "mean_gap_pct": None if gap is None else round(float(gap), 2),
                         "mean_seconds": round(float(np.mean(times)), 2)})
            LOGGER.info("E4 wp=%d pop=%d gen=%d: found %d/3, gap %s, %.1f s", waypoints,
                        population, generations, found, rows[-1]["mean_gap_pct"], np.mean(times))
    return {"course": course, "optimum_cost": round(optimum.total_cost, 2),
            "grid": cost_map.spec.to_dict(), "rows": rows}


def e6_via_points(planner) -> dict:
    out = {}
    for name, points in VIA_COURSES.items():
        row = {"points": points}
        for algorithm in ("astar", "genetic"):
            for label, pts in (("direct", [points[0], points[-1]]), ("via", points)):
                plan, _ = planner.plan_via(pts, algorithm)
                row[f"{algorithm}_{label}"] = summary(plan)
                LOGGER.info("E6 %s %s %s: %s %.0f km %.1f s", name, algorithm, label, plan.status,
                            plan.distance_km, plan.runtime_s)
        out[name] = row
    return out


def e7_audit(planner) -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    import route_audit                                   # noqa: E402 (same verdict rules)

    audit = json.loads((FIGURE_DIR / "route_audit.json").read_text(encoding="utf-8"))
    rows, verdicts = [], {"correct": 0, "grazing": 0, "crossing": 0, "not_found": 0}
    for v in audit["voyages"]:
        t0 = time.perf_counter()
        plan, _ = planner.plan(tuple(v["start"]), tuple(v["end"]), "astar")
        seconds = time.perf_counter() - t0
        if plan.feasible and len(plan.waypoints) > 1:
            runs, _, _ = route_audit.land_runs(plan.waypoints)
            verdict = route_audit.verdict_of(runs)
            run = round(max(runs), 1) if runs else 0.0
        else:
            verdict, run = "not_found", None
        verdicts[verdict] += 1
        rows.append({"id": v["id"], "region": v["region"], "verdict": verdict,
                     "v12_verdict": v["guarded"]["verdict"], "v10_verdict": v["verdict"],
                     "max_land_run_km": run, "distance_km": round(plan.distance_km, 1),
                     "seconds": round(seconds, 2), "lattices": len(plan.attempts),
                     "warning": bool(plan.warning)})
        LOGGER.info("E7 %3d %-40s %s (v1.2 %s) %.1f s", v["id"], v["region"][:40], verdict,
                    v["guarded"]["verdict"], seconds)
    secs = [r["seconds"] for r in rows]
    return {"verdicts": verdicts, "mean_seconds": round(float(np.mean(secs)), 2),
            "max_seconds": round(float(np.max(secs)), 2),
            "changed": [r for r in rows if r["verdict"] != r["v12_verdict"]],
            "warnings": sum(r["warning"] for r in rows), "rows": rows}


def e5_stop_latency(planner) -> dict:
    rows = []
    cases = [("Jeddah → Dubai", "dynamic", 0.05), ("Jeddah → Dubai", "dijkstra", 0.04),
             ("Gdańsk → Bergen", "genetic", 0.05)]
    for course, algorithm, resolution in cases:
        start, end, _ = COURSES[course]
        params = SearchParameters.from_dict({"time_limit_s": 1.0, "max_grid_cells": 4_000_000,
                                             "ga_generations": 50_000, "ga_stagnation": 0})
        t0 = time.perf_counter()
        plan, _ = planner.plan(start, end, algorithm, resolution, params=params)
        wall = time.perf_counter() - t0
        rows.append({"course": course, "algorithm": algorithm, "resolution_deg": resolution,
                     "time_limit_s": 1.0, "status": plan.status, "seconds": round(wall, 3),
                     "latency_s": round(wall - 1.0, 3)})
        LOGGER.info("E5 %s %s: %s after %.2f s", course, algorithm, plan.status, wall)
    # The Stop button: cancel from another thread after 1 s.
    start, end, _ = COURSES["Jeddah → Dubai"]
    control = RunControl()
    import threading
    threading.Timer(1.0, control.cancel).start()
    t0 = time.perf_counter()
    plan, _ = planner.plan(start, end, "dijkstra", 0.04, control=control,
                           params=SearchParameters.from_dict({"max_grid_cells": 4_000_000}))
    wall = time.perf_counter() - t0
    rows.append({"course": "Jeddah → Dubai", "algorithm": "dijkstra", "resolution_deg": 0.04,
                 "stop_after_s": 1.0, "status": plan.status, "seconds": round(wall, 3),
                 "latency_s": round(wall - 1.0, 3)})
    return {"rows": rows}


# ---------------------------------------------------------------- figures
ZONE_CMAP = ListedColormap(["#cfe3f3", "#9cc9e6", "#e8d9a8", "#c9a36b"])


def fig_routes(planner, e2, path):
    names = list(COURSES)
    cols = 4
    rows_n = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2 * cols, 3.9 * rows_n))
    for ax, name in zip(axes.ravel(), names):
        start, end, _ = COURSES[name]
        data = e2[name]
        # The same request as the warm-up of the comparison: its map is the
        # shared cost map of the methods (deterministic).
        _, cost_map = planner.plan(start, end, "astar")
        spec = cost_map.spec
        ax.imshow(cost_map.class_index, origin="lower", cmap=ZONE_CMAP, vmin=-0.5, vmax=3.5,
                  extent=(spec.lon_min, spec.lon_max, spec.lat_min, spec.lat_max), aspect="auto")
        for r in data["results"]:
            if len(r.get("waypoints", [])) > 1:
                w = np.asarray(r["waypoints"])
                ax.plot(w[:, 1], w[:, 0], color=METHOD_COLOURS[r["algorithm"]],
                        lw=1.0 if r["algorithm"] == "great_circle" else 1.6,
                        ls="--" if r["algorithm"] == "great_circle" else "-",
                        label=METHOD_NAMES[r["algorithm"]])
        ax.plot([start[1]], [start[0]], "o", color="#2ca02c", ms=5)
        ax.plot([end[1]], [end[0]], "s", color="#d62728", ms=5)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=METHOD_COLOURS[k], lw=1.6, ls="--" if k == "great_circle" else "-",
                      label=METHOD_NAMES[k]) for k in METHOD_COLOURS]
    handles.append(Line2D([], [], color="none", label="(A*, Dijkstra and DP coincide)"))
    axes.ravel()[-1].legend(handles=handles, loc="center", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_step_limits(e3, path):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for algorithm, marker in (("astar", "o"), ("dijkstra", "s")):
        rows = [r for r in e3["rows"] if r["algorithm"] == algorithm]
        x = [r["max_steps"] for r in rows]
        y = [r["steps"] for r in rows]
        ax.plot(x, y, marker=marker, color=METHOD_COLOURS[algorithm], label=METHOD_NAMES[algorithm])
        for r in rows:
            ax.annotate("found" if r["status"] == "found" else "limit", (r["max_steps"], r["steps"]),
                        textcoords="offset points", xytext=(4, -11 if algorithm == "astar" else 5),
                        fontsize=7, color=METHOD_COLOURS[algorithm])
    ax.plot(STEP_LIMITS, STEP_LIMITS, ls=":", color="grey", lw=0.8, label="steps = limit")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("maximum number of steps (parameter)")
    ax.set_ylabel("steps made")
    ax.set_title(f"{e3['course']}, lattice {e3['grid']['resolution_deg']}° "
                 f"({e3['navigable_cells']:,} navigable cells)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def fig_ga(e4, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for waypoints, colour in zip(GA_WAYPOINTS, ("#ffbf69", "#ff9f1c", "#b5651d")):
        rows = [r for r in e4["rows"] if r["max_waypoints"] == waypoints]
        evals = [r["population"] * r["generations"] for r in rows]
        a1.plot(evals, [r["mean_gap_pct"] if r["mean_gap_pct"] is not None else np.nan for r in rows],
                "o-", color=colour, label=f"max {waypoints} waypoints")
        a2.plot(evals, [r["mean_seconds"] for r in rows], "o-", color=colour,
                label=f"max {waypoints} waypoints")
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("population × generations")
        ax.grid(alpha=0.3, which="both")
    a1.set_ylabel("cost above the A* optimum, %")
    a2.set_ylabel("search time, s")
    a2.set_yscale("log")
    a1.legend(fontsize=8)
    fig.suptitle(f"Genetic algorithm parameters, {e4['course']} (mean of 3 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for noisy in ("maritime_route",):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    planner = RoutePlanner(ZoneClassifierService.instance())
    out_path = FIGURE_DIR / "complex_routes.json"
    report = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    steps = sys.argv[1:] or ["e1", "e2", "e3", "e4", "e5", "e6", "e7"]
    t0 = time.perf_counter()
    if "e1" in steps:
        report["e1_planner_versions"] = e1_planner_versions(planner)
    if "e2" in steps:
        report["e2_methods"] = e2_methods(planner)
    if "e3" in steps:
        report["e3_step_limits"] = e3_step_limits(planner)
    if "e4" in steps:
        report["e4_genetic"] = e4_genetic(planner)
    if "e5" in steps:
        report["e5_stop"] = e5_stop_latency(planner)
    if "e6" in steps:
        report["e6_via"] = e6_via_points(planner)
    if "e7" in steps:
        report["e7_audit"] = e7_audit(planner)
    report["seconds"] = round(time.perf_counter() - t0, 1)
    out_path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    if "e2_methods" in report:
        fig_routes(planner, report["e2_methods"], FIGURE_DIR / "fig_complex_routes.png")
    if "e3_step_limits" in report:
        fig_step_limits(report["e3_step_limits"], FIGURE_DIR / "fig_step_limits.png")
    if "e4_genetic" in report:
        fig_ga(report["e4_genetic"], FIGURE_DIR / "fig_ga_parameters.png")
    LOGGER.info("written %s", out_path)


if __name__ == "__main__":
    main()
