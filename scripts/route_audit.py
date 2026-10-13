"""Audit of the route planner on 100 random voyages in the eleven sea regions.

Purpose
-------
The supervisor asked whether the application ever lays a course incorrectly.
This script answers the question with an experiment that does not rely on the
neural network for the verdict:

1. 100 voyages are drawn at random (fixed seed) between water points of the
   labelled corpus, distributed over the eleven sea regions; both ends must
   also be water according to an independent land mask and lie 150-1200 km
   apart.
2. Every voyage is planned exactly as a user would plan it: A* on the default
   lattice of 0.25 deg with automatic refinement.
3. Every route is sampled every kilometre and each sample is tested against
   the GLOBE-based 1 km land mask of the package ``global-land-mask``. The
   mask is independent of the classifier, so it plays the role of the ground
   truth. Its resolution is about 1 km: a channel narrower than that (the
   Bosporus, parts of the Danish straits) appears as land.
4. Each voyage receives one of four verdicts:

   ``correct``   no sample on land;
   ``grazing``   land runs shorter than ``CROSSING_KM`` - the route touches a
                 coast, a narrow channel or a small islet at the resolution of
                 the mask;
   ``crossing``  at least one continuous land run of ``CROSSING_KM`` or more -
                 the course is laid incorrectly;
   ``not_found`` the planner returned no route.

5. Voyages with the verdict ``crossing`` or ``not_found`` are planned again on
   a 0.1 deg lattice, to see whether a finer lattice removes the error.

   Steps 2-5 are run with the planner's own coastline check switched off
   (``coast_guard=False``), i.e. as the program worked before version 1.2.
   Step 7 repeats the test with the check switched on.
7. Every voyage is planned once more with the automatic coastline check of
   version 1.2 (:mod:`maritime_route.routing.coast_check`): a route that
   crosses land is corrected by blocking cells and searching again, then by a
   finer lattice; a route that still crosses land carries a warning. The
   result is judged by the same land-mask test as in step 3.
6. The cause of every error is attributed automatically:

   * a land run is put down to the **lattice** when the classifier, asked at
     the sample point itself, also answers "land": the route steps diagonally
     between two water cells across land narrower than a cell; it is put
     down to the **classifier** when the classifier answers "water" there;
   * a voyage without a route is checked for a sea connection on the land
     mask (2 km raster, the bounding box of the two ports widened by 8 deg).
     ``no_open_connection``: the ports are not connected by water wider than
     the raster resolves - for example a Caribbean and a Pacific port (the
     Panama Canal is not modelled) or the Black Sea and the Mediterranean
     (the Bosporus is under 1 km wide); ``classifier``: the classifier labels
     the area of a port as land; ``closed_on_map``: a connection exists, but
     the classified lattice closes it (A* is complete, so a passage open on
     the cost map would have been found).

Outputs
-------
``docs/figures/route_audit.json``  every voyage with its numbers and verdict;
``docs/figures/route_audit.csv``   the same as a table (one row per voyage);
``docs/figures/fig_audit_<region>.png``  maps of the voyages of each region;
``docs/figures/fig_audit_errors.png``    the incorrect courses on the lattice;
``docs/figures/fig_audit_guard.png``     incorrect courses and their automatic
                                          correction by the coastline check.

Usage::

    pip install global-land-mask
    python scripts/route_audit.py            # about 20 minutes on a CPU

对 11 个海域中随机抽取的 100 条航线做自动测试：用默认参数规划，再用独立的
1 km 陆地掩膜（global-land-mask，基于 GLOBE 高程模型）逐公里检查航线是否穿过陆地，
并记录每条航线的结果。
"""
from __future__ import annotations

import csv
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from maritime_route.config import FIGURE_DIR, RAW_DIR, REGIONS, WATER_CLASSES  # noqa: E402
from maritime_route.data.ais_loader import load_geojson, validate  # noqa: E402
from maritime_route.data.cleaning import assign_regions, conflict_mask  # noqa: E402
from maritime_route.model.inference import ZoneClassifierService  # noqa: E402
from maritime_route.routing.geodesy import haversine_km  # noqa: E402
from maritime_route.routing.planner import RoutePlanner  # noqa: E402

try:
    from global_land_mask import globe
except ImportError:                                   # pragma: no cover
    sys.exit("This script needs the land mask: pip install global-land-mask")

#: Number of voyages and the random seed. 航线数量与随机种子。
N_VOYAGES = 100
SEED = 2026
#: Great-circle distance between the two ends, km.
MIN_KM, MAX_KM = 150.0, 1200.0
#: Sampling step along a route and the length ignored at each port, km.
STEP_KM, PORT_KM = 1.0, 5.0
#: A continuous land run of this length or more makes the course incorrect, km.
CROSSING_KM = 3.0
#: Lattice used to re-plan the incorrect voyages, degrees.
FINE_DEG = 0.1


def mask_is_water(lat, lon):
    """True where the independent 1 km land mask shows water."""
    return ~globe.is_land(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))


def clear_water(lat, lon, radius_deg=0.03):
    """Water at the point and at four points around it (about 3 km)."""
    ok = mask_is_water(lat, lon)
    for dlat, dlon in ((radius_deg, 0), (-radius_deg, 0), (0, radius_deg), (0, -radius_deg)):
        ok &= mask_is_water(lat + dlat, lon + dlon)
    return ok


def draw_voyages(rng):
    """Draw the voyages: water points of the corpus, both ends clear water."""
    frame, _ = validate(load_geojson(RAW_DIR / "merged.geojson"), source="raw")
    frame = frame[~conflict_mask(frame)].reset_index(drop=True)
    frame = frame[frame["class_code"].isin(WATER_CLASSES)].reset_index(drop=True)
    lat, lon = frame["latitude"].to_numpy(), frame["longitude"].to_numpy()
    region = assign_regions(lat, lon)
    good = clear_water(lat, lon)
    names = [name for name, _ in REGIONS]
    voyages = []
    k = 0
    while len(voyages) < N_VOYAGES:
        name = names[k % len(names)]
        k += 1
        idx = np.flatnonzero((region == name) & good)
        for _ in range(10000):
            a, b = rng.choice(idx, 2, replace=False)
            d = float(haversine_km(lat[a], lon[a], lat[b], lon[b]))
            if MIN_KM <= d <= MAX_KM:
                voyages.append({"id": len(voyages) + 1, "region": name,
                                "start": [round(float(lat[a]), 4), round(float(lon[a]), 4)],
                                "end": [round(float(lat[b]), 4), round(float(lon[b]), 4)],
                                "great_circle_km": round(d, 1)})
                break
    return voyages


def land_runs(waypoints):
    """Lengths (km) of the continuous land runs along a route.

    The poly-line is sampled every ``STEP_KM``; ``PORT_KM`` at each end is
    ignored, because a port lies on the coast by definition.
    """
    pts = []
    for (a1, b1), (a2, b2) in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(math.ceil(float(haversine_km(a1, b1, a2, b2)) / STEP_KM)))
        t = np.arange(n) / n
        pts.append(np.column_stack([a1 + (a2 - a1) * t, b1 + (b2 - b1) * t]))
    pts.append(np.array([waypoints[-1]]))
    pts = np.vstack(pts)
    seg = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    dist = np.r_[0.0, np.cumsum(seg)]
    keep = (dist >= PORT_KM) & (dist <= dist[-1] - PORT_KM)
    land = ~mask_is_water(pts[:, 0], pts[:, 1]) & keep
    runs, current = [], 0.0
    for i in range(1, len(pts)):
        if land[i] and land[i - 1]:
            current += seg[i - 1]
        elif land[i]:
            current = STEP_KM
        if not land[i] and current > 0:
            runs.append(current)
            current = 0.0
    if current > 0:
        runs.append(current)
    return runs, int(land.sum()), int(keep.sum())


def sea_connected(start, end, step=0.02, margin=8.0):
    """True if the land mask connects the two ports by water (2 km raster)."""
    from scipy import ndimage
    lat0, lat1 = min(start[0], end[0]) - margin, max(start[0], end[0]) + margin
    lon0, lon1 = min(start[1], end[1]) - margin, max(start[1], end[1]) + margin
    lats = np.arange(lat0, lat1 + step, step)
    lons = np.arange(lon0, lon1 + step, step)
    gl, ga = np.meshgrid(lons, lats)
    labels, _ = ndimage.label(mask_is_water(ga, gl))
    def cell(p):
        return int(round((p[0] - lat0) / step)), int(round((p[1] - lon0) / step))
    a, b = labels[cell(start)], labels[cell(end)]
    return bool(a and a == b)


def land_samples(waypoints):
    """Coordinates of the samples on land (the same sampling as land_runs)."""
    pts = []
    for (a1, b1), (a2, b2) in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(math.ceil(float(haversine_km(a1, b1, a2, b2)) / STEP_KM)))
        t = np.arange(n) / n
        pts.append(np.column_stack([a1 + (a2 - a1) * t, b1 + (b2 - b1) * t]))
    pts = np.vstack(pts + [np.array([waypoints[-1]])])
    seg = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    dist = np.r_[0.0, np.cumsum(seg)]
    keep = (dist >= PORT_KM) & (dist <= dist[-1] - PORT_KM)
    return pts[keep & ~mask_is_water(pts[:, 0], pts[:, 1])]


def crossing_cause(classifier, waypoints):
    """'lattice' or 'classifier', by the classifier's answer at the land samples."""
    from maritime_route.config import CLASS_NAMES
    pts = land_samples(waypoints)
    water = np.array([c in WATER_CLASSES for c in CLASS_NAMES])
    pred = classifier.predict_coordinates(pts[:, 0], pts[:, 1]).argmax(axis=1)
    taken_for_water = float(water[pred].mean())
    return ("classifier" if taken_for_water >= 0.5 else "lattice"), round(taken_for_water, 2)


def not_found_cause(classifier, voyage):
    """'no_open_connection', 'classifier' (a port area labelled land) or 'closed_on_map'."""
    from maritime_route.config import CLASS_NAMES
    if not sea_connected(voyage["start"], voyage["end"]):
        return "no_open_connection"
    water = np.array([c in WATER_CLASSES for c in CLASS_NAMES])
    # Classifier's answer in a 30 km neighbourhood of each port. 港口附近 30 km 内的分类结果。
    for p in (voyage["start"], voyage["end"]):
        offs = np.linspace(-0.27, 0.27, 7)
        la, lo = np.meshgrid(p[0] + offs, p[1] + offs / max(math.cos(math.radians(p[0])), 0.2))
        pred = classifier.predict_coordinates(la.ravel(), lo.ravel()).argmax(axis=1)
        if water[pred].mean() < 0.2:
            return "classifier"
    return "closed_on_map"


def verdict_of(runs):
    if not runs:
        return "correct"
    return "crossing" if max(runs) >= CROSSING_KM else "grazing"


def plan_and_check(planner, voyage, resolution=None, guard=False):
    """Plan one voyage and test the route against the land mask.

    ``guard`` switches the planner's own coastline check on (version 1.2).
    """
    t0 = time.perf_counter()
    try:
        plan, cost_map = planner.plan(tuple(voyage["start"]), tuple(voyage["end"]), "astar",
                                      resolution, coast_guard=guard)
    except Exception as exc:                          # a port the planner cannot use
        return {"verdict": "not_found", "message": str(exc)[:160],
                "seconds": round(time.perf_counter() - t0, 2)}, None, None
    seconds = round(time.perf_counter() - t0, 2)
    if not plan.feasible or len(plan.waypoints) < 2:
        return {"verdict": "not_found", "message": plan.message[:160], "seconds": seconds}, plan, cost_map
    runs, on_land, samples = land_runs(plan.waypoints)
    return {
        "verdict": verdict_of(runs),
        "distance_km": round(plan.distance_km, 1),
        "detour_pct": round(100 * (plan.distance_km / plan.great_circle_km - 1), 1),
        "resolution_deg": round(plan.grid["resolution_deg"], 4),
        "samples": samples,
        "samples_on_land": on_land,
        "land_runs_km": [round(r, 1) for r in sorted(runs, reverse=True)[:5]],
        "max_land_run_km": round(max(runs), 1) if runs else 0.0,
        "classifier_interior_land": plan.interior_land_waypoints,
        "seconds": seconds,
        "coast_check": {k: plan.coast_check.get(k) for k in
                        ("verdict", "initial_verdict", "corrections", "refined", "cells_blocked")},
        "warning": bool(plan.warning),
    }, plan, cost_map


# ------------------------------------------------------------------ figures
COLOR = {"correct": "#1a9850", "grazing": "#fdae61", "crossing": "#d73027", "not_found": "#000000"}


def land_background(ax, box):
    """Land of the independent mask as a grey raster."""
    lat0, lat1, lon0, lon1 = box
    lats = np.linspace(lat0, lat1, 300)
    lons = np.linspace(lon0, lon1, 400)
    grid_lon, grid_lat = np.meshgrid(lons, lats)
    land = globe.is_land(grid_lat, grid_lon)
    ax.imshow(land, origin="lower", extent=(lon0, lon1, lat0, lat1), cmap="Greys",
              vmin=0, vmax=2.2, aspect="auto", interpolation="nearest")


def region_figure(region, rows, plans, path):
    lats = [p for r in rows for p in (r["start"][0], r["end"][0])]
    lons = [p for r in rows for p in (r["start"][1], r["end"][1])]
    for r in rows:
        plan = plans.get(r["id"])
        if plan is not None and plan.waypoints:
            lats += [w[0] for w in plan.waypoints]
            lons += [w[1] for w in plan.waypoints]
    pad = 1.0
    box = (min(lats) - pad, max(lats) + pad, min(lons) - pad, max(lons) + pad)
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    land_background(ax, box)
    for r in rows:
        plan = plans.get(r["id"])
        colour = COLOR[r["verdict"]]
        if plan is not None and plan.waypoints:
            w = np.array(plan.waypoints)
            ax.plot(w[:, 1], w[:, 0], color=colour, lw=1.4)
        ax.plot(*r["start"][::-1], "o", ms=3.5, color=colour)
        ax.plot(*r["end"][::-1], "s", ms=3.5, color=colour)
        mid = plan.waypoints[len(plan.waypoints) // 2] if plan is not None and plan.waypoints else r["start"]
        ax.annotate(str(r["id"]), (mid[1], mid[0]), fontsize=7, color="#111",
                    xytext=(2, 2), textcoords="offset points")
    ax.set_xlim(box[2], box[3])
    ax.set_ylim(box[0], box[1])
    # Equal distances in both directions at the middle latitude. 按中纬度保持横纵比例。
    ax.set_aspect(1.0 / max(math.cos(math.radians(0.5 * (box[0] + box[1]))), 0.2))
    ax.set_xlabel("Longitude, deg")
    ax.set_ylabel("Latitude, deg")
    counts = {v: sum(r["verdict"] == v for r in rows) for v in COLOR}
    ax.set_title(f"{region}: correct {counts['correct']}, grazing {counts['grazing']}, "
                 f"crossing {counts['crossing']}, not found {counts['not_found']}", fontsize=8.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=c, lw=2, label=v.replace("_", " ")) for v, c in COLOR.items()],
              fontsize=6.5, loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def error_figure(rows, plans, maps, path):
    """The incorrect courses over the classified lattice of their cost map."""
    bad = [r for r in rows if r["verdict"] == "crossing"][:6]
    if not bad:
        return False
    n = len(bad)
    cols = 3 if n > 2 else n
    nrows = math.ceil(n / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(8.3, 3.0 * nrows), squeeze=False)
    from maritime_route.config import CLASS_COLOR, CLASS_NAMES
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([CLASS_COLOR[c] for c in CLASS_NAMES])
    for ax, r in zip(axes.flat, bad):
        cm = maps[r["id"]]
        spec = cm.spec
        ax.imshow(cm.class_index, origin="lower", cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto",
                  extent=(spec.lon_min, spec.lon_max, spec.lat_min, spec.lat_max), interpolation="nearest")
        lats = np.linspace(spec.lat_min, spec.lat_max, 300)
        lons = np.linspace(spec.lon_min, spec.lon_max, 400)
        gl, ga = np.meshgrid(lons, lats)
        ax.contour(gl, ga, globe.is_land(ga, gl).astype(float), levels=[0.5], colors="black", linewidths=0.6)
        w = np.array(plans[r["id"]].waypoints)
        ax.plot(w[:, 1], w[:, 0], color="#d73027", lw=1.6)
        ax.set_title(f"No. {r['id']} - {r['region'].split(' (')[0]}\nlongest land run "
                     f"{r['max_land_run_km']} km", fontsize=7.5)
        ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def guard_figure(rows, plans, guarded, path):
    """Incorrect courses (red) and the routes after the coastline check (blue)."""
    # The six longest land runs of the test (the most instructive cases).
    bad = sorted([r for r in rows if r["verdict"] == "crossing" and r["id"] in guarded],
                 key=lambda r: -r["max_land_run_km"])[:6]
    if not bad:
        return False
    n = len(bad)
    cols = 3 if n > 2 else n
    nrows = math.ceil(n / cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(8.3, 3.1 * nrows), squeeze=False)
    for ax, r in zip(axes.flat, bad):
        old_w = np.array(plans[r["id"]].waypoints)
        new_w = np.array(guarded[r["id"]].waypoints)
        both = np.vstack([old_w, new_w])
        pad = 0.4
        box = (both[:, 0].min() - pad, both[:, 0].max() + pad,
               both[:, 1].min() - pad, both[:, 1].max() + pad)
        land_background(ax, box)
        ax.plot(old_w[:, 1], old_w[:, 0], color="#d73027", lw=1.5, label="without the check")
        ax.plot(new_w[:, 1], new_w[:, 0], color="#2166ac", lw=1.3, label="with the check")
        ax.set_xlim(box[2], box[3])
        ax.set_ylim(box[0], box[1])
        ax.set_aspect(1.0 / max(math.cos(math.radians(0.5 * (box[0] + box[1]))), 0.2))
        g = r["guarded"]
        ax.set_title(f"No. {r['id']} - {r['region'].split(' (')[0]}\n"
                     f"land {r['max_land_run_km']} km -> {g.get('max_land_run_km', 0)} km "
                     f"({g['verdict']})", fontsize=7.5)
        ax.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6.5, loc="lower left")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return True


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    rng = np.random.default_rng(SEED)
    voyages = draw_voyages(rng)
    planner = RoutePlanner(ZoneClassifierService.instance())
    rows, plans, maps, guarded = [], {}, {}, {}
    for v in voyages:
        result, plan, cost_map = plan_and_check(planner, v)
        row = {**v, **result}
        if row["verdict"] == "crossing":
            row["cause"], row["share_taken_for_water"] = crossing_cause(planner.classifier, plan.waypoints)
        elif row["verdict"] == "not_found":
            row["cause"] = not_found_cause(planner.classifier, v)
        if row["verdict"] in ("crossing", "not_found"):
            fine, _, _ = plan_and_check(planner, v, FINE_DEG)
            row["fine"] = {k: fine.get(k) for k in ("verdict", "distance_km", "max_land_run_km", "resolution_deg")}
        # Step 7: the same voyage with the coastline check of version 1.2.
        g, gplan, _ = plan_and_check(planner, v, guard=True)
        row["guarded"] = {k: g.get(k) for k in ("verdict", "distance_km", "detour_pct",
                                                "max_land_run_km", "resolution_deg",
                                                "seconds", "coast_check", "warning")}
        if gplan is not None and gplan.waypoints:
            guarded[v["id"]] = gplan
        rows.append(row)
        if plan is not None:
            plans[v["id"]] = plan
            maps[v["id"]] = cost_map
        print(f"{v['id']:3d} {v['region'][:28]:28s} {row['verdict']:9s} "
              f"{row.get('distance_km', '-')!s:>7} km  max land {row.get('max_land_run_km', '-')}"
              + (f"  cause {row['cause']}" if "cause" in row else "")
              + (f"  -> 0.1 deg: {row['fine']['verdict']}" if "fine" in row else "")
              + f"  | with check: {g['verdict']}", flush=True)

    verdicts = ("correct", "grazing", "crossing", "not_found")
    by_region = {}
    for name, _ in REGIONS:
        sub = [r for r in rows if r["region"] == name]
        by_region[name] = {"n": len(sub), **{k: sum(r["verdict"] == k for r in sub) for k in verdicts}}
    fixed = [r for r in rows if "fine" in r and r["fine"]["verdict"] in ("correct", "grazing")]
    summary = {
        "n_voyages": len(rows), "seed": SEED, "min_km": MIN_KM, "max_km": MAX_KM,
        "crossing_threshold_km": CROSSING_KM, "port_margin_km": PORT_KM, "step_km": STEP_KM,
        "fine_resolution_deg": FINE_DEG,
        "verdicts": {k: sum(r["verdict"] == k for r in rows) for k in verdicts},
        "by_region": by_region,
        "fixed_by_fine_lattice": len(fixed),
        "causes": {
            "crossing": {c: sum(r.get("cause") == c for r in rows if r["verdict"] == "crossing")
                         for c in ("lattice", "classifier")},
            "not_found": {c: sum(r.get("cause") == c for r in rows if r["verdict"] == "not_found")
                          for c in ("no_open_connection", "classifier", "closed_on_map")},
        },
        "fixed_by_fine_lattice_by_cause": {
            c: sum(1 for r in fixed if r.get("cause") == c) for c in ("lattice", "classifier", "closed_on_map")},
        "mean_detour_pct": round(float(np.mean([r["detour_pct"] for r in rows if "detour_pct" in r])), 1),
        "mean_seconds": round(float(np.mean([r["seconds"] for r in rows])), 2),
        "max_seconds": round(float(max(r["seconds"] for r in rows)), 2),
        "refined": sum(1 for r in rows if r.get("resolution_deg", 0.25) < 0.25),
    }
    # ---- step 7: the same voyages with the coastline check ---------------- #
    g_rows = [r["guarded"] for r in rows]
    summary["with_coast_check"] = {
        "verdicts": {k: sum(g["verdict"] == k for g in g_rows) for k in verdicts},
        "by_region": {name: {k: sum(r["guarded"]["verdict"] == k for r in rows if r["region"] == name)
                             for k in verdicts} for name, _ in REGIONS},
        # Routes the check found crossing land and changed (initial verdict 'crossing').
        "corrected": sum(1 for g in g_rows if (g.get("coast_check") or {}).get("initial_verdict") == "crossing"
                         and g["verdict"] in ("correct", "grazing")),
        "corrected_by_refinement": sum(1 for g in g_rows if (g.get("coast_check") or {}).get("refined")
                                       and g["verdict"] in ("correct", "grazing")),
        "warnings": sum(1 for g in g_rows if g.get("warning")),
        # Crossings of the first test that the check removed / left.
        "crossings_removed": sum(1 for r in rows if r["verdict"] == "crossing"
                                 and r["guarded"]["verdict"] in ("correct", "grazing")),
        "crossings_left": sum(1 for r in rows if r["verdict"] == "crossing"
                              and r["guarded"]["verdict"] == "crossing"),
        "crossings_now_not_found": sum(1 for r in rows if r["verdict"] == "crossing"
                                       and r["guarded"]["verdict"] == "not_found"),
        "unchanged_correct": sum(1 for r in rows if r["verdict"] in ("correct", "grazing")
                                 and r["guarded"]["verdict"] == r["verdict"]),
        "mean_detour_pct": round(float(np.mean([g["detour_pct"] for g in g_rows
                                                if g.get("detour_pct") is not None])), 1),
        "mean_seconds": round(float(np.mean([g["seconds"] for g in g_rows])), 2),
        "max_seconds": round(float(max(g["seconds"] for g in g_rows)), 2),
        "extra_km_of_corrected": round(float(np.mean(
            [r["guarded"]["distance_km"] - r["distance_km"] for r in rows
             if r["verdict"] == "crossing" and r["guarded"].get("distance_km")])), 1)
        if any(r["verdict"] == "crossing" and r["guarded"].get("distance_km") for r in rows) else None,
    }
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIGURE_DIR / "route_audit.json").write_text(json.dumps({"summary": summary, "voyages": rows}, indent=1),
                                                 encoding="utf-8")
    with open(FIGURE_DIR / "route_audit.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "region", "start_lat", "start_lon", "end_lat", "end_lon", "great_circle_km",
                    "route_km", "detour_pct", "resolution_deg", "verdict", "max_land_run_km",
                    "samples_on_land", "cause", "seconds", "verdict_at_0.1deg",
                    "verdict_with_check", "route_km_with_check", "max_land_run_km_with_check",
                    "corrections", "warning"])
        for r in rows:
            w.writerow([r["id"], r["region"], *r["start"], *r["end"], r["great_circle_km"],
                        r.get("distance_km", ""), r.get("detour_pct", ""), r.get("resolution_deg", ""),
                        r["verdict"], r.get("max_land_run_km", ""), r.get("samples_on_land", ""),
                        r.get("cause", ""),
                        r["seconds"], r.get("fine", {}).get("verdict", ""),
                        r["guarded"]["verdict"], r["guarded"].get("distance_km", ""),
                        r["guarded"].get("max_land_run_km", ""),
                        (r["guarded"].get("coast_check") or {}).get("corrections", ""),
                        int(bool(r["guarded"].get("warning")))])
    for name, _ in REGIONS:
        sub = [r for r in rows if r["region"] == name]
        slug = name.split(" (")[0].lower().replace(" and ", "_").replace(" ", "_")
        region_figure(name, sub, plans, FIGURE_DIR / f"fig_audit_{slug}.png")
    error_figure(rows, plans, maps, FIGURE_DIR / "fig_audit_errors.png")
    guard_figure(rows, plans, guarded, FIGURE_DIR / "fig_audit_guard.png")
    print("\n", json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
