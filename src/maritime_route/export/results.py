"""Result files: every search and every comparison is saved as JSON.

搜索结果文件：每次搜索（无论成功、失败还是被停止）和每次算法对比都保存为 JSON 文件。

Why JSON
--------
The file is meant for further processing (the supervisor's requirement):
JSON keeps the structure - the request, the parameters, the status, the
timing and the list of points with their coordinates and zones - in one
self-describing document that Python (``json.load``), MATLAB
(``jsondecode``), Excel / Power Query, JavaScript or any other tool reads
without a custom parser. A CSV file would hold only the table of points and
lose the rest. The GeoJSON and CSV exports of the route remain available in
the interface for GIS software and spreadsheets.

Where
-----
``OUTPUT_DIR/results`` (``outputs/results`` in the project folder), or the
directory given by the environment variable ``MR_RESULTS_DIR``. The absolute
path of every file is returned to the interface and shown on screen with a
*Copy path* button and a download link.

File names
----------
``route_<start>_<end>_<algorithm>_<status>_<seconds>s_<UTC time>.json``, for
example::

    route_N54.3233E10.1228_N59.4370E24.7536_astar_found_2.14s_20261014-101530.json
    route_N65.0100E25.4000_N54.3300E10.1400_dijkstra_step_limit_0.31s_20261014-101702.json
    compare_N54.3233E10.1228_N59.4370E24.7536_5-methods_12.80s_20261014-102011.json

so that the files of one voyage sort together and the outcome and the search
time can be read from the name without opening the file. A multi-leg route
adds ``_via<k>`` after the end point.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import APP_VERSION, OUTPUT_DIR

#: Directory of the result files. 结果文件目录。
RESULTS_DIR: Path = Path(os.environ.get("MR_RESULTS_DIR", OUTPUT_DIR / "results"))
#: Identifier written into every file, so that a reader can recognise it.
FORMAT_ROUTE = "maritime-route-result"
FORMAT_COMPARISON = "maritime-route-comparison"
FORMAT_VERSION = 1

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.+-]+\.json$")


def coordinate_tag(lat: float, lon: float) -> str:
    """``N54.3233E10.1228`` - a point as it appears in a file name."""
    return (f"{'N' if lat >= 0 else 'S'}{abs(lat):.4f}"
            f"{'E' if lon >= 0 else 'W'}{abs(lon):.4f}")


def _stamp(t: Optional[float] = None) -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.gmtime(t))


def _finite(value):
    """JSON has no infinity / NaN: write ``null`` instead."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean(value: Any) -> Any:
    """Recursively replace non-finite floats and tuples for ``json.dump``."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):   # NumPy scalar
        try:
            value = value.item()
        except (TypeError, ValueError):
            pass
    return _finite(value)


def unique_path(directory: Path, name: str) -> Path:
    """``directory/name``, with ``-2``, ``-3``... appended if the file exists."""
    path = directory / name
    k = 2
    while path.exists():
        path = directory / f"{name[:-5]}-{k}.json"
        k += 1
    return path


def route_file_name(start: Sequence[float], end: Sequence[float], algorithm: str,
                    status: str, seconds: float, n_via: int = 0,
                    t: Optional[float] = None) -> str:
    """File name of a route result (see the module docstring)."""
    via = f"_via{n_via}" if n_via else ""
    return (f"route_{coordinate_tag(*start)}_{coordinate_tag(*end)}{via}_{algorithm}_"
            f"{status}_{seconds:.2f}s_{_stamp(t)}.json")


def route_document(plan, request: Optional[Dict] = None) -> Dict[str, Any]:
    """Content of the result file of one search.

    Parameters
    ----------
    plan : RoutePlan
        Outcome of :meth:`~maritime_route.routing.planner.RoutePlanner.plan`
        (found, not found, step limit, stopped by the user or by the time limit).
    request : dict, optional
        The request as received (resolution, speed...); stored for reproducibility.

    Returns
    -------
    dict
        ``format``, ``request``, ``parameters``, ``status``, ``timing``,
        ``search`` (steps, lattices tried), ``route`` (length, cost, zones,
        coastline check) and ``points`` - every waypoint with its coordinates,
        zone, cumulative distance and course.
    """
    from ..routing.control import STATUS_TEXT

    finished = time.time()
    points: List[Dict[str, Any]] = []
    for i, leg in enumerate(plan.legs):
        points.append({
            "index": i,
            "latitude": round(leg.latitude, 6),
            "longitude": round(leg.longitude, 6),
            "zone": leg.class_code,
            "zone_cost": _finite(round(leg.zone_cost, 3)),
            "cumulative_km": round(leg.cumulative_km, 3),
            "bearing_deg": round(leg.bearing_deg, 1),
        })
    endpoints = ([{"role": "start", "latitude": plan.start[0], "longitude": plan.start[1]}]
                 + [{"role": f"via {k + 1}", "latitude": p[0], "longitude": p[1]}
                    for k, p in enumerate(plan.via)]
                 + [{"role": "end", "latitude": plan.end[0], "longitude": plan.end[1]}])
    return _clean({
        "format": FORMAT_ROUTE,
        "format_version": FORMAT_VERSION,
        "application": f"MaritimeRouteAI {APP_VERSION}",
        "route_id": plan.route_id,
        "request": {
            "start": list(plan.start), "end": list(plan.end),
            "via": [list(p) for p in plan.via],
            "algorithm": plan.algorithm,
            "resolution_deg": (request or {}).get("resolution_deg"),
            "speed_knots": plan.speed_knots,
            **{k: v for k, v in (request or {}).items()
               if k in ("simplify_km",)},
        },
        "parameters": plan.parameters,
        "status": plan.status,
        "status_text": STATUS_TEXT.get(plan.status, plan.status),
        "found": bool(plan.feasible),
        "message": plan.message,
        "warning": plan.warning,
        "timing": {
            "search_seconds": round(plan.runtime_s, 4),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                         time.gmtime(finished - plan.runtime_s)),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(finished)),
        },
        "search": {
            "steps": plan.nodes_expanded,
            "steps_meaning": {"genetic": "fitness evaluations",
                              "dynamic": "Bellman updates (sweeps x cells)"}.get(
                                  plan.algorithm, "cells expanded"),
            "nodes_generated": plan.nodes_generated,
            "step_limit": (plan.parameters or {}).get({
                "astar": "astar_max_steps", "dijkstra": "dijkstra_max_steps",
                "dynamic": "dp_max_sweeps", "genetic": "ga_generations"}.get(plan.algorithm, ""), None),
            "lattice": plan.grid,
            "attempts": plan.attempts,
        },
        "route": {
            "distance_km": round(plan.distance_km, 3),
            "great_circle_km": round(plan.great_circle_km, 3),
            "detour_ratio": plan.detour_ratio,
            "total_cost": plan.total_cost,
            "estimated_hours": plan.estimated_hours,
            "zone_profile": plan.zone_profile,
            "coast_check": {k: v for k, v in (plan.coast_check or {}).items()},
        },
        "endpoints": endpoints,
        "n_points": len(points),
        "points": points,
    })


def save_route_result(plan, request: Optional[Dict] = None,
                      directory: Optional[Path] = None) -> Path:
    """Write the result file of one search and return its absolute path."""
    directory = Path(directory or RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    name = route_file_name(plan.start, plan.end, plan.algorithm, plan.status,
                           plan.runtime_s, len(plan.via))
    path = unique_path(directory, name)
    document = route_document(plan, request)
    document["file"] = str(path.resolve())
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return path.resolve()


def save_comparison_result(data: Dict, directory: Optional[Path] = None) -> Path:
    """Write the result file of an algorithm comparison (see ``planner.benchmark``)."""
    directory = Path(directory or RESULTS_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    n = len(data.get("results", []))
    name = (f"compare_{coordinate_tag(*data['start'])}_{coordinate_tag(*data['end'])}_"
            f"{n}-methods_{data.get('total_seconds', 0.0):.2f}s_{_stamp()}.json")
    path = unique_path(directory, name)
    document = _clean({"format": FORMAT_COMPARISON, "format_version": FORMAT_VERSION,
                       "application": f"MaritimeRouteAI {APP_VERSION}",
                       "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "file": str(path.resolve()), **data})
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return path.resolve()


def list_results(limit: int = 50, directory: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Most recent result files, newest first (name, absolute path, size, time)."""
    directory = Path(directory or RESULTS_DIR)
    if not directory.exists():
        return []
    files = sorted(directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"name": p.name, "path": str(p.resolve()), "size_kb": round(p.stat().st_size / 1024, 1),
             "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))}
            for p in files[:limit]]


def result_path(name: str, directory: Optional[Path] = None) -> Optional[Path]:
    """Path of a result file by its name; ``None`` for an unknown or unsafe name."""
    if not _SAFE_NAME.match(name):
        return None
    path = Path(directory or RESULTS_DIR) / name
    return path if path.is_file() else None


def load_result(path) -> Tuple[Dict[str, Any], List[Tuple[float, float]]]:
    """Read a result file back: the document and its points as ``(lat, lon)``.

    Example of further processing::

        doc, pts = load_result("outputs/results/route_...json")
        print(doc["status"], doc["timing"]["search_seconds"], len(pts))
    """
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    points = [(p["latitude"], p["longitude"]) for p in document.get("points", [])]
    return document, points
