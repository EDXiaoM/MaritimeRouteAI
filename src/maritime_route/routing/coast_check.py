"""Independent coastline check of a planned route.

航线的独立海岸线检查：规划完成后，用与神经网络无关的 1 km 陆地掩膜逐公里检查航线。

Why
---
The planner only knows what the neural classifier tells it. The test on
100 random voyages (``scripts/route_audit.py``, section 3.5.6 of the note)
showed that the classifier sometimes takes sparsely sampled land - small
islands, narrow peninsulas - for water, and that on a coarse lattice a
diagonal step can cut a corner of land between two water cells. The route
is then optimal *on the classified lattice* but crosses land in reality.

This module is the safeguard against such courses. After every search the
route is sampled every :data:`STEP_KM` kilometres and each sample is looked
up in an independent land mask: the 1 km GLOBE-based raster of the package
``global-land-mask`` (Karin, 2020; Hastings and Dunbar, 1999). The mask
plays no part in training or classification, so it is an outside witness.

The planner (:meth:`~.planner.RoutePlanner.plan`) uses the verdict to
correct the route automatically:

1. the lattice cells under the land samples, and the route vertices of the
   offending legs, are made impassable and the same method searches again;
2. if that does not help, the lattice is made finer and the route planned
   again;
3. if the route still crosses land, it is returned with a warning that the
   UI, the PDF report and the database show to the navigator.

Verdicts
--------
``clear``     no sample on land;
``grazing``   land runs shorter than :data:`CROSSING_KM` - the route touches
              a coast, a channel narrower than the mask resolves, or an islet;
              accepted;
``crossing``  at least one continuous land run of :data:`CROSSING_KM` or more
              - the course is incorrect;
``unchecked`` the land mask is not installed (``pip install global-land-mask``);
              the planner then works exactly as before.

The first and the last :data:`PORT_KM` kilometres are not checked, because
a voyage begins and ends at a quay, which lies on the coast by definition.

Port approaches. The planner joins the exact quay to the centre of the
nearest navigable lattice cell by a straight connector (up to 60 km, see
``RoutePlanner.SNAP_RADIUS_KM``). When the quay lies inside a harbour basin,
a canal or a city centre, that connector necessarily runs over land on a
1 km mask. A land stretch that begins at the quay (the mask shows land
right up to the unchecked :data:`PORT_KM`) and ends within the connector is
therefore reported separately as a *port approach* (``approach_land_km``):
it is shown to the navigator, but it is not counted as an incorrect course,
because no lattice search can change it - the way into the harbour is
found by pilotage along the harbour fairway.

Units: kilometres and degrees. 单位：千米与度。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .geodesy import haversine_km

LOGGER = logging.getLogger(__name__)

try:                                    # optional at import time, see is_available()
    from global_land_mask import globe as _GLOBE
except ImportError:                     # pragma: no cover - depends on the installation
    _GLOBE = None

#: Sampling step along the route, km (the mask resolution is about 1 km).
#: 沿航线的采样步长，千米。
STEP_KM: float = 1.0
#: Length at each end of the voyage that is not checked, km. 首尾不检查的长度。
PORT_KM: float = 5.0
#: A continuous land run of this length or more makes the course incorrect, km.
#: 连续穿陆长度达到该值即判为航线错误。
CROSSING_KM: float = 3.0
#: Largest number of land points returned for drawing on the map.
MAX_REPORTED_POINTS: int = 60

Coordinate = Tuple[float, float]


def is_available() -> bool:
    """True when the independent land mask is installed. 陆地掩膜是否可用。"""
    return _GLOBE is not None


def is_land(lat, lon) -> np.ndarray:
    """Look points up in the 1 km land mask.

    Parameters
    ----------
    lat, lon : array_like
        Coordinates, degrees; longitudes outside -180..180 are wrapped.

    Returns
    -------
    numpy.ndarray of bool
        True where the mask shows land.
    """
    lat = np.clip(np.asarray(lat, dtype=float), -90.0, 90.0)
    lon = (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0
    return np.asarray(_GLOBE.is_land(lat, lon), dtype=bool)


@dataclass
class LandRun:
    """One continuous stretch of the route over land. 一段连续穿陆的航段。

    Attributes
    ----------
    length_km : float
        Length of the stretch, km.
    start_km : float
        Distance from the departure to the first land sample, km.
    legs : list of int
        Indices ``i`` of the route legs ``waypoints[i] -> waypoints[i+1]`` that
        contain samples of this stretch.
    points : list of Coordinate
        The land samples, ``(lat, lon)``.
    """

    length_km: float
    start_km: float
    legs: List[int] = field(default_factory=list)
    points: List[Coordinate] = field(default_factory=list)


#: Longest connector between a quay and the lattice that may count as a
#: port approach, km (snap radius 60 km plus the unchecked 5 km).
APPROACH_MAX_KM: float = 65.0


@dataclass
class CoastCheck:
    """Outcome of the coastline check of one route. 一条航线的检查结果。

    Attributes
    ----------
    verdict : str
        ``clear``, ``grazing``, ``crossing`` or ``unchecked``.
    max_land_run_km : float
        Longest continuous land stretch, km (0 when clear).
    land_km : float
        Total length over land, km.
    samples, samples_on_land : int
        Checked samples and those on land.
    runs : list of LandRun
        All land stretches, longest first (port approaches excluded).
    approach_runs : list of LandRun
        Land stretches of the port approaches (see the module notes).
    """

    verdict: str
    max_land_run_km: float = 0.0
    land_km: float = 0.0
    samples: int = 0
    samples_on_land: int = 0
    runs: List[LandRun] = field(default_factory=list)
    approach_runs: List[LandRun] = field(default_factory=list)

    @property
    def crossing_runs(self) -> List[LandRun]:
        """Stretches long enough to make the course incorrect."""
        return [r for r in self.runs if r.length_km >= CROSSING_KM]

    def to_dict(self, corrections: int = 0, refined: bool = False,
                cells_blocked: int = 0, initial: Optional[str] = None) -> Dict:
        """JSON-ready summary stored with the route and sent to the client.

        Parameters
        ----------
        corrections : int
            Number of automatic re-searches that were made.
        refined : bool
            Whether a finer lattice was used because of the check.
        cells_blocked : int
            Lattice cells made impassable by the corrections.
        initial : str, optional
            Verdict of the first route, before any correction.

        Returns
        -------
        dict
            ``verdict``, ``initial_verdict``, ``max_land_run_km``, ``land_km``,
            ``samples``, ``samples_on_land``, ``corrections``, ``refined``,
            ``cells_blocked``, ``threshold_km`` and up to
            :data:`MAX_REPORTED_POINTS` ``land_points`` ``[lat, lon]``.
        """
        points = [p for r in self.crossing_runs for p in r.points] or \
                 [p for r in self.runs for p in r.points]
        step = max(1, math.ceil(len(points) / MAX_REPORTED_POINTS))
        return {
            "verdict": self.verdict,
            "initial_verdict": initial or self.verdict,
            "max_land_run_km": round(self.max_land_run_km, 1),
            "land_km": round(self.land_km, 1),
            "samples": self.samples,
            "samples_on_land": self.samples_on_land,
            "corrections": int(corrections),
            "refined": bool(refined),
            "cells_blocked": int(cells_blocked),
            "threshold_km": CROSSING_KM,
            "approach_land_km": round(sum(r.length_km for r in self.approach_runs), 1),
            "land_points": [[round(a, 4), round(b, 4)] for a, b in points[::step]],
        }


def sample_route(waypoints: Sequence[Coordinate], step_km: float = STEP_KM
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points every ``step_km`` along the poly-line.

    Each leg is divided into ``ceil(length / step_km)`` equal parts by linear
    interpolation in degrees, which for legs of a lattice (at most a few tens
    of kilometres) differs from the great circle by metres.

    Returns
    -------
    tuple
        ``(points, leg, dist)`` - an ``(n, 2)`` array of ``(lat, lon)``, the
        index of the leg of every point and the distance of every point from
        the departure along the route, km.
    """
    pts, legs = [], []
    for i, ((a1, b1), (a2, b2)) in enumerate(zip(waypoints[:-1], waypoints[1:])):
        n = max(1, int(math.ceil(float(haversine_km(a1, b1, a2, b2)) / step_km)))
        t = np.arange(n) / n
        pts.append(np.column_stack([a1 + (a2 - a1) * t, b1 + (b2 - b1) * t]))
        legs.append(np.full(n, i))
    pts.append(np.array([waypoints[-1]], dtype=float))
    legs.append(np.array([max(len(waypoints) - 2, 0)]))
    pts = np.vstack(pts).astype(float)
    leg = np.concatenate(legs)
    seg = haversine_km(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    dist = np.r_[0.0, np.cumsum(seg)]
    return pts, leg, dist


def check_route(waypoints: Sequence[Coordinate], approach: bool = False) -> CoastCheck:
    """Check a route against the independent land mask. 检查航线是否穿越陆地。

    Parameters
    ----------
    waypoints : sequence of Coordinate
        Route poly-line from the departure to the destination, ``(lat, lon)``.
    approach : bool, default False
        Treat the first and the last leg as the connectors between the quays
        and the lattice: land that begins at a quay and ends within its
        connector is a port approach, not an incorrect course (see the module
        notes). The planner sets it; the 100-voyage test, whose ends are
        clear water, does not need it.

    Returns
    -------
    CoastCheck
        Verdict and the land stretches; ``unchecked`` when the mask is not
        installed or the route has fewer than two points.
    """
    if not is_available() or len(waypoints) < 2:
        return CoastCheck(verdict="unchecked")
    pts, leg, dist = sample_route(waypoints)
    keep = (dist >= PORT_KM) & (dist <= dist[-1] - PORT_KM)
    on_land = is_land(pts[:, 0], pts[:, 1])
    land = on_land & keep

    runs: List[LandRun] = []
    current: Optional[LandRun] = None
    for i in range(len(pts)):
        if land[i]:
            if current is None:
                # A run starts: one sample stands for STEP_KM of route.
                current = LandRun(length_km=STEP_KM, start_km=float(dist[i]))
            else:
                current.length_km += float(dist[i] - dist[i - 1])
            current.points.append((float(pts[i, 0]), float(pts[i, 1])))
            if not current.legs or current.legs[-1] != int(leg[i]):
                current.legs.append(int(leg[i]))
        elif current is not None:
            runs.append(current)
            current = None
    if current is not None:
        runs.append(current)

    approach_runs: List[LandRun] = []
    if approach and runs and keep.any():
        total = float(dist[-1])
        first_leg = min(float(haversine_km(*waypoints[0], *waypoints[1])), APPROACH_MAX_KM)
        last_leg = min(float(haversine_km(*waypoints[-2], *waypoints[-1])), APPROACH_MAX_KM)
        first_kept, last_kept = np.flatnonzero(keep)[[0, -1]]
        # Land at the boundary of the unchecked stretch = the quay is on land.
        quay_start_on_land = bool(first_kept > 0 and on_land[first_kept - 1])
        quay_end_on_land = bool(last_kept + 1 < len(pts) and on_land[last_kept + 1])
        rest = []
        for run in runs:
            end_km = run.start_km + run.length_km
            at_start = (quay_start_on_land and run.start_km <= dist[first_kept] + 1e-6
                        and end_km <= max(first_leg, PORT_KM) + STEP_KM)
            at_end = (quay_end_on_land and end_km >= dist[last_kept] - STEP_KM - 1e-6
                      and run.start_km >= total - max(last_leg, PORT_KM) - STEP_KM)
            (approach_runs if at_start or at_end else rest).append(run)
        runs = rest

    runs.sort(key=lambda r: r.length_km, reverse=True)
    longest = runs[0].length_km if runs else 0.0
    verdict = "clear" if not runs else ("crossing" if longest >= CROSSING_KM else "grazing")
    return CoastCheck(
        verdict=verdict,
        max_land_run_km=longest,
        land_km=float(sum(r.length_km for r in runs)),
        samples=int(keep.sum()),
        samples_on_land=int(land.sum()),
        runs=runs,
        approach_runs=approach_runs,
    )


#: Half-width of the window around a land stretch in which navigable cells
#: centred on land of the mask are blocked as well, in cells.
BLOCK_WINDOW_CELLS: int = 2


def cells_to_block(check: CoastCheck, cost_map, path_cells: Sequence[Tuple[int, int]],
                   protected: Set[Tuple[int, int]],
                   window: int = BLOCK_WINDOW_CELLS) -> Set[Tuple[int, int]]:
    """Lattice cells to make impassable so that the next search avoids the land.

    Three kinds of cells are taken for every stretch of :attr:`CoastCheck.crossing_runs`:

    * the cell nearest to each land sample - this removes land that the
      classifier took for water;
    * the route vertices at both ends of the legs that contain the stretch -
      when the leg is a diagonal step between two water cells across a corner
      of land, blocking its ends forces the search to go round;
    * every navigable cell whose centre lies on land of the mask within
      ``window`` cells of the stretch - where the classifier is wrong about
      one cell it is usually wrong about its neighbours too (a whole island
      chain), and without this the route would move one cell per round.
      Cells whose centre is water are not touched, so a strait stays open.

    The snapped departure and destination cells (``protected``) are never
    blocked.
    为每段穿陆航段选出需要封锁的网格：陆地采样点所在格子、该航段两端的航路点格子，
    以及附近中心点位于陆地上的可航行格子。

    Parameters
    ----------
    check : CoastCheck
        Result of :func:`check_route` for the route.
    cost_map : CostMap
        Map the route was searched on.
    path_cells : sequence of (row, col)
        Lattice path of the search; ``path_cells[i]`` corresponds to the
        route vertex ``i`` (the planner replaces only the coordinates of the
        first and the last vertex by the exact ports).
    protected : set of (row, col)
        Cells that must stay navigable.

    Returns
    -------
    set of (row, col)
        Cells that are navigable now and should be blocked.
    """
    spec = cost_map.spec
    passable = cost_map.passable
    cells: Set[Tuple[int, int]] = set()
    for run in check.crossing_runs:
        run_cells = [spec.coord_to_cell(lat, lon) for lat, lon in run.points]
        cells.update(run_cells)
        if window > 0 and run_cells:
            r0 = max(min(r for r, _ in run_cells) - window, 0)
            r1 = min(max(r for r, _ in run_cells) + window, spec.n_rows - 1)
            c0 = max(min(c for _, c in run_cells) - window, 0)
            c1 = min(max(c for _, c in run_cells) + window, spec.n_cols - 1)
            rows, cols = np.mgrid[r0:r1 + 1, c0:c1 + 1]
            lat, lon = spec.cell_to_coord(rows, cols)
            on_land = is_land(lat, lon) & passable[rows, cols]
            cells.update(zip(rows[on_land].tolist(), cols[on_land].tolist()))
        if path_cells:
            for i in run.legs:
                for j in (i, i + 1):
                    if 0 <= j < len(path_cells):
                        cells.add(tuple(path_cells[j]))
    return {c for c in cells if c not in protected and passable[c]}


def block_cells(cost_map, cells: Set[Tuple[int, int]]):
    """Copy of the cost map with ``cells`` made impassable (the original is kept).

    Only the cost array changes; the classification (``class_index``) is
    left as the classifier predicted it, so the map overlay in the UI still
    shows the classifier's opinion.

    Parameters
    ----------
    cost_map : CostMap
        Map to copy.
    cells : set of (row, col)
        Cells to block.

    Returns
    -------
    CostMap
        New map sharing ``spec``, ``class_index`` and ``confidence``.
    """
    from dataclasses import replace

    from ..config import IMPASSABLE_COST

    cost = cost_map.cost.copy()
    for row, col in cells:
        cost[row, col] = IMPASSABLE_COST
    return replace(cost_map, cost=cost)
