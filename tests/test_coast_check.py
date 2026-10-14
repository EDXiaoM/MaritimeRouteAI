"""Tests of the independent coastline check and the automatic correction.

独立海岸线检查与自动修正的测试。

The planner is driven by small stand-in classifiers built on the land mask
itself, so that the tests do not depend on the trained network:

* ``TruthClassifier`` answers "land" exactly where the mask shows land;
* ``IslandBlindClassifier`` is the truth except that it takes Bornholm for
  water - the kind of mistake the 100-voyage test found (sparsely sampled
  islands taken for water);
* ``BlindClassifier`` takes everything for water.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pytest

pytest.importorskip("global_land_mask")

from maritime_route.config import CLASS_NAMES, IMPASSABLE_COST  # noqa: E402
from maritime_route.routing import coast_check  # noqa: E402
from maritime_route.routing.cost_map import CostMap, GridSpec  # noqa: E402
from maritime_route.routing.parameters import SearchParameters  # noqa: E402
from maritime_route.routing.planner import RoutePlanner  # noqa: E402

OPEN, LAND = CLASS_NAMES.index("OPEN_SEA"), CLASS_NAMES.index("COASTLINE")


def _one_hot(index: np.ndarray) -> np.ndarray:
    probs = np.zeros((index.size, len(CLASS_NAMES)), dtype=np.float32)
    probs[np.arange(index.size), index] = 1.0
    return probs


class TruthClassifier:
    """Land exactly where the independent mask shows land."""

    def predict_coordinates(self, lat, lon):
        land = coast_check.is_land(lat, lon)
        return _one_hot(np.where(land, LAND, OPEN))


class IslandBlindClassifier(TruthClassifier):
    """The truth, except that Bornholm (55.0-55.3 N, 14.7-15.2 E) is 'water'."""

    def predict_coordinates(self, lat, lon):
        lat, lon = np.asarray(lat), np.asarray(lon)
        land = coast_check.is_land(lat, lon)
        island = (lat > 54.95) & (lat < 55.35) & (lon > 14.6) & (lon < 15.25)
        return _one_hot(np.where(land & ~island, LAND, OPEN))


class BlindClassifier:
    """Everything is open sea."""

    def predict_coordinates(self, lat, lon):
        return _one_hot(np.full(np.asarray(lat).size, OPEN))


# ---------------------------------------------------------------- the check
def test_open_sea_route_is_clear():
    check = coast_check.check_route([(40.0, -40.0), (40.0, -35.0)])
    assert check.verdict == "clear"
    assert check.max_land_run_km == 0.0
    assert check.samples > 400


def test_route_across_jutland_is_a_crossing():
    # North Sea -> Kattegat straight across Denmark (about 130 km of land).
    check = coast_check.check_route([(56.2, 7.5), (56.2, 11.0)])
    assert check.verdict == "crossing"
    assert check.max_land_run_km > 100
    assert check.crossing_runs and check.crossing_runs[0].points


def test_the_quays_are_not_checked():
    # Departure 2 km inland of the Danish west coast, then out to sea: the
    # first PORT_KM are ignored, so the voyage is clear.
    check = coast_check.check_route([(56.0, 8.17), (56.0, 7.0)])
    assert check.verdict in ("clear", "grazing")


def test_summary_is_json_ready():
    check = coast_check.check_route([(56.2, 7.5), (56.2, 11.0)])
    summary = check.to_dict(corrections=2, refined=True, cells_blocked=7, initial="crossing")
    assert summary["verdict"] == "crossing" and summary["initial_verdict"] == "crossing"
    assert 0 < len(summary["land_points"]) <= coast_check.MAX_REPORTED_POINTS
    assert summary["corrections"] == 2 and summary["refined"] is True


def test_blocking_copies_the_map():
    spec = GridSpec(lat_min=55.0, lat_max=56.0, lon_min=10.0, lon_max=11.0, resolution_deg=0.25)
    shape = (spec.n_rows, spec.n_cols)
    cost_map = CostMap(spec=spec, class_index=np.zeros(shape, np.int8),
                       confidence=np.ones(shape, np.float32), cost=np.ones(shape, np.float32))
    blocked = coast_check.block_cells(cost_map, {(1, 1), (2, 3)})
    assert not blocked.passable[1, 1] and not blocked.passable[2, 3]
    assert cost_map.passable.all()                      # the original is unchanged
    assert blocked.cost[1, 1] == IMPASSABLE_COST


# ---------------------------------------------------------- the safeguard
#: The safeguard is tested on its own: without the land mask in the cost map
#: (version 1.3), which would already keep the route off the missed island.
NO_MASK = SearchParameters.from_dict({"use_land_mask": False})


def test_route_missed_island_is_corrected_automatically():
    planner = RoutePlanner(IslandBlindClassifier())
    plan, _ = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1, params=NO_MASK)
    assert plan.feasible
    check = plan.coast_check
    assert check["initial_verdict"] == "crossing"   # the first route ran over Bornholm
    assert check["verdict"] != "crossing"           # ... the final one does not
    assert check["corrections"] >= 1
    assert plan.warning == ""
    assert "corrected automatically" in plan.message
    # Independent confirmation of the final route.
    assert coast_check.check_route(plan.waypoints).verdict != "crossing"


def test_correct_route_is_left_unchanged():
    planner = RoutePlanner(TruthClassifier())
    plan, _ = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1, params=NO_MASK)
    assert plan.coast_check["initial_verdict"] != "crossing"
    assert plan.coast_check["corrections"] == 0
    assert "coastline check passed" in plan.message


@pytest.mark.parametrize("algorithm", ["dijkstra", "dynamic"])
def test_safeguard_works_for_the_other_exact_methods(algorithm):
    planner = RoutePlanner(IslandBlindClassifier())
    plan, _ = planner.plan((55.12, 13.9), (55.12, 16.0), algorithm, 0.1, params=NO_MASK)
    assert plan.feasible and plan.coast_check["verdict"] != "crossing"


def test_uncorrectable_route_carries_a_warning():
    class NoCorrection(RoutePlanner):
        MAX_COAST_CORRECTIONS = 0

    plan, _ = NoCorrection(BlindClassifier()).plan((56.2, 7.5), (56.2, 11.0), "astar", 0.25,
                                                  auto_refine=False, params=NO_MASK)
    assert plan.feasible
    assert plan.coast_check["verdict"] == "crossing"
    assert "crosses land" in plan.warning
    assert plan.to_dict()["warning"] == plan.warning


def test_benchmark_map_is_checked_but_not_changed():
    planner = RoutePlanner(IslandBlindClassifier())
    _, cost_map = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1, coast_guard=False,
                                params=NO_MASK)
    plan, used = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1,
                              reuse_cost_map=cost_map, auto_refine=False, params=NO_MASK)
    assert plan.coast_check["verdict"] == "crossing"
    assert plan.coast_check["corrections"] == 0
    assert used is cost_map


def test_switching_the_guard_off_keeps_the_old_behaviour():
    planner = RoutePlanner(IslandBlindClassifier())
    plan, _ = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1, coast_guard=False,
                                params=NO_MASK)
    assert plan.coast_check["verdict"] == "crossing" and plan.coast_check["corrections"] == 0


# ------------------------------------------------------------ persistence
def test_check_is_stored_with_the_route(repository):
    planner = RoutePlanner(IslandBlindClassifier())
    plan, _ = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1, params=NO_MASK)
    repository.save_route(plan)
    stored = repository.get_route(plan.route_id)
    assert stored["coast_check"]["initial_verdict"] == "crossing"
    assert stored["coast_check"]["verdict"] == plan.coast_check["verdict"]


def test_old_database_gets_the_new_column(tmp_path):
    from maritime_route.storage.repository import RouteRepository

    path = tmp_path / "old.db"
    with sqlite3.connect(path) as conn:
        # Route table of version 1.0: no coast_check_json column.
        conn.execute("""CREATE TABLE route (id INTEGER PRIMARY KEY AUTOINCREMENT,
            route_uid TEXT NOT NULL UNIQUE, session_id INTEGER, algorithm TEXT NOT NULL
            CHECK (algorithm IN ('astar','dijkstra','great_circle')),
            start_lat REAL NOT NULL, start_lon REAL NOT NULL, end_lat REAL NOT NULL,
            end_lon REAL NOT NULL, total_cost REAL, distance_km REAL NOT NULL,
            great_circle_km REAL NOT NULL, detour_ratio REAL, estimated_hours REAL,
            speed_knots REAL, nodes_expanded INTEGER, runtime_s REAL NOT NULL,
            grid_resolution REAL, grid_cells INTEGER, feasible INTEGER NOT NULL DEFAULT 1,
            baseline_json TEXT, zone_profile_json TEXT, created_utc TEXT NOT NULL)""")
        conn.execute("""INSERT INTO route(route_uid, algorithm, start_lat, start_lon, end_lat,
            end_lon, distance_km, great_circle_km, runtime_s, created_utc)
            VALUES ('old0000000000001','astar',54,10,55,11,100,90,0.1,'2026-01-01T00:00:00Z')""")
    repo = RouteRepository(path)        # the constructor runs the migrations
    with sqlite3.connect(path) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(route)")]
    assert columns[-4:] == ["coast_check_json", "status", "parameters_json", "result_file"]
    assert repo.get_route("old0000000000001")["coast_check"] == {}


def test_pdf_line_describes_the_check():
    from maritime_route.export.exporters import _coast_text

    assert _coast_text({}).startswith("not performed")
    assert _coast_text({"verdict": "clear"}).startswith("passed")
    text = _coast_text({"verdict": "clear", "initial_verdict": "crossing", "corrections": 2})
    assert "corrected automatically" in text
    assert _coast_text({"verdict": "crossing", "max_land_run_km": 12}).startswith("FAILED")
