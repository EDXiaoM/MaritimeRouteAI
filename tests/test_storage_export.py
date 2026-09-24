"""Tests of persistence and of the three export formats. 存储与导出测试。

Covers ``maritime_route.export.exporters`` (route GeoJSON, route CSV, PDF
report, classified-point GeoJSON/CSV) and
``maritime_route.storage.repository.RouteRepository`` (SQLite schema, route
and point round trips, statistics, cascading deletes, handling of non-finite
numbers and migration of an old schema). The routes and points are built by
hand with :func:`make_plan` and :func:`make_results`, and every storage test
uses a fresh temporary database from the ``repository`` fixture
(``conftest.py``), so no model is needed.
测试导出格式与 SQLite 存储；数据为手工构造，每个测试使用独立的临时数据库。
"""
from __future__ import annotations

import csv
import io
import json

import pytest

from maritime_route.config import CLASS_NAMES
from maritime_route.export.exporters import (
    classified_points_to_geojson, points_to_csv_string, route_to_csv_string,
    route_to_geojson, route_to_pdf,
)
from maritime_route.model.inference import ClassificationResult
from maritime_route.routing.planner import RouteLeg, RoutePlan


def make_plan() -> RoutePlan:
    """Build a small hand-written feasible RoutePlan (3 waypoints) for the tests.

    Returns
    -------
    RoutePlan
        Plan with fixed values (distance 320 km, great circle 300 km, a
        baseline that crosses land) so that exported and stored values can be
        compared with exact numbers. No planner or model is involved.
    """
    legs = [
        RouteLeg(54.0, 10.0, "COASTAL_SEA", 1.8, 0.0, 45.0),
        RouteLeg(55.0, 12.0, "OPEN_SEA", 1.0, 160.0, 44.0),
        RouteLeg(56.0, 14.0, "COASTAL_SEA", 1.8, 320.0, 43.0),
    ]
    return RoutePlan(
        route_id="testroute000001", algorithm="astar", start=(54.0, 10.0), end=(56.0, 14.0),
        waypoints=[(l.latitude, l.longitude) for l in legs], legs=legs,
        total_cost=430.0, distance_km=320.0, great_circle_km=300.0, detour_ratio=1.0667,
        estimated_hours=12.3, nodes_expanded=512, runtime_s=0.42,
        grid={"resolution_deg": 0.25, "n_cells": 4000, "n_rows": 50, "n_cols": 80},
        zone_profile={"OPEN_SEA": 1, "COASTAL_SEA": 2, "NEAR_COAST": 0, "COASTLINE": 0},
        baseline={"distance_km": 300.0, "is_navigable": False, "land_share": 0.25,
                  "land_samples": 40, "samples": 160, "extra_distance_pct": 6.67,
                  "verdict": "Great-circle route crosses land and is not navigable",
                  "weighted_cost": None, "cost_saving_pct": None,
                  "zone_profile": {c: 40 for c in CLASS_NAMES}},
        cost_map_stats={}, created_utc="2026-09-20T12:00:00Z", speed_knots=14.0,
    )


def make_results():
    """Build two classification results (one COASTAL_SEA, one OPEN_SEA).

    Returns
    -------
    list of ClassificationResult
        Points with depth values and uniform class probabilities.
    """
    return [
        ClassificationResult(54.0, 10.0, 1, "COASTAL_SEA", 0.91,
                             {c: 0.25 for c in CLASS_NAMES}, -25.0),
        ClassificationResult(55.0, 12.0, 0, "OPEN_SEA", 0.98,
                             {c: 0.25 for c in CLASS_NAMES}, -180.0),
    ]


# --------------------------------------------------------------------- export
def test_geojson_structure_and_geometry_order():
    """Route GeoJSON is a FeatureCollection with route and great-circle features in [lon, lat] order."""
    document = route_to_geojson(make_plan())
    assert document["type"] == "FeatureCollection"
    kinds = [f["properties"]["feature_type"] for f in document["features"]]
    assert "optimal_route" in kinds and "great_circle_reference" in kinds
    route = next(f for f in document["features"]
                 if f["properties"]["feature_type"] == "optimal_route")
    # GeoJSON stores [longitude, latitude]
    assert route["geometry"]["coordinates"][0] == [10.0, 54.0]
    json.dumps(document)          # must be serialisable


def test_geojson_contains_no_infinities():
    """Serialised route GeoJSON contains no "Infinity" or "NaN" (both are invalid JSON)."""
    payload = json.dumps(route_to_geojson(make_plan()))
    assert "Infinity" not in payload and "NaN" not in payload


def test_route_csv_has_one_row_per_waypoint():
    """Route CSV has the expected header and one row per waypoint."""
    rows = list(csv.reader(io.StringIO(route_to_csv_string(make_plan()))))
    assert rows[0][:4] == ["seq", "latitude", "longitude", "class_code"]
    assert len(rows) == 1 + 3
    assert rows[1][3] == "COASTAL_SEA"


def test_impassable_cost_is_exported_as_infinity():
    """A berth in a shore zone must not export as the internal 1e6 constant."""
    plan = make_plan()
    plan.legs[0].zone_cost = 1.0e6
    rows = list(csv.reader(io.StringIO(route_to_csv_string(plan))))
    assert rows[1][4] == "inf"
    feature = next(f for f in route_to_geojson(plan)["features"]
                   if f["properties"].get("seq") == 0)
    assert feature["properties"]["zone_cost"] is None
    assert feature["properties"]["navigable"] is False


def test_points_exports():
    """Point exports: one GeoJSON feature per point, CSV header + rows with per-class probability columns."""
    results = make_results()
    document = classified_points_to_geojson(results)
    assert len(document["features"]) == 2
    rows = list(csv.reader(io.StringIO(points_to_csv_string(results))))
    assert len(rows) == 3
    assert "p_OPEN_SEA" in rows[0]


def test_pdf_is_written_and_looks_like_a_pdf(tmp_path):
    """PDF report is written, starts with the %PDF signature and is not trivially small."""
    path = route_to_pdf(make_plan(), tmp_path / "report.pdf",
                        model_summary={"architecture": "ResidualMLP", "n_features": 28,
                                       "parameters": 98364, "accuracy": 0.9665,
                                       "macro_f1": 0.9665, "trained_utc": "2026-09-20"})
    assert path.exists()
    data = path.read_bytes()
    assert data.startswith(b"%PDF")
    assert len(data) > 3000


# -------------------------------------------------------------------- storage
def test_zone_table_is_seeded(repository):
    """A new database has the zone table filled with all four class codes."""
    with repository.connect() as conn:
        codes = [r["code"] for r in conn.execute("SELECT code FROM zone")]
    assert set(codes) == set(CLASS_NAMES)


def test_route_round_trip(repository):
    """A saved route is read back with its scalar fields, ordered waypoints and baseline."""
    plan = make_plan()
    repository.save_route(plan)
    stored = repository.get_route(plan.route_id)
    assert stored is not None
    assert stored["algorithm"] == "astar"
    assert stored["distance_km"] == pytest.approx(320.0)
    assert len(stored["waypoints"]) == 3
    assert stored["waypoints"][0]["seq"] == 0
    assert stored["baseline"]["is_navigable"] is False


def test_points_and_histogram(repository):
    """Saved points are counted and summarised correctly in the class histogram."""
    session_id = repository.create_session("t", "unit-test")
    assert repository.save_points(session_id, make_results()) == 2
    histogram = repository.class_histogram(session_id)
    assert histogram["OPEN_SEA"] == 1 and histogram["COASTAL_SEA"] == 1


def test_statistics_and_delete(repository):
    """Statistics count a stored route; deleting it removes it and resets the count."""
    plan = make_plan()
    repository.save_route(plan)
    stats = repository.statistics()
    assert stats["routes"] == 1 and stats["waypoints"] == 3
    assert "astar" in stats["by_algorithm"]
    assert repository.delete_route(plan.route_id) is True
    assert repository.get_route(plan.route_id) is None
    assert repository.statistics()["routes"] == 0


def test_cascade_delete_removes_waypoints(repository):
    """Deleting a route also deletes its waypoints (ON DELETE CASCADE)."""
    plan = make_plan()
    repository.save_route(plan)
    repository.delete_route(plan.route_id)
    with repository.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM waypoint").fetchone()[0] == 0


def test_non_finite_values_become_null(repository):
    """inf/NaN route metrics are stored as NULL and read back as None."""
    plan = make_plan()
    plan.total_cost = float("inf")
    plan.detour_ratio = float("nan")
    repository.save_route(plan)
    stored = repository.get_route(plan.route_id)
    assert stored["total_cost"] is None
    assert stored["detour_ratio"] is None


def test_genetic_route_can_be_stored(repository):
    """A route produced by the genetic algorithm is accepted by the schema."""
    plan = make_plan()
    plan.algorithm = "genetic"
    repository.save_route(plan)
    assert repository.get_route(plan.route_id)["algorithm"] == "genetic"


def test_old_database_is_upgraded_for_the_genetic_algorithm(tmp_path):
    """A database created before the GA existed is migrated on start-up."""
    import sqlite3

    from maritime_route.storage.repository import SCHEMA_PATH, RouteRepository

    path = tmp_path / "old.db"
    old_schema = SCHEMA_PATH.read_text(encoding="utf-8").replace(
        "'astar','dijkstra','genetic','dynamic','great_circle'", "'astar','dijkstra','great_circle'")
    with sqlite3.connect(path) as conn:
        conn.executescript(old_schema)
    repository = RouteRepository(path)
    plan = make_plan()
    plan.algorithm = "genetic"
    repository.save_route(plan)
    assert repository.get_route(plan.route_id)["algorithm"] == "genetic"
