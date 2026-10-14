"""Tests of version 1.3: parameters, step limits, Stop, time limit, result files,
via points, adaptive search area, land-mask fusion and the two-level search.

1.3 版测试：参数、步数上限、停止、时间限制、结果文件、途经点、自适应搜索区域、
陆地掩膜融合与两级（走廊）搜索。

The tests use small synthetic cost maps and fake classifiers built on the
independent land mask, so they do not need the trained network and run in a
few seconds. The WebSocket tests at the end use the real model and are
skipped when it is missing.
"""
from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest

from conftest import requires_model
from maritime_route.config import CLASS_NAMES, GENETIC
from maritime_route.export import results
from maritime_route.routing import coast_check
from maritime_route.routing.astar import astar
from maritime_route.routing.control import RunControl, SearchStopped
from maritime_route.routing.cost_map import (CostMap, GridSpec, build_cost_map, corridor_mask,
                                             fuse_land_mask, mask_land_share)
from maritime_route.routing.dijkstra import dijkstra
from maritime_route.routing.dynamic_programming import dynamic_programming
from maritime_route.routing.genetic import genetic
from maritime_route.routing.parameters import FIELDS, SearchParameters, parameter_spec
from maritime_route.routing.planner import RoutePlanner, benchmark

OPEN, LAND = CLASS_NAMES.index("OPEN_SEA"), CLASS_NAMES.index("COASTLINE")


def _one_hot(index):
    probs = np.zeros((index.size, len(CLASS_NAMES)), dtype=np.float32)
    probs[np.arange(index.size), index] = 1.0
    return probs


class TruthClassifier:
    """Land exactly where the independent mask shows land."""

    def predict_coordinates(self, lat, lon):
        return _one_hot(np.where(coast_check.is_land(lat, lon), LAND, OPEN))


class IslandBlindClassifier:
    """The truth, except that Bornholm is taken for water."""

    def predict_coordinates(self, lat, lon):
        lat, lon = np.asarray(lat), np.asarray(lon)
        island = (lat > 54.95) & (lat < 55.35) & (lon > 14.6) & (lon < 15.25)
        return _one_hot(np.where(coast_check.is_land(lat, lon) & ~island, LAND, OPEN))


class SlowClassifier(TruthClassifier):
    """The truth, slowly: every batch takes 50 ms (for the Stop tests)."""

    def predict_coordinates(self, lat, lon):
        time.sleep(0.05)
        return super().predict_coordinates(lat, lon)


def open_map(rows=60, cols=80, wall=True):
    """Open-sea lattice with a wall that has one gap (a long detour)."""
    spec = GridSpec(50.0, 50.0 + (rows - 1) * 0.1, 0.0, (cols - 1) * 0.1, 0.1)
    cost = np.ones((spec.n_rows, spec.n_cols), np.float32)
    if wall:
        cost[5:, cols // 2] = 1e9                      # gap only in rows 0-4
    shape = cost.shape
    return CostMap(spec=spec, class_index=np.zeros(shape, np.int8),
                   confidence=np.ones(shape, np.float32), cost=cost)


START, GOAL = (50, 5), (50, 75)


# ------------------------------------------------------------ run control
def test_control_cancel_and_time_limit():
    control = RunControl()
    control.check()                                    # nothing happens
    control.cancel()
    with pytest.raises(SearchStopped) as stop:
        control.check()
    assert stop.value.reason == "cancelled"

    limited = RunControl(time_limit_s=0.01)
    time.sleep(0.03)
    with pytest.raises(SearchStopped) as stop:
        limited.check()
    assert stop.value.reason == "time_limit"


def test_parent_stop_reaches_the_child():
    parent = RunControl()
    child = RunControl(time_limit_s=100, parent=parent)
    parent.cancel()
    with pytest.raises(SearchStopped):
        child.check()


# ------------------------------------------------------------ step limits
@pytest.mark.parametrize("search", [astar, dijkstra])
def test_step_limit_stops_the_search(search):
    cost_map = open_map()
    limited = search(cost_map, START, GOAL, max_steps=100)
    assert not limited.found and limited.status == "step_limit"
    assert limited.nodes_expanded == 100
    assert "Step limit" in limited.message

    full = search(cost_map, START, GOAL)
    raised = search(cost_map, START, GOAL, max_steps=full.nodes_expanded + 10)
    assert raised.found and raised.total_cost == pytest.approx(full.total_cost)


def test_dijkstra_needs_more_steps_than_astar():
    cost_map = open_map()
    a, d = astar(cost_map, START, GOAL), dijkstra(cost_map, START, GOAL)
    assert d.nodes_expanded > a.nodes_expanded
    # A limit between the two: A* succeeds, Dijkstra does not.
    limit = (a.nodes_expanded + d.nodes_expanded) // 2
    assert astar(cost_map, START, GOAL, max_steps=limit).found
    assert dijkstra(cost_map, START, GOAL, max_steps=limit).status == "step_limit"


def test_dynamic_programming_sweep_limit():
    cost_map = open_map()
    few = dynamic_programming(cost_map, START, GOAL, max_sweeps=5)
    assert not few.found and few.status == "step_limit"
    exact = dynamic_programming(cost_map, START, GOAL)
    assert exact.found and exact.total_cost == pytest.approx(astar(cost_map, START, GOAL).total_cost)


def test_genetic_generation_limit_and_parameters():
    cost_map = open_map()
    params = SearchParameters.from_dict({"ga_population": 20, "ga_generations": 3,
                                         "ga_stagnation": 0, "ga_seed": 7})
    result = genetic(cost_map, START, GOAL, config=params.genetic_config())
    assert result.nodes_generated == 3
    if not result.found:
        assert result.status == "step_limit"


@pytest.mark.parametrize("search", [astar, dijkstra])
def test_search_stops_on_cancel(search):
    control = RunControl()
    control.cancel()
    with pytest.raises(SearchStopped):
        search(open_map(), START, GOAL, control=control, check_every=10)


def test_weighted_astar_is_faster_and_bounded():
    cost_map = open_map(wall=False)
    exact = astar(cost_map, START, GOAL)
    params = SearchParameters.from_dict({"astar_heuristic_weight": 2.0})
    fast = astar(cost_map, START, GOAL, params.routing_config(0.1))
    assert fast.found and fast.nodes_expanded <= exact.nodes_expanded
    assert fast.total_cost <= 2.0 * exact.total_cost + 1e-6


# ------------------------------------------------------------ parameters
def test_parameters_are_validated_and_clamped():
    params = SearchParameters.from_dict({"ga_population": 1, "ga_tournament": 50,
                                         "dijkstra_max_steps": "1000", "coast_guard": "off"})
    assert params.ga_population == FIELDS["ga_population"]["min"]
    assert params.ga_tournament <= params.ga_population
    assert params.dijkstra_max_steps == 1000 and params.coast_guard is False
    with pytest.raises(ValueError):
        SearchParameters.from_dict({"warp_speed": 9})
    with pytest.raises(ValueError):
        SearchParameters.from_dict({"astar_max_steps": "many"})


def test_defaults_equal_the_previous_constants():
    params = SearchParameters()
    assert params.genetic_config() == GENETIC
    assert params.changed() == {}


def test_parameter_spec_describes_every_field():
    spec = parameter_spec()
    keys = [f["key"] for g in spec["groups"] for f in g["fields"]]
    assert sorted(keys) == sorted(FIELDS)
    assert [g["id"] for g in spec["groups"]] == ["common", "astar", "dijkstra", "dynamic", "genetic"]
    assert all(f["help"] for g in spec["groups"] for f in g["fields"])
    only_ga = SearchParameters().for_algorithm("genetic")
    assert "ga_population" in only_ga and "astar_max_steps" not in only_ga


# ------------------------------------------------------------ planner
def test_planner_reports_step_limit_without_refining():
    planner = RoutePlanner(TruthClassifier())
    params = SearchParameters.from_dict({"dijkstra_max_steps": 50})
    plan, _ = planner.plan((55.12, 13.9), (55.12, 16.0), "dijkstra", 0.1, params=params)
    assert not plan.feasible and plan.status == "step_limit"
    assert len(plan.attempts) == 1                      # no finer lattice was tried
    assert plan.parameters["dijkstra_max_steps"] == 50


def test_planner_stopped_by_the_user():
    planner = RoutePlanner(SlowClassifier())
    control = RunControl()
    threading.Timer(0.2, control.cancel).start()
    t0 = time.perf_counter()
    plan, _ = planner.plan((55.12, 13.9), (57.0, 19.0), "astar", 0.05, control=control)
    assert time.perf_counter() - t0 < 3.0               # stopped quickly
    assert plan.status == "cancelled" and not plan.feasible
    assert "Stopped by the user" in plan.message


def test_planner_time_limit():
    planner = RoutePlanner(SlowClassifier())
    params = SearchParameters.from_dict({"time_limit_s": 0.2})
    plan, _ = planner.plan((55.12, 13.9), (57.0, 19.0), "astar", 0.05, params=params)
    assert plan.status == "time_limit" and plan.runtime_s < 3.0


def test_land_mask_fusion_corrects_the_island():
    # The classifier misses Bornholm; fusing the land mask (without a second
    # classification) blocks the island and the route goes round it.
    planner = RoutePlanner(IslandBlindClassifier())
    plan, cost_map = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1)
    assert plan.feasible and plan.coast_check["verdict"] != "crossing"
    assert plan.coast_check["initial_verdict"] == "crossing"
    assert any(a.get("mask_fused") for a in plan.attempts)
    assert cost_map.mask_land_cells > 0


def test_fuse_land_mask_keeps_the_zones():
    spec = GridSpec(54.8, 55.5, 14.4, 15.5, 0.05)
    blind = build_cost_map(spec, IslandBlindClassifier())
    fused = fuse_land_mask(blind, RoutePlanner(TruthClassifier()).config, 0.75)
    assert fused.mask_land_cells > 0
    assert fused.passable.sum() < blind.passable.sum()
    assert np.array_equal(fused.raw_class_index, blind.class_index)
    share = mask_land_share(spec)
    assert share.min() >= 0 and share.max() <= 1


def test_corridor_map_classifies_only_the_corridor():
    spec = GridSpec(54.0, 59.0, 13.0, 21.0, 0.05)
    route = [(55.5, 15.0), (58.0, 19.0)]
    region = corridor_mask(spec, route, 30.0)
    assert 0 < region.sum() < spec.n_cells / 3
    cost_map = build_cost_map(spec, TruthClassifier(), region=region)
    assert cost_map.classified_cells == int(region.sum())
    assert not (cost_map.passable & ~region).any()      # nothing outside is navigable


def test_adaptive_area_widens_the_box():
    # The direct box of the two ports holds no passage: a wall of land across
    # it except outside the box. The planner widens the area instead of
    # refining the lattice.
    class WallClassifier:
        def predict_coordinates(self, lat, lon):
            lat, lon = np.asarray(lat), np.asarray(lon)
            wall = (np.abs(lon - 20.0) < 0.6) & (lat > 38.0) & (lat < 46.0)
            return _one_hot(np.where(wall, LAND, OPEN))

    planner = RoutePlanner(WallClassifier())
    params = SearchParameters.from_dict({"coast_guard": False, "margin_deg": 1.0})
    plan, _ = planner.plan((42.0, 18.0), (42.0, 22.0), "astar", 0.25, params=params)
    assert plan.feasible
    assert any("widened" in a.get("next", "") for a in plan.attempts)
    assert plan.grid["resolution_deg"] == 0.25          # not refined


def test_route_through_via_points():
    planner = RoutePlanner(TruthClassifier())
    via = (56.2, 18.5)
    plan, _ = planner.plan_via([(55.12, 13.9), via, (57.5, 20.0)], "astar", 0.1)
    assert plan.feasible and plan.via == [via]
    assert {a["leg"] for a in plan.attempts} == {1, 2}
    # The route passes through the via point (within one cell).
    d = min(abs(lat - via[0]) + abs(lon - via[1]) for lat, lon in plan.waypoints)
    assert d < 0.15
    cumulative = [leg.cumulative_km for leg in plan.legs]
    assert cumulative == sorted(cumulative)
    assert cumulative[-1] == pytest.approx(plan.distance_km, rel=1e-6)


def test_benchmark_with_parameters_and_complexity():
    planner = RoutePlanner(TruthClassifier())
    params = SearchParameters.from_dict({"dijkstra_max_steps": 30, "ga_generations": 20})
    data = benchmark(planner, (55.12, 13.9), (55.12, 16.0), 0.1, params=params)
    rows = {r["algorithm"]: r for r in data["results"]}
    assert rows["dijkstra"]["status"] == "step_limit"
    assert rows["astar"]["status"] == "found"
    assert rows["astar"]["complexity"]["time"] == "O((V + E) log V)"
    assert rows["astar"]["theoretical_bound_ops"] >= rows["astar"]["nodes_expanded"]
    assert rows["astar"]["gap_to_optimum_pct"] == 0.0
    assert data["parameters"]["dijkstra_max_steps"] == 30


def test_benchmark_can_be_stopped():
    control = RunControl()
    control.cancel()
    data = benchmark(RoutePlanner(TruthClassifier()), (55.12, 13.9), (55.12, 16.0), 0.1,
                     control=control)
    assert data["status"] == "cancelled"


# ------------------------------------------------------------ result files
def test_result_file_of_a_found_route(tmp_path):
    planner = RoutePlanner(TruthClassifier())
    plan, _ = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1)
    path = results.save_route_result(plan, {"resolution_deg": 0.1}, tmp_path)
    assert path.name.startswith("route_N55.1200E13.9000_N55.1200E16.0000_astar_found_")
    document, points = results.load_result(path)
    assert document["format"] == results.FORMAT_ROUTE
    assert document["status"] == "found" and document["found"] is True
    assert document["file"] == str(path)
    assert len(points) == len(plan.waypoints) == document["n_points"]
    assert points[0] == pytest.approx(plan.start)
    assert document["timing"]["search_seconds"] > 0
    assert document["parameters"]["astar_max_steps"] == SearchParameters().astar_max_steps


def test_result_file_of_a_failed_search(tmp_path):
    planner = RoutePlanner(TruthClassifier())
    params = SearchParameters.from_dict({"astar_max_steps": 10})
    plan, _ = planner.plan((55.12, 13.9), (55.12, 16.0), "astar", 0.1, params=params)
    path = results.save_route_result(plan, None, tmp_path)
    assert "_astar_step_limit_" in path.name
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["found"] is False and document["points"] == []
    assert [e["role"] for e in document["endpoints"]] == ["start", "end"]
    assert document["search"]["step_limit"] == 10


def test_result_names_are_safe(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    assert results.result_path("a.json", tmp_path) is not None
    assert results.result_path("../a.json", tmp_path) is None
    assert results.result_path("a.txt", tmp_path) is None
    assert [f["name"] for f in results.list_results(directory=tmp_path)] == ["a.json"]


# ------------------------------------------------------------ WebSocket (model)
@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from maritime_route.web.app import app
    with TestClient(app) as test_client:
        yield test_client


def _frames_until(socket, kinds, limit=2000):
    seen = []
    for _ in range(limit):
        frame = socket.receive_json()
        seen.append(frame)
        if frame["type"] in kinds:
            return frame, seen
    pytest.fail(f"no {kinds} frame")


@requires_model
def test_parameters_endpoint(client):
    data = client.get("/api/parameters").json()
    assert {g["id"] for g in data["groups"]} >= {"common", "astar", "genetic"}
    assert data["results_dir"]


@requires_model
def test_websocket_stop_and_result_file(client):
    with client.websocket_connect("/ws/plan") as socket:
        socket.receive_json()                                    # hello
        socket.send_json({"action": "plan", "start_lat": 54.3233, "start_lon": 10.1228,
                          "end_lat": 59.4370, "end_lon": 24.7536, "algorithm": "dijkstra",
                          "resolution_deg": 0.04, "parameters": {"max_grid_cells": 4000000},
                          "include_overlay": False, "persist": False})
        assert socket.receive_json()["type"] == "started"
        socket.send_json({"action": "cancel"})
        frame, seen = _frames_until(socket, ("result", "error"))
        assert frame["type"] == "result", frame
        assert frame["plan"]["status"] == "cancelled"
        assert frame["result_file"].endswith(".json")
        assert any(f["type"] == "cancelling" for f in seen)
        name = frame["result_file"].replace("\\", "/").split("/")[-1]
        assert client.get(f"/api/results/{name}").json()["status"] == "cancelled"


@requires_model
def test_websocket_step_limit_and_benchmark(client):
    with client.websocket_connect("/ws/plan") as socket:
        socket.receive_json()
        socket.send_json({"action": "plan", "start_lat": 54.3233, "start_lon": 10.1228,
                          "end_lat": 59.4370, "end_lon": 24.7536, "algorithm": "dijkstra",
                          "resolution_deg": 0.3, "parameters": {"dijkstra_max_steps": 100},
                          "include_overlay": False, "persist": False})
        frame, _ = _frames_until(socket, ("result", "error"))
        assert frame["plan"]["status"] == "step_limit"
        assert "step_limit" in frame["result_file"]

        socket.send_json({"action": "benchmark", "start_lat": 54.3233, "start_lon": 10.1228,
                          "end_lat": 59.4370, "end_lon": 24.7536, "resolution_deg": 0.3,
                          "parameters": {"ga_generations": 30}})
        frame, seen = _frames_until(socket, ("benchmark", "error"))
        assert frame["type"] == "benchmark", frame
        assert len(frame["data"]["results"]) == 5
        assert frame["result_file"].split("/")[-1].startswith("compare_")
        assert any(f["type"] == "progress" and "elapsed_s" in f for f in seen)


@requires_model
def test_rest_route_with_via_point(client):
    response = client.post("/api/route", json={
        "start_lat": 54.3233, "start_lon": 10.1228, "end_lat": 59.4370, "end_lon": 24.7536,
        "via": [[57.0, 19.0]], "resolution_deg": 0.3, "persist": False})
    assert response.status_code == 200
    plan = response.json()
    assert plan["via"] == [[57.0, 19.0]] and plan["result_file"]
    bad = client.post("/api/route", json={"start_lat": 54.3, "start_lon": 10.1, "end_lat": 59.4,
                                          "end_lon": 24.7, "parameters": {"nonsense": 1}})
    assert bad.status_code == 422
