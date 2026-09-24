"""Integration tests of the trained model and of the HTTP/WebSocket API.

模型推理与 Web 接口的集成测试（需要已训练的模型）。

Three groups of tests:

* model - the trained ``ZoneClassifierService``: probability output, plausible
  classes for known places, reported quality, determinism, batch consistency;
* routing - ``RoutePlanner`` on the real model for Kiel -> Tallinn: no land
  between the berths, correct end points, land-crossing great-circle baseline,
  A*/Dijkstra agreement and automatic lattice refinement;
* API - every group of endpoints of ``maritime_route.web.app`` through
  FastAPI's ``TestClient`` (REST requests, validation errors, exports, upload,
  offline tiles) and the ``/ws/plan`` WebSocket protocol.

The whole module is skipped (``pytestmark = requires_model``) when the trained
model files are missing.
整个模块在缺少已训练模型时跳过。
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import requires_model
from maritime_route.config import CLASS_NAMES

# Module-level marker: applies the skip condition to every test in this file.
pytestmark = requires_model


@pytest.fixture(scope="module")
def client():
    """FastAPI TestClient bound to the real application, shared by the module.

    Using the client as a context manager runs the application's start-up and
    shutdown events. Requests are served in-process, without a network socket.
    Note that routes planned here are stored in the application's configured
    database (the ``STATE`` repository), not in a temporary one.

    Yields
    ------
    fastapi.testclient.TestClient
    """
    from maritime_route.web.app import app
    with TestClient(app) as test_client:
        yield test_client


# ------------------------------------------------------------------- model
def test_probabilities_are_a_distribution(classifier):
    """Predicted class probabilities have shape (N, 4), are non-negative and sum to 1 per point."""
    probs = classifier.predict_coordinates([54.5, 57.0, 52.5], [10.2, 19.0, 13.4])
    assert probs.shape == (3, len(CLASS_NAMES))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert (probs >= 0).all()


def test_known_locations_are_classified_plausibly(classifier):
    """Deep Atlantic must be sea; a point deep inland must be land."""
    results = classifier.classify_coordinates(
        [58.0, 52.52], [-25.0, 13.40])          # mid-Atlantic, Berlin
    assert results[0].class_code in ("OPEN_SEA", "COASTAL_SEA")
    assert results[1].class_code in ("COASTLINE", "NEAR_COAST")


def test_model_summary_reports_trained_quality(classifier):
    """Model summary reports 28 features, a positive parameter count and test accuracy above 90 %."""
    summary = classifier.summary()
    assert summary["n_features"] == 28
    assert summary["parameters"] > 0
    assert summary["accuracy"] > 0.90


def test_inference_is_deterministic(classifier):
    """Repeated inference on the same coordinate gives identical probabilities (eval mode, no dropout)."""
    a = classifier.predict_coordinates([55.0], [12.0])
    b = classifier.predict_coordinates([55.0], [12.0])
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_batched_inference_matches_single(classifier):
    """Classifying points in one batch gives the same result as one call per point."""
    lat = [54.0, 55.0, 56.0, 57.0]
    lon = [10.0, 12.0, 14.0, 16.0]
    batch = classifier.predict_coordinates(lat, lon)
    single = np.vstack([classifier.predict_coordinates([a], [b]) for a, b in zip(lat, lon)])
    np.testing.assert_allclose(batch, single, rtol=1e-5, atol=1e-6)


# ------------------------------------------------------------------ routing
def test_planned_route_never_crosses_land(planner):
    """Every intermediate waypoint must be navigable.

    The two end points are the berths the user asked for, and a quay legitimately
    lies in a shore zone, so the invariant is stated over the interior of the route.
    首尾为用户指定的泊位，允许位于岸线区；中间航路点必须全部可航行。
    """
    plan, _ = planner.plan((54.3233, 10.1228), (59.4370, 24.7536), "astar", 0.25)
    assert plan.feasible
    assert plan.interior_land_waypoints == 0
    for leg in plan.legs[1:-1]:
        assert leg.class_code in ("OPEN_SEA", "COASTAL_SEA")
    assert plan.distance_km >= plan.great_circle_km


def test_departure_berth_may_lie_on_the_shore(planner):
    """Kiel is a port: its exact coordinates fall in a shore cell by design."""
    plan, _ = planner.plan((54.3233, 10.1228), (59.4370, 24.7536), "astar", 0.25)
    assert plan.legs[0].latitude == pytest.approx(54.3233)
    assert plan.legs[-1].latitude == pytest.approx(59.4370)
    assert plan.interior_land_waypoints == 0


def test_baseline_reports_land_crossing(planner):
    """The great-circle baseline Kiel -> Tallinn is reported as crossing land with no weighted cost."""
    plan, _ = planner.plan((54.3233, 10.1228), (59.4370, 24.7536), "astar", 0.25)
    assert plan.baseline["is_navigable"] is False
    assert plan.baseline["land_samples"] > 0
    assert plan.baseline["weighted_cost"] is None


def test_astar_and_dijkstra_return_the_same_optimum(planner):
    """On one shared cost map A* and Dijkstra reach the same cost, A* with no more expanded nodes."""
    plan_a, cost_map = planner.plan((54.3233, 10.1228), (59.4370, 24.7536), "astar", 0.3)
    plan_d, _ = planner.plan((54.3233, 10.1228), (59.4370, 24.7536), "dijkstra",
                             cost_map.spec.resolution_deg, reuse_cost_map=cost_map,
                             auto_refine=False)
    assert plan_a.total_cost == pytest.approx(plan_d.total_cost, rel=1e-6)
    assert plan_a.nodes_expanded <= plan_d.nodes_expanded


def test_auto_refinement_recovers_a_narrow_strait(planner):
    """At 0.5° the Danish straits are narrower than a cell; refinement must recover."""
    plan, cost_map = planner.plan((54.3233, 10.1228), (59.4370, 24.7536), "astar", 0.5)
    assert plan.feasible
    assert cost_map.spec.resolution_deg < 0.5


# ---------------------------------------------------------------------- API
def test_health_and_zones(client):
    """GET /api/health reports "ok"; GET /api/zones lists the classes in model order, two of them navigable."""
    assert client.get("/api/health").json()["status"] == "ok"
    zones = client.get("/api/zones").json()["classes"]
    assert [z["code"] for z in zones] == CLASS_NAMES
    assert sum(1 for z in zones if z["navigable"]) == 2


def test_index_page_is_served(client):
    """GET / returns the single-page client HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Maritime Route Planner" in response.text


def test_classify_endpoint(client):
    """POST /api/classify returns one result per point and a histogram with the same total."""
    response = client.post("/api/classify", json={"points": [[54.5, 10.2], [52.5, 13.4]]})
    assert response.status_code == 200
    body = response.json()
    assert body["n"] == 2
    assert sum(body["histogram"].values()) == 2


def test_classify_rejects_bad_coordinates(client):
    """POST /api/classify answers 422 for an out-of-range latitude and for a one-number point."""
    assert client.post("/api/classify", json={"points": [[200.0, 10.0]]}).status_code == 422
    assert client.post("/api/classify", json={"points": [[10.0]]}).status_code == 422


def test_route_endpoint_and_exports(client):
    """POST /api/route plans a feasible route that exports as GeoJSON, CSV and PDF; an unknown format gives 400."""
    response = client.post("/api/route", json={
        "start_lat": 54.3233, "start_lon": 10.1228,
        "end_lat": 59.4370, "end_lon": 24.7536,
        "algorithm": "astar", "resolution_deg": 0.3,
    })
    assert response.status_code == 200
    plan = response.json()
    assert plan["feasible"] is True
    route_id = plan["route_id"]

    geojson = client.get(f"/api/export/{route_id}.geojson")
    assert geojson.status_code == 200
    assert json.loads(geojson.text)["type"] == "FeatureCollection"

    csv_response = client.get(f"/api/export/{route_id}.csv")
    assert csv_response.status_code == 200
    assert csv_response.text.startswith("seq,latitude,longitude")

    pdf = client.get(f"/api/export/{route_id}.pdf")
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF")

    assert client.get(f"/api/export/{route_id}.docx").status_code == 400


def test_route_validation_rejects_bad_input(client):
    """POST /api/route answers 422 for a latitude above 90 and for an unknown algorithm."""
    bad = {"start_lat": 91.0, "start_lon": 0.0, "end_lat": 0.0, "end_lon": 0.0}
    assert client.post("/api/route", json=bad).status_code == 422
    bad_alg = {"start_lat": 54.0, "start_lon": 10.0, "end_lat": 55.0, "end_lon": 11.0,
               "algorithm": "bfs"}
    assert client.post("/api/route", json=bad_alg).status_code == 422


def test_routes_are_listed_and_retrievable(client):
    """Stored routes are listed, retrievable by uid, and an unknown uid gives 404."""
    client.post("/api/route", json={
        "start_lat": 54.3233, "start_lon": 10.1228,
        "end_lat": 57.0, "end_lon": 18.0, "resolution_deg": 0.4})
    routes = client.get("/api/routes?limit=5").json()["routes"]
    assert routes
    uid = routes[0]["route_uid"]
    assert client.get(f"/api/routes/{uid}").json()["route_uid"] == uid
    assert client.get("/api/routes/does-not-exist").status_code == 404


def test_upload_geojson_and_classify(client, tmp_path):
    """POST /api/upload classifies an uploaded GeoJSON and compares against its labels."""
    document = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {"class_code": "OPEN_SEA"},
             "geometry": {"type": "Point", "coordinates": [-25.0, 58.0]}},
            {"type": "Feature", "properties": {"class_code": "COASTLINE"},
             "geometry": {"type": "Point", "coordinates": [13.40, 52.52]}},
        ],
    }
    path = tmp_path / "track.geojson"
    path.write_text(json.dumps(document), encoding="utf-8")
    with path.open("rb") as handle:
        response = client.post("/api/upload",
                               files={"file": ("track.geojson", handle, "application/json")})
    assert response.status_code == 200
    body = response.json()
    assert body["classified"] == 2
    assert body["accuracy_vs_labels"]["n_labelled"] == 2


def test_upload_rejects_a_file_without_coordinates(client, tmp_path):
    """POST /api/upload answers 400 for a CSV without coordinate columns."""
    path = tmp_path / "bad.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with path.open("rb") as handle:
        response = client.post("/api/upload", files={"file": ("bad.csv", handle, "text/csv")})
    assert response.status_code == 400


def test_on_board_tile_is_a_png(client):
    """Offline tile endpoint returns a PNG (checked by signature) and 400 for zoom > 10."""
    response = client.get("/api/tiles/zones/4/8/5.png")
    assert response.status_code == 200
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    # Zoom 14 is above the on-board chart limit (0..10).
    assert client.get("/api/tiles/zones/14/1/1.png").status_code == 400


def test_websocket_live_planning(client):
    """WebSocket protocol: hello, ping/pong, classify, then started/progress/overlay/result for a plan."""
    with client.websocket_connect("/ws/plan") as socket:
        assert socket.receive_json()["type"] == "hello"

        socket.send_json({"action": "ping"})
        assert socket.receive_json()["type"] == "pong"

        socket.send_json({"action": "classify", "points": [[54.5, 10.2]]})
        frame = socket.receive_json()
        assert frame["type"] == "classified"
        assert frame["results"][0]["class_code"] in CLASS_NAMES

        socket.send_json({
            "action": "plan", "start_lat": 54.3233, "start_lon": 10.1228,
            "end_lat": 59.4370, "end_lon": 24.7536, "algorithm": "astar",
            "resolution_deg": 0.3, "include_overlay": True,
        })
        # Collect frame types until the result arrives; the 400-frame bound
        # keeps a protocol bug from hanging the test suite forever.
        # 限制最多读取 400 帧，避免协议错误导致测试无限等待。
        seen = set()
        for _ in range(400):
            frame = socket.receive_json()
            seen.add(frame["type"])
            if frame["type"] == "result":
                assert frame["plan"]["feasible"] is True
                break
            if frame["type"] == "error":
                pytest.fail(frame["detail"])
        assert {"started", "progress", "overlay", "result"} <= seen


def test_websocket_reports_a_bad_request(client):
    """WebSocket answers an invalid plan request and an unknown action with "error" frames."""
    with client.websocket_connect("/ws/plan") as socket:
        socket.receive_json()
        socket.send_json({"action": "plan", "start_lat": 500.0})
        assert socket.receive_json()["type"] == "error"
        socket.send_json({"action": "teleport"})
        assert socket.receive_json()["type"] == "error"


def test_statistics_endpoint(client):
    """GET /api/statistics contains the session, point, route and waypoint counters."""
    stats = client.get("/api/statistics").json()
    assert {"sessions", "points", "routes", "waypoints"} <= set(stats)
