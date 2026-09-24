"""FastAPI application: REST endpoints + WebSocket for real-time interaction.

Web 服务层：REST 接口 + WebSocket 实时交互。
The WebSocket channel streams planning progress (grid construction, neural
network classification, graph search) to the browser so that the user sees the
route being built instead of waiting for a single blocking response.

Place in the application
------------------------
This module is the outermost layer of the system. It does not implement any
navigation logic itself; it validates HTTP/WebSocket input, forwards the work
to the lower layers and serialises their results to JSON:

* ``maritime_route.model.inference.ZoneClassifierService`` - the trained
  PyTorch zone classifier (coordinates -> navigation zone probabilities);
* ``maritime_route.routing.planner.RoutePlanner`` - cost-map construction and
  A* / Dijkstra / genetic / great-circle route search;
* ``maritime_route.storage.repository.RouteRepository`` - SQLite persistence of
  upload sessions, classified points and planned routes;
* ``maritime_route.export.exporters`` - GeoJSON / CSV / PDF export;
* ``maritime_route.web.tiles`` - offline map tiles rendered from the model.

Main objects
------------
``PlanRequest`` / ``ClassifyRequest``
    Pydantic models that validate request bodies (FastAPI answers 422 when
    validation fails).
``AppState`` / ``STATE``
    Process-wide singleton with the lazily loaded model, the planner, the
    repository and two small in-memory caches (recent plans, recent uploads).
``create_app()``
    Application factory that registers every route; ``app`` at the bottom of
    the module is the instance served by Uvicorn (``maritime_route.web:app``).
``_ws_plan`` / ``_ws_classify``
    Helpers that implement the WebSocket actions.

Concurrency model
-----------------
All endpoints are ``async``. CPU-bound work (neural-network inference, graph
search, file parsing, SQLite writes, PDF rendering) is moved to the default
thread pool with :func:`asyncio.to_thread` so that the event loop keeps serving
other requests and, for the WebSocket, keeps forwarding progress frames while
a route is being computed.
所有耗时计算都通过 asyncio.to_thread 放到线程池执行，避免阻塞事件循环。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from ..config import (
    APP_AUTHOR,
    APP_VERSION,
    CLASS_COLOR,
    CLASS_COLOR_DARK,
    CLASS_DESCRIPTION,
    CLASS_NAMES,
    OUTPUT_DIR,
    ROUTING,
    SUPPORTED_ALGORITHMS,
    SERVER,
    ZONE_COST,
)
from ..data.ais_loader import DataLoadError, load_any, validate
from ..export.exporters import (
    classified_points_to_geojson, points_to_csv_string, route_to_csv_string,
    route_to_geojson, route_to_pdf,
)
from ..model.inference import ModelNotTrainedError, ZoneClassifierService
from ..routing.planner import RoutePlanner, benchmark
from ..storage.repository import RouteRepository
from .tiles import cache_info as tile_cache_info, render_tile

LOGGER = logging.getLogger(__name__)
#: Directory with index.html, css/, js/ and vendor/ (served under /static).
#: 前端静态文件目录。
STATIC_DIR = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- #
# Request models                                                              #
# --------------------------------------------------------------------------- #
class PlanRequest(BaseModel):
    """Body of ``POST /api/route``. 航线规划请求体。

    Also used by ``POST /api/benchmark`` and, field by field, by the
    ``"plan"`` action of the ``/ws/plan`` WebSocket. Range checks are declared
    with ``Field`` constraints, so an out-of-range value is rejected by
    FastAPI with HTTP 422 before the endpoint body runs.

    Attributes
    ----------
    start_lat, start_lon : float
        Departure point in decimal degrees (WGS-84), latitude in [-90, 90],
        longitude in [-180, 180].
    end_lat, end_lon : float
        Destination point, same ranges.
    algorithm : str
        One of ``SUPPORTED_ALGORITHMS`` (``"astar"``, ``"dijkstra"``,
        ``"genetic"``, ``"great_circle"``); case-insensitive, normalised to
        lower case by the validator.
    resolution_deg : float
        Cell size of the routing lattice in degrees, 0.01 < value <= 3.0.
        The planner may refine it automatically when a strait is too narrow.
    speed_knots : float
        Vessel speed used only for the travel-time estimate, 0.1 < value <= 60.
    simplify_km : float
        Douglas-Peucker tolerance for the final poly-line; 0 disables
        simplification.
    persist : bool
        If true the finished plan is also written to the SQLite database.
    """

    start_lat: float = Field(..., ge=-90, le=90)
    start_lon: float = Field(..., ge=-180, le=180)
    end_lat: float = Field(..., ge=-90, le=90)
    end_lon: float = Field(..., ge=-180, le=180)
    algorithm: str = Field("astar")
    resolution_deg: float = Field(ROUTING.grid_resolution_deg, gt=0.01, le=3.0)
    speed_knots: float = Field(14.0, gt=0.1, le=60.0)
    simplify_km: float = Field(0.0, ge=0.0, le=200.0)
    persist: bool = True

    @field_validator("algorithm")
    @classmethod
    def _check_algorithm(cls, value: str) -> str:
        """Normalise the algorithm name and reject unknown ones.

        Parameters
        ----------
        value : str
            Algorithm name as sent by the client.

        Returns
        -------
        str
            The lower-case name, guaranteed to be in ``SUPPORTED_ALGORITHMS``.

        Raises
        ------
        ValueError
            If the name is not supported (FastAPI turns it into HTTP 422).
        """
        value = value.lower()
        if value not in SUPPORTED_ALGORITHMS:
            raise ValueError("algorithm must be one of: " + ", ".join(SUPPORTED_ALGORITHMS))
        return value


class ClassifyRequest(BaseModel):
    """Body of ``POST /api/classify``. 坐标分类请求体。

    Attributes
    ----------
    points : list of [float, float]
        Between 1 and 50 000 ``[latitude, longitude]`` pairs. The upper bound
        keeps a single request from occupying the model for too long.
    """

    points: List[List[float]] = Field(..., min_length=1, max_length=50000)

    @field_validator("points")
    @classmethod
    def _check_points(cls, value):
        """Check that every item is a valid ``[lat, lon]`` pair.

        Parameters
        ----------
        value : list of list of float
            Raw point list from the request body.

        Returns
        -------
        list of list of float
            The same list, unchanged, if all points are valid.

        Raises
        ------
        ValueError
            If a point does not have exactly two numbers or lies outside the
            valid latitude/longitude range (reported to the client as 422).
        """
        for p in value:
            if len(p) != 2:
                raise ValueError("each point must be [latitude, longitude]")
            if not (-90 <= p[0] <= 90 and -180 <= p[1] <= 180):
                raise ValueError(f"coordinate out of range: {p}")
        return value


# --------------------------------------------------------------------------- #
# Application state                                                           #
# --------------------------------------------------------------------------- #
class AppState:
    """Process-wide singletons. 应用级共享状态。

    One instance (``STATE``) is created at import time and shared by all
    requests of the process. The model is *not* loaded here: loading is
    deferred to :meth:`ensure_model` so that the server can start (and serve
    the UI and ``/api/health``) even before a model has been trained.
    模型延迟加载，未训练模型时服务器依然可以启动。

    Attributes
    ----------
    classifier : ZoneClassifierService or None
        Trained zone classifier, ``None`` until the first successful
        :meth:`ensure_model` call.
    planner : RoutePlanner or None
        Route planner bound to ``classifier``.
    repository : RouteRepository
        SQLite repository (sessions, points, routes, waypoints).
    plans : dict of str to RoutePlan
        The most recent plans (at most 64) keyed by ``route_id``; needed by
        ``/api/export/{route_id}.{fmt}`` because exporters work on the full
        in-memory ``RoutePlan`` object, not on the database row.
    uploads : dict of str to pandas.DataFrame
        Uploaded AIS frames keyed by ``"upload-<session_id>"``, used by
        ``/api/export/points/{token}.{fmt}``. This cache is not trimmed; it
        lives as long as the process.
    started : float
        UNIX time when the state was created, used for ``uptime_s``.
    """

    def __init__(self) -> None:
        """Create empty state; the model is loaded lazily later."""
        self.classifier: Optional[ZoneClassifierService] = None
        self.planner: Optional[RoutePlanner] = None
        self.repository = RouteRepository()
        self.plans: Dict[str, Any] = {}          # route_id -> RoutePlan
        self.uploads: Dict[str, pd.DataFrame] = {}
        self.started = time.time()

    def ensure_model(self) -> RoutePlanner:
        """Load the classifier and planner on first use and return the planner.

        ``ZoneClassifierService.instance()`` is itself a thread-safe
        singleton, so concurrent first calls end up sharing one model object.

        Returns
        -------
        RoutePlanner
            The shared planner instance.

        Raises
        ------
        HTTPException
            Status 503 (Service Unavailable) if no trained model files exist;
            the detail text tells the user to run the training script.
        """
        if self.planner is None:
            try:
                self.classifier = ZoneClassifierService.instance()
            except ModelNotTrainedError as exc:
                # 503 = the service is up but its model is missing. 模型缺失返回 503。
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            self.planner = RoutePlanner(self.classifier)
        return self.planner

    def remember(self, plan) -> None:
        """Keep a plan in memory so that it can be exported later.

        Parameters
        ----------
        plan : RoutePlan
            Finished plan; stored under ``plan.route_id``.

        Notes
        -----
        Python dictionaries keep insertion order, so the first keys are the
        oldest plans. When more than 64 plans are stored, everything except
        the last 64 keys is dropped (a first-in-first-out trim).
        字典保持插入顺序，超过 64 条时删除最早的记录。
        """
        self.plans[plan.route_id] = plan
        if len(self.plans) > 64:                        # simple LRU trim
            for key in list(self.plans)[:-64]:
                self.plans.pop(key, None)


#: The single shared state object used by every endpoint. 全局唯一状态对象。
STATE = AppState()


def create_app() -> FastAPI:
    """Application factory. 创建 FastAPI 应用。

    Builds a :class:`fastapi.FastAPI` instance, enables CORS, mounts the static
    single-page client and registers all REST and WebSocket endpoints as
    closures. All endpoints share the module-level ``STATE`` object.

    Endpoint overview
    -----------------
    ======  ==================================  ================================
    Method  Path                                Purpose
    ======  ==================================  ================================
    GET     /                                   single-page client (index.html)
    GET     /static/...                         CSS, JS, vendor libraries
    GET     /favicon.ico                        inline SVG icon
    GET     /api/health                         liveness + model availability
    GET     /api/zones                          zone classes, colours, costs
    GET     /api/model                          model summary and metrics
    GET     /api/statistics                     database statistics
    GET     /api/tiles/zones/{z}/{x}/{y}.png    offline chart tile
    GET     /api/tiles/info                     tile cache size
    POST    /api/classify                       classify coordinates
    POST    /api/upload                         upload + classify AIS file
    POST    /api/route                          plan one route
    POST    /api/benchmark                      compare all algorithms
    GET     /api/routes                         list stored routes
    GET     /api/routes/{route_uid}             one stored route
    DELETE  /api/routes/{route_uid}             delete stored route
    GET     /api/export/{route_id}.{fmt}        export route (geojson/csv/pdf)
    GET     /api/export/points/{token}.{fmt}    export upload (geojson/csv)
    WS      /ws/plan                            live planning with progress
    ======  ==================================  ================================

    Returns
    -------
    FastAPI
        Fully configured application.
    """
    app = FastAPI(title=SERVER.title, version=SERVER.version,
                  description="Neural-network based maritime route planning service")
    # CORS is fully open: the API is meant to be called from the bundled page
    # or from local tools/notebooks, and it carries no user credentials.
    # 允许任意来源跨域访问（接口不涉及用户凭据）。
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                       allow_headers=["*"])

    # ------------------------------------------------------------------ #
    # Static UI                                                          #
    # ------------------------------------------------------------------ #
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> HTMLResponse:
        """Serve the single-page client.

        ``GET /`` - returns ``static/index.html``. The file is read on every
        request (it is small), so edits to the page are visible without a
        server restart. Hidden from the OpenAPI schema.

        Returns
        -------
        HTMLResponse
            The HTML page, status 200.
        """
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # Metadata                                                           #
    # ------------------------------------------------------------------ #
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon() -> Response:
        """Inline SVG favicon so that no external asset is required.

        ``GET /favicon.ico`` - returns a 24x24 SVG (anchor symbol on a dark
        square) with a one-week ``Cache-Control`` header. Browsers request this
        path automatically; answering it avoids 404 lines in the log.

        Returns
        -------
        Response
            ``image/svg+xml`` body.
        """
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
            '<rect width="24" height="24" rx="5" fill="#0e1621"/>'
            '<path d="M12 4v11M12 6h4l-4 3M5 17c2 2 4 2 4 2h6s2 0 4-2l-2 4H7z" '
            'fill="none" stroke="#4da3ff" stroke-width="1.5" '
            'stroke-linecap="round" stroke-linejoin="round"/></svg>'
        )
        return Response(svg, media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=604800"})

    @app.get("/api/health")
    async def health() -> Dict:
        """Report whether the service and its model are available.

        ``GET /api/health``. The call also triggers lazy model loading, so the
        first health check after start-up may take a moment. It never fails
        with an error status: a missing model is reported in the body.

        Returns
        -------
        dict
            ``{"status": "ok" | "model_missing", "detail": str,
            "version": str, "uptime_s": float}``; ``detail`` is ``"ready"``
            or the reason why the model could not be loaded.
        """
        ready = True
        detail = "ready"
        try:
            STATE.ensure_model()
        except HTTPException as exc:
            ready, detail = False, str(exc.detail)
        return {"status": "ok" if ready else "model_missing", "detail": detail,
                "version": APP_VERSION, "uptime_s": round(time.time() - STATE.started, 1)}

    @app.get("/api/zones")
    async def zones() -> Dict:
        """Describe the navigation-zone classes predicted by the model.

        ``GET /api/zones``. Does not need the model; the data comes from
        ``config``. The client uses it to build the map legend.

        Returns
        -------
        dict
            ``{"classes": [{"code", "index", "description", "colour", "cost",
            "navigable"}, ...]}`` in model output order. ``cost`` is ``None``
            (JSON ``null``) for impassable zones because JSON has no infinity;
            ``navigable`` is ``False`` for the same zones.
        """
        return {
            "classes": [
                {
                    "code": name,
                    "index": i,
                    "description": CLASS_DESCRIPTION[name],
                    "colour": CLASS_COLOR_DARK[name],
                    # float("inf") is not valid JSON, so impassable zones get null.
                    # JSON 不支持无穷大，不可航行区的代价输出为 null。
                    "cost": None if ZONE_COST[name] == float("inf") else ZONE_COST[name],
                    "navigable": ZONE_COST[name] != float("inf"),
                }
                for i, name in enumerate(CLASS_NAMES)
            ]
        }

    @app.get("/api/model")
    async def model_info() -> Dict:
        """Return the model summary and the metadata saved at training time.

        ``GET /api/model``.

        Returns
        -------
        dict
            ``{"summary", "metrics", "hyperparameters", "dataset",
            "history"}``. ``summary`` comes from
            ``ZoneClassifierService.summary()`` (architecture, parameter
            count); the other keys are copied from the metadata JSON written by
            the trainer (``history`` holds per-epoch loss/accuracy curves used
            by the charts on the "Model" tab). Missing keys become ``{}``.

        Raises
        ------
        HTTPException
            503 if the model has not been trained.
        """
        STATE.ensure_model()
        metadata = STATE.classifier.metadata
        return {
            "summary": STATE.classifier.summary(),
            "metrics": metadata.get("metrics", {}),
            "hyperparameters": metadata.get("hyperparameters", {}),
            "dataset": metadata.get("dataset", {}),
            "history": metadata.get("history", {}),
        }

    @app.get("/api/statistics")
    async def statistics() -> Dict:
        """Return aggregate statistics of the SQLite database.

        ``GET /api/statistics``. Does not need the model.

        Returns
        -------
        dict
            Output of ``RouteRepository.statistics()``: counts of sessions,
            points, routes and waypoints, the class histogram of stored
            points, per-algorithm averages (``by_algorithm``) and the database
            file name and size.
        """
        return STATE.repository.statistics()

    @app.get("/api/tiles/zones/{z}/{x}/{y}.png", include_in_schema=False)
    async def zone_tile(z: int, x: int, y: int):
        """On-board chart tile: the network's classification rendered as PNG.

        离线海图瓦片接口，使应用在无网络环境下仍可显示底图。

        ``GET /api/tiles/zones/{z}/{x}/{y}.png`` - standard slippy-map
        (XYZ, Web Mercator) addressing, as used by Leaflet ``L.tileLayer``.

        Parameters
        ----------
        z : int
            Zoom level, 0..10. Higher levels are refused because the model
            works on a coarse lattice and deeper tiles would add no detail.
        x, y : int
            Tile column and row, each in ``[0, 2**z)``; ``y = 0`` is the
            northern edge.

        Returns
        -------
        Response
            ``image/png`` 256x256 RGBA tile with a one-day ``Cache-Control``.

        Raises
        ------
        HTTPException
            503 if the model is missing, 400 for a zoom outside 0..10, 404 for
            ``x``/``y`` outside the tile matrix of that zoom level.
        """
        STATE.ensure_model()
        if not (0 <= z <= 10):
            raise HTTPException(400, "Zoom level out of range for the on-board chart")
        # At zoom z the world is a 2**z x 2**z matrix of tiles. 第 z 级共有 2^z × 2^z 张瓦片。
        limit = 2 ** z
        if not (0 <= x < limit and 0 <= y < limit):
            raise HTTPException(404, "Tile out of range")
        # Rendering runs inference + PNG encoding: done in a worker thread so the
        # event loop can serve the other tiles Leaflet requests in parallel.
        # 渲染在线程池中执行，不阻塞其它并发的瓦片请求。
        payload = await asyncio.to_thread(render_tile, STATE.classifier, z, x, y)
        return Response(payload, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/tiles/info")
    async def tile_info() -> Dict:
        """Report the state of the in-memory tile cache.

        ``GET /api/tiles/info``.

        Returns
        -------
        dict
            ``{"tiles_cached": int, "limit": int}``.
        """
        return tile_cache_info()

    # ------------------------------------------------------------------ #
    # Classification                                                     #
    # ------------------------------------------------------------------ #
    @app.post("/api/classify")
    async def classify(request: ClassifyRequest) -> Dict:
        """Classify arbitrary coordinates with the neural network.

        ``POST /api/classify``

        Parameters
        ----------
        request : ClassifyRequest
            JSON body ``{"points": [[lat, lon], ...]}`` (1..50 000 points).

        Returns
        -------
        dict
            ``{"n": int, "histogram": {class_code: count},
            "results": [ClassificationResult.to_dict(), ...]}``; each result
            carries ``latitude``, ``longitude``, ``class_index``,
            ``class_code``, ``confidence``, ``probabilities`` and ``depth_m``.

        Raises
        ------
        HTTPException
            503 if the model is missing; 422 for an invalid body.
        """
        STATE.ensure_model()
        array = np.asarray(request.points, dtype=float)
        results = await asyncio.to_thread(
            STATE.classifier.classify_coordinates, array[:, 0], array[:, 1]
        )
        # Start from zero for every class so that absent classes still appear.
        # 所有类别初始化为 0，保证未出现的类别也在直方图中。
        histogram: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
        for r in results:
            histogram[r.class_code] += 1
        return {"n": len(results), "histogram": histogram,
                "results": [r.to_dict() for r in results]}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)) -> Dict:
        """Upload an AIS file (GeoJSON/CSV/JSON) and classify its points.

        上传 AIS 数据文件并即时分类。

        ``POST /api/upload`` - ``multipart/form-data`` with one field ``file``.

        Processing steps: size check -> parse by file type (``load_any``) ->
        clean and validate (``validate``) -> keep the first 20 000 rows ->
        classify -> store a new session and its points in SQLite -> cache the
        frame in ``STATE.uploads`` for later export.

        Parameters
        ----------
        file : UploadFile
            The uploaded file; the extension of ``file.filename`` selects the
            parser.

        Returns
        -------
        dict
            ``token`` (``"upload-<session_id>"``, used by the points export
            endpoint), ``session_id``, ``report`` (``LoadReport.to_dict()``:
            ``source``, ``n_features``, ``n_valid``, ``n_dropped``,
            ``labelled``, ``columns``, ``warnings``), ``classified`` (number of classified
            points), ``histogram`` (count per class), ``accuracy_vs_labels``
            (``{"n_labelled", "agreement"}`` or ``None`` when the file has no
            ``class_code`` column) and ``points`` (at most about 6 000 evenly
            thinned classified points for drawing on the map).

        Raises
        ------
        HTTPException
            503 if the model is missing, 413 if the file exceeds 80 MB, 400 if
            the file cannot be parsed or contains no valid AIS rows.
        """
        STATE.ensure_model()
        # The whole body is read into memory, hence the explicit size limit.
        # 文件整体读入内存，因此设置 80 MB 上限。
        content = await file.read()
        if len(content) > 80 * 1024 * 1024:
            raise HTTPException(413, "File too large (limit 80 MB)")
        try:
            frame = await asyncio.to_thread(load_any, file.filename or "upload.json", content)
            frame, report = validate(frame, source=file.filename or "upload")
        except DataLoadError as exc:
            raise HTTPException(400, str(exc)) from exc

        # Only the first 20 000 rows are classified and stored interactively;
        # larger data sets belong to the offline preparation scripts.
        # 交互模式只处理前 20000 行，大数据集应使用离线脚本。
        preview = frame.head(20000).reset_index(drop=True)
        results = await asyncio.to_thread(STATE.classifier.classify_frame, preview)
        session_id = STATE.repository.create_session(
            name=file.filename or "upload", source="web-upload", notes="interactive upload")
        await asyncio.to_thread(STATE.repository.save_points, session_id, results)
        token = f"upload-{session_id}"
        STATE.uploads[token] = preview
        # Keep only the 16 most recent uploads in memory (dicts preserve insertion
        # order), so a long-running server does not grow without bound.
        # 内存中只保留最近 16 次上传，避免长时间运行时内存无限增长。
        while len(STATE.uploads) > 16:
            STATE.uploads.pop(next(iter(STATE.uploads)))

        histogram: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
        for r in results:
            histogram[r.class_code] += 1
        # Thin the returned points to roughly 6 000 so the browser stays fluid;
        # the histogram above is still computed on all points.
        # 返回给前端的点抽稀到约 6000 个，直方图仍基于全部点。
        step = max(1, len(results) // 6000)
        return {
            "token": token,
            "session_id": session_id,
            "report": report.to_dict(),
            "classified": len(results),
            "histogram": histogram,
            "accuracy_vs_labels": _label_agreement(preview, results),
            "points": [r.to_dict() for r in results[::step]],
        }

    # ------------------------------------------------------------------ #
    # Routing                                                            #
    # ------------------------------------------------------------------ #
    @app.post("/api/route")
    async def plan_route(request: PlanRequest) -> Dict:
        """Plan one route in a single blocking request (no progress frames).

        ``POST /api/route`` - the non-streaming alternative to ``/ws/plan``;
        useful for scripts and tests.

        Parameters
        ----------
        request : PlanRequest
            JSON body with start/end coordinates and planning options.

        Returns
        -------
        dict
            ``RoutePlan.to_dict()``: ``route_id``, ``algorithm``, ``start``,
            ``end``, ``waypoints`` ([[lat, lon], ...]), ``legs`` (per-waypoint
            zone, cost, cumulative distance, bearing), ``total_cost``,
            ``distance_km``, ``great_circle_km``, ``detour_ratio``,
            ``estimated_hours``, ``nodes_expanded``, ``runtime_s``, ``grid``,
            ``zone_profile``, ``baseline``, ``cost_map_stats``,
            ``created_utc``, ``speed_knots``, ``interior_land_waypoints``,
            ``feasible`` and ``message``. An impossible voyage is *not* an
            HTTP error: it is returned with ``feasible = false`` and a
            ``message`` explaining why.

        Raises
        ------
        HTTPException
            503 if the model is missing; 422 for an invalid body.
        """
        planner = STATE.ensure_model()
        plan, _ = await asyncio.to_thread(
            planner.plan,
            (request.start_lat, request.start_lon),
            (request.end_lat, request.end_lon),
            request.algorithm, request.resolution_deg, request.speed_knots,
            request.simplify_km,
        )
        STATE.remember(plan)
        if request.persist:
            await asyncio.to_thread(STATE.repository.save_route, plan)
        return plan.to_dict()

    @app.post("/api/benchmark")
    async def run_benchmark(request: PlanRequest) -> Dict:
        """Run every algorithm on the same cost map and compare them.

        ``POST /api/benchmark`` - uses only the coordinates and
        ``resolution_deg`` of the body; ``algorithm``, ``speed_knots``,
        ``simplify_km`` and ``persist`` are ignored and nothing is stored.

        Parameters
        ----------
        request : PlanRequest
            Same body as ``/api/route``.

        Returns
        -------
        dict
            Output of :func:`maritime_route.routing.planner.benchmark`: one
            row per algorithm (``feasible``, ``distance_km``, ``total_cost``,
            ``nodes_expanded``, ``runtime_s``, ``detour_ratio``, ``note``)
            plus derived comparisons such as the A*/Dijkstra node-expansion
            speed-up and the genetic algorithm's gap to the optimum.

        Raises
        ------
        HTTPException
            503 if the model is missing; 422 for an invalid body.
        """
        planner = STATE.ensure_model()
        return await asyncio.to_thread(
            benchmark, planner, (request.start_lat, request.start_lon),
            (request.end_lat, request.end_lon), request.resolution_deg,
        )

    @app.get("/api/routes")
    async def list_routes(limit: int = Query(30, ge=1, le=200)) -> Dict:
        """List the most recent stored routes.

        ``GET /api/routes?limit=30``

        Parameters
        ----------
        limit : int
            Maximum number of routes, 1..200 (default 30); other values give
            HTTP 422.

        Returns
        -------
        dict
            ``{"routes": [...]}`` - summary rows from
            ``RouteRepository.list_routes`` (newest first).
        """
        return {"routes": STATE.repository.list_routes(limit)}

    @app.get("/api/routes/{route_uid}")
    async def get_route(route_uid: str) -> Dict:
        """Fetch one stored route with its waypoints.

        ``GET /api/routes/{route_uid}``

        Parameters
        ----------
        route_uid : str
            The ``route_id`` assigned by the planner.

        Returns
        -------
        dict
            The stored route as returned by ``RouteRepository.get_route``.

        Raises
        ------
        HTTPException
            404 if no route with this id is stored.
        """
        route = STATE.repository.get_route(route_uid)
        if route is None:
            raise HTTPException(404, "Route not found")
        return route

    @app.delete("/api/routes/{route_uid}")
    async def delete_route(route_uid: str) -> Dict:
        """Delete a stored route and forget its in-memory copy.

        ``DELETE /api/routes/{route_uid}``

        Parameters
        ----------
        route_uid : str
            The ``route_id`` to delete.

        Returns
        -------
        dict
            ``{"deleted": route_uid}``.

        Raises
        ------
        HTTPException
            404 if the route does not exist in the database.
        """
        if not STATE.repository.delete_route(route_uid):
            raise HTTPException(404, "Route not found")
        # Also drop it from the export cache so it cannot be exported any more.
        # 同时从内存缓存移除，之后不能再导出。
        STATE.plans.pop(route_uid, None)
        return {"deleted": route_uid}

    # ------------------------------------------------------------------ #
    # Export                                                             #
    # ------------------------------------------------------------------ #
    @app.get("/api/export/{route_id}.{fmt}")
    async def export_route(route_id: str, fmt: str):
        """Download a planned route as GeoJSON, CSV or PDF. 导出航线。

        ``GET /api/export/{route_id}.{fmt}`` - all formats are sent with a
        ``Content-Disposition: attachment`` header so the browser saves them.

        Parameters
        ----------
        route_id : str
            Id of a plan still held in ``STATE.plans`` (the last 64 plans of
            this server process). Plans only present in the database cannot be
            exported, because the exporters need the full ``RoutePlan``.
        fmt : str
            ``geojson``, ``csv`` or ``pdf`` (case-insensitive).

        Returns
        -------
        Response or FileResponse
            ``application/geo+json``, ``text/csv`` or ``application/pdf``.
            The PDF is first written to ``OUTPUT_DIR/route_<id>.pdf`` and then
            streamed from disk.

        Raises
        ------
        HTTPException
            404 if the plan is not in memory, 400 for an unsupported format.
        """
        plan = STATE.plans.get(route_id)
        if plan is None:
            raise HTTPException(404, "Route not in memory; re-plan it before exporting")
        fmt = fmt.lower()
        if fmt == "geojson":
            payload = json.dumps(route_to_geojson(plan), indent=2, ensure_ascii=False)
            return Response(payload, media_type="application/geo+json", headers={
                "Content-Disposition": f'attachment; filename="route_{route_id}.geojson"'})
        if fmt == "csv":
            return Response(route_to_csv_string(plan), media_type="text/csv", headers={
                "Content-Disposition": f'attachment; filename="route_{route_id}.csv"'})
        if fmt == "pdf":
            path = OUTPUT_DIR / f"route_{route_id}.pdf"
            # The model summary is printed in the PDF report when available.
            # PDF 报告中附带模型摘要（若模型已加载）。
            summary = STATE.classifier.summary() if STATE.classifier else None
            await asyncio.to_thread(route_to_pdf, plan, path, None, summary)
            return FileResponse(path, media_type="application/pdf",
                                filename=f"route_{route_id}.pdf")
        raise HTTPException(400, "Supported formats: geojson, csv, pdf")

    @app.get("/api/export/points/{token}.{fmt}")
    async def export_points(token: str, fmt: str):
        """Download the classified points of an earlier upload.

        ``GET /api/export/points/{token}.{fmt}``

        Parameters
        ----------
        token : str
            ``token`` returned by ``POST /api/upload`` (``"upload-<id>"``).
        fmt : str
            ``csv`` or ``geojson`` (case-insensitive).

        Returns
        -------
        Response
            ``text/csv`` or ``application/geo+json`` attachment with every
            stored point (not the thinned map sample) and its predicted class.

        Raises
        ------
        HTTPException
            404 if the token is unknown to this process, 503 if the model is
            missing, 400 for an unsupported format.

        Notes
        -----
        The points are classified again rather than read back from SQLite;
        the model is deterministic, so the output matches the upload result.
        重新分类而非读取数据库，模型是确定性的，结果一致。
        """
        frame = STATE.uploads.get(token)
        if frame is None:
            raise HTTPException(404, "Upload not found")
        STATE.ensure_model()
        results = await asyncio.to_thread(STATE.classifier.classify_frame, frame)
        if fmt.lower() == "csv":
            return Response(points_to_csv_string(results), media_type="text/csv", headers={
                "Content-Disposition": f'attachment; filename="{token}_points.csv"'})
        if fmt.lower() == "geojson":
            payload = json.dumps(classified_points_to_geojson(results), indent=2)
            return Response(payload, media_type="application/geo+json", headers={
                "Content-Disposition": f'attachment; filename="{token}_points.geojson"'})
        raise HTTPException(400, "Supported formats: geojson, csv")

    # ------------------------------------------------------------------ #
    # WebSocket - real-time planning                                     #
    # ------------------------------------------------------------------ #
    @app.websocket("/ws/plan")
    async def ws_plan(websocket: WebSocket) -> None:
        """Bidirectional channel streaming live planning progress.

        实时规划通道：客户端发送请求，服务端持续推送进度与结果。

        ``WS /ws/plan`` - all messages are JSON text frames. One connection
        can serve any number of requests; requests are processed one after
        another (a new message is read only after the previous job finished).

        Protocol
        --------
        Server -> client, immediately after the connection is accepted::

            {"type": "hello", "version": str,
             "zones": [{"code": str, "colour": "#rrggbb"}, ...]}

        Client -> server messages are objects with an ``action`` key
        (``"plan"`` when the key is missing):

        ``{"action": "ping"}``
            Answered with ``{"type": "pong", "t": <server UNIX time>}``.
            Used by the client as a keep-alive.
        ``{"action": "classify", "points": [[lat, lon], ...]}``
            Answered with ``{"type": "classified", "results": [...]}`` (items
            as in ``POST /api/classify``) or an ``error`` frame.
        ``{"action": "plan", <PlanRequest fields>, "include_overlay": bool}``
            Starts a planning job; see :func:`_ws_plan` for the frames
            ``started`` -> ``progress``* -> ``overlay``? -> ``result``.
        any other ``action``
            Answered with ``{"type": "error", "detail": "unknown action '...'"}``;
            the connection stays open.

        Errors inside one request are reported as ``{"type": "error",
        "detail": str}`` and the loop continues. The loop ends when the client
        disconnects (``WebSocketDisconnect``, logged at INFO level). Any other
        unexpected exception is logged, reported as an ``error`` frame if the
        socket is still writable, and ends the handler.

        Parameters
        ----------
        websocket : WebSocket
            The accepted connection supplied by FastAPI.
        """
        await websocket.accept()
        # The running loop is captured here because the planner's progress
        # callback runs in a worker thread and must post frames back to it.
        # 记录事件循环，供工作线程中的进度回调跨线程投递消息。
        loop = asyncio.get_running_loop()
        try:
            await websocket.send_json({"type": "hello", "version": APP_VERSION,
                                       "zones": [{"code": c, "colour": CLASS_COLOR_DARK[c]}
                                                 for c in CLASS_NAMES]})
            while True:
                message = await websocket.receive_json()
                action = message.get("action", "plan")

                if action == "ping":
                    await websocket.send_json({"type": "pong", "t": time.time()})
                    continue

                if action == "classify":
                    await _ws_classify(websocket, message)
                    continue

                if action != "plan":
                    await websocket.send_json({"type": "error",
                                               "detail": f"unknown action '{action}'"})
                    continue

                await _ws_plan(websocket, loop, message)

        except WebSocketDisconnect:
            LOGGER.info("WebSocket client disconnected")
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.exception("WebSocket failure")
            # The socket may already be closed; a second failure is ignored.
            # 连接可能已关闭，此时发送失败直接忽略。
            try:
                await websocket.send_json({"type": "error", "detail": str(exc)})
            except Exception:
                pass

    return app


# --------------------------------------------------------------------------- #
# WebSocket helpers                                                           #
# --------------------------------------------------------------------------- #
async def _ws_classify(websocket: WebSocket, message: Dict) -> None:
    """Classify a batch of coordinates pushed over the socket.

    Implements the ``"classify"`` WebSocket action.

    Parameters
    ----------
    websocket : WebSocket
        Open connection used for the reply.
    message : dict
        Client message; ``message["points"]`` must be a non-empty list of
        ``[lat, lon]`` pairs.

    Notes
    -----
    Sends exactly one frame: ``{"type": "classified", "results": [...]}`` on
    success, or ``{"type": "error", "detail": str}`` when the model is missing,
    the points have the wrong shape, lie outside the valid coordinate range, or
    there are more than 50 000 of them (the same limits as ``POST /api/classify``).
    """
    try:
        STATE.ensure_model()
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "detail": str(exc.detail)})
        return
    points = np.asarray(message.get("points", []), dtype=float)
    # Accept only an (N, 2) array with N > 0. 仅接受形状为 (N, 2) 的数组。
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        await websocket.send_json({"type": "error", "detail": "points must be [[lat, lon], ...]"})
        return
    # Same limits as the REST endpoint: at most 50 000 points, valid lat/lon range.
    # 与 REST 接口相同的限制：最多 50000 个点，经纬度在有效范围内。
    if len(points) > 50000 or not (np.all(np.abs(points[:, 0]) <= 90)
                                   and np.all(np.abs(points[:, 1]) <= 180)):
        await websocket.send_json({"type": "error",
                                   "detail": "at most 50000 points with |lat| <= 90 and |lon| <= 180"})
        return
    results = await asyncio.to_thread(
        STATE.classifier.classify_coordinates, points[:, 0], points[:, 1])
    await websocket.send_json({"type": "classified",
                               "results": [r.to_dict() for r in results]})


async def _ws_plan(websocket: WebSocket, loop: asyncio.AbstractEventLoop, message: Dict) -> None:
    """Run a planning job in a worker thread, streaming progress frames.

    Implements the ``"plan"`` WebSocket action.

    Parameters
    ----------
    websocket : WebSocket
        Open connection used for all frames of this job.
    loop : asyncio.AbstractEventLoop
        The event loop that owns ``websocket``; the progress callback, which
        runs in a worker thread, uses it to hand frames over safely.
    message : dict
        Client message. Keys that are fields of :class:`PlanRequest` are
        validated by it; unknown keys (``action``, ``include_overlay``) are
        filtered out first. ``include_overlay`` (default ``True``) selects
        whether the classified lattice is sent back.

    Notes
    -----
    Frames sent to the client, in order:

    1. ``{"type": "started", "algorithm": str, "resolution_deg": float}``
    2. zero or more ``{"type": "progress", "stage": str, "progress": float,
       "detail": dict}`` frames. ``stage`` is one of ``"grid"``,
       ``"classify"`` (``detail`` has ``cells_done`` / ``cells_total``),
       ``"cost_map"`` (``detail`` = cost-map statistics), ``"search"``,
       ``"astar"`` / ``"dijkstra"`` / ``"genetic"`` (search progress),
       ``"refine"`` (the lattice is refined and planning restarts) and
       ``"done"``. ``progress`` is a fraction in [0, 1] for the current stage.
    3. if ``include_overlay``: ``{"type": "overlay", "cells": [{"lat", "lon",
       "c", "p"}, ...], "step": int, "cell_size_deg": float, "grid": {...}}``
       where ``c`` is the class index and ``p`` the mean confidence of a
       down-sampled block (at most about 6 000 cells).
    4. ``{"type": "result", "plan": RoutePlan.to_dict()}``.

    Instead of this sequence a single ``{"type": "error", "detail": str}`` is
    sent when the request is invalid or the model is missing; if the planner
    raises, ``error`` is sent after the progress frames already delivered.

    Threading
    ---------
    ``planner.plan`` is synchronous and CPU-bound, so it runs in a thread via
    :func:`asyncio.to_thread`. Its progress callback executes in that thread,
    where calling ``websocket.send_json`` (a coroutine bound to the event loop)
    is not allowed. The callback therefore only puts the frame into an
    :class:`asyncio.Queue` using ``loop.call_soon_threadsafe`` (asyncio queues
    are not thread-safe themselves), and this coroutine drains the queue and
    sends the frames.
    工作线程不能直接调用 send_json，只能通过 call_soon_threadsafe 把消息放入队列，
    再由协程从队列取出并发送。
    """
    try:
        # Keep only PlanRequest fields; extra keys would otherwise be ignored
        # silently by Pydantic anyway, but filtering keeps the intent explicit.
        # 只保留 PlanRequest 字段，再交给 Pydantic 校验。
        request = PlanRequest(**{k: v for k, v in message.items()
                                 if k in PlanRequest.model_fields})
    except Exception as exc:
        await websocket.send_json({"type": "error", "detail": f"invalid request: {exc}"})
        return

    try:
        planner = STATE.ensure_model()
    except HTTPException as exc:
        await websocket.send_json({"type": "error", "detail": str(exc.detail)})
        return

    queue: asyncio.Queue = asyncio.Queue()
    # Mutable holder (dict) so the nested callback can update the timestamp
    # without a ``nonlocal`` declaration. 用字典保存时间戳，便于闭包内修改。
    last_sent = {"t": 0.0}

    def progress(stage: str, fraction: float, detail: Dict) -> None:
        """Called from the worker thread; throttled to the configured interval.

        Parameters
        ----------
        stage : str
            Planning stage name (see :func:`_ws_plan` notes).
        fraction : float
            Progress of the stage in [0, 1].
        detail : dict
            Stage-specific data; may contain NumPy values, which are converted
            by :func:`_jsonable`.

        Notes
        -----
        At most one frame per ``SERVER.progress_interval_s`` (0.15 s) is
        forwarded, so a search that reports thousands of expansions does not
        flood the socket. ``"done"`` and ``"cost_map"`` frames are always
        forwarded because the client needs them (final state, statistics).
        按时间间隔节流，但 done 与 cost_map 两个阶段总是发送。
        """
        now = time.monotonic()
        if stage not in ("done", "cost_map") and now - last_sent["t"] < SERVER.progress_interval_s:
            return
        last_sent["t"] = now
        # Thread-safe hand-over: put_nowait itself runs later on the loop thread.
        # 线程安全地把消息交给事件循环线程执行 put_nowait。
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"type": "progress", "stage": stage,
             "progress": round(float(fraction), 4), "detail": _jsonable(detail)},
        )

    await websocket.send_json({"type": "started", "algorithm": request.algorithm,
                               "resolution_deg": request.resolution_deg})

    def work():
        """Blocking planning call executed in the worker thread."""
        return planner.plan(
            (request.start_lat, request.start_lon), (request.end_lat, request.end_lon),
            request.algorithm, request.resolution_deg, request.speed_knots,
            request.simplify_km, progress=progress,
        )

    task = asyncio.create_task(asyncio.to_thread(work))
    # Forward progress frames while the job runs. The 0.1 s timeout lets the
    # loop re-check ``task.done()``; the ``queue.empty()`` condition makes sure
    # frames queued just before the job finished are still delivered.
    # 任务运行期间循环转发进度；超时 0.1 秒用于重新检查任务状态，
    # 任务结束后继续发送队列中剩余的消息。
    while not task.done() or not queue.empty():
        try:
            frame = await asyncio.wait_for(queue.get(), timeout=0.1)
            await websocket.send_json(frame)
        except asyncio.TimeoutError:
            continue

    try:
        # Awaiting the finished task returns its value or re-raises its exception.
        # 等待已完成的任务：返回结果或重新抛出线程中的异常。
        plan, cost_map = await task
    except Exception as exc:
        LOGGER.exception("Planning failed")
        await websocket.send_json({"type": "error", "detail": str(exc)})
        return

    STATE.remember(plan)
    if request.persist:
        await asyncio.to_thread(STATE.repository.save_route, plan)

    if message.get("include_overlay", True):
        # Down-sample the lattice to <= ~6000 cells so the browser can draw it.
        # 将分类网格降采样至约 6000 个单元供前端绘制。
        overlay = cost_map.to_overlay(max_cells=6000)
        await websocket.send_json({"type": "overlay", **overlay,
                                   "grid": cost_map.spec.to_dict()})
    await websocket.send_json({"type": "result", "plan": plan.to_dict()})


def _jsonable(value: Any) -> Any:
    """Make NumPy scalars/arrays JSON serialisable.

    Converts recursively: dicts and lists/tuples are walked, NumPy integers
    and floats become Python ``int``/``float``, arrays become nested lists and
    positive infinity becomes ``None`` (JSON has no infinity). Other values
    are returned unchanged.

    Parameters
    ----------
    value : Any
        Progress detail or any nested structure produced by the planner.

    Returns
    -------
    Any
        A structure that ``json.dumps`` accepts.
    """
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and value == float("inf"):
        return None
    return value


def _label_agreement(frame: pd.DataFrame, results: List) -> Optional[Dict]:
    """If the upload carries ground-truth labels, report the agreement rate.

    若上传数据带有真实标签，则统计模型与标签的一致率。

    Parameters
    ----------
    frame : pandas.DataFrame
        The validated upload; labels are read from an optional
        ``class_code`` column.
    results : list of ClassificationResult
        Predictions for the same rows, in the same order.

    Returns
    -------
    dict or None
        ``{"n_labelled": int, "agreement": float}`` where ``agreement`` is
        the share (0..1) of labelled rows whose predicted class equals the
        label. ``None`` if there is no ``class_code`` column or no row holds a
        known class name (unknown labels are ignored).
    """
    if "class_code" not in frame.columns:
        return None
    truth = frame["class_code"].to_numpy()
    predicted = np.array([r.class_code for r in results])
    # Rows with empty or unknown labels are excluded from the comparison.
    # 排除空标签或未知标签的行。
    mask = np.isin(truth, CLASS_NAMES)
    if not mask.any():
        return None
    return {
        "n_labelled": int(mask.sum()),
        "agreement": round(float((truth[mask] == predicted[mask]).mean()), 4),
    }


#: Module-level application instance served by Uvicorn (``maritime_route.web.app:app``).
#: 供 Uvicorn 加载的应用实例。
app = create_app()
