"""Web layer: FastAPI application and static single-page client.

Web 层：FastAPI 服务与单页前端。

Contents of the package
-----------------------
``app.py``
    FastAPI application: REST endpoints (classification, upload, routing,
    benchmark, stored routes, export, offline tiles) and the ``/ws/plan``
    WebSocket that streams planning progress.
``tiles.py``
    Offline map tiles rendered by the zone classifier (used when no internet
    tile server is reachable).
``static/``
    The browser client: ``index.html``, ``css/app.css``, ``js/api.js`` (HTTP and
    WebSocket client), ``js/app.js`` (Leaflet map and UI logic),
    ``js/charts.js`` (canvas charts) and ``vendor/`` (bundled Leaflet so the
    page also works offline).

Exports
-------
``app``
    Ready-made application instance, e.g. ``uvicorn maritime_route.web:app``.
``create_app``
    Factory that builds a fresh application instance.
"""
from .app import app, create_app

__all__ = ["app", "create_app"]
