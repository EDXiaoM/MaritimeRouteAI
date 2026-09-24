"""Persistence package.

持久化层：SQLite 仓储。

Stores classification sessions, classified points, planned routes with their
waypoints, and classifier training runs in a single SQLite file
(``config.DATABASE_PATH``).

Contents
--------
repository.RouteRepository
    All database access of the application (create/list/get/delete routes,
    save points, statistics, schema creation and migration).
schema.sql
    The relational schema (tables ``session``, ``point``, ``zone``,
    ``route``, ``waypoint``, ``model_run`` and the view ``v_route_summary``),
    executed by the repository at start-up.
"""
from .repository import RouteRepository

# Public name exported by ``from maritime_route.storage import *``.
__all__ = ["RouteRepository"]
