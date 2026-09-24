"""SQLite persistence layer.

数据持久化层：负责会话、点位、航线与航路点的存取。
SQLite was selected because the application is a single-user desktop/web tool:
it needs zero administration, ships as one file, and still provides full
transactional SQL with foreign keys and views.

Role in the pipeline
--------------------
The web application stores every classification upload (``session`` +
``point`` rows) and every planned route (``route`` + ``waypoint`` rows) through
:class:`RouteRepository`, and reads them back for the history list, the route
detail view, re-export and the statistics page. The table definitions live in
``schema.sql`` next to this module; see the comments there for every column.

Design
------
* Repository pattern: SQL is confined to this class; callers exchange plain
  dictionaries and the planner's dataclasses.
* One short-lived connection per operation (:meth:`RouteRepository.connect`):
  SQLite connections must not be shared between threads, and the web
  application calls the repository both from the event loop and from worker
  threads (``asyncio.to_thread``).
* A lock owned by the repository instance serialises *writes*, so that
  concurrent requests do not fail with "database is locked"; reads run
  without it.
* WAL journal mode lets readers proceed while a write is in progress.

设计：仓储模式；每次操作使用独立连接；写操作加锁串行化；WAL 模式允许读写并发。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..config import (
    CLASS_COLOR,
    CLASS_DESCRIPTION,
    CLASS_NAMES,
    DATABASE_PATH,
    NAVIGABLE_CLASSES,
    SUPPORTED_ALGORITHMS,
    ZONE_COST,
)

LOGGER = logging.getLogger(__name__)
#: DDL script executed at start-up. 建表脚本路径。
SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class RouteRepository:
    """Thin repository over the SQLite database. 仓储模式封装。

    Parameters
    ----------
    path : Path or str, default ``config.DATABASE_PATH``
        Database file; its parent directory is created if missing, and the
        schema is created or upgraded immediately.

    Attributes
    ----------
    path : Path
        Location of the database file.
    """

    def __init__(self, path: Path | str = DATABASE_PATH) -> None:
        """Open (or create) the database file and bring the schema up to date.

        Parameters
        ----------
        path : Path or str
            Database file location.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Serialises write transactions within this process. 写操作互斥锁。
        self._lock = threading.Lock()
        self.initialise()

    # ------------------------------------------------------------------ #
    @contextmanager
    def connect(self):
        """Yield a connection with foreign keys and row factory configured.

        The connection is committed when the ``with`` block finishes normally,
        rolled back if it raises, and always closed. Every block is therefore
        one transaction.
        上下文管理器：正常结束则提交，异常则回滚，最后关闭连接。

        Yields
        ------
        sqlite3.Connection
            Connection whose rows are :class:`sqlite3.Row` (accessible by
            column name and convertible with ``dict(row)``).

        Raises
        ------
        sqlite3.Error
            Any database error is re-raised after the rollback.
        """
        # timeout=30 s: wait for a lock held by another process instead of
        # failing immediately.
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        # Foreign keys are off by default in SQLite and the setting is per
        # connection, so it is switched on for every new connection.
        connection.execute("PRAGMA foreign_keys = ON")
        # Write-ahead log: readers are not blocked by a writer. The mode is
        # stored in the file, so repeating the pragma is cheap.
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialise(self) -> None:
        """Create the schema and seed the zone reference table.

        Runs ``schema.sql`` (idempotent), upgrades an old ``route`` table if
        necessary, and upserts the four zone rows from ``config`` so that the
        stored weights always match the ones used by the planner.
        建表、必要时迁移旧表，并按配置写入/更新四个区域的参照数据。
        """
        with self._lock, self.connect() as conn:
            conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            self._migrate_algorithm_check(conn)
            for name in CLASS_NAMES:
                cost = ZONE_COST[name]
                # UPSERT: insert the zone, or refresh cost and navigability if
                # it already exists (title, description and colour are kept).
                conn.execute(
                    "INSERT INTO zone(code, title, description, traversal_cost, navigable, colour) "
                    "VALUES (?,?,?,?,?,?) ON CONFLICT(code) DO UPDATE SET "
                    "traversal_cost=excluded.traversal_cost, navigable=excluded.navigable",
                    (
                        name,
                        name.replace("_", " ").title(),     # OPEN_SEA -> "Open Sea"
                        CLASS_DESCRIPTION[name],
                        # SQL REAL has no infinity: -1 marks an impassable zone.
                        float(cost) if cost != float("inf") else -1.0,
                        1 if name in NAVIGABLE_CLASSES else 0,
                        CLASS_COLOR[name],
                    ),
                )

    @staticmethod
    def _migrate_algorithm_check(conn) -> None:
        """Upgrade a database created before the genetic algorithm existed.

        Older databases constrain ``route.algorithm`` to three values, so a
        genetic-algorithm route could not be saved. SQLite cannot alter a CHECK
        constraint in place; the table is therefore rebuilt with the current
        definition from ``schema.sql`` and the stored rows are copied across.
        升级旧数据库：旧表的 CHECK 约束不含 genetic，SQLite 无法直接修改约束，
        因此按 schema.sql 的新定义重建 route 表并复制原有数据。

        Procedure (the standard SQLite "12-step" table rebuild, reduced):

        1. Detect: read the stored ``CREATE TABLE route`` text from
           ``sqlite_master``; nothing to do if it already names every method
           of ``SUPPORTED_ALGORITHMS`` or the table does not exist.
        2. Cut the current ``route`` definition out of ``schema.sql`` and
           rename it to ``route_new``.
        3. Switch foreign keys off, so that dropping ``route`` does not cascade
           into ``waypoint``, and drop the views that reference ``route``.
        4. Create ``route_new``, copy all rows, drop ``route``, rename.
           ``SELECT *`` relies on the column order being unchanged between the
           old and the new definition (only the CHECK differs). The row ids
           are preserved, so ``waypoint.route_id`` still points at the right
           routes.
        5. Recreate the indexes and the views (by re-running the idempotent
           schema) and switch foreign keys back on.

        Parameters
        ----------
        conn : sqlite3.Connection
            Open connection inside the initialisation transaction.
        """
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='route'").fetchone()
        # Up to date when every method the planner offers appears, quoted, in the
        # stored CHECK constraint. 存储的 CHECK 约束已包含全部算法名时无需升级。
        if row is None or all(f"'{name}'" in row[0] for name in SUPPORTED_ALGORITHMS):
            return
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        # Text of the route table definition: from its CREATE line to the
        # closing ");". 从 schema.sql 中截取 route 表的定义。
        start = schema.index("CREATE TABLE IF NOT EXISTS route (")
        definition = schema[start:schema.index(");", start) + 2]
        definition = definition.replace("CREATE TABLE IF NOT EXISTS route (", "CREATE TABLE route_new (")
        # Prevent ON DELETE CASCADE into waypoint when the old table is dropped.
        conn.execute("PRAGMA foreign_keys = OFF")
        # Views that read the table are dropped first and recreated from the
        # schema afterwards. 先删除依赖该表的视图，重建后再按 schema 恢复。
        views = [v for (v,) in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")]
        for view in views:
            conn.execute(f'DROP VIEW IF EXISTS "{view}"')
        conn.execute(definition)
        conn.execute("INSERT INTO route_new SELECT * FROM route")
        conn.execute("DROP TABLE route")
        conn.execute("ALTER TABLE route_new RENAME TO route")
        # Indexes were dropped together with the old table.
        conn.execute("CREATE INDEX IF NOT EXISTS ix_route_algorithm ON route(algorithm)")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_route_created   ON route(created_utc)")
        conn.executescript(schema)                 # recreates the dropped views
        conn.execute("PRAGMA foreign_keys = ON")
        LOGGER.info("Database upgraded: route.algorithm now accepts %s", SUPPORTED_ALGORITHMS)

    # -- sessions -------------------------------------------------------- #
    def create_session(self, name: str, source: str, n_points: int = 0,
                       notes: str = "") -> int:
        """Insert a new ingestion session. 新建会话。

        Parameters
        ----------
        name : str
            Display name.
        source : str
            Origin of the data (file name or ``"interactive"``).
        n_points : int, default 0
            Initial point counter; normally 0 and increased by
            :meth:`save_points`.
        notes : str, default ""
            Free text.

        Returns
        -------
        int
            The new ``session.id``.
        """
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO session(name, source, n_points, notes) VALUES (?,?,?,?)",
                (name, source, int(n_points), notes),
            )
            return int(cur.lastrowid)

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """Most recent sessions first. 按时间倒序列出会话。

        Parameters
        ----------
        limit : int, default 50
            Maximum number of rows.

        Returns
        -------
        list of dict
            One dictionary per ``session`` row (all columns).
        """
        with self.connect() as conn:
            # id DESC = newest first (ids grow monotonically with AUTOINCREMENT).
            rows = conn.execute(
                "SELECT * FROM session ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # -- points ---------------------------------------------------------- #
    def save_points(self, session_id: int, results: Sequence) -> int:
        """Bulk-insert classification results. 批量写入分类结果。

        All rows and the session counter update are written in one
        transaction with ``executemany``, which is much faster than one
        insert per point.

        Parameters
        ----------
        session_id : int
            Owning session (must exist - enforced by the foreign key).
        results : sequence
            :class:`~maritime_route.model.inference.ClassificationResult`
            objects with ``latitude``, ``longitude``, ``depth_m``,
            ``class_code`` and ``confidence``.

        Returns
        -------
        int
            Number of points written.

        Raises
        ------
        sqlite3.IntegrityError
            If ``session_id`` does not exist or a value violates a CHECK.
        """
        # Column order: session_id, latitude, longitude, elevation_m (not
        # available -> NULL), depth_m, class_code, confidence, observed_utc (NULL).
        payload = [
            (session_id, r.latitude, r.longitude, None, r.depth_m,
             r.class_code, r.confidence, None)
            for r in results
        ]
        with self._lock, self.connect() as conn:
            conn.executemany(
                "INSERT INTO point(session_id, latitude, longitude, elevation_m, depth_m, "
                "class_code, confidence, observed_utc) VALUES (?,?,?,?,?,?,?,?)",
                payload,
            )
            # Keep the denormalised counter in step with the inserted rows.
            conn.execute("UPDATE session SET n_points = n_points + ? WHERE id = ?",
                         (len(payload), session_id))
        return len(payload)

    def class_histogram(self, session_id: Optional[int] = None) -> Dict[str, int]:
        """Number of stored points per zone class. 各类别点数统计。

        Parameters
        ----------
        session_id : int, optional
            Restrict to one session; ``None`` counts all points.

        Returns
        -------
        dict
            ``{class_code: count}``; classes without points are absent.
        """
        query = "SELECT class_code, COUNT(*) AS n FROM point"
        params: tuple = ()
        if session_id is not None:
            # Parameter placeholder, never string formatting (SQL injection safe).
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " GROUP BY class_code"
        with self.connect() as conn:
            return {row["class_code"]: row["n"] for row in conn.execute(query, params)}

    # -- routes ---------------------------------------------------------- #
    def save_route(self, plan, session_id: Optional[int] = None) -> int:
        """Persist a :class:`RoutePlan` together with its waypoints.

        The route row and all waypoint rows are written in one transaction,
        so a route is never stored without its waypoints. Non-finite numbers
        (cost, ratio and time of an infeasible plan) are stored as NULL.
        The nested ``baseline`` and ``zone_profile`` dictionaries are stored
        as JSON text.
        在同一事务中写入航线及其全部航路点。

        Parameters
        ----------
        plan : RoutePlan
            Result of :meth:`~maritime_route.routing.planner.RoutePlanner.plan`.
        session_id : int, optional
            Session the route belongs to.

        Returns
        -------
        int
            Internal ``route.id`` of the new row.

        Raises
        ------
        sqlite3.IntegrityError
            If ``plan.route_id`` already exists (UNIQUE) or the algorithm is
            not allowed by the CHECK constraint.
        """
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                """INSERT INTO route(route_uid, session_id, algorithm, start_lat, start_lon,
                                     end_lat, end_lon, total_cost, distance_km, great_circle_km,
                                     detour_ratio, estimated_hours, speed_knots, nodes_expanded,
                                     runtime_s, grid_resolution, grid_cells, feasible,
                                     baseline_json, zone_profile_json, created_utc)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    plan.route_id, session_id, plan.algorithm,
                    plan.start[0], plan.start[1], plan.end[0], plan.end[1],
                    _finite(plan.total_cost), plan.distance_km, plan.great_circle_km,
                    _finite(plan.detour_ratio), _finite(plan.estimated_hours),
                    plan.speed_knots, plan.nodes_expanded, plan.runtime_s,
                    plan.grid.get("resolution_deg"), plan.grid.get("n_cells"),
                    1 if plan.feasible else 0,
                    json.dumps(plan.baseline), json.dumps(plan.zone_profile),
                    plan.created_utc,
                ),
            )
            route_id = int(cur.lastrowid)
            # seq = position of the leg in the route (0 = departure).
            conn.executemany(
                "INSERT INTO waypoint(route_id, seq, latitude, longitude, class_code, "
                "zone_cost, cumulative_km, bearing_deg) VALUES (?,?,?,?,?,?,?,?)",
                [
                    (route_id, i, leg.latitude, leg.longitude, leg.class_code,
                     leg.zone_cost, leg.cumulative_km, leg.bearing_deg)
                    for i, leg in enumerate(plan.legs)
                ],
            )
            return route_id

    def list_routes(self, limit: int = 50) -> List[Dict]:
        """Newest routes first, from the ``v_route_summary`` view.

        ``route_uid`` is the tie-breaker for routes created in the same second
        (timestamps have one-second resolution), which keeps the order stable.

        Parameters
        ----------
        limit : int, default 50
            Maximum number of rows.

        Returns
        -------
        list of dict
            Summary columns of each route plus ``n_waypoints``.
        """
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM v_route_summary ORDER BY created_utc DESC, route_uid DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_route(self, route_uid: str) -> Optional[Dict]:
        """Load one route with its decoded JSON fields and its waypoints.

        Parameters
        ----------
        route_uid : str
            Public route identifier.

        Returns
        -------
        dict or None
            All ``route`` columns, with ``baseline_json`` and
            ``zone_profile_json`` replaced by the parsed ``baseline`` and
            ``zone_profile`` dictionaries, plus ``waypoints`` - a list of
            dictionaries ordered by ``seq``. ``None`` if the route is unknown.
        """
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM route WHERE route_uid = ?", (route_uid,)).fetchone()
            if row is None:
                return None
            route = dict(row)
            # Replace the JSON text columns by parsed objects ("{}" for NULL).
            route["baseline"] = json.loads(route.pop("baseline_json") or "{}")
            route["zone_profile"] = json.loads(route.pop("zone_profile_json") or "{}")
            route["waypoints"] = [
                dict(w) for w in conn.execute(
                    "SELECT seq, latitude, longitude, class_code, zone_cost, cumulative_km, "
                    "bearing_deg FROM waypoint WHERE route_id = ? ORDER BY seq", (route["id"],)
                )
            ]
            return route

    def delete_route(self, route_uid: str) -> bool:
        """Delete a route; its waypoints go with it (ON DELETE CASCADE).

        Parameters
        ----------
        route_uid : str
            Public route identifier.

        Returns
        -------
        bool
            True if a route was deleted, False if it did not exist.
        """
        with self._lock, self.connect() as conn:
            cur = conn.execute("DELETE FROM route WHERE route_uid = ?", (route_uid,))
            return cur.rowcount > 0

    # -- model runs ------------------------------------------------------ #
    def record_model_run(self, metadata: Dict) -> int:
        """Log a classifier training run. 记录一次模型训练。

        Parameters
        ----------
        metadata : dict
            Contents of the model metadata JSON written by the trainer:
            ``version``, ``created_utc`` and ``metrics`` with ``accuracy``,
            ``macro_f1``, ``cohen_kappa`` and ``training`` (``parameters``,
            ``epochs_run``). Missing keys are stored as NULL.

        Returns
        -------
        int
            The new ``model_run.id``.
        """
        metrics = metadata.get("metrics", {})
        with self._lock, self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO model_run(version, accuracy, macro_f1, cohen_kappa, parameters, "
                "epochs, trained_utc, metadata_json) VALUES (?,?,?,?,?,?,?,?)",
                (
                    metadata.get("version", "1.0.0"),
                    metrics.get("accuracy"), metrics.get("macro_f1"),
                    metrics.get("cohen_kappa"),
                    (metrics.get("training") or {}).get("parameters"),
                    (metrics.get("training") or {}).get("epochs_run"),
                    metadata.get("created_utc"),
                    # The per-epoch history is large and already in the model
                    # metadata file, so it is left out of the database copy.
                    json.dumps({k: v for k, v in metadata.items() if k != "history"}),
                ),
            )
            return int(cur.lastrowid)

    # -- statistics ------------------------------------------------------ #
    def statistics(self) -> Dict:
        """Aggregate figures for the statistics page. 数据库统计信息。

        Returns
        -------
        dict
            Row counts of the four main tables, the class histogram of all
            points, per-algorithm route count and mean runtime (s), distance
            (km) and nodes expanded, the database path and its size in KiB.
        """
        with self.connect() as conn:
            def scalar(sql: str, default=0):
                """First column of the first row, or ``default`` for NULL.

                Parameters
                ----------
                sql : str
                    Query returning one value.
                default
                    Value returned when the query yields NULL.

                Returns
                -------
                object
                    The scalar result.
                """
                value = conn.execute(sql).fetchone()[0]
                return default if value is None else value

            # AVG() over an empty group is NULL, hence "or 0.0" before rounding.
            algorithms = {
                row["algorithm"]: {
                    "n": row["n"],
                    "avg_runtime_s": round(row["avg_runtime"] or 0.0, 4),
                    "avg_distance_km": round(row["avg_distance"] or 0.0, 2),
                    "avg_nodes": round(row["avg_nodes"] or 0.0, 1),
                }
                for row in conn.execute(
                    "SELECT algorithm, COUNT(*) n, AVG(runtime_s) avg_runtime, "
                    "AVG(distance_km) avg_distance, AVG(nodes_expanded) avg_nodes "
                    "FROM route GROUP BY algorithm"
                )
            }
            return {
                "sessions": scalar("SELECT COUNT(*) FROM session"),
                "points": scalar("SELECT COUNT(*) FROM point"),
                "routes": scalar("SELECT COUNT(*) FROM route"),
                "waypoints": scalar("SELECT COUNT(*) FROM waypoint"),
                "class_histogram": self.class_histogram(),
                "by_algorithm": algorithms,
                "database_file": str(self.path),
                "database_size_kb": round(self.path.stat().st_size / 1024, 1)
                if self.path.exists() else 0.0,
            }


def _finite(value) -> Optional[float]:
    """Convert inf/nan to NULL so that SQL aggregates stay meaningful.

    AVG() over a column containing a huge sentinel or NaN would be
    meaningless; NULL values are simply skipped by SQL aggregates.
    将无穷大/非数值转换为 NULL。

    Parameters
    ----------
    value
        Any value convertible with ``float()``.

    Returns
    -------
    float or None
        The finite float, or ``None`` for inf, nan and non-numeric input.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    # value == value is False only for NaN (IEEE 754).
    return value if value == value and abs(value) != float("inf") else None
