-- ---------------------------------------------------------------------------
-- Maritime Route Planner - relational schema (SQLite)
-- 数据库模式：存储会话、点位、分类结果、航线及航路点。
-- ---------------------------------------------------------------------------
-- This script is executed by RouteRepository.initialise() on every start-up.
-- Every statement uses IF NOT EXISTS, so running it against an existing
-- database is harmless and never drops data (idempotent).
-- 本脚本在每次启动时执行，所有语句均带 IF NOT EXISTS，可重复执行而不丢失数据。
--
-- Entity overview
--   session 1 --- n point        classified AIS / user points of one upload
--   session 1 --- n route        routes planned within a session (optional)
--   route   1 --- n waypoint     ordered vertices of a route
--   zone    1 --- n point / waypoint   reference table of the four zone classes
--   model_run                    independent log of classifier training runs
--
-- Conventions
--   * Coordinates are WGS 84 decimal degrees (latitude, longitude).
--   * Distances in km, speed in knots, durations in hours or seconds as named.
--   * Timestamps are ISO 8601 UTC text "YYYY-MM-DDTHH:MM:SSZ" (SQLite has no
--     native date type, and this format sorts correctly as text).
--   * Booleans are INTEGER 0/1 guarded by CHECK constraints.
--   * Free-form structured data is stored as JSON text in *_json columns.
--
-- NOTE for maintainers: RouteRepository._migrate_algorithm_check() copies the
-- text of the route table below, from its CREATE line up to the first
-- closing bracket followed by a semicolon. Comments inside that table must
-- therefore not contain that two-character sequence, and must not quote the
-- name of any search method in single quotes (the migration looks for the
-- quoted names to detect an up-to-date table).
-- ---------------------------------------------------------------------------

-- Enforce REFERENCES clauses. SQLite ignores foreign keys unless this pragma
-- is on for the connection. The repository also sets it on every connection
-- it opens, because the pragma is per connection, not stored in the file.
-- 开启外键约束（SQLite 默认关闭，且为连接级设置）。
PRAGMA foreign_keys = ON;

-- Ingestion sessions: one row per uploaded AIS file or interactive session.
-- 会话表：每次上传的 AIS 文件或一次交互会话对应一行。
CREATE TABLE IF NOT EXISTS session (
    -- Surrogate key. AUTOINCREMENT guarantees ids are never reused after a delete.
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Descriptive data: display name and origin (file name or "interactive").
    name            TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    -- Denormalised counter of point rows, incremented by save_points() so the
    -- session list does not need a COUNT(*) join. 冗余计数，避免联表统计。
    n_points        INTEGER NOT NULL DEFAULT 0,
    -- Creation time, filled by SQLite when the insert does not supply it.
    created_utc     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now')),
    notes           TEXT
);

-- Individual geographic points with their classification result.
-- 点位表：单个地理点及其神经网络分类结果。
CREATE TABLE IF NOT EXISTS point (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Owning session. ON DELETE CASCADE: deleting a session deletes its points.
    session_id      INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    -- Position in degrees, range-checked so that swapped lat/lon values are rejected
    -- whenever the "latitude" would exceed +/-90.
    latitude        REAL    NOT NULL CHECK (latitude  BETWEEN -90  AND 90),
    longitude       REAL    NOT NULL CHECK (longitude BETWEEN -180 AND 180),
    -- Physical measurements in metres (either may be unknown = NULL).
    -- elevation_m: height above sea level, depth_m: NOAA relief value
    -- (positive above sea level, negative below).
    elevation_m     REAL,
    depth_m         REAL,
    -- Classification result: zone code (foreign key to the zone reference
    -- table) and the softmax probability of that class, 0..1.
    class_code      TEXT    REFERENCES zone(code),
    confidence      REAL    CHECK (confidence IS NULL OR (confidence BETWEEN 0 AND 1)),
    -- Original AIS timestamp of the observation, if the source had one.
    observed_utc    TEXT
);
-- Indexes for the three access paths: by session (listing, cascade delete),
-- by position (spatial range queries) and by class (histogram GROUP BY).
-- 索引：按会话、按坐标、按类别查询。
CREATE INDEX IF NOT EXISTS ix_point_session ON point(session_id);
CREATE INDEX IF NOT EXISTS ix_point_coords  ON point(latitude, longitude);
CREATE INDEX IF NOT EXISTS ix_point_class   ON point(class_code);

-- Reference table of the four geographic zones and their traversal weights.
-- 区域参照表：四个地理区域类别及其通行权重（启动时由程序根据 config 同步）。
-- The rows are (re)written from config.ZONE_COST by RouteRepository.initialise()
-- with an UPSERT, so the table always matches the weights the planner uses.
CREATE TABLE IF NOT EXISTS zone (
    -- Class name, e.g. OPEN_SEA. Natural primary key referenced by point and waypoint.
    code            TEXT PRIMARY KEY,
    -- Presentation: title for the UI and a longer description.
    title           TEXT NOT NULL,
    description     TEXT,
    -- Weight w(z) of the routing cost model. An impassable zone (infinite
    -- weight in the code) is stored as -1 because SQL REAL has no infinity.
    traversal_cost  REAL NOT NULL,
    -- 1 = ships may enter the zone (OPEN_SEA, COASTAL_SEA), 0 = land/shore.
    navigable       INTEGER NOT NULL CHECK (navigable IN (0, 1)),
    -- Map colour of the zone, hex "#rrggbb".
    colour          TEXT
);

-- Planned routes with their aggregated parameters.
-- 航线表：每条规划航线的汇总参数（航路点明细见 waypoint 表）。
CREATE TABLE IF NOT EXISTS route (
    -- Internal integer key (used by waypoint.route_id) and the public
    -- 16-hex-digit identifier from RoutePlan.route_id (used in URLs and file names).
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    route_uid         TEXT    NOT NULL UNIQUE,
    -- Optional owning session. ON DELETE SET NULL: a route outlives its session.
    session_id        INTEGER REFERENCES session(id) ON DELETE SET NULL,
    -- Search method. The CHECK list must include every method the planner
    -- offers - older databases lacked the genetic and the dynamic-programming
    -- methods and are upgraded by the repository migration.
    algorithm         TEXT    NOT NULL CHECK (algorithm IN ('astar','dijkstra','genetic','dynamic','great_circle')),
    -- Requested departure and destination, degrees.
    start_lat         REAL    NOT NULL,
    start_lon         REAL    NOT NULL,
    end_lat           REAL    NOT NULL,
    end_lon           REAL    NOT NULL,
    -- Route quality: search cost (weighted km, NULL if infinite), length and
    -- great-circle distance in km, and their ratio (>= 1, NULL if undefined).
    total_cost        REAL,
    distance_km       REAL    NOT NULL,
    great_circle_km   REAL    NOT NULL,
    detour_ratio      REAL,
    -- Voyage estimate: passage time in hours at the planned speed in knots.
    estimated_hours   REAL,
    speed_knots       REAL,
    -- Search effort and wall-clock planning time in seconds.
    nodes_expanded    INTEGER,
    runtime_s         REAL    NOT NULL,
    -- Lattice used: step in degrees and total number of cells.
    grid_resolution   REAL,
    grid_cells        INTEGER,
    -- 1 = a navigable route was found, 0 = the planner reported failure.
    feasible          INTEGER NOT NULL DEFAULT 1 CHECK (feasible IN (0, 1)),
    -- JSON documents: great-circle comparison and waypoint count per zone.
    baseline_json     TEXT,
    zone_profile_json TEXT,
    created_utc       TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
-- Indexes for per-method statistics (GROUP BY algorithm) and for listing
-- routes newest first (ORDER BY created_utc).
CREATE INDEX IF NOT EXISTS ix_route_algorithm ON route(algorithm);
CREATE INDEX IF NOT EXISTS ix_route_created   ON route(created_utc);

-- Ordered waypoints belonging to a route.
-- 航路点表：属于某条航线的有序航路点。
CREATE TABLE IF NOT EXISTS waypoint (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Owning route. ON DELETE CASCADE: deleting a route removes its waypoints,
    -- which is how RouteRepository.delete_route() cleans up.
    route_id        INTEGER NOT NULL REFERENCES route(id) ON DELETE CASCADE,
    -- Position of the waypoint along the route, 0 = departure.
    seq             INTEGER NOT NULL,
    -- Waypoint position, degrees.
    latitude        REAL    NOT NULL,
    longitude       REAL    NOT NULL,
    -- Zone of the waypoint and its nominal weight (1e6 marks an impassable zone).
    class_code      TEXT    REFERENCES zone(code),
    zone_cost       REAL,
    -- Distance from the departure in km and course to the next waypoint in
    -- degrees from true north.
    cumulative_km   REAL,
    bearing_deg     REAL,
    -- A route cannot have two waypoints with the same sequence number.
    UNIQUE (route_id, seq)
);
CREATE INDEX IF NOT EXISTS ix_waypoint_route ON waypoint(route_id);

-- Model training runs, kept for traceability of the deployed classifier.
-- 模型训练记录：用于追溯当前部署的分类模型。
CREATE TABLE IF NOT EXISTS model_run (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Model version string, e.g. "1.0.0".
    version         TEXT NOT NULL,
    -- Test-set quality metrics (fractions 0..1, Cohen kappa -1..1).
    accuracy        REAL,
    macro_f1        REAL,
    cohen_kappa     REAL,
    -- Size of the network (trainable parameters) and epochs actually trained.
    parameters      INTEGER,
    epochs          INTEGER,
    trained_utc     TEXT,
    -- Full training metadata as JSON (the per-epoch history is omitted).
    metadata_json   TEXT
);

-- Convenience view used by the statistics endpoint.
-- 汇总视图：每条航线一行，并附带航路点数量。
-- One row per route with its waypoint count. LEFT JOIN keeps routes without
-- waypoints (infeasible plans) with n_waypoints = 0, and GROUP BY r.id
-- collapses the joined waypoint rows back to one row per route.
CREATE VIEW IF NOT EXISTS v_route_summary AS
SELECT r.route_uid,
       r.algorithm,
       r.distance_km,
       r.great_circle_km,
       r.detour_ratio,
       r.total_cost,
       r.runtime_s,
       r.nodes_expanded,
       r.feasible,
       r.created_utc,
       COUNT(w.id) AS n_waypoints
FROM route r
LEFT JOIN waypoint w ON w.route_id = r.id
GROUP BY r.id;
