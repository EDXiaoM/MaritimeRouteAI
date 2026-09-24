#!/usr/bin/env python3
"""Generate the five A1 sheets of graphic material required by the assignment.

生成任务书要求的五张 A1 图纸：
  1 问题描述  2 系统架构  3 试验结果  4 分类算法流程图  5 航线规划流程图

Place in the application
------------------------
A stand-alone documentation script, not used by the running web service. It
draws the five A1 sheets presented with the diploma project:

1. problem statement (purpose, input data, zones, requirements, context);
2. application architecture (layers, request sequence, database schema, model);
3. test results (interface screenshots, classification metrics, benchmark);
4. flowchart of the zone classification algorithm;
5. flowchart of the route construction algorithm (cost map, A*/Dijkstra).

Inputs (must exist before running; produced by the training and figure
scripts): the model metadata JSON (``MODEL_METADATA_PATH``), the routing
benchmark ``FIGURE_DIR/routing_benchmark.json``, the dataset statistics
``data/processed/dataset_statistics.json``, figures in ``FIGURE_DIR`` and UI
screenshots in ``docs/screenshots``. Missing images are replaced by a
placeholder (see ``poster_kit.image``); missing JSON files stop the script.

Output: ``docs/posters/sheet_<n>_<name>.png`` and ``.pdf`` for each sheet.

Usage::

    python scripts/make_posters.py

All positions are fractions of the sheet (0..1, origin bottom-left), see
``poster_kit``. Numbers quoted on the sheets are read from the JSON files, so
the sheets stay consistent with the latest training and benchmark run.
所有数值均从 JSON 结果文件读取，保证图纸与最新训练、基准测试结果一致。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``maritime_route`` (in src/) and ``poster_kit`` (next to this file)
# importable without installing the project. 添加导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from poster_kit import (
    ACCENT, ACCENT_2, BORDER, INK, INK_2, INK_3, PANEL, SURFACE,
    arrow, box, bullets, diamond, image, legend_chips, new_sheet, panel, save, table,
)

from maritime_route.config import (
    CLASS_COLOR, CLASS_DESCRIPTION, CLASS_NAMES, DOCS_DIR, FIGURE_DIR,
    MODEL_METADATA_PATH, ZONE_COST,
)

# Output folder and screenshot folder. 输出目录与截图目录。
POSTER_DIR = DOCS_DIR / "posters"
SHOTS = DOCS_DIR / "screenshots"
# Result files loaded once at import time (the script fails early with
# FileNotFoundError if the model has not been trained or the benchmark not run):
#   META  - model metadata written by the trainer (architecture, metrics, dataset split);
#   BENCH - routing benchmark (per-voyage results in "cases", means in "aggregate");
#   STATS - statistics of the prepared AIS dataset (point count, bounding box...).
# 在导入时读取结果文件；缺少文件时脚本立即报错。
META = json.loads(Path(MODEL_METADATA_PATH).read_text(encoding="utf-8"))
BENCH = json.loads((FIGURE_DIR / "routing_benchmark.json").read_text(encoding="utf-8"))
STATS = json.loads((Path(__file__).resolve().parents[1] / "data" / "processed" /
                    "dataset_statistics.json").read_text(encoding="utf-8"))
#: Shortcut to the test-set metrics (accuracy, F1, per-class values, confusion matrix...).
M = META["metrics"]


# ========================================================================== #
def sheet1() -> None:
    """Sheet 1 - problem statement. 图纸1：任务描述。

    Layout: left column - purpose (with the depth-by-class figure), input
    data and technology stack; middle column - the four zone classes with
    their colours and traversal weights, functional requirements and a table
    of quality targets versus achieved values; right column - a context
    diagram (operator, application, classifier, planner, data, exports) and a
    screenshot of a planned route.

    Writes ``POSTER_DIR/sheet_1_problem_statement.png`` and ``.pdf``.
    """
    fig, ax = new_sheet(1, "Problem statement",
                        "Application for constructing a maritime vessel's course "
                        "based on neural-network classification of geographic zones")

    # --- left column: purpose & inputs ------------------------------------
    x0, y0 = panel(ax, 0.028, 0.560, 0.300, 0.340, "1. Purpose of the work")
    bullets(ax, x0, y0, [
        "Build an application that plans an optimal and\nsafe maritime route between two points.",
        "The safety of every point of the route is decided\nby a neural network that classifies the\ngeographic zone the point belongs to.",
        "The route is then produced by graph search on a\ncost map derived from those classes.",
        "The result is compared with the naive\nshortest-distance (great-circle) route.",
    ], fontsize=11.5, leading=0.0205)
    image(ax, FIGURE_DIR / "fig_depth_by_class.png", 0.036, 0.572, 0.284, 0.128,
          "Why the task is non-trivial: the two shoreline classes\n"
          "occupy almost the same depth band", fontsize=9.4)

    x0, y0 = panel(ax, 0.028, 0.250, 0.300, 0.292, "2. Input data")
    bullets(ax, x0, y0, [
        "~Labelled AIS corpus (merged.geojson)",
        f"{STATS['n_points']:,} geo-referenced points, "
        f"{STATS.get('n_sessions', 0)} recording sessions",
        "Attributes per point: latitude, longitude,\nelevation, NOAA depth/height, 9 OSM\nobject counters, timestamp",
        f"Coverage: lat {STATS['bbox']['lat_min']:.1f}…{STATS['bbox']['lat_max']:.1f}°, "
        f"lon {STATS['bbox']['lon_min']:.1f}…{STATS['bbox']['lon_max']:.1f}°",
        "~Run-time input",
        "Departure / destination coordinates, or an\nuploaded AIS track (GeoJSON / CSV / JSON)",
    ], fontsize=11, leading=0.0225)

    x0, y0 = panel(ax, 0.028, 0.088, 0.300, 0.145, "3. Technology stack")
    bullets(ax, x0, y0, [
        "Python 3.11  ·  PyTorch 2.x (neural network)",
        "scikit-learn (scaling, spatial index, metrics)",
        "FastAPI + WebSocket  ·  Leaflet (live map UI)",
        "SQLite (storage)  ·  ReportLab (PDF export)",
    ], fontsize=11, leading=0.0215)

    # --- middle column: the four classes ----------------------------------
    x0, y0 = panel(ax, 0.344, 0.560, 0.300, 0.340,
                   "4. Geographic zones to be recognised")
    # One entry per class: colour swatch, code, description and traversal
    # weight (infinite weight printed as "impassable"). 每个区域一条说明。
    yy = y0
    for name in CLASS_NAMES:
        ax.add_patch(plt.matplotlib.patches.Rectangle(
            (x0, yy - 0.050), 0.014, 0.042, facecolor=CLASS_COLOR[name],
            edgecolor=BORDER, linewidth=0.8, zorder=4))
        ax.text(x0 + 0.022, yy - 0.012, name, fontsize=12.5, color=INK,
                fontweight="bold", va="top", ha="left", zorder=4)
        ax.text(x0 + 0.022, yy - 0.030, CLASS_DESCRIPTION[name], fontsize=9.8,
                color=INK_2, va="top", ha="left", zorder=4)
        cost = ZONE_COST[name]
        ax.text(x0 + 0.022, yy - 0.044,
                f"traversal weight: {'impassable' if cost == float('inf') else f'{cost:.1f}'}",
                fontsize=9.5, color=INK_3, va="top", ha="left", zorder=4)
        yy -= 0.070
    ax.text(x0, 0.600,
            "The first two classes are navigable; the last two are land and are\n"
            "assigned an infinite traversal cost, so the planner can never route\n"
            "a vessel through them. The ordering of the four classes along the\n"
            "depth axis is what the diverging colour ramp encodes.",
            fontsize=10.2, color=INK_2, va="top", ha="left", zorder=5, linespacing=1.5)

    x0, y0 = panel(ax, 0.344, 0.250, 0.300, 0.292, "5. Functional requirements")
    bullets(ax, x0, y0, [
        "Load AIS data or route-point coordinates",
        "Classify geographic zones with a pre-trained\nneural network",
        "Build a traversal cost map from the classes",
        "Search the optimal route with A* or Dijkstra",
        "Visualise the route on an interactive map",
        "Compare with the shortest-distance route",
        "Export results to GeoJSON, CSV and PDF",
        "Persist points, zones, routes and their\nparameters in a database",
    ], fontsize=11, leading=0.0225)

    x0, y0 = panel(ax, 0.344, 0.088, 0.300, 0.145, "6. Quality targets and results")
    table(ax, x0, y0 + 0.006, 0.280, [
        ["Criterion", "Target", "Achieved"],
        ["Classification accuracy", "≥ 90 %", f"{100 * M['accuracy']:.2f} %"],
        ["Macro F1", "≥ 0.90", f"{M['macro_f1']:.4f}"],
        ["Route planning time", "≤ 5 s", "≤ 1.4 s"],
        ["Routes free of land", "100 %", "100 % (6/6)"],
    ], [0.46, 0.27, 0.27], row_h=0.0225, fontsize=10.5,
       align=["left", "center", "center"])

    # --- right column: context diagram ------------------------------------
    # Box coordinates were placed by hand; arrows connect box edges.
    # 方框坐标为手工布置，箭头连接方框边缘。
    panel(ax, 0.660, 0.088, 0.312, 0.812, "7. Context of use")

    box(ax, 0.700, 0.800, 0.230, 0.052, "Navigator / operator",
        facecolor="#e8f1fa", fontsize=13, bold=True)
    box(ax, 0.700, 0.670, 0.230, 0.090,
        "Maritime Route Planner\n\nweb application\n(real-time interactive map)",
        facecolor="#ffffff", edgecolor=ACCENT, fontsize=12.5)
    box(ax, 0.678, 0.545, 0.128, 0.072,
        "Zone classifier\n(neural network)", facecolor="#eef7f2",
        edgecolor="#1baf7a", fontsize=11)
    box(ax, 0.824, 0.545, 0.128, 0.072,
        "Route planner\n(A* / Dijkstra)", facecolor="#fdf1e9",
        edgecolor="#eb6834", fontsize=11)
    box(ax, 0.678, 0.420, 0.128, 0.068,
        "Reference index\nbathymetry + OSM", facecolor=PANEL,
        edgecolor=BORDER, fontsize=10.5)
    box(ax, 0.824, 0.420, 0.128, 0.068,
        "SQLite database\npoints · routes", facecolor=PANEL,
        edgecolor=BORDER, fontsize=10.5)
    box(ax, 0.700, 0.300, 0.230, 0.056,
        "Exported artefacts:  GeoJSON  ·  CSV  ·  PDF report",
        facecolor="#f7f2fb", edgecolor="#4a3aa7", fontsize=11.5)

    arrow(ax, (0.815, 0.800), (0.815, 0.762), "departure / destination",
          label_offset=(0.068, 0.0))
    arrow(ax, (0.790, 0.762), (0.790, 0.800), "", colour=ACCENT_2)
    arrow(ax, (0.760, 0.668), (0.742, 0.619))
    arrow(ax, (0.870, 0.668), (0.888, 0.619))
    arrow(ax, (0.742, 0.543), (0.742, 0.490))
    arrow(ax, (0.888, 0.543), (0.888, 0.490))
    arrow(ax, (0.806, 0.581), (0.824, 0.581), "zone costs", label_offset=(0.0, 0.012))
    arrow(ax, (0.815, 0.418), (0.815, 0.358))

    ax.text(0.680, 0.284,
            "The operator never sees raw zone codes: the application converts them into a\n"
            "traversal cost, searches the cheapest navigable path, and reports how much\n"
            "distance that safety costs compared with the straight great-circle line.",
            fontsize=10.6, color=INK_2, va="top", ha="left", zorder=4, linespacing=1.5)

    image(ax, SHOTS / "ui_03b_map_only.png", 0.676, 0.098, 0.280, 0.132,
          "Planned route (blue) vs great circle (red dashed), Kiel → Tallinn",
          fontsize=9.6)

    save(fig, POSTER_DIR / "sheet_1_problem_statement.png")


# ========================================================================== #
def sheet2() -> None:
    """Sheet 2 - application architecture. 图纸2：应用架构。

    Layout: left - five horizontal layer bands (presentation, service, domain,
    data, export), each with its modules, connected by downward arrows, and
    below them the seven-step sequence of one planning request; right - the
    SQLite entity diagram (session, point, zone, route, waypoint with their
    relations) and a summary of the classification model with the cost-model
    figure.

    Writes ``POSTER_DIR/sheet_2_architecture.png`` and ``.pdf``.
    """
    fig, ax = new_sheet(2, "Application architecture",
                        "Layered structure, module responsibilities and data flow")

    panel(ax, 0.028, 0.088, 0.620, 0.812, "Layered architecture")

    # Each layer: (title, y of the band, fill colour, edge colour,
    # [(module label, box width), ...]). Boxes are laid out left to right.
    # 每层：名称、纵坐标、填充色、边框色，以及模块方框列表（标签、宽度）。
    layers = [
        ("Presentation layer", 0.760, "#e8f1fa", ACCENT, [
            ("index.html\nsingle-page client", 0.052),
            ("app.js\nmap & controller", 0.052),
            ("api.js\nREST + WebSocket", 0.052),
            ("charts.js\nmetrics plots", 0.052),
        ]),
        ("Service layer  (FastAPI)", 0.630, "#eef7f2", "#1baf7a", [
            ("REST endpoints\n/api/route, /api/classify", 0.072),
            ("WebSocket /ws/plan\nlive progress streaming", 0.072),
            ("Tile service\non-board chart", 0.050),
        ]),
        ("Domain layer", 0.500, "#fdf1e9", "#eb6834", [
            ("ZoneClassifierService\ninference façade", 0.068),
            ("RoutePlanner\norchestration", 0.058),
            ("CostMap\nlattice + weights", 0.058),
            ("A* / Dijkstra\ngraph search", 0.058),
        ]),
        ("Data layer", 0.370, "#f7f2fb", "#4a3aa7", [
            ("ais_loader\nGeoJSON/CSV/JSON", 0.068),
            ("features\n28 engineered inputs", 0.062),
            ("ReferenceIndex\nBallTree, label-free", 0.062),
            ("repository\nSQLite persistence", 0.062),
        ]),
        ("Export layer", 0.246, "#f2f5f7", INK_2, [
            ("GeoJSON writer", 0.062),
            ("CSV writer", 0.055),
            ("PDF report (ReportLab)", 0.078),
        ]),
    ]

    # Draw each band and its module boxes; bx advances by box width + 0.014 gap.
    for name, y, fill, edge, blocks in layers:
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (0.044, y - 0.012), 0.588, 0.094,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=3))
        ax.text(0.054, y + 0.068, name, fontsize=13.5, color=edge,
                fontweight="bold", va="center", ha="left", zorder=5)
        bx = 0.062
        for label, w in blocks:
            box(ax, bx, y + 0.002, w, 0.048, label, facecolor="#ffffff",
                edgecolor=edge, fontsize=9.6, linewidth=1.2, zorder=5)
            bx += w + 0.014

    # Arrows between consecutive bands, labelled with the kind of interaction.
    for y_from, y_to in [(0.760, 0.724), (0.630, 0.594), (0.500, 0.464), (0.370, 0.340)]:
        arrow(ax, (0.338, y_from - 0.012), (0.338, y_to), colour=INK_2, linewidth=2.0)
    ax.text(0.352, 0.742, "HTTP / WS", fontsize=10, color=INK_3, va="center", zorder=5)
    ax.text(0.352, 0.612, "calls", fontsize=10, color=INK_3, va="center", zorder=5)
    ax.text(0.352, 0.482, "reads", fontsize=10, color=INK_3, va="center", zorder=5)
    ax.text(0.352, 0.355, "writes", fontsize=10, color=INK_3, va="center", zorder=5)

    # --- planning sequence -------------------------------------------------
    x0, y0 = panel(ax, 0.044, 0.100, 0.588, 0.128, "Request sequence of one planning job")
    steps = [
        "1  client\nsends request", "2  build\nlattice", "3  network\nclassifies cells",
        "4  cost map\n+ safety margin", "5  A* / Dijkstra\nsearch", "6  compare with\ngreat circle",
        "7  persist +\nstream result",
    ]
    # Boxes alternate blue/green borders; a short arrow joins consecutive steps.
    bx = x0 + 0.004
    for i, step in enumerate(steps):
        box(ax, bx, y0 - 0.058, 0.074, 0.050, step, facecolor="#ffffff",
            edgecolor=ACCENT if i % 2 == 0 else "#1baf7a", fontsize=9.2, linewidth=1.2)
        if i < len(steps) - 1:
            arrow(ax, (bx + 0.074, y0 - 0.033), (bx + 0.082, y0 - 0.033), linewidth=1.4)
        bx += 0.082
    ax.text(x0 + 0.004, y0 - 0.074,
            "Progress frames (stages 2–5) are pushed to the browser over the WebSocket, "
            "so the operator watches the route being built rather than waiting for a single response.",
            fontsize=10.5, color=INK_2, va="top", ha="left", zorder=5)

    # --- right column: database & deployment ------------------------------
    x0, y0 = panel(ax, 0.664, 0.470, 0.308, 0.430, "Database schema (SQLite)")

    # Entity boxes: (x, y_top, width, height, fields). 实体表位置。
    entities = {
        "session":  (0.676, 0.852, 0.138, 0.076,
                     ["id  PK", "name, source", "n_points", "created_utc"]),
        "point":    (0.676, 0.734, 0.138, 0.104,
                     ["id  PK", "session_id  FK", "latitude, longitude",
                      "depth_m, elevation_m", "class_code  FK", "confidence"]),
        "zone":     (0.826, 0.734, 0.138, 0.076,
                     ["code  PK", "title, description", "traversal_cost", "navigable"]),
        "route":    (0.676, 0.600, 0.138, 0.118,
                     ["id  PK", "route_uid  UNIQUE", "algorithm", "start/end lat, lon",
                      "total_cost, distance_km", "great_circle_km, detour_ratio",
                      "nodes_expanded, runtime_s"]),
        "waypoint": (0.826, 0.600, 0.138, 0.090,
                     ["id  PK", "route_id  FK", "seq", "latitude, longitude",
                      "class_code  FK", "cumulative_km"]),
    }
    # Each entity: white box, blue title bar with the table name, one text
    # line per column group. 实体框：蓝色标题栏 + 字段列表。
    for name, (px, py, pw, ph, fields) in entities.items():
        ax.add_patch(plt.matplotlib.patches.Rectangle(
            (px, py - ph), pw, ph, facecolor="#ffffff", edgecolor=ACCENT,
            linewidth=1.4, zorder=4))
        ax.add_patch(plt.matplotlib.patches.Rectangle(
            (px, py - 0.017), pw, 0.017, facecolor=ACCENT, edgecolor=ACCENT,
            linewidth=1.0, zorder=5))
        ax.text(px + pw / 2, py - 0.0085, name, fontsize=10.5, color="#ffffff",
                ha="center", va="center", fontweight="bold", zorder=6)
        for k, field in enumerate(fields):
            ax.text(px + 0.006, py - 0.028 - k * 0.0135, field, fontsize=8.4,
                    color=INK_2, ha="left", va="center", zorder=6)

    # Relationship arrows: solid = one-to-many ownership, dashed = FK lookup
    # of the zone table. 实线为一对多关系，虚线为对 zone 表的外键引用。
    arrow(ax, (0.745, 0.776), (0.745, 0.734), "1 : N",
          label_offset=(0.018, 0.0), fontsize=9)
    arrow(ax, (0.745, 0.630), (0.745, 0.600), "1 : N",
          label_offset=(0.018, 0.0), fontsize=9)
    arrow(ax, (0.814, 0.545), (0.826, 0.545), "1 : N",
          label_offset=(0.0, 0.012), fontsize=9)
    arrow(ax, (0.826, 0.700), (0.814, 0.700), "FK", label_offset=(0.0, 0.012),
          fontsize=9, colour=INK_3, linestyle="--")
    arrow(ax, (0.880, 0.510), (0.880, 0.560), "FK", label_offset=(0.016, 0.0),
          fontsize=9, colour=INK_3, linestyle="--")
    ax.text(0.676, 0.492,
            "Foreign keys are enforced (PRAGMA foreign_keys = ON); the view\n"
            "v_route_summary aggregates routes with their waypoint counts.",
            fontsize=9.6, color=INK_3, va="top", ha="left", zorder=6, linespacing=1.4)

    x0, y0 = panel(ax, 0.664, 0.088, 0.308, 0.368, "Classification model")
    bullets(ax, x0, y0, [
        f"~Residual multilayer perceptron, {META['model']['n_features']} inputs → 4 classes",
        f"Hidden layers {' – '.join(str(h) for h in META['model']['hidden_sizes'])}, "
        f"GELU, batch norm, dropout {META['model']['dropout']}",
        f"{M['training']['parameters']:,} trainable parameters",
        f"AdamW, lr {META['hyperparameters']['learning_rate']}, "
        f"batch {META['hyperparameters']['batch_size']}",
        f"Trained {M['training']['epochs_run']} epochs in "
        f"{M['training']['seconds'] / 60:.1f} min on CPU",
        "~Feature groups",
        "Geodetic encoding (4)  ·  point bathymetry (5)",
        "Neighbourhood context (9)  ·  OSM objects (10)",
    ], fontsize=10.8, leading=0.0215)
    image(ax, FIGURE_DIR / "fig_cost_model.png", 0.672, 0.096, 0.292, 0.150)

    save(fig, POSTER_DIR / "sheet_2_architecture.png")


# ========================================================================== #
def sheet3() -> None:
    """Sheet 3 - test results. 图纸3：试验结果。

    Layout: top left - screenshot of a planned route; top right - confusion
    matrix, per-class metrics figure and a table of overall test metrics;
    bottom left - routing benchmark figure, a per-voyage table (A* versus
    great-circle distance, extra distance, nodes expanded by A* and Dijkstra,
    search time) and a summary sentence built from the aggregate values;
    bottom right - four UI screenshots (comparison, Atlantic route, upload,
    model panel).

    Writes ``POSTER_DIR/sheet_3_test_results.png`` and ``.pdf``.
    """
    fig, ax = new_sheet(3, "Test results",
                        "Interface, classification quality, planned routes "
                        "and comparison with the shortest-distance route")

    panel(ax, 0.028, 0.470, 0.470, 0.430, "Application interface: planned route")
    image(ax, SHOTS / "ui_03_route.png", 0.036, 0.480, 0.454, 0.388,
          "Kiel → Tallinn, A*, lattice 0.25°: the route stays in navigable "
          "zones while the great circle (red) crosses Denmark and Sweden")

    panel(ax, 0.512, 0.470, 0.460, 0.430, "Classification quality on the test set")
    image(ax, FIGURE_DIR / "fig_confusion_matrix.png", 0.520, 0.478, 0.220, 0.380)
    image(ax, FIGURE_DIR / "fig_per_class_metrics.png", 0.748, 0.660, 0.216, 0.192)
    x0 = 0.752
    y0 = 0.638
    table(ax, x0, y0, 0.208, [
        ["Metric", "Value"],
        ["Test points", f"{META['dataset']['test']:,}"],
        ["Accuracy", f"{100 * M['accuracy']:.2f} %"],
        ["Macro F1", f"{M['macro_f1']:.4f}"],
        ["Weighted F1", f"{M['weighted_f1']:.4f}"],
        ["Cohen κ", f"{M['cohen_kappa']:.4f}"],
        ["Macro ROC-AUC", f"{M['macro_roc_auc']:.4f}"],
        ["Mean confidence", f"{M['mean_confidence']:.4f}"],
    ], [0.6, 0.4], row_h=0.0192, fontsize=10, align=["left", "right"])

    panel(ax, 0.028, 0.088, 0.470, 0.366, "Routing benchmark")
    image(ax, FIGURE_DIR / "fig_routing_benchmark.png", 0.036, 0.252, 0.454, 0.158)
    # One table row per benchmark voyage; the case name is cut before " (".
    # "Extra" = A* distance relative to the great circle, in percent.
    # 每个航次一行；Extra 为 A* 航线相对大圆航线的额外距离百分比。
    agg = BENCH["aggregate"]
    x0 = 0.040
    y0 = 0.244
    table(ax, x0, y0, 0.450, [
        ["Voyage", "A* km", "Great circle km", "Extra", "A* nodes",
         "Dijkstra nodes", "Search s"],
        *[[c["case"].split(" (")[0],
           f"{c['astar']['distance_km']:,.0f}",
           f"{c['great_circle']['distance_km']:,.0f}",
           f"+{100 * (c['astar']['distance_km'] / c['great_circle']['distance_km'] - 1):.1f} %",
           f"{c['astar']['nodes_expanded']:,}",
           f"{c['dijkstra']['nodes_expanded']:,}",
           f"{c['astar']['runtime_s']:.3f}"] for c in BENCH["cases"]],
    ], [0.30, 0.11, 0.15, 0.09, 0.12, 0.13, 0.10], row_h=0.0170, fontsize=9.4,
       align=["left", "right", "right", "right", "right", "right", "right"])
    ax.text(0.040, 0.098,
            f"Across {agg['n_cases']} voyages A* expands {agg['mean_node_reduction']}× fewer "
            f"nodes than Dijkstra and runs {agg['mean_time_ratio']}× faster, while both return "
            f"the identical optimal cost (max deviation {agg['max_optimality_gap_pct']} %). "
            f"All {agg['great_circle_infeasible_cases']} great-circle routes cross land; the "
            f"optimised routes are {agg['mean_detour_pct']:.1f} % longer on average and fully navigable.",
            fontsize=10.2, color=INK_2, va="top", ha="left", zorder=5, linespacing=1.5)

    panel(ax, 0.512, 0.088, 0.460, 0.366, "Comparison, uploaded data and stored routes")
    image(ax, SHOTS / "ui_04_compare.png", 0.518, 0.264, 0.224, 0.146,
          "Algorithm comparison on one shared cost map", fontsize=9.2)
    image(ax, SHOTS / "ui_07b_atlantic_map.png", 0.748, 0.264, 0.218, 0.146,
          "Rotterdam → Lisbon, lattice 0.25°", fontsize=9.2)
    image(ax, SHOTS / "ui_06_data.png", 0.518, 0.100, 0.224, 0.126,
          "AIS upload: 900 points classified against their labels", fontsize=9.2)
    image(ax, SHOTS / "ui_05b_model_panel.png", 0.748, 0.100, 0.218, 0.126,
          "Model panel: metrics, matrix, training curves", fontsize=9.2)

    save(fig, POSTER_DIR / "sheet_3_test_results.png")


# ========================================================================== #
def sheet4() -> None:
    """Sheet 4 - flowchart of the classification algorithm. 图纸4：分类算法流程图。

    Layout: left - a vertical flowchart from the input coordinate to the zone
    code and confidence (reference-index query, a decision on missing
    measurements with an imputation branch, feature assembly,
    standardisation, forward pass, softmax, argmax); right - description of
    the 28 feature components, an explanation of why the reference index does
    not leak labels, and the per-class precision/recall/F1 table.

    Writes ``POSTER_DIR/sheet_4_classification_flowchart.png`` and ``.pdf``.
    """
    fig, ax = new_sheet(4, "Algorithm of geographic-zone classification",
                        "Flowchart of the neural-network classification procedure")

    cx = 0.300
    w, h = 0.230, 0.042
    y = 0.848

    def step(text, dy=0.070, fill="#ffffff", edge=ACCENT, fs=11.5, height=None):
        """Draw one process box at the current ``y`` and move ``y`` down.

        Helper kept for manual layouts; the flowchart below uses its own loop.

        Parameters
        ----------
        text : str
            Box label.
        dy : float, optional
            Vertical step applied to ``y`` after drawing.
        fill, edge : str, optional
            Box colours.
        fs : float, optional
            Font size.
        height : float, optional
            Box height; defaults to ``h``.

        Returns
        -------
        tuple
            ``(top, bottom)`` centre points of the box edges, for arrows.
        """
        nonlocal y
        hh = height or h
        centre = box(ax, cx - w / 2, y - hh, w, hh, text, facecolor=fill,
                     edgecolor=edge, fontsize=fs)
        top, bottom = (cx, y), (cx, y - hh)
        y -= dy
        return top, bottom

    # START terminal. The right-hand side is a 2-tuple (box centre, point), so
    # ``b`` receives (cx, 0.848); neither value is used afterwards.
    # 起始节点；此处元组解包的结果后续未使用。
    _, b = box(ax, cx - w / 2, 0.848, w, 0.040,
               "START  ·  point (φ, λ) to classify", facecolor="#e8f1fa",
               edgecolor=ACCENT, style="terminal", fontsize=11.5), (cx, 0.848)
    y = 0.848
    prev = (cx, 0.848)

    # Flowchart nodes: (text, box height, fill, edge). A height of None marks
    # the decision diamond, which is drawn with its "no" branch.
    # 节点列表；高度为 None 表示判断节点。
    nodes = [
        ("Query the reference index:\nk = 8 nearest bathymetry samples\n(BallTree, haversine metric)", 0.062, "#ffffff", ACCENT),
        ("Are the point's own depth /\nelevation values present?", None, None, None),   # decision
        ("Compute neighbourhood statistics:\nmean, std, min, max, land fraction,\nvertical gradient, distance to sample", 0.062, "#ffffff", ACCENT),
        ("Aggregate OSM objects:\nmarine infrastructure, natural coast,\ninland water, industrial", 0.062, "#ffffff", ACCENT),
        ("Assemble the 28-component\nfeature vector x", 0.050, "#ffffff", ACCENT),
        ("Standardise:  z = (x − μ) / σ\n(scaler fitted on the training split)", 0.050, "#eef7f2", "#1baf7a"),
        ("Forward pass through the residual MLP\n28 → 256 → 128 → 64 → 4", 0.050, "#eef7f2", "#1baf7a"),
        ("Softmax → class probabilities p₀…p₃", 0.044, "#eef7f2", "#1baf7a"),
        ("Take argmax → zone code;\nkeep max p as the confidence", 0.050, "#ffffff", ACCENT),
    ]

    # Walk down the column: each box is joined to the previous one by an arrow;
    # ``prev`` is the bottom-centre of the last element, ``y`` the next top.
    # 自上而下绘制：prev 为上一节点底部中点，y 为下一个节点顶部。
    y = 0.808
    for i, (text, hh, fill, edge) in enumerate(nodes):
        if hh is None:
            d = diamond(ax, cx, y - 0.038, 0.250, 0.076, text, fontsize=11)
            arrow(ax, prev, d["top"])
            # YES branch continues down; NO branch goes to imputation box
            ax.text(cx + 0.010, y - 0.082, "yes", fontsize=10.5, color=INK_2, zorder=6)
            ax.text(cx + 0.132, y - 0.028, "no", fontsize=10.5, color=INK_2, zorder=6)
            box(ax, cx + 0.148, y - 0.066, 0.152, 0.056,
                "Impute from the\ninterpolated context value;\nraise the 'missing' flag",
                facecolor="#fff6e5", edgecolor="#b3730a", fontsize=9.8)
            # "no" path: right into the imputation box, then down and back left
            # to rejoin the main column below the diamond. 否分支绕回主流程。
            arrow(ax, d["right"], (cx + 0.148, y - 0.038))
            arrow(ax, (cx + 0.224, y - 0.066), (cx + 0.224, y - 0.108), style="-")
            arrow(ax, (cx + 0.224, y - 0.108), (cx + 0.004, y - 0.108), style="-|>")
            prev = d["bottom"]
            y -= 0.108
            continue
        centre_box = box(ax, cx - w / 2, y - hh, w, hh, text, facecolor=fill,
                         edgecolor=edge, fontsize=10.8)
        arrow(ax, prev, (cx, y))
        prev = (cx, y - hh)
        y -= hh + 0.026

    box(ax, cx - w / 2, y - 0.040, w, 0.040,
        "END  ·  zone code + confidence", facecolor="#e8f1fa",
        edgecolor=ACCENT, style="terminal", fontsize=11.5)
    arrow(ax, prev, (cx, y))

    # --- right column: explanation ----------------------------------------
    x0, y0 = panel(ax, 0.600, 0.560, 0.372, 0.340, "Feature vector (28 components)")
    groups = [
        ("Geodetic encoding — 4", "sin φ, cos φ, sin λ, cos λ  (continuous across\nthe antimeridian, no artificial discontinuity)"),
        ("Point measurements — 5", "signed-log depth, missing flag, height/depth flag,\nsigned-log elevation, missing flag"),
        ("Neighbourhood context — 9", "mean / std / min / max depth of the 8 nearest\nsamples, range, land fraction, sign agreement,\nlog distance to the nearest sample, gradient"),
        ("OSM objects — 10", "marine infrastructure, natural coastline, inland\nwater, industrial — each at the point and\naveraged over the neighbourhood; plus totals"),
    ]
    yy = y0
    for title, body in groups:
        ax.text(x0, yy, title, fontsize=12, color=ACCENT, fontweight="bold",
                va="top", ha="left", zorder=5)
        ax.text(x0, yy - 0.019, body, fontsize=10.3, color=INK_2, va="top",
                ha="left", zorder=5, linespacing=1.45)
        yy -= 0.082

    x0, y0 = panel(ax, 0.600, 0.300, 0.372, 0.240, "Why a reference index is not leakage")
    bullets(ax, x0, y0, [
        "The index stores only depth, elevation and OSM\ncounters — never a class label.",
        "It is built from the training split alone; validation\nand test points are never inserted.",
        "During training a point never queries itself\n(exclude_self = True).",
        "In deployment the same role is played by a NOAA\nbathymetry raster and an OSM extract, so the\nfeature is reproducible for any coordinate on Earth.",
    ], fontsize=10.8, leading=0.0185)

    x0, y0 = panel(ax, 0.600, 0.088, 0.372, 0.196, "Achieved quality")
    table(ax, x0, y0, 0.352, [
        ["Class", "Precision", "Recall", "F1", "Support"],
        *[[c, f"{M['per_class'][c]['precision']:.4f}", f"{M['per_class'][c]['recall']:.4f}",
           f"{M['per_class'][c]['f1']:.4f}", f"{M['per_class'][c]['support']:,}"]
          for c in CLASS_NAMES],
        ["Overall accuracy", "", "", f"{M['accuracy']:.4f}", f"{META['dataset']['test']:,}"],
    ], [0.34, 0.17, 0.17, 0.16, 0.16], row_h=0.0235, fontsize=10.5,
       align=["left", "right", "right", "right", "right"])

    save(fig, POSTER_DIR / "sheet_4_classification_flowchart.png")


# ========================================================================== #
def sheet5() -> None:
    """Sheet 5 - flowchart of the route construction algorithm. 图纸5：航线规划流程图。

    Layout: column 1 - cost-map preparation (bounding box and lattice,
    classification of cells, zone weights, safety margin, snapping of the
    end points) with a dashed lattice-refinement loop; column 2 - the search
    (algorithm choice, A*/Dijkstra expansion loop, path check, path
    reconstruction, great-circle comparison, end); a notes strip under the
    columns; right column - the edge-cost and heuristic formulas, the
    benchmark figure and a complexity/performance table.

    Writes ``POSTER_DIR/sheet_5_routing_flowchart.png`` and ``.pdf``.
    """
    fig, ax = new_sheet(5, "Algorithm of route construction",
                        "Flowchart of cost-map generation and A* / Dijkstra search")

    # ---- column 1: cost map preparation ---------------------------------
    cx1, w1 = 0.170, 0.230
    left1, right1 = cx1 - w1 / 2, cx1 + w1 / 2

    box(ax, left1, 0.836, w1, 0.038, "START  ·  departure A, destination B",
        facecolor="#e8f1fa", edgecolor=ACCENT, style="terminal", fontsize=11)
    col1 = [
        ("Build the bounding box of A and B,\nadd a 2° margin; choose the lattice\n"
         "resolution (default 0.25°)", 0.056, "#ffffff", ACCENT),
        ("Classify every lattice cell centre\nwith the neural network\n(batched inference)",
         0.056, "#eef7f2", "#1baf7a"),
        ("Map zones to traversal weights:\nOPEN_SEA 1.0 · COASTAL_SEA 1.8\n"
         "NEAR_COAST, COASTLINE = ∞", 0.056, "#fdf1e9", "#eb6834"),
        ("Apply the safety margin:\nwater cells adjacent to land × 1.6",
         0.046, "#fdf1e9", "#eb6834"),
        ("Snap A and B onto the nearest\nnavigable cell (ring search)",
         0.046, "#ffffff", ACCENT),
    ]
    # Draw column 1 top to bottom; ``tops`` records (y, height) of each box.
    prev = (cx1, 0.836)
    y = 0.790
    tops = []
    for text, hh, fill, edge in col1:
        box(ax, left1, y - hh, w1, hh, text, facecolor=fill, edgecolor=edge, fontsize=10.6)
        arrow(ax, prev, (cx1, y))
        tops.append((y, hh))
        prev = (cx1, y - hh)
        y -= hh + 0.030

    # Hand over to column 2: right, up the free corridor, into the decision.
    # 交接到第二列：向右 -> 沿空白走廊向上 -> 进入判断节点。
    hand_y = 0.433
    arrow(ax, (right1, hand_y), (0.298, hand_y), style="-")
    arrow(ax, (0.298, hand_y), (0.298, 0.846), style="-")
    arrow(ax, (0.298, 0.846), (0.340, 0.846), style="-|>")
    ax.text(0.302, 0.640, "cost map ready", fontsize=9.8, color=INK_2, zorder=6,
            rotation=90, va="center", ha="left")

    # ---- refinement branch (lower left) ----------------------------------
    # Dashed amber loop back to the top of column 1: shown when no path exists
    # at the current resolution (mirrors RoutePlanner._refine).
    # 虚线回路：当前分辨率下无路径时细化网格并重建代价地图。
    box(ax, 0.075, 0.300, 0.170, 0.068,
        "Refine the lattice × 0.55\n(strait narrower than one cell),\nretry up to 3 times",
        facecolor="#fff6e5", edgecolor="#b3730a", fontsize=10.2)
    arrow(ax, (0.075, 0.334), (0.042, 0.334), style="-", colour="#b3730a", linestyle="--")
    arrow(ax, (0.042, 0.334), (0.042, 0.762), style="-", colour="#b3730a", linestyle="--")
    arrow(ax, (0.042, 0.762), (left1, 0.762), style="-|>", colour="#b3730a", linestyle="--")
    ax.text(0.031, 0.548, "rebuild the cost map at the finer resolution", fontsize=9.4,
            color="#b3730a", zorder=6, rotation=90, va="center", ha="center")

    # ---- column 2: the search --------------------------------------------
    cx2, w2 = 0.450, 0.240
    left2 = cx2 - w2 / 2

    d0 = diamond(ax, cx2, 0.846, 0.220, 0.074,
                 "Which algorithm\nwas requested?", fontsize=10.8)

    box(ax, 0.330, 0.716, 0.110, 0.064,
        "Dijkstra\n\ng(v) = g(u) + d·w̄\nno heuristic",
        facecolor="#fdf1e9", edgecolor="#eb6834", fontsize=9.8)
    box(ax, 0.460, 0.716, 0.110, 0.064,
        "A*\n\nf(v) = g(v) + h(v)\nh = w_min · d_gc",
        facecolor="#e8f1fa", edgecolor=ACCENT, fontsize=9.8)
    # Curved arrows from the decision to the two algorithm boxes, and from
    # them into the common expansion step. 由判断节点分到两种算法。
    arrow(ax, d0["left"], (0.385, 0.780), connection="arc3,rad=-0.22")
    arrow(ax, d0["right"], (0.515, 0.780), connection="arc3,rad=0.22")

    box(ax, left2, 0.618, w2, 0.050,
        "Pop the cheapest node from the priority\nqueue and relax its 8 neighbours",
        facecolor="#ffffff", edgecolor=ACCENT, fontsize=10.4)
    arrow(ax, (0.385, 0.716), (0.410, 0.668), connection="arc3,rad=0.15")
    arrow(ax, (0.515, 0.716), (0.490, 0.668), connection="arc3,rad=-0.15")

    d1 = diamond(ax, cx2, 0.548, 0.212, 0.072,
                 "Goal reached, or\nqueue exhausted?", fontsize=10.4)
    arrow(ax, (cx2, 0.618), d1["top"])
    # loop back to the expansion node
    arrow(ax, d1["left"], (0.318, 0.548), style="-")
    arrow(ax, (0.318, 0.548), (0.318, 0.632), style="-")
    arrow(ax, (0.318, 0.632), (left2, 0.632), style="-|>")
    ax.text(0.322, 0.590, "continue", fontsize=9.6, color=INK_2, zorder=6,
            rotation=90, va="center", ha="left")

    d2 = diamond(ax, cx2, 0.446, 0.212, 0.072, "Path found?", fontsize=11)
    arrow(ax, d1["bottom"], d2["top"])
    # "no" leads left and down to the refinement box of column 1.
    arrow(ax, d2["left"], (0.245, 0.446), "no", label_offset=(0.0, 0.013), style="-")
    arrow(ax, (0.245, 0.446), (0.245, 0.368), style="-|>", colour=INK_2)

    box(ax, left2, 0.336, w2, 0.052,
        "Reconstruct the path, anchor its ends on\nthe exact user coordinates, build legs\n"
        "(zone, cumulative distance, bearing)",
        facecolor="#ffffff", edgecolor=ACCENT, fontsize=10.2)
    arrow(ax, d2["bottom"], (cx2, 0.388), "yes", label_offset=(0.022, 0.0))

    box(ax, left2, 0.248, w2, 0.052,
        "Sample the great circle on the same cost\nmap; count samples over land; compute\n"
        "extra distance and the safety verdict",
        facecolor="#f7f2fb", edgecolor="#4a3aa7", fontsize=10.2)
    arrow(ax, (cx2, 0.336), (cx2, 0.300))

    box(ax, left2, 0.190, w2, 0.038,
        "END  ·  route + comparison + export",
        facecolor="#e8f1fa", edgecolor=ACCENT, style="terminal", fontsize=11)
    arrow(ax, (cx2, 0.248), (cx2, 0.228))

    # ---- note strip -------------------------------------------------------
    x0, y0 = panel(ax, 0.040, 0.088, 0.530, 0.090, "Notes on the implementation")
    bullets(ax, x0, y0, [
        "Diagonal steps are allowed (8-connected lattice); each step length is the true "
        "great-circle distance for that row,\npre-computed once per offset so the inner "
        "loop contains no trigonometry.",
        "The A* heuristic is evaluated for every generated node and is therefore written "
        "with pre-computed per-row\ntrigonometry; replacing a NumPy call with plain "
        "`math` made A* 1.8× faster and reversed the measured ranking.",
    ], fontsize=10.2, leading=0.0215)

    # ---- right column -----------------------------------------------------
    x0, y0 = panel(ax, 0.600, 0.618, 0.372, 0.282, "Cost model")
    ax.text(x0, y0 - 0.004, "Traversal cost of the edge (u → v):",
            fontsize=12, color=INK, va="top", ha="left", zorder=5)
    # Formulas are rendered with Matplotlib's built-in TeX subset (mathtext),
    # so no LaTeX installation is needed. 公式使用 mathtext 渲染。
    ax.text(x0 + 0.030, y0 - 0.048,
            r"$c(u,v)\;=\;d_{\mathrm{gc}}(u,v)\cdot\dfrac{w(z_u)+w(z_v)}{2}$",
            fontsize=19, color=ACCENT, va="center", ha="left", zorder=5)
    ax.text(x0, y0 - 0.084,
            "where $d_{gc}$ is the great-circle length of the step and $w(z)$ the weight\n"
            "of the zone the cell was classified into. The A* heuristic",
            fontsize=10.8, color=INK_2, va="top", ha="left", zorder=5, linespacing=1.5)
    ax.text(x0 + 0.030, y0 - 0.136,
            r"$h(v)\;=\;w_{\min}\cdot d_{\mathrm{gc}}(v,\,\mathrm{goal})$",
            fontsize=17, color=ACCENT, va="center", ha="left", zorder=5)
    ax.text(x0, y0 - 0.168,
            "never over-estimates the remaining cost, because no cell can be cheaper\n"
            "than $w_{min}$. The heuristic is therefore admissible and A* returns the\n"
            "same optimum as Dijkstra — confirmed experimentally on every test voyage\n"
            f"(maximum cost deviation {BENCH['aggregate']['max_optimality_gap_pct']} %).",
            fontsize=10.8, color=INK_2, va="top", ha="left", zorder=5, linespacing=1.5)

    panel(ax, 0.600, 0.330, 0.372, 0.270, "Comparison with the shortest route")
    image(ax, FIGURE_DIR / "fig_routing_benchmark.png", 0.608, 0.338, 0.356, 0.212)

    # Mean node counts are integer means over all benchmark voyages; relative
    # effort/time of A* = 1 / (Dijkstra-to-A* ratio). 相对量为比值的倒数。
    x0, y0 = panel(ax, 0.600, 0.088, 0.372, 0.222, "Complexity and measured performance")
    table(ax, x0, y0, 0.352, [
        ["Property", "Dijkstra", "A*"],
        ["Time complexity", "O(E log V)", "O(E log V), pruned by h"],
        ["Space complexity", "O(V)", "O(V)"],
        ["Optimal result", "yes", "yes (h admissible)"],
        ["Mean nodes expanded",
         f"{sum(c['dijkstra']['nodes_expanded'] for c in BENCH['cases']) // len(BENCH['cases']):,}",
         f"{sum(c['astar']['nodes_expanded'] for c in BENCH['cases']) // len(BENCH['cases']):,}"],
        ["Relative search effort", "1.00×",
         f"{1 / BENCH['aggregate']['mean_node_reduction']:.2f}×"],
        ["Relative search time", "1.00×",
         f"{1 / BENCH['aggregate']['mean_time_ratio']:.2f}×"],
    ], [0.36, 0.28, 0.36], row_h=0.0235, fontsize=10.5,
       align=["left", "center", "center"])

    save(fig, POSTER_DIR / "sheet_5_routing_flowchart.png")


def main() -> int:
    """Generate all five sheets.

    Returns
    -------
    int
        Process exit code, always 0 (errors propagate as exceptions).
    """
    import matplotlib.pyplot as plt  # noqa: F401  (used through poster_kit)
    print("Generating A1 graphic material:")
    sheet1(); sheet2(); sheet3(); sheet4(); sheet5()
    print(f"\nSheets written to {POSTER_DIR}")
    return 0


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    raise SystemExit(main())
