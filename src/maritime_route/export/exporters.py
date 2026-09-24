"""Export of planning results to GeoJSON, CSV and PDF.

结果导出模块：GeoJSON / CSV / PDF 三种格式，满足任务书的导出要求。

Role in the pipeline
--------------------
The last stage of the application: a
:class:`~maritime_route.routing.planner.RoutePlan` produced by the planner (or
re-loaded from the database) and, optionally, the classified points of an AIS
upload are converted into files. The web application calls these functions
from its download endpoints; :func:`export_all` writes every format at once.

Formats
-------
GeoJSON (RFC 7946)
    A ``FeatureCollection`` with the optimised route (``LineString``), one
    ``Point`` per waypoint and the great-circle reference line. RFC 7946
    requires positions in ``[longitude, latitude]`` order (x, y), the reverse
    of the ``(lat, lon)`` tuples used everywhere else in the code, so every
    coordinate is swapped when written. Styling properties (``stroke``,
    ``marker-color`` ...) follow the informal "simplestyle" convention
    understood by geojson.io and GitHub.
CSV
    One row per waypoint (see :data:`ROUTE_CSV_HEADER`), or one row per
    classified point with the probability of every class.
PDF
    An A4 voyage report rendered with ReportLab (summary table, optional map,
    zone profile, comparison with the great circle, optional model summary
    and a waypoint appendix).

Numeric conventions: coordinates are rounded to 6 decimals (about 0.1 m),
distances are in km (the PDF also gives nautical miles, 1 NM = 1.852 km),
speed in knots, time in hours. Values that cannot be represented (infinite
cost of an impassable zone, NaN of an infeasible plan) are written as JSON
``null``, CSV ``inf`` or "-" in the PDF.
数值约定：坐标保留 6 位小数；距离单位千米；不可表示的数值写为 null / inf / "-"。
"""
from __future__ import annotations

import csv
import io
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..config import (
    APP_AUTHOR,
    IMPASSABLE_COST,
    APP_UNIVERSITY,
    APP_VERSION,
    CLASS_COLOR,
    CLASS_DESCRIPTION,
    CLASS_NAMES,
    OUTPUT_DIR,
    ZONE_COST,
)

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# GeoJSON                                                                     #
# --------------------------------------------------------------------------- #
def route_to_geojson(plan, include_points: bool = True,
                     include_baseline: bool = True) -> Dict:
    """Build a GeoJSON FeatureCollection describing the planned route.

    生成 GeoJSON：包含航线 LineString、各航路点 Point 以及大圆参考航线。

    Feature order: the optimised route (only if the plan has waypoints), the
    waypoints, then the great-circle reference (only if the plan has a
    baseline comparison). Each feature carries ``feature_type`` so a reader
    can tell them apart.

    Parameters
    ----------
    plan : RoutePlan
        Planning result.
    include_points : bool, default True
        Add a ``Point`` feature for every waypoint with its zone, cost,
        cumulative distance (km) and bearing (degrees).
    include_baseline : bool, default True
        Add the great-circle line (128 samples) with its navigability verdict.

    Returns
    -------
    dict
        GeoJSON document, ready for :func:`json.dumps`.
    """
    features: List[Dict] = []

    if plan.waypoints:
        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "optimal_route",
                "route_id": plan.route_id,
                "algorithm": plan.algorithm,
                "distance_km": round(plan.distance_km, 3),
                "great_circle_km": round(plan.great_circle_km, 3),
                "detour_ratio": _clean(plan.detour_ratio),
                "total_cost": _clean(plan.total_cost),
                "estimated_hours": _clean(plan.estimated_hours),
                "speed_knots": plan.speed_knots,
                "nodes_expanded": plan.nodes_expanded,
                "runtime_s": round(plan.runtime_s, 4),
                "created_utc": plan.created_utc,
                "stroke": "#0b6efd", "stroke-width": 4, "stroke-opacity": 0.9,
            },
            "geometry": {
                "type": "LineString",
                # RFC 7946 order is [lon, lat]; the plan stores (lat, lon).
                # GeoJSON 坐标顺序为 [经度, 纬度]，与内部 (纬度, 经度) 相反。
                "coordinates": [[round(lon, 6), round(lat, 6)] for lat, lon in plan.waypoints],
            },
        })

    if include_points:
        for i, leg in enumerate(plan.legs):
            features.append({
                "type": "Feature",
                "properties": {
                    "feature_type": "waypoint",
                    "seq": i,
                    "class_code": leg.class_code,
                    "class_name": CLASS_DESCRIPTION.get(leg.class_code, ""),
                    # JSON has no representation for infinity - use null.
                    # JSON 无法表示无穷大，此处写为 null。
                    "zone_cost": (None if leg.zone_cost >= IMPASSABLE_COST
                                  else round(leg.zone_cost, 3)),
                    "navigable": leg.zone_cost < IMPASSABLE_COST,
                    "cumulative_km": round(leg.cumulative_km, 2),
                    "bearing_deg": round(leg.bearing_deg, 1),
                    "marker-color": CLASS_COLOR.get(leg.class_code, "#666666"),
                },
                "geometry": {"type": "Point",
                             "coordinates": [round(leg.longitude, 6), round(leg.latitude, 6)]},
            })

    if include_baseline and plan.baseline:
        # Imported here to keep the export package free of a module-level
        # dependency on the routing package.
        from ..routing.geodesy import great_circle_points
        line = great_circle_points(plan.start, plan.end, 128)
        features.append({
            "type": "Feature",
            "properties": {
                "feature_type": "great_circle_reference",
                "distance_km": plan.baseline.get("distance_km"),
                "is_navigable": plan.baseline.get("is_navigable"),
                "land_share": plan.baseline.get("land_share"),
                "verdict": plan.baseline.get("verdict"),
                "stroke": "#d62828", "stroke-width": 2, "stroke-dasharray": "6,6",
            },
            "geometry": {"type": "LineString",
                         "coordinates": [[round(lon, 6), round(lat, 6)] for lat, lon in line]},
        })

    # "name", "crs" and "metadata" are foreign members (RFC 7946, 6.1): readers
    # that do not know them ignore them. "crs" is the pre-RFC (2008) way of
    # declaring the reference system; CRS84 = WGS 84 in lon/lat order, which
    # is also the RFC 7946 default, so it only helps older tools such as QGIS 2.
    return {
        "type": "FeatureCollection",
        "name": f"maritime_route_{plan.route_id}",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "metadata": {
            "generator": f"Maritime Route Planner {APP_VERSION}",
            "author": APP_AUTHOR,
            "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            # Cost model used for the route; infinite weights become null.
            "zone_weights": {k: (None if math.isinf(v) else v) for k, v in ZONE_COST.items()},
        },
        "features": features,
    }


def classified_points_to_geojson(results: Sequence) -> Dict:
    """GeoJSON of classified AIS points. 分类后的 AIS 点导出。

    One ``Point`` feature per point with the predicted class, its
    confidence, the NOAA depth in metres and the probability of every class
    (``p_OPEN_SEA`` ...). Coordinates are in RFC 7946 ``[lon, lat]`` order.

    Parameters
    ----------
    results : sequence of ClassificationResult
        Output of the classifier service.

    Returns
    -------
    dict
        GeoJSON ``FeatureCollection``.
    """
    return {
        "type": "FeatureCollection",
        "name": "classified_points",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "class_code": r.class_code,
                    "class_name": CLASS_DESCRIPTION.get(r.class_code, ""),
                    "confidence": round(r.confidence, 4),
                    "depth_m": None if r.depth_m is None else round(r.depth_m, 1),
                    "marker-color": CLASS_COLOR.get(r.class_code, "#666666"),
                    **{f"p_{k}": round(v, 4) for k, v in r.probabilities.items()},
                },
                "geometry": {"type": "Point",
                             "coordinates": [round(r.longitude, 6), round(r.latitude, 6)]},
            }
            for r in results
        ],
    }


def write_geojson(document: Dict, path: Path | str) -> Path:
    """Write a GeoJSON document to disk as UTF-8 text.

    ``ensure_ascii=False`` keeps non-ASCII characters (e.g. Chinese or
    Russian zone descriptions) readable instead of ``\\uXXXX`` escapes;
    ``indent=2`` makes the file human-readable.

    Parameters
    ----------
    document : dict
        GeoJSON object.
    path : Path or str
        Target file; missing parent directories are created.

    Returns
    -------
    Path
        The written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# CSV                                                                          #
# --------------------------------------------------------------------------- #
#: Columns of the route CSV, one row per waypoint:
#: ``seq`` (0 = departure), ``latitude``/``longitude`` (degrees, 6 decimals),
#: ``class_code`` (zone), ``zone_cost`` (nominal weight or ``inf``),
#: ``cumulative_km`` (distance from departure, km), ``bearing_deg`` (course to
#: the next waypoint, degrees from north), and the ``route_id`` and
#: ``algorithm`` of the plan repeated on every row so that files can be
#: concatenated. 航线 CSV 的列定义。
ROUTE_CSV_HEADER = [
    "seq", "latitude", "longitude", "class_code", "zone_cost",
    "cumulative_km", "bearing_deg", "route_id", "algorithm",
]


def route_to_csv_string(plan) -> str:
    """Serialise the waypoint table to CSV text. 航路点 CSV。

    Parameters
    ----------
    plan : RoutePlan
        Planning result.

    Returns
    -------
    str
        CSV text with a header row (:data:`ROUTE_CSV_HEADER`) and ``\\n``
        line endings; comma separator, dot as decimal mark.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(ROUTE_CSV_HEADER)
    for i, leg in enumerate(plan.legs):
        # The planner stores impassability as a large finite constant so that the
        # weights fit in a numeric array; the export reports it as "inf", which
        # pandas and NumPy both read back as infinity.
        # 内部用大常数表示不可通行，导出时写作 "inf"，便于下游正确解析。
        cost = ("inf" if leg.zone_cost >= IMPASSABLE_COST else f"{leg.zone_cost:.3f}")
        writer.writerow([
            i, f"{leg.latitude:.6f}", f"{leg.longitude:.6f}", leg.class_code,
            cost, f"{leg.cumulative_km:.2f}", f"{leg.bearing_deg:.1f}",
            plan.route_id, plan.algorithm,
        ])
    return buffer.getvalue()


def points_to_csv_string(results: Sequence) -> str:
    """Serialise classification results to CSV text. 分类结果 CSV。

    Columns: ``latitude``, ``longitude`` (degrees), ``class_code``,
    ``confidence`` (0..1), ``depth_m`` (empty when unknown) and one
    ``p_<CLASS>`` probability column per class in ``CLASS_NAMES`` order.

    Parameters
    ----------
    results : sequence of ClassificationResult
        Output of the classifier service.

    Returns
    -------
    str
        CSV text with a header row.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["latitude", "longitude", "class_code", "confidence", "depth_m",
                     *[f"p_{c}" for c in CLASS_NAMES]])
    for r in results:
        writer.writerow([
            f"{r.latitude:.6f}", f"{r.longitude:.6f}", r.class_code, f"{r.confidence:.4f}",
            "" if r.depth_m is None else f"{r.depth_m:.1f}",
            *[f"{r.probabilities.get(c, 0.0):.4f}" for c in CLASS_NAMES],
        ])
    return buffer.getvalue()


def write_csv(text: str, path: Path | str) -> Path:
    """Write CSV text to disk.

    Parameters
    ----------
    text : str
        CSV content.
    path : Path or str
        Target file; missing parent directories are created.

    Returns
    -------
    Path
        The written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # "utf-8-sig" prepends a byte-order mark so that Microsoft Excel detects
    # UTF-8 instead of the local ANSI code page. 带 BOM，便于 Excel 正确识别编码。
    path.write_text(text, encoding="utf-8-sig")
    return path


# --------------------------------------------------------------------------- #
# PDF                                                                          #
# --------------------------------------------------------------------------- #
def route_to_pdf(plan, path: Path | str, map_png: Optional[bytes] = None,
                 model_summary: Optional[Dict] = None) -> Path:
    """Render a one-file PDF voyage report using ReportLab.

    生成 PDF 航行报告：包含摘要表、区域分布、对比结论与航路点清单。

    Layout (A4 portrait, margins 18 mm left/right and 16 mm top/bottom, so
    the text column is 174 mm wide; every table is at most 168 mm):

    1. Title, generator line and route identifier.
    2. *Voyage summary* - two-column parameter table.
    3. *Route chart* - only when ``map_png`` is given (168 x 100 mm image).
    4. *Geographic zone profile* - waypoints per zone with weight and share.
    5. *Comparison with the shortest-distance route* - only when the plan has
       a baseline.
    6. *Classification model* - only when ``model_summary`` is given
       (unnumbered).
    7. *Appendix* on a new page - the waypoint list, thinned to about 120 rows.

    Section numbers shift by one when the map is absent. ReportLab
    ("platypus") lays out the flowables in ``story`` and breaks pages
    automatically.

    Parameters
    ----------
    plan : RoutePlan
        Planning result.
    path : Path or str
        Output PDF file; missing parent directories are created.
    map_png : bytes, optional
        PNG image of the route map, supplied by the caller (the web
        application currently passes ``None``, so the section is omitted).
    model_summary : dict, optional
        ``architecture``, ``n_features``, ``parameters``, ``accuracy``,
        ``macro_f1`` and ``trained_utc`` of the classifier.

    Returns
    -------
    Path
        The written PDF.

    Raises
    ------
    ImportError
        If ReportLab is not installed (imported lazily so that the other
        formats work without it).
    """
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Paragraph styles: dark-blue headings, 9.5 pt body text, 8 pt grey notes.
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=17, spaceAfter=8,
                        textColor=colors.HexColor("#0b3c68"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10,
                        spaceAfter=5, textColor=colors.HexColor("#0b3c68"))
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("Small", parent=body, fontSize=8, textColor=colors.HexColor("#555555"))

    document = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"Voyage report {plan.route_id}", author=APP_AUTHOR,
    )
    story: List = []

    story.append(Paragraph("Maritime Route Planning Report", h1))
    story.append(Paragraph(
        f"Generated by Maritime Route Planner v{APP_VERSION} &mdash; {APP_UNIVERSITY}<br/>"
        f"Route identifier <b>{plan.route_id}</b>, created {plan.created_utc}", small))
    story.append(Spacer(1, 6))

    def table(rows: List[List[str]], widths: List[float], header: bool = True) -> Table:
        """Build a report table in the common style.

        Thin grid, 8.5 pt text; with ``header`` the first row is white bold
        text on dark blue and the body rows alternate white / light blue.
        Cell coordinates in ``TableStyle`` are ``(column, row)``, and
        ``-1`` means the last column/row.

        Parameters
        ----------
        rows : list of list of str
            Table content, header row first.
        widths : list of float
            Column widths in points (``mm`` multiples).
        header : bool, default True
            Style the first row as a header.

        Returns
        -------
        reportlab.platypus.Table
            Styled table flowable.
        """
        t = Table(rows, colWidths=widths, hAlign="LEFT")
        style = [
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b9c6d4")),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        if header:
            style += [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3c68")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
            ]
        t.setStyle(TableStyle(style))
        return t

    # --- 1. voyage summary -------------------------------------------------
    story.append(Paragraph("1. Voyage summary", h2))
    story.append(table([
        ["Parameter", "Value"],
        ["Departure (lat, lon)", f"{plan.start[0]:.4f}, {plan.start[1]:.4f}"],
        ["Destination (lat, lon)", f"{plan.end[0]:.4f}, {plan.end[1]:.4f}"],
        ["Search algorithm", plan.algorithm.upper()],
        # 1 nautical mile = 1.852 km. 1 海里 = 1.852 千米。
        ["Route length", f"{plan.distance_km:,.1f} km ({plan.distance_km / 1.852:,.1f} NM)"],
        ["Great-circle distance", f"{plan.great_circle_km:,.1f} km"],
        ["Detour ratio", _fmt(plan.detour_ratio, "{:.3f}")],
        ["Weighted traversal cost", _fmt(plan.total_cost, "{:,.1f}")],
        ["Planned speed", f"{plan.speed_knots:.1f} kn"],
        ["Estimated passage time", _fmt(plan.estimated_hours, "{:,.1f} h")],
        ["Nodes expanded", f"{plan.nodes_expanded:,}"],
        ["Planning runtime", f"{plan.runtime_s:.3f} s"],
        ["Grid", f"{plan.grid.get('n_rows')} x {plan.grid.get('n_cols')} cells at "
                 f"{plan.grid.get('resolution_deg')}°"],
        ["Feasible", "Yes" if plan.feasible else "No"],
    ], [58 * mm, 110 * mm]))

    # --- 2. map ------------------------------------------------------------
    if map_png:
        story.append(Paragraph("2. Route chart", h2))
        story.append(Image(io.BytesIO(map_png), width=168 * mm, height=100 * mm))
        story.append(Paragraph("Blue: optimised route. Red dashed: great-circle reference.",
                               small))

    # --- 3. zone profile ---------------------------------------------------
    story.append(Paragraph(f"{'3' if map_png else '2'}. Geographic zone profile", h2))
    # max(..., 1) avoids a division by zero for an infeasible plan without legs.
    total_legs = max(sum(plan.zone_profile.values()), 1)
    rows = [["Zone", "Description", "Weight", "Waypoints", "Share"]]
    for name in CLASS_NAMES:
        count = plan.zone_profile.get(name, 0)
        weight = ZONE_COST[name]
        rows.append([
            name, CLASS_DESCRIPTION[name],
            "impassable" if math.isinf(weight) else f"{weight:.1f}",
            str(count), f"{100.0 * count / total_legs:.1f}%",
        ])
    story.append(table(rows, [28 * mm, 72 * mm, 22 * mm, 22 * mm, 22 * mm]))

    # --- 4. comparison -----------------------------------------------------
    if plan.baseline:
        story.append(Paragraph(f"{'4' if map_png else '3'}. Comparison with the "
                               "shortest-distance route", h2))
        b = plan.baseline
        # The optimised column shows 0 land samples by construction: the
        # search never enters land cells (interior waypoints are navigable).
        story.append(table([
            ["Criterion", "Great-circle route", "Optimised route"],
            ["Length, km", f"{b.get('distance_km', 0):,.1f}", f"{plan.distance_km:,.1f}"],
            ["Extra length", "-", f"+{b.get('extra_distance_pct', 0):.2f} %"],
            ["Samples over land", f"{b.get('land_samples', 0)} of {b.get('samples', 0)}", "0"],
            ["Share over land", f"{100.0 * b.get('land_share', 0):.1f} %", "0.0 %"],
            ["Navigable", "Yes" if b.get("is_navigable") else "No", "Yes"],
        ], [45 * mm, 60 * mm, 60 * mm]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Conclusion.</b> {b.get('verdict', '')} "
                               f"The optimised route accepts "
                               f"{b.get('extra_distance_pct', 0):.2f}% additional distance in "
                               f"exchange for a passage that remains inside navigable zones "
                               f"over its whole length.", body))

    # --- 5. model ----------------------------------------------------------
    if model_summary:
        story.append(Paragraph("Classification model", h2))
        story.append(table([
            ["Property", "Value"],
            ["Architecture", str(model_summary.get("architecture"))],
            ["Input features", str(model_summary.get("n_features"))],
            ["Trainable parameters", f"{model_summary.get('parameters', 0):,}"],
            ["Test accuracy", _fmt(model_summary.get("accuracy"), "{:.4f}")],
            ["Macro F1", _fmt(model_summary.get("macro_f1"), "{:.4f}")],
            ["Trained", str(model_summary.get("trained_utc"))],
        ], [58 * mm, 110 * mm]))

    # --- 6. waypoint table -------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("Appendix. Waypoint list", h2))
    rows = [["#", "Latitude", "Longitude", "Zone", "Cum. km", "Bearing"]]
    # Thin long routes to every step-th waypoint (about 120 rows, 3-4 pages).
    # 航路点过多时按步长抽样，约 120 行。
    step = max(1, len(plan.legs) // 120)
    for i in range(0, len(plan.legs), step):
        leg = plan.legs[i]
        rows.append([str(i), f"{leg.latitude:.4f}", f"{leg.longitude:.4f}", leg.class_code,
                     f"{leg.cumulative_km:.1f}", f"{leg.bearing_deg:.0f}°"])
    # Always list the destination, even if thinning skipped the last index.
    if len(plan.legs) and (len(plan.legs) - 1) % step:
        leg = plan.legs[-1]
        rows.append([str(len(plan.legs) - 1), f"{leg.latitude:.4f}", f"{leg.longitude:.4f}",
                     leg.class_code, f"{leg.cumulative_km:.1f}", f"{leg.bearing_deg:.0f}°"])
    story.append(table(rows, [12 * mm, 30 * mm, 30 * mm, 36 * mm, 26 * mm, 24 * mm]))
    if step > 1:
        story.append(Spacer(1, 3))
        story.append(Paragraph(f"Table shows every {step}-th waypoint of "
                               f"{len(plan.legs)} in total.", small))

    # Lay out all flowables and write the file.
    document.build(story)
    LOGGER.info("PDF report written to %s", path)
    return path


# --------------------------------------------------------------------------- #
def export_all(plan, results: Optional[Sequence] = None,
               directory: Path | str = OUTPUT_DIR,
               map_png: Optional[bytes] = None,
               model_summary: Optional[Dict] = None) -> Dict[str, str]:
    """Write GeoJSON + CSV + PDF for one plan and return the file paths.

    Files are named ``route_<route_id>.<ext>``; with ``results`` two more
    files ``route_<route_id>_points.geojson`` and ``..._points.csv`` hold the
    classified points.
    一次性导出全部格式并返回文件路径。

    Parameters
    ----------
    plan : RoutePlan
        Planning result.
    results : sequence of ClassificationResult, optional
        Classified points to export alongside the route.
    directory : Path or str, default ``config.OUTPUT_DIR``
        Output directory.
    map_png : bytes, optional
        Map image for the PDF.
    model_summary : dict, optional
        Model section of the PDF.

    Returns
    -------
    dict
        Format key (``geojson``, ``csv``, ``pdf``, ``points_geojson``,
        ``points_csv``) -> file path as string.
    """
    directory = Path(directory)
    stem = f"route_{plan.route_id}"
    paths = {
        "geojson": str(write_geojson(route_to_geojson(plan), directory / f"{stem}.geojson")),
        "csv": str(write_csv(route_to_csv_string(plan), directory / f"{stem}.csv")),
        "pdf": str(route_to_pdf(plan, directory / f"{stem}.pdf", map_png, model_summary)),
    }
    if results:
        paths["points_geojson"] = str(write_geojson(
            classified_points_to_geojson(results), directory / f"{stem}_points.geojson"))
        paths["points_csv"] = str(write_csv(
            points_to_csv_string(results), directory / f"{stem}_points.csv"))
    return paths


def _clean(value) -> Optional[float]:
    """Round a number for JSON, mapping inf/nan/non-numbers to ``None``.

    JSON (RFC 8259) has no literal for infinity or NaN; Python would write
    the non-standard tokens ``Infinity``/``NaN``, which strict parsers reject.

    Parameters
    ----------
    value
        Any value.

    Returns
    -------
    float or None
        ``round(value, 4)`` for a finite number, else ``None``.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return round(value, 4) if math.isfinite(value) else None


def _fmt(value, spec: str) -> str:
    """Format a number for the PDF, or "-" when it is missing or not finite.

    Parameters
    ----------
    value
        Any value.
    spec : str
        :meth:`str.format` pattern such as ``"{:.3f}"``.

    Returns
    -------
    str
        Formatted text or ``"-"``.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "-"
    return spec.format(value) if math.isfinite(value) else "-"
