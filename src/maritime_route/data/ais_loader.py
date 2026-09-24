"""Loading of AIS / route-point data from GeoJSON, CSV and JSON sources.

AIS 数据加载模块：支持 GeoJSON(FeatureCollection)、CSV 与 JSON 数组三种输入。
The loader is streaming-capable so that files of several hundred megabytes can
be ingested without materialising the whole document in memory.

Role in the pipeline
--------------------
This is the first stage of both the training pipeline
(:func:`maritime_route.data.dataset.prepare`) and the web upload endpoint.
Every loader returns a :class:`pandas.DataFrame` with one row per point and at
least the columns ``latitude`` and ``longitude`` (decimal degrees, WGS84).
Optional columns understood by the rest of the code are:

* ``class_code`` - one of :data:`~maritime_route.config.CLASS_NAMES` (labelled
  corpus only);
* ``noaa_depth`` - NOAA/ETOPO relief value, metres, positive above sea level;
* ``noaa_type`` - ``"height"`` or ``"depth"``, which kind of relief sample it is;
* ``elevation`` - terrain elevation, metres;
* the OSM count columns of :data:`~maritime_route.config.OSM_COLUMNS`.

After loading, :func:`validate` drops rows with missing or out-of-range
coordinates and returns a :class:`LoadReport` that the UI shows to the user.

Main objects
------------
:func:`load_any`
    Entry point; dispatches on the file extension, accepts in-memory uploads.
:func:`load_geojson`, :func:`load_csv`, :func:`load_json`
    Format-specific readers.
:func:`validate`
    Coordinate range check and report.
:class:`DataLoadError`, :class:`LoadReport`
    Error type and summary object.

在流水线中的位置：训练预处理与网页上传的第一步。所有读取函数返回每行一个点、
至少含 latitude/longitude（十进制度）的 DataFrame，随后由 validate 剔除无效坐标。
"""
from __future__ import annotations

import csv
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..config import CLASS_NAMES, OSM_COLUMNS

LOGGER = logging.getLogger(__name__)

#: Column aliases accepted in user supplied CSV files. 兼容常见列名写法。
#: The first alias present in the file is used; matching falls back to a
#: case-insensitive comparison (see :func:`_first_present`).
LAT_ALIASES = ("latitude", "lat", "y", "LAT", "Latitude")
LON_ALIASES = ("longitude", "lon", "lng", "long", "x", "LON", "Longitude")

#: Columns without which a row cannot be used at all. 必需列。
REQUIRED_COLUMNS = ("latitude", "longitude")
#: Columns converted to numbers on load; unparsable cells become NaN.
#: 加载时转换为数值的列，无法解析的值变为 NaN。
NUMERIC_COLUMNS = ("latitude", "longitude", "elevation", "noaa_depth", *OSM_COLUMNS)


class DataLoadError(ValueError):
    """Raised when an input file cannot be interpreted as AIS point data.

    Subclasses :class:`ValueError` (the input value is wrong, not the
    program). The web upload endpoint catches it and answers HTTP 400 with the
    message, instead of a server error.
    无法将输入解析为点数据时抛出；Web 上传接口捕获后返回 HTTP 400。
    """


@dataclass
class LoadReport:
    """Summary returned to the UI after an upload. 加载结果摘要。

    Attributes
    ----------
    source:
        File name or other description of the input.
    n_features:
        Number of rows (GeoJSON features / CSV lines) read from the input.
    n_valid:
        Rows that passed validation.
    n_dropped:
        Rows removed because of missing or out-of-range coordinates
        (``n_features - n_valid``).
    labelled:
        ``True`` if the input has a non-empty ``class_code`` column, i.e. it
        can be used for training or for measuring accuracy.
    columns:
        Column names of the validated frame.
    warnings:
        Human-readable messages shown next to the upload in the UI.
    """

    source: str
    n_features: int
    n_valid: int
    n_dropped: int
    labelled: bool
    columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a JSON-serialisable dictionary.

        Returns
        -------
        dict
            All fields of the report, keyed by field name.

        转换为可 JSON 序列化的字典（用于 API 响应与统计文件）。
        """
        return {
            "source": self.source,
            "n_features": self.n_features,
            "n_valid": self.n_valid,
            "n_dropped": self.n_dropped,
            "labelled": self.labelled,
            "columns": self.columns,
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------- #
# GeoJSON                                                                     #
# --------------------------------------------------------------------------- #
def iter_geojson_features(path: Path | str) -> Iterator[Dict[str, Any]]:
    """Yield ``properties`` dictionaries of a GeoJSON FeatureCollection.

    使用 ijson 流式解析，避免一次性载入 100MB+ 文件。
    Falls back to a plain ``json.load`` when ijson is unavailable.

    ijson reads the file incrementally and yields one element of the
    ``features`` array at a time (prefix ``"features.item"``), so memory use
    stays proportional to one feature instead of the whole document.
    ``use_float=True`` makes ijson return ``float`` instead of
    :class:`decimal.Decimal`, which pandas and NumPy handle directly.

    Parameters
    ----------
    path:
        Path of a GeoJSON file whose top-level object is a FeatureCollection.

    Yields
    ------
    dict
        The ``properties`` of each feature, with ``latitude`` and
        ``longitude`` taken from the point geometry (see
        :func:`_merge_geometry`).
    """
    path = Path(path)
    try:
        import ijson  # type: ignore  # optional dependency 可选依赖

        with path.open("rb") as handle:
            for feature in ijson.items(handle, "features.item", use_float=True):
                yield _merge_geometry(feature)
        # The streaming branch finished; do not run the fallback below.
        # 流式解析已完成，不再执行下面的回退分支。
        return
    except ImportError:  # pragma: no cover - exercised only without ijson
        LOGGER.warning("ijson not installed, falling back to in-memory parsing")

    # Fallback: parse the whole document in memory (fine for small files).
    # 回退方案：整体读入内存解析（适用于小文件）。
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    for feature in document.get("features", []):
        yield _merge_geometry(feature)


def _merge_geometry(feature: Dict[str, Any]) -> Dict[str, Any]:
    """Copy geometry coordinates into the properties dict when they are missing.

    In practice the geometry always overrides the properties: a GeoJSON Point
    geometry is ``{"type": "Point", "coordinates": [lon, lat]}`` and is the
    authoritative position, whereas ``latitude``/``longitude`` properties may
    be stale or rounded. Features without a usable geometry keep whatever
    coordinate properties they have.

    Parameters
    ----------
    feature:
        One GeoJSON Feature object (``dict``).

    Returns
    -------
    dict
        A new dictionary (the input is not modified) with the feature
        properties plus ``latitude`` and ``longitude``.

    将几何坐标写入属性字典；几何坐标优先（GeoJSON 顺序为 经度, 纬度）。
    """
    # ``or {}`` handles both a missing key and an explicit JSON null.
    # ``or {}`` 同时处理缺失键与 JSON null。
    props: Dict[str, Any] = dict(feature.get("properties") or {})
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates")
    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        props.setdefault("longitude", coords[0])
        props.setdefault("latitude", coords[1])
        # GeoJSON stores (lon, lat) - the geometry is authoritative.
        # The assignments below overwrite any property values set above.
        # GeoJSON 坐标顺序为 (经度, 纬度)，以几何为准并覆盖属性中的值。
        props["longitude"] = coords[0]
        props["latitude"] = coords[1]
    return props


def load_geojson(path: Path | str, limit: Optional[int] = None) -> pd.DataFrame:
    """Read a GeoJSON point FeatureCollection into a DataFrame.

    Parameters
    ----------
    path:
        GeoJSON file.
    limit:
        If given, only the first ``limit`` features are read (quick
        experiments on a large corpus).

    Returns
    -------
    pandas.DataFrame
        One row per feature; columns are the union of all property names plus
        ``latitude`` and ``longitude``. Column names are *not* normalised here.

    Raises
    ------
    DataLoadError
        If the collection contains no features.

    读取 GeoJSON 点集合为 DataFrame，可用 limit 限制读取数量。
    """
    rows: List[Dict[str, Any]] = []
    for i, props in enumerate(iter_geojson_features(path)):
        if limit is not None and i >= limit:
            break
        rows.append(props)
    if not rows:
        raise DataLoadError(f"No features found in {path}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# CSV / JSON                                                                  #
# --------------------------------------------------------------------------- #
def load_csv(path_or_buffer: Any) -> pd.DataFrame:
    """Read a CSV file of route points, normalising the coordinate columns.

    Parameters
    ----------
    path_or_buffer:
        File path or text buffer accepted by :func:`pandas.read_csv`.

    Returns
    -------
    pandas.DataFrame
        Frame with canonical ``latitude``/``longitude`` columns (see
        :func:`_normalise_columns`).

    Raises
    ------
    DataLoadError
        If no latitude/longitude column can be identified.

    读取 CSV 并统一坐标列名。
    """
    frame = pd.read_csv(path_or_buffer)
    return _normalise_columns(frame)


def load_json(path_or_buffer: Any) -> pd.DataFrame:
    """Read a JSON document: either a FeatureCollection or a list of records.

    Parameters
    ----------
    path_or_buffer:
        File path, or any object with a ``read`` method (e.g.
        :class:`io.StringIO` holding an uploaded file).

    Returns
    -------
    pandas.DataFrame
        Normalised frame, one row per point.

    Raises
    ------
    DataLoadError
        If the document is neither a FeatureCollection nor a list, or if it
        contains no points, or if coordinate columns are missing.

    读取 JSON：支持 FeatureCollection 或记录数组两种结构（整体读入内存）。
    """
    # Duck typing: anything with ``read`` is treated as an open file object.
    # 鸭子类型：具有 read 方法的对象视为已打开的文件。
    if hasattr(path_or_buffer, "read"):
        document = json.load(path_or_buffer)
    else:
        with open(path_or_buffer, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    if isinstance(document, dict) and document.get("type") == "FeatureCollection":
        rows = [_merge_geometry(f) for f in document.get("features", [])]
    elif isinstance(document, list):
        # A plain array of records: [{"latitude": .., "longitude": ..}, ...]
        # 普通记录数组。
        rows = document
    else:
        raise DataLoadError("Unsupported JSON structure for AIS data")
    if not rows:
        raise DataLoadError("JSON document contains no points")
    return _normalise_columns(pd.DataFrame(rows))


def load_any(source: Path | str, content: Optional[bytes] = None) -> pd.DataFrame:
    """Dispatch on the file extension. ``content`` allows in-memory uploads.

    Parameters
    ----------
    source:
        File path, or, when ``content`` is given, only the original file name
        (used to pick the format).
    content:
        Raw bytes of an uploaded file. When present the file system is not
        touched.

    Returns
    -------
    pandas.DataFrame
        Normalised frame with ``latitude``/``longitude`` columns. Call
        :func:`validate` afterwards to drop invalid rows.

    Raises
    ------
    DataLoadError
        Unsupported extension or content that cannot be parsed.

    按文件扩展名选择读取方式；content 参数用于处理网页上传的内存数据。
    """
    name = str(source).lower()
    if content is not None:
        # Uploaded bytes. "utf-8-sig" strips a byte-order mark that some
        # spreadsheet programs write at the start of CSV/JSON exports;
        # errors="replace" keeps a single bad byte from rejecting the file.
        # 上传的字节流：utf-8-sig 去除 BOM；errors="replace" 容忍个别非法字节。
        buffer: Any
        if name.endswith(".csv"):
            buffer = io.StringIO(content.decode("utf-8-sig", errors="replace"))
            return load_csv(buffer)
        # Every other upload (.json, .geojson) is parsed as JSON; load_json
        # recognises FeatureCollections itself.
        # 其他上传（.json/.geojson）一律按 JSON 解析，load_json 可识别 FeatureCollection。
        buffer = io.StringIO(content.decode("utf-8-sig", errors="replace"))
        return load_json(buffer)
    if name.endswith(".csv"):
        return load_csv(source)
    if name.endswith((".geojson", ".json")):
        # Try the streaming GeoJSON reader first; if the file is a plain JSON
        # array (no "features"), it raises DataLoadError and load_json is used.
        # 先尝试流式 GeoJSON 读取；若为普通 JSON 数组则回退到 load_json。
        try:
            return _normalise_columns(load_geojson(source))
        except DataLoadError:
            return load_json(source)
    raise DataLoadError(f"Unsupported file type: {source}")


# --------------------------------------------------------------------------- #
# Normalisation & validation                                                  #
# --------------------------------------------------------------------------- #
def _first_present(frame: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    """Return the first column of ``frame`` that matches one of ``aliases``.

    Exact matches are tried first (in alias order), then a case-insensitive
    match, so ``"LaT"`` is still recognised as latitude.

    Parameters
    ----------
    frame:
        Input frame.
    aliases:
        Candidate column names in order of preference.

    Returns
    -------
    str or None
        The actual column name in ``frame``, or ``None`` if nothing matches.

    按别名顺序查找列名：先精确匹配，再忽略大小写匹配。
    """
    for alias in aliases:
        if alias in frame.columns:
            return alias
    # Map lower-cased names back to the real column names. 小写名 -> 实际列名。
    lowered = {c.lower(): c for c in frame.columns}
    for alias in aliases:
        if alias.lower() in lowered:
            return lowered[alias.lower()]
    return None


def _normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename coordinate aliases and coerce numeric dtypes.

    Steps: rename the detected coordinate columns to ``latitude`` /
    ``longitude``; convert :data:`NUMERIC_COLUMNS` to numbers (bad values ->
    NaN); add missing OSM count columns filled with 0 (no objects known); add
    an empty ``noaa_type`` column. After this every downstream function can
    rely on the same column set.

    Parameters
    ----------
    frame:
        Raw frame from any reader.

    Returns
    -------
    pandas.DataFrame
        A normalised copy; the input frame is not modified.

    Raises
    ------
    DataLoadError
        If no latitude or no longitude column can be found.

    统一列名与数据类型，并补齐缺失的 OSM 计数列（填 0）与 noaa_type 列。
    """
    frame = frame.copy()
    lat_col = _first_present(frame, LAT_ALIASES)
    lon_col = _first_present(frame, LON_ALIASES)
    if lat_col is None or lon_col is None:
        # Show at most 12 column names so the error message stays readable.
        # 最多列出 12 个列名，保持错误信息简洁。
        raise DataLoadError(
            "Input must provide latitude and longitude columns "
            f"(got: {list(frame.columns)[:12]})"
        )
    frame = frame.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    for column in NUMERIC_COLUMNS:
        if column in frame.columns:
            # errors="coerce": text such as "n/a" becomes NaN instead of failing.
            # 无法解析的文本转为 NaN，而不是抛出异常。
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in OSM_COLUMNS:
        if column not in frame.columns:
            frame[column] = 0.0
    if "noaa_type" not in frame.columns:
        frame["noaa_type"] = None
    return frame


def validate(frame: pd.DataFrame, source: str = "upload") -> tuple[pd.DataFrame, LoadReport]:
    """Drop invalid rows and build a :class:`LoadReport`.

    校验经纬度范围并剔除缺失/越界的记录。

    A row is kept when both coordinates are present and inside the valid WGS84
    range: latitude in [-90, 90] degrees, longitude in [-180, 180] degrees.
    If the frame is labelled, unknown class codes are reported as a warning
    (they are not removed here; :func:`~maritime_route.data.dataset.prepare`
    filters them out).

    Parameters
    ----------
    frame:
        Normalised frame (output of one of the loaders).
    source:
        Description of the input, copied into the report.

    Returns
    -------
    (pandas.DataFrame, LoadReport)
        The valid rows (index reset to 0..n-1) and the report.

    Raises
    ------
    DataLoadError
        If a required column is missing or no valid row remains.
    """
    n_in = len(frame)
    warnings: List[str] = []
    for column in REQUIRED_COLUMNS:
        if column not in frame.columns:
            raise DataLoadError(f"Missing required column '{column}'")

    # Boolean row mask: both coordinates present and within the WGS84 range
    # (``between`` is inclusive on both ends).
    # 行掩码：坐标非空且位于合法范围内（between 两端均包含）。
    mask = frame["latitude"].notna() & frame["longitude"].notna()
    mask &= frame["latitude"].between(-90.0, 90.0)
    mask &= frame["longitude"].between(-180.0, 180.0)
    clean = frame.loc[mask].reset_index(drop=True)
    dropped = n_in - len(clean)
    if dropped:
        warnings.append(f"{dropped} row(s) dropped: missing or out-of-range coordinates")
    if clean.empty:
        raise DataLoadError("No valid coordinates remain after validation")

    # The data counts as labelled if at least one row carries a class code.
    # 只要至少一行带有类别代码即视为带标签数据。
    labelled = "class_code" in clean.columns and clean["class_code"].notna().any()
    if labelled:
        unknown = sorted(set(clean["class_code"].dropna().unique()) - set(CLASS_NAMES))
        if unknown:
            warnings.append(f"Unknown class codes ignored: {unknown}")

    report = LoadReport(
        source=source,
        n_features=n_in,
        n_valid=len(clean),
        n_dropped=dropped,
        labelled=bool(labelled),
        columns=list(clean.columns),
        warnings=warnings,
    )
    return clean, report


def to_coordinate_array(frame: pd.DataFrame) -> np.ndarray:
    """Return an ``(n, 2)`` float array of (latitude, longitude).

    Parameters
    ----------
    frame:
        Frame with ``latitude`` and ``longitude`` columns.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 2)``, dtype float64, degrees; column 0 is latitude.

    返回 (n, 2) 的 (纬度, 经度) 数组，单位为度。
    """
    return frame[["latitude", "longitude"]].to_numpy(dtype=float)
