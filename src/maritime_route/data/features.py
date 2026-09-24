"""Feature engineering for the geographic-zone classifier.

特征工程模块。
Two feature sources are combined for every geographic point:

1. **Geodetic encoding** - latitude/longitude expressed in a form that has no
   artificial discontinuity at the antimeridian.
2. **Environmental context** - bathymetry/orography statistics and OSM object
   counts sampled from a reference index around the query point.

The reference index (:class:`ReferenceIndex`) is built once from the training
corpus and is *label free*: it only stores depth, elevation and OSM counts, so
using it at inference time introduces no target leakage. It plays the role that
a NOAA bathymetry raster and an OpenStreetMap extract would play in a
production deployment.

Role in the pipeline
--------------------
* Training: :func:`maritime_route.data.dataset.prepare` builds the index from
  the training split, then calls :func:`build_features` for the training,
  validation and test points.
* Inference: :class:`maritime_route.model.inference.ZoneClassifierService`
  loads the saved index and calls :func:`build_features` (uploaded points
  with measurements) or :func:`features_for_coordinates` (bare grid cells of
  the route planner).

The output is always an ``(n, 28)`` float32 matrix whose columns are listed in
:data:`FEATURE_NAMES`. The network never sees raw values; the matrix is
standardised by the scaler fitted in :mod:`maritime_route.model.trainer`.

Units used throughout: coordinates in degrees (radians only inside the
BallTree), distances in km, depth/elevation in metres with the NOAA sign
convention (positive = above sea level, negative = below).

流水线位置：训练时由 dataset.prepare 调用，推理时由 ZoneClassifierService 调用。
输出恒为 (n, 28) float32 矩阵。单位：坐标为度（BallTree 内为弧度），距离为 km，
水深/高程为米（正值在海平面以上，负值在海平面以下）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from ..config import BATHY_NEIGHBOURS, EARTH_RADIUS_KM, OSM_COLUMNS, OSM_RADIUS_KM

#: Two neighbours closer than this are treated as the same location. The index
#: stores float32 coordinates, whose rounding error reaches about half a metre.
#: 小于该距离视为同一位置（float32 存储带来的舍入误差约 0.5 m）。
#: Value: 0.002 km = 2 m, i.e. a few times the float32 rounding error but far
#: below the spacing of distinct corpus points. (Index files written by the
#: current :meth:`ReferenceIndex.save` keep float64 coordinates; the tolerance
#: keeps older float32 files working and costs nothing for new ones.)
#: 取值 2 m：大于 float32 误差、远小于样本间距；当前 save 已保存 float64，
#: 该容差用于兼容旧的 float32 索引文件。
SELF_MATCH_EPS_KM: float = 0.002

LOGGER = logging.getLogger(__name__)

#: Names of the 28 columns of the feature matrix, in column order.
#: The order is part of the trained model (it is stored in the metadata) and
#: must not change without retraining. Formulas, with d = depth (m), e =
#: elevation (m), k = BATHY_NEIGHBOURS neighbours from the reference index:
#:
#: ======  =========================  ==========================================
#: col     name                       definition
#: ======  =========================  ==========================================
#: 0-3     lat_sin .. lon_cos         sin/cos of latitude and longitude (radians)
#: 4       abs_lat_norm               |lat| / 90, distance from the equator 0..1
#: 5       depth_signed_log           slog(d); d replaced by ctx mean if missing
#: 6       depth_is_missing           1 if the point had no measured depth
#: 7       is_height_sample           1 if noaa_type == "height" (land sample)
#: 8       elevation_signed_log       slog(e); e replaced by filled d if missing
#: 9       elevation_is_missing       1 if the point had no measured elevation
#: 10-13   ctx_depth_mean..max        slog of mean/min/max, log1p of std of the
#:                                    k neighbour depths
#: 14      ctx_depth_range            log1p(max - min) of neighbour depths
#: 15      ctx_land_fraction          share of neighbours with depth > 0
#: 16      ctx_sign_agreement         1 if sign(d) == sign(neighbour mean)
#: 17      ctx_distance_km_log        log1p(distance to nearest neighbour, km)
#: 18      ctx_gradient               slog((max - min) / mean distance), m/km
#: 19-24   osm_* / ctx_osm_*          log1p of grouped OSM counts, own point and
#:                                    neighbour mean (marine infra, natural
#:                                    coast, inland water)
#: 25      osm_industrial_log         log1p(own industrial count)
#: 26      ctx_osm_total              log1p(mean total OSM count of neighbours)
#: 27      ctx_osm_density            log1p(mean total count / span km)
#: ======  =========================  ==========================================
#:
#: slog(x) = sign(x) * log(1 + |x|), see :func:`signed_log`.
#: 28 个特征的名称（列顺序为模型的一部分，更改需重新训练）。
FEATURE_NAMES: List[str] = [
    "lat_sin",
    "lat_cos",
    "lon_sin",
    "lon_cos",
    "abs_lat_norm",
    "depth_signed_log",
    "depth_is_missing",
    "is_height_sample",
    "elevation_signed_log",
    "elevation_is_missing",
    "ctx_depth_mean",
    "ctx_depth_std",
    "ctx_depth_min",
    "ctx_depth_max",
    "ctx_depth_range",
    "ctx_land_fraction",
    "ctx_sign_agreement",
    "ctx_distance_km_log",
    "ctx_gradient",
    "osm_marine_infra",
    "ctx_osm_marine_infra",
    "osm_natural_coastal",
    "ctx_osm_natural_coastal",
    "osm_inland_water",
    "ctx_osm_inland_water",
    "osm_industrial_log",
    "ctx_osm_total",
    "ctx_osm_density",
]
#: Width of the feature matrix (28). 特征维数。
N_FEATURES: int = len(FEATURE_NAMES)

#: OSM groups. 海事基础设施 / 自然海岸 / 内陆水体。
#: The nine raw OSM columns are summed into three semantically meaningful
#: groups; single object types are too sparse to be informative on their own.
#: 九个原始 OSM 列合并为三组，单一类型过于稀疏。
MARINE_INFRA = ["osm_piers", "osm_lighthouses", "osm_breakwaters", "osm_harbors"]
NATURAL_COASTAL = ["osm_beaches", "osm_cliffs"]
INLAND_WATER = ["osm_rivers", "osm_lakes"]


def signed_log(values: np.ndarray) -> np.ndarray:
    """``sign(x) * log1p(|x|)`` - compresses the huge bathymetric range.

    Relief values span roughly -11 000 m (ocean trenches) to +8 800 m, while
    the class boundaries lie within a few metres of zero. The signed logarithm
    keeps the sign (land vs water), is continuous and monotone through zero
    (``log1p(0) = 0``), and maps the full range to about [-9.3, +9.1], so
    that small depths near the coast stay distinguishable after
    standardisation. Example: -4000 m -> -8.29, -10 m -> -2.40, +3 m -> +1.39.

    Parameters
    ----------
    values:
        Array of any shape, metres (or any signed quantity).

    Returns
    -------
    numpy.ndarray
        Same shape, float64.

    带符号对数变换：保留正负号（陆地/水域），压缩数千米的水深范围，
    同时保持近海平面处的分辨率。
    """
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


# --------------------------------------------------------------------------- #
# Reference index                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class ReferenceIndex:
    """Label-free spatial index over environmental measurements.

    参考索引：仅保存水深/高程/OSM 计数，不含类别标签，因此不会造成标签泄漏。
    Queried with a haversine :class:`~sklearn.neighbors.BallTree`.

    A BallTree partitions the points into nested hyperspheres, so a
    k-nearest-neighbour query costs about O(log n) instead of O(n). With
    ``metric="haversine"`` it works on the sphere: inputs are
    ``[latitude, longitude]`` in **radians** and returned distances are
    central angles in radians, converted to km by multiplying with
    :data:`~maritime_route.config.EARTH_RADIUS_KM`. This avoids the distortion
    of a Euclidean distance on raw degrees (one degree of longitude shrinks
    with cos(latitude)).

    Attributes
    ----------
    coordinates_rad:
        ``(n, 2)`` float64, radians, columns (lat, lon).
    depth:
        ``(n,)`` relief value, metres, positive above sea level.
    elevation:
        ``(n,)`` terrain elevation, metres.
    osm_counts:
        ``(n, 9)`` OSM object counts in the order of
        :data:`~maritime_route.config.OSM_COLUMNS`.
    tree:
        The BallTree; built automatically in :meth:`__post_init__` when not
        supplied.

    BallTree 采用 haversine 距离：输入为弧度制 (纬度, 经度)，返回球面角距离（弧度），
    乘以地球半径得到公里。
    """

    coordinates_rad: np.ndarray  # (n, 2) radians, (lat, lon)
    depth: np.ndarray            # (n,) metres, positive = above sea level
    elevation: np.ndarray        # (n,) metres
    osm_counts: np.ndarray       # (n, len(OSM_COLUMNS))
    tree: Optional[BallTree] = None

    def __post_init__(self) -> None:
        """Build the haversine BallTree if none was passed in.

        Building takes O(n log n) and is done once per index (at training time
        and when the inference service starts). The tree is not saved to disk;
        it is rebuilt by :meth:`load`.
        若未提供则构建 BallTree（O(n log n)，每个索引只构建一次，不写入磁盘）。
        """
        if self.tree is None:
            self.tree = BallTree(self.coordinates_rad, metric="haversine")

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> "ReferenceIndex":
        """Build an index from a frame of measured points.

        The ``class_code`` column, if present, is ignored, which is what makes
        the index label-free. ``noaa_depth`` and ``elevation`` describe the
        same relief surface, so a gap in one is filled from the other; points
        with neither value are dropped because they add no information.

        Parameters
        ----------
        frame:
            Points with ``latitude``, ``longitude`` (degrees) and optionally
            ``noaa_depth``, ``elevation`` and the OSM count columns. At
            training time this is the training split only.

        Returns
        -------
        ReferenceIndex
            Index over the points that have a depth or elevation value.

        从 DataFrame 构建索引：忽略类别标签；水深与高程互相补缺；两者皆缺的点被丢弃。
        """
        # (n, 2) degrees -> radians, as required by the haversine BallTree.
        # 度 -> 弧度（haversine BallTree 的要求）。
        coords = np.radians(frame[["latitude", "longitude"]].to_numpy(dtype=float))
        depth = pd.to_numeric(frame.get("noaa_depth"), errors="coerce").to_numpy(dtype=float)
        elevation = pd.to_numeric(frame.get("elevation"), errors="coerce").to_numpy(dtype=float)
        # Fill gaps in one source from the other; both encode the same surface.
        # 两个数据源描述同一地表，互相填补缺失值。
        depth = np.where(np.isnan(depth), elevation, depth)
        elevation = np.where(np.isnan(elevation), depth, elevation)
        # (n, 9) matrix of OSM counts; a missing column becomes all zeros.
        # (n, 9) OSM 计数矩阵；缺失列视为 0。
        osm = np.stack(
            [pd.to_numeric(frame.get(c, 0.0), errors="coerce").fillna(0.0).to_numpy(float)
             for c in OSM_COLUMNS],
            axis=1,
        )
        # Keep only points that have some relief value. 仅保留有高程/水深的点。
        keep = ~np.isnan(depth)
        return cls(coords[keep], depth[keep], np.nan_to_num(elevation[keep]), osm[keep])

    # -- persistence -------------------------------------------------------- #
    def save(self, path: Path | str) -> None:
        """Write the index arrays to a compressed ``.npz`` file.

        Measurements are stored as float32 (half the size; centimetre
        precision is irrelevant for relief values in metres). The BallTree is
        not stored and is rebuilt on :meth:`load`.

        Parameters
        ----------
        path:
            Target file, normally :data:`~maritime_route.config.BATHY_INDEX_PATH`.

        保存为压缩 npz：测量值用 float32，坐标保持 float64；BallTree 不保存。
        """
        np.savez_compressed(
            path,
            # Coordinates keep full precision: rounding them to float32
            # would displace a point from itself by up to ~1.5 m.
            # 坐标保持双精度：float32 会使点与自身产生最多约 1.5 m 的偏差。
            coordinates_rad=self.coordinates_rad.astype(np.float64),
            depth=self.depth.astype(np.float32),
            elevation=self.elevation.astype(np.float32),
            osm_counts=self.osm_counts.astype(np.float32),
        )

    @classmethod
    def load(cls, path: Path | str) -> "ReferenceIndex":
        """Read an index written by :meth:`save` and rebuild its BallTree.

        All arrays are converted to float64 so that the arithmetic in
        :func:`build_features` is identical to the one done at training time.

        Parameters
        ----------
        path:
            ``.npz`` file written by :meth:`save`.

        Returns
        -------
        ReferenceIndex
            The loaded index.

        读取索引并重建 BallTree；统一转换为 float64 以保证与训练时计算一致。
        """
        blob = np.load(path)
        return cls(
            coordinates_rad=blob["coordinates_rad"].astype(np.float64),
            depth=blob["depth"].astype(np.float64),
            elevation=blob["elevation"].astype(np.float64),
            osm_counts=blob["osm_counts"].astype(np.float64),
        )

    # -- queries ------------------------------------------------------------ #
    def query(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        k: int = BATHY_NEIGHBOURS,
        exclude_self: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(distance_km, neighbour_index)`` arrays of shape (n, k).

        The query point itself is always excluded, and the decision is taken by
        distance rather than by a flag. This matters because the index stores
        coordinates as float32: a point that *is* in the index matches itself at
        a distance of roughly half a metre rather than exactly zero, so an
        equality test would miss it. Dropping by distance makes the features of
        a point identical whether or not that point happens to be part of the
        index, which is what keeps training and inference consistent.

        查询点自身总是被剔除，且依据距离而非标志位判断：索引以 float32 存储
        坐标，点与自身的距离约为 0.5 m 而非严格为 0，用相等判断会漏掉。
        按距离剔除可保证"点是否在索引中"不影响其特征，从而使训练与推理一致。

        ``exclude_self`` is retained for backward compatibility and no longer
        changes the result.

        Algorithm: ask the BallTree for ``k + 1`` neighbours. If the nearest
        one is within :data:`SELF_MATCH_EPS_KM` it is the query point itself
        and is dropped; otherwise the farthest of the ``k + 1`` is dropped.
        Either way exactly ``k`` neighbours remain, sorted by distance.

        Parameters
        ----------
        lat, lon:
            Query coordinates, degrees, arrays of length n.
        k:
            Number of neighbours wanted. Clipped to [1, number of indexed
            points].
        exclude_self:
            Ignored (see above).

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            ``distance_km`` - shape (n, k), float64, great-circle distance in
            km, ascending along axis 1; ``neighbour_index`` - shape (n, k),
            int, row indices into ``depth``/``elevation``/``osm_counts``.
        """
        # (n, 2) radians, columns (lat, lon) - the BallTree's input convention.
        # (n, 2) 弧度 (纬度, 经度)，BallTree 的输入格式。
        query_rad = np.radians(np.stack([np.asarray(lat, float), np.asarray(lon, float)], axis=1))
        n_samples = len(self.depth)
        # Cannot ask for more neighbours than there are points in the index.
        # 请求的邻居数不能超过索引中的点数。
        k = max(1, min(k, n_samples))
        # One spare neighbour, so that the point itself can be discarded.
        # 多取一个邻居，以便剔除查询点自身。
        k_eff = min(k + 1, n_samples)
        dist, idx = self.tree.query(query_rad, k=k_eff)
        # Haversine distances are central angles (radians); arc length = R * angle.
        # haversine 距离为圆心角（弧度），弧长 = 地球半径 × 角度。
        dist = dist * EARTH_RADIUS_KM

        if k_eff > k:
            # Drop the first neighbour when it is the query point itself,
            # otherwise drop the farthest one; either way exactly k remain.
            # 若最近邻即自身则去掉第一个，否则去掉最远的一个，始终保留 k 个。
            # is_self has shape (n, 1) so it broadcasts over the k columns.
            is_self = (dist[:, 0] <= SELF_MATCH_EPS_KM)[:, None]
            dist = np.where(is_self, dist[:, 1:], dist[:, :-1])
            idx = np.where(is_self, idx[:, 1:], idx[:, :-1])
        return dist, idx

    def sample_depth(self, lat: np.ndarray, lon: np.ndarray, k: int = 4) -> np.ndarray:
        """Inverse-distance weighted depth at arbitrary coordinates.

        反距离加权插值，用于为航线网格的任意单元估计水深。

        Inverse-distance weighting (IDW) with power 2::

            w_i = 1 / max(d_i, 0.001 km)^2
            depth = sum(w_i * depth_i) / sum(w_i)

        The 1 m floor on the distance prevents division by zero when a query
        coincides with a reference point. Power 2 makes the nearest
        neighbours dominate, which suits a surface that changes quickly at the
        coast.

        Parameters
        ----------
        lat, lon:
            Query coordinates, degrees, length n.
        k:
            Number of neighbours used for the interpolation.

        Returns
        -------
        numpy.ndarray
            ``(n,)`` interpolated relief, metres (positive above sea level).
        """
        dist_km, idx = self.query(lat, lon, k=k)
        weights = 1.0 / np.maximum(dist_km, 1e-3) ** 2   # (n, k), 1/km^2
        values = self.depth[idx]                          # (n, k), metres
        return np.sum(values * weights, axis=1) / np.sum(weights, axis=1)


# --------------------------------------------------------------------------- #
# Feature matrix                                                              #
# --------------------------------------------------------------------------- #
def build_features(
    frame: pd.DataFrame,
    index: ReferenceIndex,
    exclude_self: bool = False,
) -> np.ndarray:
    """Compute the ``(n, N_FEATURES)`` design matrix for a frame of points.

    构建特征矩阵。``exclude_self=True`` 用于训练集，避免点查询到自身。
    (Since :meth:`ReferenceIndex.query` now always removes a self-match by
    distance, the flag is passed through but no longer changes the result.)
    （目前 query 总是按距离剔除自身，该参数已不影响结果。）

    Feature groups (see :data:`FEATURE_NAMES` for the full table):

    * **Geodetic** - latitude and longitude are encoded as sin/cos pairs. A raw
      longitude jumps from +180 to -180 at the antimeridian although the two
      points are neighbours; (sin, cos) places the angle on the unit circle,
      so neighbours stay close. ``|lat| / 90`` adds a climate-zone proxy.
    * **Point measurements** - the point's own depth and elevation (signed
      log) plus 0/1 flags that say whether each value was measured or
      imputed, so the network can learn how much to trust it.
    * **Neighbourhood relief** - statistics of the depths of the
      ``BATHY_NEIGHBOURS`` nearest reference points: mean/min/max (signed
      log), std and range (log1p, both >= 0), the share of land neighbours,
      whether the point's sign agrees with the neighbourhood, the log
      distance to the nearest neighbour and a slope proxy (relief range
      divided by mean neighbour distance, metres per km).
    * **OSM context** - grouped OSM counts of the point and the mean over its
      neighbours, compressed with log1p because counts are heavy-tailed.

    Missing values: a missing depth is replaced by the neighbourhood mean, a
    missing elevation by the (filled) depth; any remaining NaN/inf in the
    final matrix is set to 0.

    Parameters
    ----------
    frame:
        Points with ``latitude``/``longitude`` (degrees) and optionally
        ``noaa_depth``, ``elevation``, ``noaa_type`` and OSM count columns.
    index:
        Reference index built from the training split.
    exclude_self:
        Passed to :meth:`ReferenceIndex.query`; has no effect any more.

    Returns
    -------
    numpy.ndarray
        Shape ``(n, 28)``, dtype float32, columns in :data:`FEATURE_NAMES`
        order. Not standardised (the scaler is applied by the model code).
    """
    lat = frame["latitude"].to_numpy(dtype=float)    # (n,) degrees
    lon = frame["longitude"].to_numpy(dtype=float)   # (n,) degrees
    n = len(lat)

    lat_rad, lon_rad = np.radians(lat), np.radians(lon)

    # Point measurements; a missing column is replaced by an all-NaN series.
    # 点自身的测量值；缺失列以全 NaN 序列代替。
    depth_raw = pd.to_numeric(frame.get("noaa_depth", pd.Series(np.nan, index=frame.index)),
                              errors="coerce").to_numpy(dtype=float)
    elev_raw = pd.to_numeric(frame.get("elevation", pd.Series(np.nan, index=frame.index)),
                             errors="coerce").to_numpy(dtype=float)

    # Missing-value indicators (0.0 / 1.0), computed before imputation.
    # 缺失指示变量（0/1），在填补之前计算。
    depth_missing = np.isnan(depth_raw).astype(float)
    elev_missing = np.isnan(elev_raw).astype(float)

    # 1.0 when the NOAA sample is a land height rather than a sea depth.
    # NOAA 样本为陆地高程时取 1。
    noaa_type = frame.get("noaa_type")
    if noaa_type is None:
        is_height = np.zeros(n)
    else:
        # astype("string") gives a nullable string column so None/NaN survive
        # .str.lower(); fillna(False) turns those into "not a height sample".
        # 转为可空字符串类型，None/NaN 最终视为非高程样本。
        is_height = (noaa_type.astype("string").str.lower() == "height").fillna(False).to_numpy(float)

    # Contextual statistics from the reference index.
    # 来自参考索引的邻域统计量。
    dist_km, idx = index.query(lat, lon, k=BATHY_NEIGHBOURS, exclude_self=exclude_self)
    ctx_depth = index.depth[idx]                       # (n, k)
    ctx_osm = index.osm_counts[idx]                    # (n, k, 9)

    # Relief statistics over the k neighbours (axis 1), metres. 邻域高程统计（米）。
    ctx_mean = ctx_depth.mean(axis=1)
    ctx_std = ctx_depth.std(axis=1)
    ctx_min = ctx_depth.min(axis=1)
    ctx_max = ctx_depth.max(axis=1)
    # Share of neighbours above sea level, 0..1. 邻居中陆地点的比例。
    ctx_land_fraction = (ctx_depth > 0.0).mean(axis=1)
    nearest_km = dist_km[:, 0]
    # Mean neighbour distance (km), floored at 1 m to avoid division by zero.
    # 邻居平均距离（km），下限 1 m 以避免除零。
    span_km = np.maximum(dist_km.mean(axis=1), 1e-3)
    # Slope proxy: relief range over the neighbourhood size, metres per km.
    # Steep values indicate a shelf edge or a cliff coast.
    # 坡度近似：邻域高差 / 邻域尺度（米/公里），大值对应陆架边缘或陡峭海岸。
    ctx_gradient = (ctx_max - ctx_min) / span_km

    # Missing point measurements fall back to the interpolated context value.
    # 缺失的点测量值用邻域均值代替；缺失高程再用已填补的水深代替。
    depth_filled = np.where(np.isnan(depth_raw), ctx_mean, depth_raw)
    elev_filled = np.where(np.isnan(elev_raw), depth_filled, elev_raw)
    # 1.0 when the point and its neighbourhood are on the same side of sea
    # level; 0.0 marks points at a land/water transition (np.sign(0) = 0).
    # 点与邻域位于海平面同侧时为 1；为 0 表示处于水陆交界。
    sign_agreement = (np.sign(depth_filled) == np.sign(ctx_mean)).astype(float)

    # OSM aggregation: the point's own counts plus the neighbourhood.
    # OSM 聚合：点自身计数 + 邻域均值。
    own_osm = np.stack(
        [pd.to_numeric(frame.get(c, 0.0), errors="coerce").fillna(0.0).to_numpy(float)
         for c in OSM_COLUMNS],
        axis=1,
    )                                                  # (n, 9)
    col = {name: i for i, name in enumerate(OSM_COLUMNS)}  # OSM column name -> position

    def group(mat: np.ndarray, names: Sequence[str], axis_sum: bool = False) -> np.ndarray:
        """Sum the OSM columns listed in ``names`` along the last axis.

        Works for both the per-point matrix ``(n, 9) -> (n,)`` and the
        neighbourhood tensor ``(n, k, 9) -> (n, k)`` because the columns are
        selected with ``...`` (ellipsis) indexing.

        Parameters
        ----------
        mat:
            Array whose last axis follows :data:`OSM_COLUMNS`.
        names:
            OSM column names to add up.
        axis_sum:
            Unused; kept for signature compatibility.

        Returns
        -------
        numpy.ndarray
            ``mat`` with the last axis replaced by the group total.

        按组对 OSM 列求和（最后一个轴），同时适用于 (n, 9) 与 (n, k, 9)。
        """
        cols = [col[nm] for nm in names]
        sub = mat[..., cols]
        return sub.sum(axis=-1)

    # log1p compresses the heavy-tailed counts (0 stays 0; 100 -> 4.6).
    # log1p 压缩长尾计数分布（0 仍为 0）。
    own_marine = np.log1p(group(own_osm, MARINE_INFRA))
    own_natural = np.log1p(group(own_osm, NATURAL_COASTAL))
    own_inland = np.log1p(group(own_osm, INLAND_WATER))
    own_industrial = np.log1p(own_osm[:, col["osm_industrial"]])

    # Group totals per neighbour (n, k), averaged over the k neighbours -> (n,).
    # 每个邻居的组合计 (n, k)，再对 k 个邻居取平均 -> (n,)。
    ctx_marine = np.log1p(group(ctx_osm, MARINE_INFRA).mean(axis=1))
    ctx_natural = np.log1p(group(ctx_osm, NATURAL_COASTAL).mean(axis=1))
    ctx_inland = np.log1p(group(ctx_osm, INLAND_WATER).mean(axis=1))
    # All nine OSM types summed per neighbour, then averaged. 全部 OSM 类型的邻域平均总数。
    ctx_osm_total = np.log1p(ctx_osm.sum(axis=2).mean(axis=1))
    # Objects per km of neighbourhood size; the denominator is at least
    # 0.1 * OSM_RADIUS_KM = 1.2 km so that very dense sampling does not
    # produce extreme densities.
    # 每公里对象数；分母至少为 1.2 km，避免密集采样导致极端值。
    ctx_osm_density = ctx_osm.sum(axis=2).mean(axis=1) / np.maximum(span_km, OSM_RADIUS_KM * 0.1)

    # Assemble the 28 columns in FEATURE_NAMES order -> (n, 28).
    # 按 FEATURE_NAMES 顺序拼接 28 列。
    matrix = np.stack(
        [
            np.sin(lat_rad),                 # 0  lat_sin
            np.cos(lat_rad),                 # 1  lat_cos
            np.sin(lon_rad),                 # 2  lon_sin
            np.cos(lon_rad),                 # 3  lon_cos
            np.abs(lat) / 90.0,              # 4  abs_lat_norm, 0 at equator .. 1 at pole
            signed_log(depth_filled),        # 5  depth_signed_log
            depth_missing,                   # 6  depth_is_missing
            is_height,                       # 7  is_height_sample
            signed_log(elev_filled),         # 8  elevation_signed_log
            elev_missing,                    # 9  elevation_is_missing
            signed_log(ctx_mean),            # 10 ctx_depth_mean
            np.log1p(ctx_std),               # 11 ctx_depth_std (std >= 0, plain log1p)
            signed_log(ctx_min),             # 12 ctx_depth_min
            signed_log(ctx_max),             # 13 ctx_depth_max
            np.log1p(ctx_max - ctx_min),     # 14 ctx_depth_range (>= 0)
            ctx_land_fraction,               # 15 ctx_land_fraction
            sign_agreement,                  # 16 ctx_sign_agreement
            np.log1p(nearest_km),            # 17 ctx_distance_km_log
            np.log1p(np.abs(ctx_gradient)) * np.sign(ctx_gradient),  # 18 ctx_gradient (signed log)
            own_marine,                      # 19 osm_marine_infra
            ctx_marine,                      # 20 ctx_osm_marine_infra
            own_natural,                     # 21 osm_natural_coastal
            ctx_natural,                     # 22 ctx_osm_natural_coastal
            own_inland,                      # 23 osm_inland_water
            ctx_inland,                      # 24 ctx_osm_inland_water
            own_industrial,                  # 25 osm_industrial_log
            ctx_osm_total,                   # 26 ctx_osm_total
            np.log1p(ctx_osm_density),       # 27 ctx_osm_density
        ],
        axis=1,
    )
    # Safety net: any NaN/inf left (e.g. from an empty or corrupt input value)
    # becomes 0; float32 halves memory and matches the PyTorch model dtype.
    # 兜底：残留的 NaN/inf 置 0；float32 节省内存并与 PyTorch 模型精度一致。
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def features_for_coordinates(
    lat: Sequence[float],
    lon: Sequence[float],
    index: ReferenceIndex,
) -> np.ndarray:
    """Build features for bare coordinates (no measured attributes available).

    仅有经纬度时的特征构建：全部环境属性来自参考索引。

    Used for the cells of the routing grid, which have only a position. The
    missing point attributes are synthesised from the 4 nearest reference
    points before calling :func:`build_features`:

    * ``noaa_depth`` and ``elevation`` - inverse-distance-weighted (power 2)
      average of the neighbours, as in :meth:`ReferenceIndex.sample_depth`;
    * ``noaa_type`` - ``"height"`` if the interpolated value is above sea
      level, else ``"depth"``;
    * OSM counts - copied from the single nearest neighbour (counts cannot be
      meaningfully interpolated).

    Because the imputed values are marked as measured (the ``*_is_missing``
    flags are 0), a grid cell looks to the network like an ordinary sample at
    that location.

    Parameters
    ----------
    lat, lon:
        Coordinates, degrees, length n.
    index:
        Reference index.

    Returns
    -------
    numpy.ndarray
        ``(n, 28)`` float32 feature matrix.
    """
    frame = pd.DataFrame({"latitude": np.asarray(lat, float), "longitude": np.asarray(lon, float)})
    dist_km, idx = index.query(frame["latitude"].to_numpy(), frame["longitude"].to_numpy(), k=4)
    # IDW weights 1/d^2 with a 1 m distance floor, shape (n, 4).
    # 反距离平方权重，距离下限 1 m，形状 (n, 4)。
    weights = 1.0 / np.maximum(dist_km, 1e-3) ** 2
    frame["noaa_depth"] = np.sum(index.depth[idx] * weights, axis=1) / np.sum(weights, axis=1)
    frame["elevation"] = np.sum(index.elevation[idx] * weights, axis=1) / np.sum(weights, axis=1)
    frame["noaa_type"] = np.where(frame["noaa_depth"] > 0, "height", "depth")
    # OSM counts of the nearest neighbour (column 0 of idx), shape (n, 9).
    # 取最近邻（idx 第 0 列）的 OSM 计数。
    nearest_osm = index.osm_counts[idx[:, 0]]
    for i, column in enumerate(OSM_COLUMNS):
        frame[column] = nearest_osm[:, i]
    return build_features(frame, index, exclude_self=False)
