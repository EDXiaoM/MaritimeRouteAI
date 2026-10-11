"""Cleaning of the labelled corpus before training.

The labelled corpus contains points whose class contradicts the physical
measurements stored with them. The most frequent case, pointed out by the
project supervisor, is a ``COASTAL_SEA`` (sea) point that lies several metres
*above* sea level; the symmetric case is a land point (``NEAR_COAST`` or
``COASTLINE``) that lies below the sea surface. Such points teach the network
something that is physically impossible, so they are removed before the
dataset is split and the model is trained.

Every point carries two independent height measurements, both in metres,
positive above sea level and negative below it:

* ``noaa_depth`` - the NOAA/ETOPO relief value;
* ``elevation`` - the terrain elevation recorded by the data collector.

Each measurement is checked separately and a point is removed if *either* of
them contradicts the class. A tolerance
(:data:`~maritime_route.config.LABEL_CHECK_TOLERANCE_M`, 5 m by default)
absorbs tides and the vertical error of the relief models, so that a beach
point at +1 m labelled as coastal sea is *not* flagged.

======================  =============================================
Rule                     Condition (for each measurement h)
======================  =============================================
``sea_above_level``      class OPEN_SEA or COASTAL_SEA and h > +tol
``land_below_level``     class NEAR_COAST or COASTLINE and h < -tol
======================  =============================================

With the 5 m tolerance the rule removes 5 427 of the 156 838 points of the
corpus delivered with the Aegean Sea extension and leaves 151 411 points,
exactly the number of "clean" points stated by the supervisor for that
corpus. Using ``noaa_depth`` alone would leave 332 more points.

A missing measurement is simply not checked; a point without any
measurement is kept, and the feature extractor marks the missing value
explicitly.

Main objects
------------
:func:`remove_label_conflicts`
    Entry point called by :func:`maritime_route.data.dataset.prepare`.
:func:`conflict_mask`
    The rule itself, as a boolean mask (also used for the sensitivity table).
:func:`assign_regions`
    Maps points to the eleven sea regions of
    :data:`~maritime_route.config.REGIONS`; also used for per-region
    statistics and evaluation.
:class:`CleaningReport`
    Counts written to ``dataset_statistics.json``.

数据清洗：删除类别与实测高程/水深相矛盾的样本（例如位于海平面以上数米的
"近岸海"点，或位于海平面以下的陆地点）。noaa_depth 与 elevation 两个测量值分别检查，
任意一个矛盾即删除；容差 5 米用于吸收潮汐和高程模型误差。新语料清洗后剩 151 411 点，
与指导老师给出的数字一致。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from ..config import (
    CLASS_NAMES,
    LABEL_CHECK_TOLERANCE_M,
    LAND_CLASSES,
    REGIONS,
    WATER_CLASSES,
)

#: Name used for points that fall outside every region box. 不属于任何海域的点。
OTHER_REGION = "Other (inland samples)"


@dataclass
class CleaningReport:
    """Outcome of :func:`remove_label_conflicts`.

    Attributes
    ----------
    n_before, n_after:
        Number of points before and after cleaning.
    tolerance_m:
        Tolerance that was applied, metres.
    removed_by_class:
        Number of removed points per class (the rule that fired is implied by
        the class: a water class means "above sea level", a land class means
        "below sea level").
    removed_by_region:
        Number of removed points per geographic region.
    removed_by_measure:
        Which measurement revealed the conflict: ``noaa_depth_only``,
        ``elevation_only`` or ``both``.
    sensitivity:
        How many points *would* be removed at other tolerances; reported in
        the thesis to justify the chosen value.

    清洗报告：清洗前后点数、按类别与海域统计的删除数量，以及不同容差下的敏感性分析。
    """

    n_before: int
    n_after: int
    tolerance_m: float
    removed_by_class: Dict[str, int] = field(default_factory=dict)
    removed_by_region: Dict[str, int] = field(default_factory=dict)
    removed_by_measure: Dict[str, int] = field(default_factory=dict)
    sensitivity: Dict[str, int] = field(default_factory=dict)

    @property
    def n_removed(self) -> int:
        """Total number of removed points. 删除总数。"""
        return self.n_before - self.n_after

    def to_dict(self) -> Dict:
        """JSON-serialisable form, stored in ``dataset_statistics.json``.

        Returns
        -------
        dict
            All fields plus the derived ``n_removed`` and ``share_removed``
            (fraction 0..1 of the input that was removed, 6 decimals).

        转换为可 JSON 序列化的字典，附加删除总数与删除比例。
        """
        return {
            "n_before": self.n_before,
            "n_after": self.n_after,
            "n_removed": self.n_removed,
            # max(.., 1) avoids division by zero for an empty corpus. 防止空语料除零。
            "share_removed": round(self.n_removed / max(self.n_before, 1), 6),
            "tolerance_m": self.tolerance_m,
            "removed_by_class": self.removed_by_class,
            "removed_by_region": self.removed_by_region,
            "removed_by_measure": self.removed_by_measure,
            "sensitivity": self.sensitivity,
        }


def assign_regions(latitude: np.ndarray, longitude: np.ndarray) -> np.ndarray:
    """Return the region name of every point.

    The boxes of :data:`~maritime_route.config.REGIONS` are tested in order and
    the first box that contains the point wins, so overlapping boxes are
    resolved deterministically.

    Parameters
    ----------
    latitude, longitude:
        Arrays of equal length, degrees.

    Returns
    -------
    numpy.ndarray of str
        One region name per point, :data:`OTHER_REGION` where no box matches.

    为每个点分配所属海域：按顺序检查各经纬度范围，先匹配者优先。
    """
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    # dtype=object stores Python strings of any length (a fixed-width NumPy
    # string dtype would truncate the longer region names).
    # 使用 object 类型存放任意长度的字符串（定长字符串会截断较长的海域名）。
    out = np.full(lat.shape, OTHER_REGION, dtype=object)
    # Points not yet claimed by an earlier box. 尚未被前面范围匹配的点。
    unassigned = np.ones(lat.shape, dtype=bool)
    for name, (lat_min, lat_max, lon_min, lon_max) in REGIONS:
        # Half-open box [min, max): a point on a shared border belongs to one box only.
        # 半开区间 [min, max)：位于公共边界上的点只属于一个范围。
        inside = (lat >= lat_min) & (lat < lat_max) & (lon >= lon_min) & (lon < lon_max)
        hit = inside & unassigned
        out[hit] = name
        unassigned &= ~hit
    return out


#: Height measurements checked against the class, in this order.
#: 与类别对照检查的两个高度测量值。
MEASUREMENTS: Tuple[str, ...] = ("noaa_depth", "elevation")


def measurement_conflicts(frame: pd.DataFrame, column: str,
                          tolerance_m: float = LABEL_CHECK_TOLERANCE_M) -> np.ndarray:
    """Boolean mask of points whose class contradicts one height measurement.

    Parameters
    ----------
    frame:
        Validated corpus with a ``class_code`` column.
    column:
        Name of the measurement column (``noaa_depth`` or ``elevation``);
        metres, positive above sea level. A missing column or value never
        produces a conflict.
    tolerance_m:
        Allowed deviation from sea level in metres.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where the measurement contradicts the class.

    单个测量值与类别的矛盾掩码；缺失值不参与判断。
    """
    if column in frame:
        height = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    else:
        height = np.full(len(frame), np.nan)
    cls = frame["class_code"].astype(str).to_numpy()
    # NaN compares False with everything, so a missing value is never flagged.
    # NaN 与任何数比较均为 False，因此缺失值不会被标记。
    with np.errstate(invalid="ignore"):
        sea_above = np.isin(cls, WATER_CLASSES) & (height > tolerance_m)
        land_below = np.isin(cls, LAND_CLASSES) & (height < -tolerance_m)
    return sea_above | land_below


def conflict_mask(frame: pd.DataFrame, tolerance_m: float = LABEL_CHECK_TOLERANCE_M) -> np.ndarray:
    """Boolean mask of points whose class contradicts their height/depth.

    A point is flagged when *either* measurement of :data:`MEASUREMENTS`
    contradicts its class (see :func:`measurement_conflicts`).

    Parameters
    ----------
    frame:
        Validated corpus with ``class_code`` and the measurement columns.
    tolerance_m:
        Allowed deviation from sea level in metres.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` for every point that should be removed.

    返回类别与高程/水深矛盾的样本掩码（任一测量值矛盾即为 True，表示应删除）。
    """
    mask = np.zeros(len(frame), dtype=bool)
    for column in MEASUREMENTS:
        mask |= measurement_conflicts(frame, column, tolerance_m)
    return mask


def remove_label_conflicts(
    frame: pd.DataFrame,
    tolerance_m: float = LABEL_CHECK_TOLERANCE_M,
) -> Tuple[pd.DataFrame, CleaningReport]:
    """Drop mislabelled points and describe what was dropped.

    Besides the counts per class and per region, the report contains a
    sensitivity table: the number of points that :func:`conflict_mask` would
    flag at tolerances of 0, 2, 5, 10 and 20 m. It shows how strongly the
    result depends on the chosen tolerance.

    Parameters
    ----------
    frame:
        Validated corpus (output of :func:`~maritime_route.data.ais_loader.validate`).
    tolerance_m:
        Tolerance of the check, metres.

    Returns
    -------
    (pandas.DataFrame, CleaningReport)
        The cleaned frame (index reset) and the report.

    删除标注错误的样本并返回清洗报告。
    """
    by_depth = measurement_conflicts(frame, "noaa_depth", tolerance_m)
    by_elevation = measurement_conflicts(frame, "elevation", tolerance_m)
    mask = by_depth | by_elevation
    regions = assign_regions(frame["latitude"].to_numpy(), frame["longitude"].to_numpy())

    # Counts of the removed points by class and by region. 按类别与海域统计删除数。
    removed_cls = frame.loc[mask, "class_code"].value_counts()
    removed_reg = pd.Series(regions[mask]).value_counts()
    # int(...) converts NumPy integers to plain int so the report is JSON-serialisable.
    # 转换为 Python int，便于 JSON 序列化。
    report = CleaningReport(
        n_before=int(len(frame)),
        n_after=int((~mask).sum()),
        tolerance_m=float(tolerance_m),
        removed_by_class={c: int(removed_cls.get(c, 0)) for c in CLASS_NAMES},
        removed_by_region={str(k): int(v) for k, v in removed_reg.items()},
        removed_by_measure={
            "noaa_depth_only": int((by_depth & ~by_elevation).sum()),
            "elevation_only": int((by_elevation & ~by_depth).sum()),
            "both": int((by_depth & by_elevation).sum()),
        },
        sensitivity={f"{t:g} m": int(conflict_mask(frame, t).sum()) for t in (0.0, 2.0, 5.0, 10.0, 20.0)},
    )
    cleaned = frame.loc[~mask].reset_index(drop=True)
    return cleaned, report
