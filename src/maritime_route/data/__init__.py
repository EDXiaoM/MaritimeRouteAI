"""Data access layer: loading, validation and feature engineering.

数据访问层：加载、校验与特征工程。

Modules
-------
:mod:`~maritime_route.data.ais_loader`
    Reads GeoJSON / CSV / JSON point files into a :class:`pandas.DataFrame`,
    normalises column names and drops rows with invalid coordinates.
:mod:`~maritime_route.data.cleaning`
    Removes labelled points whose class contradicts their measured
    height/depth, and assigns every point to one of the ten sea regions.
:mod:`~maritime_route.data.features`
    Builds the label-free :class:`ReferenceIndex` and the 28-column feature
    matrix that is the input of the neural network.
:mod:`~maritime_route.data.dataset`
    Runs the whole preparation pipeline (load -> validate -> clean -> split ->
    index -> features) and saves the train/validation/test matrices.

Only the names used by other layers (web service, model, routing) are
re-exported here; ``cleaning`` and ``dataset`` are imported explicitly by the
scripts that need them.

子模块：ais_loader（读取与校验）、cleaning（清洗与海域划分）、features（参考索引
与 28 维特征）、dataset（完整预处理流程）。此处仅导出其他层需要的名称。
"""
# Re-export the public API so callers can write ``from maritime_route.data import ...``.
# 重新导出公共接口，便于上层直接从 maritime_route.data 导入。
from .ais_loader import DataLoadError, LoadReport, load_any, load_geojson, validate
from .features import FEATURE_NAMES, N_FEATURES, ReferenceIndex, build_features, features_for_coordinates

__all__ = [
    "DataLoadError", "LoadReport", "load_any", "load_geojson", "validate",
    "FEATURE_NAMES", "N_FEATURES", "ReferenceIndex", "build_features",
    "features_for_coordinates",
]
