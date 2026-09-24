"""Maritime Route Planner — neural-network based maritime route planning.

海上航线规划应用包。
The package is organised in layers: :mod:`~maritime_route.data` (loading,
validation, feature engineering), :mod:`~maritime_route.model` (the zone
classifier), :mod:`~maritime_route.routing` (cost map and graph search),
:mod:`~maritime_route.storage` (persistence), :mod:`~maritime_route.export`
(GeoJSON/CSV/PDF) and :mod:`~maritime_route.web` (HTTP and WebSocket service).

Data flow through the layers
----------------------------
1. ``data``: a labelled GeoJSON corpus is loaded, validated, cleaned of
   mislabelled points and turned into a 28-column feature matrix.
2. ``model``: a residual multilayer perceptron (PyTorch) is trained on that
   matrix to predict one of the four zone classes of
   :data:`~maritime_route.config.CLASS_NAMES`.
3. ``routing``: the trained classifier labels every cell of a latitude/longitude
   grid; zone labels are converted into traversal costs and A*, Dijkstra or a
   genetic algorithm searches the cheapest path between two ports.
4. ``storage`` / ``export`` / ``web``: routes are saved, exported and served to
   the browser.

数据流：data（加载/清洗/特征）-> model（区域分类网络）-> routing（代价栅格与路径搜索）
-> storage / export / web（持久化、导出与网页服务）。

Only the package-level metadata is re-exported here; importing the package
does not load PyTorch or the trained model.
包级别只导出元数据，导入本包不会加载 PyTorch 或已训练模型。
"""
# Version and author come from config so there is one single source of truth.
# 版本号与作者统一来自 config，避免多处重复定义。
from .config import APP_AUTHOR, APP_VERSION, CLASS_NAMES

#: Package version string (PEP 396 convention), e.g. "1.0.0". 包版本号。
__version__ = APP_VERSION
#: Author of the application. 作者。
__author__ = APP_AUTHOR

# Public names exported by ``from maritime_route import *``. 公开导出的名称。
__all__ = ["__version__", "__author__", "CLASS_NAMES"]
