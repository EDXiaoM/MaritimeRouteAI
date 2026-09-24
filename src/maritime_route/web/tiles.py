"""On-board chart tiles rendered from the model itself (offline mode).

离线海图瓦片：直接用神经网络对瓦片像素分类并渲染 PNG。
A shipboard planner cannot rely on an internet tile service, so the server can
render its own slippy-map tiles: every tile is a small lattice of coordinates
classified by the network and painted with the zone palette. Tiles are cached
in an LRU so panning stays responsive.

Place in the application
------------------------
Used only by the endpoint ``GET /api/tiles/zones/{z}/{x}/{y}.png`` in
:mod:`maritime_route.web.app`; the browser shows these tiles through a Leaflet
``L.tileLayer`` ("On-board chart" base layer) when no internet map is
available.

Main objects
------------
``tile_bounds(z, x, y)``
    Geographic extent of a Web-Mercator tile.
``render_tile(classifier, z, x, y)``
    Classify a ``SAMPLE x SAMPLE`` lattice inside the tile and return PNG bytes.
``cache_info()``
    Size of the in-memory tile cache.

Tile addressing (Web Mercator / XYZ)
------------------------------------
At zoom ``z`` the world between about 85.05 deg S and 85.05 deg N is split into
``n = 2**z`` columns and ``n`` rows of 256x256-pixel tiles; ``x`` grows
eastwards from 180 deg W and ``y`` grows southwards from the northern edge.
Longitude is linear in ``x``::

    lon = x / n * 360 - 180

Latitude is not linear in ``y``; the inverse Mercator projection gives::

    lat = degrees(atan(sinh(pi * (1 - 2 * y / n))))

For fractional ``y`` the same formula gives the latitude of any pixel row.
瓦片编号规则：经度与 x 成线性关系，纬度需通过反墨卡托公式由 y 计算。
"""
from __future__ import annotations

import io
import logging
import math
import threading
from collections import OrderedDict
from typing import Dict, Tuple

import numpy as np
from PIL import Image

from ..config import CLASS_COLOR_DARK, CLASS_NAMES

LOGGER = logging.getLogger(__name__)

#: Output tile edge in pixels (the Leaflet default). 瓦片边长（像素）。
TILE_SIZE = 256
#: Coordinates are classified on a coarse lattice and then upsampled.
#: 48 x 48 = 2304 model evaluations per tile instead of 65 536 per-pixel ones;
#: zone boundaries are smooth enough that bilinear upsampling hides the difference.
#: 每张瓦片仅分类 48×48 个点，再双线性放大到 256×256。
SAMPLE = 48
#: Maximum number of PNG tiles kept in memory (each is a few kilobytes).
#: 缓存瓦片数量上限。
CACHE_LIMIT = 900

#: Class index -> RGB colour. Built from the ``#rrggbb`` strings of the dark
#: theme: characters 1-2, 3-4 and 5-6 are parsed as hexadecimal R, G, B.
#: Shape (n_classes, 3), so ``_PALETTE[classes]`` maps a class grid to an image.
#: 调色板：把 "#rrggbb" 解析为 RGB，按类别索引直接查表上色。
_PALETTE = np.array(
    [[int(CLASS_COLOR_DARK[name][i:i + 2], 16) for i in (1, 3, 5)] for name in CLASS_NAMES],
    dtype=np.uint8,
)
#: Slightly darker shade used for the 1-pixel graticule inside each tile.
#: (Currently not used by ``render_tile``; kept for an optional grid overlay.)
#: （当前未被 render_tile 使用。）
_GRID_RGB = np.array([12, 20, 30], dtype=np.uint8)

#: LRU cache ``(z, x, y) -> PNG bytes``. An ``OrderedDict`` keeps usage order:
#: a hit is moved to the end, eviction removes from the front (least recent).
#: The lock is needed because tiles are rendered in several worker threads at
#: once (``asyncio.to_thread`` in the endpoint).
#: LRU 缓存；多个线程同时渲染瓦片，因此需要加锁。
_cache: "OrderedDict[Tuple[int, int, int], bytes]" = OrderedDict()
_cache_lock = threading.Lock()


def tile_bounds(z: int, x: int, y: int) -> Tuple[float, float, float, float]:
    """Web-Mercator tile bounds as (lat_north, lat_south, lon_west, lon_east).

    Parameters
    ----------
    z : int
        Zoom level; the tile matrix has ``2**z`` columns and rows.
    x : int
        Tile column, 0 at 180 deg W, increasing eastwards.
    y : int
        Tile row, 0 at the northern edge (about 85.05 deg N), increasing
        southwards.

    Returns
    -------
    tuple of float
        ``(lat_north, lat_south, lon_west, lon_east)`` in decimal degrees.
        The north edge is row ``y``, the south edge is row ``y + 1``.
    """
    n = 2.0 ** z
    # Longitude is a linear function of the column. 经度与列号线性相关。
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    # Inverse Mercator: y/n in [0, 1] maps to Mercator ordinate pi*(1 - 2y/n)
    # in [pi, -pi]; latitude = atan(sinh(ordinate)) (the Gudermannian function).
    # 反墨卡托投影：纬度 = atan(sinh(π(1 − 2y/n)))。
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lat_n, lat_s, lon_w, lon_e


def render_tile(classifier, z: int, x: int, y: int, opacity: int = 235) -> bytes:
    """Render one PNG tile of the zone classification. 渲染单张分类瓦片。

    The tile is covered by a ``SAMPLE x SAMPLE`` lattice of coordinates that
    are evenly spaced in *projected* (Mercator) space, so they line up with the
    pixels Leaflet will display. Each coordinate is classified by the neural
    network; the most probable class selects the colour and the probability of
    that class (confidence) controls how much the colour is faded to grey.
    The small image is then resized to ``TILE_SIZE`` with bilinear filtering
    and PNG-encoded. Results are cached per ``(z, x, y)``.

    Parameters
    ----------
    classifier : ZoneClassifierService
        Object with ``predict_coordinates(lat, lon) -> (N, n_classes)``
        probability array.
    z, x, y : int
        Tile address (validated by the calling endpoint).
    opacity : int, optional
        Alpha channel value 0..255 for every pixel (default 235), so the tile
        is slightly transparent over the page background.

    Returns
    -------
    bytes
        PNG image, ``TILE_SIZE x TILE_SIZE`` RGBA.

    Notes
    -----
    The cache key does not include ``opacity``; all callers use the default.
    Two threads that miss the cache for the same tile at the same moment both
    render it; the second result simply overwrites the first (the images are
    identical), which is cheaper than holding the lock during rendering.
    缓存未命中时不持锁渲染，两个线程可能重复渲染同一瓦片，但结果相同。
    """
    key = (z, x, y)
    # Fast path: return a cached tile and mark it as most recently used.
    # 命中缓存：直接返回并移到末尾（最近使用）。
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached

    lat_n, lat_s, lon_w, lon_e = tile_bounds(z, x, y)
    # Mercator is non-linear in latitude: sample in projected space, then map back.
    # ``rows`` are fractional tile-row offsets 0..1, i.e. global rows y..y+1;
    # the inverse-Mercator formula of tile_bounds turns each into a latitude.
    # 在投影空间中等间距取样（y 到 y+1），再用反墨卡托公式换算成纬度。
    rows = np.linspace(0.0, 1.0, SAMPLE)
    n = 2.0 ** z
    lat_grid = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * (y + rows) / n))))
    # Longitude is linear, so a plain linspace between the edges is exact.
    lon_grid = np.linspace(lon_w, lon_e, SAMPLE)
    # indexing="ij": first axis = image row (north -> south), second = column.
    # 第一维为图像行（北到南），第二维为列（西到东）。
    mesh_lat, mesh_lon = np.meshgrid(lat_grid, lon_grid, indexing="ij")

    probs = classifier.predict_coordinates(mesh_lat.ravel(), mesh_lon.ravel())
    classes = probs.argmax(axis=1).reshape(SAMPLE, SAMPLE).astype(np.uint8)
    confidence = probs.max(axis=1).reshape(SAMPLE, SAMPLE)

    # Palette lookup: (SAMPLE, SAMPLE) class indices -> (SAMPLE, SAMPLE, 3) RGB.
    rgb = _PALETTE[classes]
    # Low-confidence cells are blended towards grey so uncertainty is visible.
    # blend = 1 (pure class colour) for confidence >= 0.90, falls linearly to
    # 0.25 at confidence <= 0.5625; grey RGB(70, 84, 98) fills the remainder.
    # 置信度 ≥ 0.90 时为纯类别色，越低越接近灰色（最低保留 25% 类别色）。
    blend = np.clip((confidence - 0.45) / 0.45, 0.25, 1.0)[..., None]
    rgb = (rgb * blend + np.array([70, 84, 98]) * (1 - blend)).astype(np.uint8)

    # Append a constant alpha channel -> RGBA, then upsample to the tile size.
    # 添加统一的透明度通道，然后双线性放大到 256×256。
    alpha = np.full((SAMPLE, SAMPLE, 1), opacity, dtype=np.uint8)
    image = Image.fromarray(np.concatenate([rgb, alpha], axis=2), mode="RGBA")
    image = image.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    payload = buffer.getvalue()

    # Store and evict the least recently used tiles beyond CACHE_LIMIT.
    # 写入缓存；超过上限时从头部（最久未用）淘汰。
    with _cache_lock:
        _cache[key] = payload
        while len(_cache) > CACHE_LIMIT:
            _cache.popitem(last=False)
    return payload


def cache_info() -> Dict:
    """Return the current tile-cache occupancy. 返回瓦片缓存状态。

    Returns
    -------
    dict
        ``{"tiles_cached": int, "limit": int}``; served by
        ``GET /api/tiles/info``.
    """
    with _cache_lock:
        return {"tiles_cached": len(_cache), "limit": CACHE_LIMIT}
