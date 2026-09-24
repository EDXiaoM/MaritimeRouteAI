"""Dataset preparation: raw GeoJSON -> stratified train/val/test tensors.

数据集准备：原始 GeoJSON -> 分层划分的训练/验证/测试集。

This module runs the offline part of the machine-learning pipeline, invoked by
``scripts/prepare_dataset.py``:

1. load the labelled corpus (:func:`~maritime_route.data.ais_loader.load_geojson`);
2. validate coordinates (:func:`~maritime_route.data.ais_loader.validate`);
3. keep only rows with one of the four known class codes;
4. remove mislabelled points
   (:func:`~maritime_route.data.cleaning.remove_label_conflicts`);
5. split 70 / 15 / 15 into train / validation / test, stratified by class;
6. build the label-free :class:`~maritime_route.data.features.ReferenceIndex`
   from the *training* split only and save it;
7. compute the 28 features of every point and save the matrices, labels,
   test-point regions and corpus statistics.

Outputs
-------
:data:`PROCESSED_NPZ`
    ``x_train``/``x_val``/``x_test`` (float32, shape (n, 28)),
    ``y_train``/``y_val``/``y_test`` (int64 class indices), ``region_test``
    and ``feature_names``. Read back with :func:`load_prepared`.
:data:`STATS_JSON`
    Descriptive statistics of the corpus before and after cleaning, per
    region, the split sizes and the cleaning report (used in the thesis).
:data:`~maritime_route.config.BATHY_INDEX_PATH`
    The reference index needed at inference time.

流程：加载 -> 校验 -> 过滤未知类别 -> 清洗 -> 70/15/15 分层划分 -> 仅用训练集建立
参考索引 -> 提取特征并保存矩阵、标签与统计信息。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import (
    BATHY_INDEX_PATH,
    CLASS_NAMES,
    CLASS_TO_INDEX,
    PROCESSED_DIR,
    TRAINING,
)
from .ais_loader import load_geojson, validate
from .cleaning import assign_regions, remove_label_conflicts
from .features import FEATURE_NAMES, ReferenceIndex, build_features

LOGGER = logging.getLogger(__name__)

#: Feature matrices and labels of the three splits (compressed NumPy archive).
#: 三个数据子集的特征矩阵与标签。
PROCESSED_NPZ = PROCESSED_DIR / "zone_dataset.npz"
#: Corpus statistics written by :func:`prepare`. 语料统计信息。
STATS_JSON = PROCESSED_DIR / "dataset_statistics.json"


@dataclass
class SplitData:
    """Container for one train/val/test split. 数据划分容器。

    Attributes
    ----------
    x_train, x_val, x_test:
        Feature matrices, shape ``(n_split, 28)``, float32, *not*
        standardised.
    y_train, y_val, y_test:
        Class indices 0..3 (see :data:`~maritime_route.config.CLASS_NAMES`),
        shape ``(n_split,)``, int64.
    feature_names:
        Column names of the feature matrices, in order.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list

    @property
    def n_features(self) -> int:
        """Number of feature columns (28). 特征维数。"""
        return self.x_train.shape[1]

    def describe(self) -> Dict[str, int]:
        """Return the size of each split and the feature count.

        Returns
        -------
        dict
            ``{"train": .., "validation": .., "test": .., "features": ..}``;
            stored in the statistics file and the model metadata.

        返回各子集样本数与特征维数。
        """
        return {
            "train": int(len(self.y_train)),
            "validation": int(len(self.y_val)),
            "test": int(len(self.y_test)),
            "features": self.n_features,
        }


def compute_statistics(frame: pd.DataFrame) -> Dict:
    """Descriptive statistics reported in chapter 3 of the thesis.

    Parameters
    ----------
    frame:
        Validated corpus; ``class_code``, ``noaa_depth``, ``elevation`` and
        ``source_session`` columns are used when present.

    Returns
    -------
    dict
        JSON-serialisable dictionary with keys:

        * ``n_points`` - number of rows;
        * ``class_distribution`` - count and share per class;
        * ``bbox`` - bounding box of the points, degrees;
        * ``missing`` - number of missing elevation / depth values;
        * ``depth_by_class`` - mean, std and 5 / 50 / 95 % quantiles of
          ``noaa_depth`` per class, metres (shows how well depth alone
          separates the classes);
        * ``n_sessions`` - number of distinct collection sessions.

    描述性统计：样本数、类别分布、范围、缺失值、各类别水深分位数、采集会话数。
    """
    stats: Dict = {
        "n_points": int(len(frame)),
        "class_distribution": {},
        "bbox": {
            "lat_min": float(frame["latitude"].min()),
            "lat_max": float(frame["latitude"].max()),
            "lon_min": float(frame["longitude"].min()),
            "lon_max": float(frame["longitude"].max()),
        },
        "missing": {
            "elevation": int(frame["elevation"].isna().sum()) if "elevation" in frame else 0,
            "noaa_depth": int(frame["noaa_depth"].isna().sum()) if "noaa_depth" in frame else 0,
        },
    }
    if "class_code" in frame.columns:
        counts = frame["class_code"].value_counts()
        for name in CLASS_NAMES:
            n = int(counts.get(name, 0))
            stats["class_distribution"][name] = {
                "count": n,
                # max(.., 1) guards against division by zero on an empty frame.
                # 防止空表时除零。
                "share": round(n / max(len(frame), 1), 6),
            }
    if "noaa_depth" in frame.columns and "class_code" in frame.columns:
        grouped = frame.groupby("class_code")["noaa_depth"]
        stats["depth_by_class"] = {
            str(k): {
                "mean": float(v.mean()),
                "std": float(v.std()),
                "p05": float(v.quantile(0.05)),
                "p50": float(v.quantile(0.50)),
                "p95": float(v.quantile(0.95)),
            }
            for k, v in grouped
        }
    if "source_session" in frame.columns:
        stats["n_sessions"] = int(frame["source_session"].nunique())
    return stats


def prepare(
    raw_path: Path | str,
    output_npz: Path | str = PROCESSED_NPZ,
    index_path: Path | str = BATHY_INDEX_PATH,
    limit: Optional[int] = None,
    seed: int = TRAINING.random_seed,
    clean: bool = True,
) -> Tuple[SplitData, Dict]:
    """Full preparation pipeline. Returns the split and the statistics dict.

    Steps: load the GeoJSON corpus, validate coordinates, remove mislabelled
    points (``clean=True``), split 70/15/15 with stratification by class, build
    the label-free reference index from the training part only, extract the
    28 features of every point and save everything to ``output_npz``.

    Stratification keeps the class proportions equal in the three splits,
    which matters because the classes are imbalanced. Building the reference
    index only from training points ensures that no validation or test point
    contributes its own measurements to its neighbourhood features, so the
    test accuracy is an honest estimate for unseen locations.

    Parameters
    ----------
    raw_path:
        Labelled corpus (GeoJSON FeatureCollection of points).
    output_npz:
        Where the feature matrices are written.
    index_path:
        Where the reference index is written.
    limit:
        Read only the first ``limit`` features (for quick experiments).
    seed:
        Random seed of the split.
    clean:
        Remove points whose class contradicts their height/depth.

    Returns
    -------
    (SplitData, dict)
        The feature matrices / labels and the statistics dictionary (also
        written to :data:`STATS_JSON`).

    Raises
    ------
    ValueError
        If no row carries a known class code.
    DataLoadError
        If the input file contains no valid points.

    完整预处理流程：加载 -> 校验 -> 清洗 -> 分层划分 -> 建立参考索引 -> 特征提取。
    """
    LOGGER.info("Loading raw dataset from %s", raw_path)
    frame = load_geojson(raw_path, limit=limit)
    frame, report = validate(frame, source=str(raw_path))
    LOGGER.info("Validated %d points (%d dropped)", report.n_valid, report.n_dropped)

    # Training needs a label: drop unlabelled rows and unknown class codes.
    # 训练需要标签：删除无标签或类别未知的行。
    frame = frame[frame["class_code"].isin(CLASS_NAMES)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Dataset contains no rows with a known class_code")

    # Statistics of the corpus as delivered, then removal of the points whose
    # class contradicts their measured height/depth (labelling errors).
    # 先统计原始语料，再删除类别与高程/水深矛盾的标注错误样本。
    raw_statistics = compute_statistics(frame)
    if clean:
        frame, cleaning = remove_label_conflicts(frame)
        LOGGER.info("Removed %d mislabelled points (%.2f %%)",
                    cleaning.n_removed, 100.0 * cleaning.n_removed / max(cleaning.n_before, 1))
    else:
        cleaning = None

    # Encode class names as integers 0..3, shape (n,), int64 (PyTorch's
    # CrossEntropyLoss expects int64 targets). 类别名编码为整数 0..3。
    labels = frame["class_code"].map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    statistics = compute_statistics(frame)
    statistics["raw"] = raw_statistics
    statistics["cleaning"] = cleaning.to_dict() if cleaning else None
    # Region name of every point, and the class counts per region.
    # 每个点所属海域及各海域的类别计数。
    regions = assign_regions(frame["latitude"].to_numpy(), frame["longitude"].to_numpy())
    statistics["regions"] = {
        str(name): {
            "count": int((regions == name).sum()),
            "classes": {c: int(((regions == name) & (frame["class_code"].to_numpy() == c)).sum())
                        for c in CLASS_NAMES},
        }
        for name in pd.unique(regions)
    }

    # Split *before* the index is built so that the reference index is derived
    # from training data only. 先划分再建索引，保证参考索引只来自训练集。
    # Two-stage split on row indices: first 70 % train vs 30 % hold-out, then
    # the hold-out is halved into validation and test (rel = 0.15 / 0.30 = 0.5).
    # Both stages are stratified by class and seeded for reproducibility.
    # 两阶段划分：先 70% 训练 / 30% 保留，再将保留部分对半分为验证与测试；均按类别分层。
    idx_all = np.arange(len(frame))
    idx_train, idx_hold = train_test_split(
        idx_all, test_size=TRAINING.test_size + TRAINING.val_size,
        random_state=seed, stratify=labels,
    )
    rel = TRAINING.test_size / (TRAINING.test_size + TRAINING.val_size)
    idx_val, idx_test = train_test_split(
        idx_hold, test_size=rel, random_state=seed, stratify=labels[idx_hold],
    )

    LOGGER.info("Building reference index from %d training points", len(idx_train))
    index = ReferenceIndex.from_frame(frame.iloc[idx_train])
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    index.save(index_path)

    # Feature matrices, each (n_split, 28) float32. reset_index gives every
    # sub-frame a clean 0..n-1 index, so row i of the matrix is row i of labels.
    # 各子集特征矩阵 (n, 28) float32；reset_index 使子表索引为 0..n-1，与标签逐行对应。
    LOGGER.info("Extracting features")
    x_train = build_features(frame.iloc[idx_train].reset_index(drop=True), index, exclude_self=True)
    x_val = build_features(frame.iloc[idx_val].reset_index(drop=True), index, exclude_self=False)
    x_test = build_features(frame.iloc[idx_test].reset_index(drop=True), index, exclude_self=False)

    split = SplitData(
        x_train=x_train, y_train=labels[idx_train],
        x_val=x_val, y_val=labels[idx_val],
        x_test=x_test, y_test=labels[idx_test],
        feature_names=list(FEATURE_NAMES),
    )

    Path(output_npz).parent.mkdir(parents=True, exist_ok=True)
    # The region of every test point is stored so that the accuracy can be
    # reported per sea region. 保存测试点所属海域，便于按海域统计准确率。
    np.savez_compressed(
        output_npz,
        x_train=split.x_train, y_train=split.y_train,
        x_val=split.x_val, y_val=split.y_val,
        x_test=split.x_test, y_test=split.y_test,
        region_test=np.array(regions[idx_test], dtype=str),
        feature_names=np.array(FEATURE_NAMES),
    )
    statistics["split"] = split.describe()
    statistics["load_report"] = report.to_dict()
    # ensure_ascii=False keeps non-ASCII text readable in the JSON file.
    # ensure_ascii=False 使 JSON 中的非 ASCII 文本保持可读。
    STATS_JSON.write_text(json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved processed dataset to %s", output_npz)
    return split, statistics


def load_prepared(path: Path | str = PROCESSED_NPZ) -> SplitData:
    """Load a previously prepared split from disk.

    Parameters
    ----------
    path:
        ``.npz`` archive written by :func:`prepare`.

    Returns
    -------
    SplitData
        The six arrays and the feature names. ``region_test`` is not part of
        :class:`SplitData`; ``scripts/evaluate_regions.py`` reads it directly.

    读取已预处理的数据集。``allow_pickle=True`` 用于读取字符串数组。
    """
    blob = np.load(path, allow_pickle=True)
    return SplitData(
        x_train=blob["x_train"], y_train=blob["y_train"],
        x_val=blob["x_val"], y_val=blob["y_val"],
        x_test=blob["x_test"], y_test=blob["y_test"],
        feature_names=list(blob["feature_names"]),
    )
