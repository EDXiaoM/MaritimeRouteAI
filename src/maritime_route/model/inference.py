"""Inference service wrapping the trained classifier.

推理服务：加载已训练权重 + 标准化器 + 参考索引，对任意坐标进行区域分类。
The service is a process-wide singleton so that the 100k-point reference index
is memory-resident only once.

Role in the pipeline
--------------------
This is the bridge between the neural network and the rest of the
application:

* the route planner (:mod:`maritime_route.routing.cost_map`) calls
  :meth:`ZoneClassifierService.predict_coordinates` for every cell of the
  routing grid and turns the most probable class into a traversal cost;
* the web API calls :meth:`ZoneClassifierService.classify_frame` for
  uploaded AIS files and :meth:`ZoneClassifierService.classify_coordinates`
  for single clicked points.

Every prediction runs the same chain as training: features
(:mod:`maritime_route.data.features`) -> standardisation with the saved
scaler -> network -> softmax.

Main objects: :class:`ZoneClassifierService`, :class:`ClassificationResult`,
:class:`ModelNotTrainedError`.

用途：路线规划器对网格单元分类、Web 接口对上传文件与单点分类。
推理链与训练一致：特征提取 -> 标准化 -> 网络 -> softmax。
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import joblib
import numpy as np
import pandas as pd
import torch

from ..config import (
    BATHY_INDEX_PATH,
    CLASS_NAMES,
    INDEX_TO_CLASS,
    MODEL_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    SCALER_PATH,
)
from ..data.features import ReferenceIndex, build_features, features_for_coordinates
from .network import ZoneClassifier

LOGGER = logging.getLogger(__name__)


class ModelNotTrainedError(RuntimeError):
    """Raised when the weights file is missing. 模型尚未训练。

    The web application catches it when the model is first needed and answers
    HTTP 503 with the message, which tells the user to run
    ``scripts/train_model.py``. Tests that need a model are skipped when the
    weights file is absent (see ``tests/conftest.py``).
    Web 应用在首次需要模型时捕获并返回 HTTP 503。
    """


@dataclass
class ClassificationResult:
    """Per-point classification output. 单点分类结果。

    Attributes
    ----------
    latitude, longitude:
        Position, degrees.
    class_index:
        Index of the most probable class (0..3).
    class_code:
        Name of that class, e.g. ``"COASTAL_SEA"``.
    confidence:
        Probability of the predicted class, 0..1.
    probabilities:
        Probability of every class, keyed by class name; sums to 1.
    depth_m:
        Relief at the point, metres, positive above sea level; the measured
        value if the input had one, otherwise interpolated from the reference
        index. ``None`` if unknown.
    """

    latitude: float
    longitude: float
    class_index: int
    class_code: str
    confidence: float
    probabilities: Dict[str, float]
    depth_m: Optional[float] = None

    def to_dict(self) -> Dict:
        """Return a JSON-ready dictionary with rounded values.

        Rounding: coordinates to 6 decimals (about 0.1 m), probabilities to 4
        decimals, depth to 0.1 m; this keeps API responses compact without
        losing meaningful precision.

        Returns
        -------
        dict
            All fields of the result.

        转换为 JSON 字典并做适当舍入（坐标 6 位小数约 0.1 m）。
        """
        return {
            "latitude": round(self.latitude, 6),
            "longitude": round(self.longitude, 6),
            "class_index": self.class_index,
            "class_code": self.class_code,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "depth_m": None if self.depth_m is None else round(float(self.depth_m), 1),
        }


class ZoneClassifierService:
    """Thread-safe façade over the trained model.

    Loads the four saved artefacts once (network weights, feature scaler,
    reference index, metadata) and offers prediction methods for data frames
    and bare coordinates.

    Concurrency: the web server runs classification in worker threads
    (``asyncio.to_thread``), so several requests can arrive at once. The
    class-level ``_lock`` guards creation of the singleton, and the
    per-instance ``_infer_lock`` serialises forward passes through the network,
    so that concurrent requests never run the same PyTorch module at the same
    time.

    Use :meth:`instance` to obtain the shared object and :meth:`reset` to
    force a reload after retraining.

    线程安全的推理门面：类锁保护单例创建，实例锁串行化网络前向计算。
    """

    #: The shared instance created by :meth:`instance`. 共享单例。
    _instance: Optional["ZoneClassifierService"] = None
    #: Guards creation / reset of the singleton. 保护单例创建与重置。
    _lock = threading.Lock()

    def __init__(
        self,
        weights_path: Path | str = MODEL_WEIGHTS_PATH,
        scaler_path: Path | str = SCALER_PATH,
        index_path: Path | str = BATHY_INDEX_PATH,
        metadata_path: Path | str = MODEL_METADATA_PATH,
    ) -> None:
        """Load the model artefacts from disk.

        Parameters
        ----------
        weights_path:
            PyTorch checkpoint ``{"state_dict", "config"}`` written by the
            trainer.
        scaler_path:
            Fitted StandardScaler (joblib).
        index_path:
            Reference index (.npz).
        metadata_path:
            Metadata JSON; optional - missing metadata only affects
            :meth:`summary`.

        Raises
        ------
        ModelNotTrainedError
            If the weights file does not exist.

        从磁盘加载权重、标准化器、参考索引与元数据。
        """
        weights_path, scaler_path, index_path = Path(weights_path), Path(scaler_path), Path(index_path)
        if not weights_path.exists():
            raise ModelNotTrainedError(
                f"Model weights not found at {weights_path}. Run scripts/train_model.py first."
            )
        # map_location="cpu" allows loading a checkpoint saved on a GPU machine.
        # weights_only=False because the checkpoint also holds the config dict
        # (the file is produced by this project, not an untrusted source).
        # 映射到 CPU 以兼容 GPU 上保存的检查点；检查点包含配置字典，故 weights_only=False。
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        # Rebuild a network of identical shape, then load the trained weights.
        # 先按保存的结构参数重建网络，再加载权重。
        self.model = ZoneClassifier(
            n_features=config["n_features"],
            n_classes=config["n_classes"],
            hidden_sizes=config["hidden_sizes"],
            dropout=config["dropout"],
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        # Evaluation mode: BatchNorm uses running statistics, Dropout is off.
        # 评估模式：BN 使用训练时的滑动统计量，关闭 Dropout。
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.index = ReferenceIndex.load(index_path)
        self.metadata: Dict = {}
        if Path(metadata_path).exists():
            self.metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        self._infer_lock = threading.Lock()
        LOGGER.info("Zone classifier ready (%d reference samples)", len(self.index.depth))

    # -- singleton ---------------------------------------------------------- #
    @classmethod
    def instance(cls) -> "ZoneClassifierService":
        """Return the process-wide service, creating it on first use.

        The check and the creation happen under ``_lock``, so two threads that
        call this at the same time cannot both load the model.

        Returns
        -------
        ZoneClassifierService
            The shared instance (default file paths).

        Raises
        ------
        ModelNotTrainedError
            If no trained model exists.

        返回进程级单例，首次调用时加载模型（加锁防止重复加载）。
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Forget the shared instance; the next :meth:`instance` call reloads.

        Intended for use after retraining, so that the new weights are picked
        up without restarting the process. 重置单例，下次调用时重新加载模型。
        """
        with cls._lock:
            cls._instance = None

    # -- prediction --------------------------------------------------------- #
    def _forward(self, matrix: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        """Standardise a raw feature matrix and return class probabilities.

        The matrix is processed in chunks of ``batch_size`` rows to bound
        memory use for large grids (hundreds of thousands of cells).

        Parameters
        ----------
        matrix:
            Unscaled features, shape ``(n, 28)``.
        batch_size:
            Rows per forward pass.

        Returns
        -------
        numpy.ndarray
            Softmax probabilities, shape ``(n, 4)``; each row sums to 1.
            Shape ``(0, 4)`` for an empty input.

        标准化后分批前向计算，返回 softmax 概率。
        """
        # Same standardisation as in training: z = (x - mean_train) / std_train.
        # 与训练时相同的标准化。
        scaled = self.scaler.transform(matrix).astype(np.float32)
        outputs: List[np.ndarray] = []
        # _infer_lock: one forward pass at a time; no_grad: no autograd graph.
        # 加锁串行化前向计算；no_grad 不构建计算图，节省内存与时间。
        with self._infer_lock, torch.no_grad():
            for start in range(0, len(scaled), batch_size):
                # from_numpy shares memory with the array (no copy).
                chunk = torch.from_numpy(scaled[start:start + batch_size])
                outputs.append(torch.softmax(self.model(chunk), dim=1).numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, len(CLASS_NAMES)))

    def predict_frame(self, frame: pd.DataFrame) -> np.ndarray:
        """Probabilities for a DataFrame that already carries measured attributes.

        Parameters
        ----------
        frame:
            Points with ``latitude``/``longitude`` and optionally the measured
            columns (``noaa_depth``, ``elevation``, OSM counts); missing values
            are imputed by :func:`~maritime_route.data.features.build_features`.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 4)`` probabilities.

        对带测量属性的数据表进行预测。
        """
        matrix = build_features(frame, self.index, exclude_self=False)
        return self._forward(matrix)

    def predict_coordinates(self, lat: Sequence[float], lon: Sequence[float]) -> np.ndarray:
        """Probabilities for bare coordinates. 仅凭经纬度预测。

        All environmental attributes are derived from the reference index
        (see :func:`~maritime_route.data.features.features_for_coordinates`).
        This is the method the route planner uses for grid cells.

        Parameters
        ----------
        lat, lon:
            Coordinates, degrees, length n.

        Returns
        -------
        numpy.ndarray
            Shape ``(n, 4)`` probabilities.
        """
        matrix = features_for_coordinates(lat, lon, self.index)
        return self._forward(matrix)

    def classify_frame(self, frame: pd.DataFrame) -> List[ClassificationResult]:
        """Classify a frame of AIS points into :class:`ClassificationResult` objects.

        The reported depth is the measured ``noaa_depth`` when the frame has
        that column (NaN entries become ``None``), otherwise it is
        interpolated from the reference index.

        Parameters
        ----------
        frame:
            Validated points.

        Returns
        -------
        list of ClassificationResult
            One result per row, in row order.

        对 AIS 数据表逐点分类并封装结果。
        """
        probs = self.predict_frame(frame)
        lat = frame["latitude"].to_numpy(float)
        lon = frame["longitude"].to_numpy(float)
        # Measured depth if available, otherwise IDW interpolation from the index.
        # 有实测水深则使用实测值，否则由参考索引插值。
        depth = (pd.to_numeric(frame["noaa_depth"], errors="coerce").to_numpy(float)
                 if "noaa_depth" in frame.columns else self.index.sample_depth(lat, lon))
        return self._package(lat, lon, probs, depth)

    def classify_coordinates(self, lat: Sequence[float], lon: Sequence[float]
                             ) -> List[ClassificationResult]:
        """Classify bare coordinates.

        Parameters
        ----------
        lat, lon:
            Coordinates, degrees, length n.

        Returns
        -------
        list of ClassificationResult
            One result per coordinate; depth interpolated from the index.

        仅凭经纬度分类，水深由参考索引插值。
        """
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        probs = self.predict_coordinates(lat, lon)
        return self._package(lat, lon, probs, self.index.sample_depth(lat, lon))

    @staticmethod
    def _package(lat, lon, probs, depth) -> List[ClassificationResult]:
        """Turn probability rows into :class:`ClassificationResult` objects.

        Parameters
        ----------
        lat, lon:
            Arrays of length n, degrees.
        probs:
            ``(n, 4)`` class probabilities.
        depth:
            ``(n,)`` depth in metres (NaN allowed) or ``None``.

        Returns
        -------
        list of ClassificationResult
            The predicted class is the argmax of each row; its probability is
            the confidence.

        将概率矩阵封装为结果对象：取每行最大概率的类别作为预测，其概率为置信度。
        """
        best = probs.argmax(axis=1)   # (n,) index of the most probable class
        results: List[ClassificationResult] = []
        for i in range(len(lat)):
            results.append(
                ClassificationResult(
                    latitude=float(lat[i]),
                    longitude=float(lon[i]),
                    class_index=int(best[i]),
                    class_code=INDEX_TO_CLASS[int(best[i])],
                    confidence=float(probs[i, best[i]]),
                    probabilities={CLASS_NAMES[c]: float(probs[i, c]) for c in range(len(CLASS_NAMES))},
                    # NaN (no measurement) is reported as None (JSON null).
                    # NaN 以 None（JSON null）表示。
                    depth_m=None if depth is None or np.isnan(depth[i]) else float(depth[i]),
                )
            )
        return results

    # -- introspection ------------------------------------------------------ #
    def summary(self) -> Dict:
        """Describe the loaded model; returned by ``GET /api/model`` in the web API.

        Returns
        -------
        dict
            Classes, input width, parameter count, architecture label, size of
            the reference index, test accuracy and macro F1 from the metadata
            (``None`` if the metadata file was missing) and the training time
            stamp (UTC).

        返回模型摘要：类别、特征维数、参数量、结构、参考样本数、测试集指标与训练时间。
        """
        metrics = self.metadata.get("metrics", {})
        return {
            "classes": CLASS_NAMES,
            "n_features": self.model.n_features,
            "parameters": self.model.count_parameters(),
            "architecture": self.model.config_dict()["architecture"],
            "reference_samples": int(len(self.index.depth)),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "trained_utc": self.metadata.get("created_utc"),
        }
