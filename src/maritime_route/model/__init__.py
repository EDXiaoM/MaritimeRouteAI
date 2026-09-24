"""Neural network model package.

神经网络模型层：结构、训练与推理。

Modules
-------
:mod:`~maritime_route.model.network`
    PyTorch definition of :class:`ZoneClassifier`, a residual multilayer
    perceptron mapping the 28 engineered features of a point to logits of the
    four zone classes.
:mod:`~maritime_route.model.trainer`
    Training loop (AdamW, class-weighted cross-entropy with label smoothing,
    learning-rate scheduling, early stopping), test-set metrics and saving of
    weights, feature scaler and metadata.
:mod:`~maritime_route.model.inference`
    :class:`ZoneClassifierService`, a thread-safe singleton that loads the
    saved artefacts once and classifies arbitrary coordinates for the route
    planner and the web API.

子模块：network（网络结构）、trainer（训练与评估）、inference（推理服务）。
"""
# Re-export the two classes used by the rest of the application. The trainer is
# not re-exported: it pulls in scikit-learn metrics that inference does not need.
# 只导出应用其余部分需要的两个类；trainer 依赖较多，不在此处导入。
from .network import ZoneClassifier
from .inference import ZoneClassifierService

__all__ = ["ZoneClassifier", "ZoneClassifierService"]
