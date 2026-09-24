"""Training loop, evaluation metrics and model persistence.

训练与评估模块：包含早停、学习率调度、类别加权与完整评估指标。

Role in the pipeline
--------------------
Called by ``scripts/train_model.py`` with the :class:`SplitData` produced by
:mod:`maritime_route.data.dataset`. :func:`train_model`:

1. fits a :class:`~sklearn.preprocessing.StandardScaler` on the training
   features (per-column ``z = (x - mean) / std``, statistics from the training
   split only, so no information leaks from validation/test);
2. trains a :class:`~maritime_route.model.network.ZoneClassifier` with AdamW
   on a class-weighted, label-smoothed cross-entropy loss;
3. halves the learning rate when the validation loss stops improving
   (ReduceLROnPlateau) and stops early when it has not improved for
   ``early_stopping_patience`` epochs, restoring the best weights;
4. evaluates the best model on the test split (:func:`evaluate`);
5. saves the weights, the scaler and a metadata JSON (hyper-parameters,
   metrics, per-epoch history) to :data:`~maritime_route.config.MODEL_DIR`.

The three saved files are what
:class:`~maritime_route.model.inference.ZoneClassifierService` loads.

Main objects: :func:`train_model`, :func:`evaluate`, :class:`EpochRecord`,
:class:`TrainingHistory`.

流程：拟合标准化器 -> AdamW 训练（类别加权 + 标签平滑的交叉熵）-> 学习率调度与早停
-> 在测试集上评估 -> 保存权重、标准化器与元数据。
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ..config import (
    CLASS_NAMES,
    MODEL_METADATA_PATH,
    MODEL_WEIGHTS_PATH,
    SCALER_PATH,
    TRAINING,
    TrainingConfig,
)
from ..data.dataset import SplitData
from ..data.features import FEATURE_NAMES
from .network import ZoneClassifier

LOGGER = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    """Metrics of one training epoch. 单轮训练记录。

    Attributes
    ----------
    epoch:
        Epoch number, starting at 1.
    train_loss:
        Mean loss over the training samples of this epoch (computed while the
        weights were still changing, with dropout active).
    val_loss:
        Mean loss on the validation split after the epoch (eval mode).
    train_accuracy, val_accuracy:
        Share of correctly classified samples, 0..1.
    val_macro_f1:
        Unweighted mean of the per-class F1 scores on validation data; unlike
        accuracy it is not dominated by the largest class.
    learning_rate:
        Learning rate after the scheduler step of this epoch.
    seconds:
        Wall-clock duration of the epoch.
    """

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    val_macro_f1: float
    learning_rate: float
    seconds: float


@dataclass
class TrainingHistory:
    """Ordered list of :class:`EpochRecord`; source of the training-curve figures.

    训练历史：按轮次保存的记录，用于绘制训练曲线。
    """

    records: List[EpochRecord] = field(default_factory=list)

    def append(self, record: EpochRecord) -> None:
        """Add the record of the epoch that has just finished. 追加一轮记录。"""
        self.records.append(record)

    def to_dict(self) -> Dict[str, List[float]]:
        """Return the history as parallel lists (one list per metric).

        This column layout is stored in the metadata JSON and can be plotted
        directly (x = ``epoch``). ``seconds`` is not included.

        Returns
        -------
        dict
            Metric name -> list of values, one per epoch.

        转换为"每个指标一个列表"的字典，便于保存与绘图。
        """
        return {
            "epoch": [r.epoch for r in self.records],
            "train_loss": [r.train_loss for r in self.records],
            "val_loss": [r.val_loss for r in self.records],
            "train_accuracy": [r.train_accuracy for r in self.records],
            "val_accuracy": [r.val_accuracy for r in self.records],
            "val_macro_f1": [r.val_macro_f1 for r in self.records],
            "learning_rate": [r.learning_rate for r in self.records],
        }


def _loaders(split: SplitData, scaler: StandardScaler, cfg: TrainingConfig):
    """Build PyTorch data loaders for the three splits.

    Every split is standardised with the *same* scaler (fitted on the
    training data) and wrapped in a :class:`~torch.utils.data.TensorDataset`.
    Only the training loader shuffles, so that every epoch sees the samples
    in a new order (better stochastic gradients); validation and test order
    is irrelevant and kept fixed.

    Parameters
    ----------
    split:
        Feature matrices and labels.
    scaler:
        Fitted scaler.
    cfg:
        Training configuration (batch size).

    Returns
    -------
    tuple of DataLoader
        ``(train_loader, val_loader, test_loader)``; each batch is
        ``(x, y)`` with ``x`` of shape ``(batch, 28)`` float32 and ``y`` of
        shape ``(batch,)`` int64.

    构建三个数据加载器：统一用训练集拟合的标准化器变换，仅训练集打乱顺序。
    """
    def make(x: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        """Standardise ``x`` and wrap ``(x, y)`` in a DataLoader. 标准化并封装。"""
        tensors = TensorDataset(
            # Standardised features as float32 (the dtype of the model weights).
            torch.tensor(scaler.transform(x), dtype=torch.float32),
            # int64 ("long") class indices, required by CrossEntropyLoss.
            torch.tensor(y, dtype=torch.long),
        )
        # drop_last=False keeps the final, smaller batch so no sample is skipped.
        # 保留最后一个不完整批次，不丢弃样本。
        return DataLoader(tensors, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False)

    return (make(split.x_train, split.y_train, True),
            make(split.x_val, split.y_val, False),
            make(split.x_test, split.y_test, False))


def _class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1. 类别不平衡加权。

    ``w_c = N / (C * n_c)`` where ``N`` is the number of samples, ``C`` the
    number of classes and ``n_c`` the count of class ``c``. A class that is
    twice as rare gets twice the weight in the loss, so rare classes are not
    ignored by the optimiser. Dividing by the mean keeps the overall loss
    scale (and therefore the effective learning rate) unchanged.

    Parameters
    ----------
    y:
        Training labels, int, shape ``(n,)``.
    n_classes:
        Number of classes.

    Returns
    -------
    torch.Tensor
        Shape ``(n_classes,)``, float32, mean 1.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    # An absent class would give a division by zero; treat it as count 1.
    # 缺失的类别计数按 1 处理，避免除零。
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights / weights.mean(), dtype=torch.float32)


@torch.no_grad()
def _evaluate_loader(model: nn.Module, loader: DataLoader, criterion, device: str):
    """Run the model over a loader and collect loss, labels and probabilities.

    The model is put into evaluation mode (BatchNorm uses running statistics,
    Dropout is off); the decorator disables gradient tracking.

    Parameters
    ----------
    model:
        The network.
    loader:
        Validation or test loader.
    criterion:
        Loss function (the same weighted, smoothed loss as in training, so
        train and validation losses are comparable).
    device:
        PyTorch device string.

    Returns
    -------
    tuple
        ``(mean_loss, y_true, y_pred, probs)`` - mean loss per sample
        (float), true labels ``(n,)``, predicted labels ``(n,)`` and softmax
        probabilities ``(n, n_classes)``.

    在评估模式下遍历数据，返回平均损失、真实标签、预测标签与概率。
    """
    model.eval()
    total_loss, n = 0.0, 0
    probs_all, y_all = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)                           # (batch, 4)
        loss = criterion(logits, yb)                 # mean over the batch
        # Multiply by the batch size to get a sum, so the final division gives
        # the exact per-sample mean even when the last batch is smaller.
        # 乘以批大小得到总和，使最终均值在最后一批较小时仍然精确。
        total_loss += float(loss.item()) * len(yb)
        n += len(yb)
        # Softmax turns logits into probabilities that sum to 1 per row.
        # softmax 将 logits 转换为每行和为 1 的概率。
        probs_all.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_all.append(yb.cpu().numpy())
    probs = np.concatenate(probs_all)
    y_true = np.concatenate(y_all)
    y_pred = probs.argmax(axis=1)                    # most probable class
    return total_loss / max(n, 1), y_true, y_pred, probs


def train_model(
    split: SplitData,
    cfg: TrainingConfig = TRAINING,
    weights_path: Path | str = MODEL_WEIGHTS_PATH,
    scaler_path: Path | str = SCALER_PATH,
    metadata_path: Path | str = MODEL_METADATA_PATH,
    progress_callback=None,
) -> Tuple[ZoneClassifier, TrainingHistory, Dict]:
    """Train the classifier and persist weights, scaler and metadata.

    训练主函数，返回 (模型, 训练历史, 评估指标)。

    Training components:

    * **Loss** - cross-entropy ``-sum_c q_c log p_c`` over the softmax
      probabilities ``p``, with

      - *class weights* from :func:`_class_weights` (compensate for class
        imbalance), and
      - *label smoothing* ``eps = cfg.label_smoothing``: the target ``q`` is
        ``1 - eps + eps/C`` for the true class and ``eps/C`` for the others
        (C = 4). The model is discouraged from producing extreme
        probabilities, which reduces overconfidence on label noise.

    * **Optimiser** - AdamW: Adam (per-parameter step sizes from running
      means of the gradient and its square) with *decoupled* weight decay,
      i.e. the weights are shrunk by ``lr * weight_decay`` directly instead
      of adding an L2 term to the gradient; this regularises more
      consistently than L2 in plain Adam.
    * **Gradient clipping** - the global gradient norm is limited to 5.0 to
      prevent a single bad batch from causing a very large update.
    * **Scheduler** - ReduceLROnPlateau halves the learning rate
      (``factor=0.5``) when the validation loss has not improved for
      ``cfg.lr_scheduler_patience`` epochs, down to ``min_lr=1e-6``.
    * **Early stopping** - training stops after
      ``cfg.early_stopping_patience`` epochs without a validation-loss
      improvement larger than 1e-5; the weights of the best epoch are then
      restored.

    Parameters
    ----------
    split:
        Prepared feature matrices and labels.
    cfg:
        Hyper-parameters.
    weights_path, scaler_path, metadata_path:
        Output files.
    progress_callback:
        Optional callable receiving each :class:`EpochRecord` after every
        epoch (a hook for progress display; the command-line script does not
        use it).

    Returns
    -------
    (ZoneClassifier, TrainingHistory, dict)
        The model with the best validation weights, the per-epoch history and
        the test metrics (see :func:`evaluate`) plus a ``training`` summary.
    """
    # Seed PyTorch (weight init, dropout masks, shuffling) and NumPy for
    # reproducible runs. 固定随机种子以保证结果可复现。
    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    device = cfg.device

    # Standardisation z = (x - mean) / std per feature column, fitted on the
    # training split only. 按列标准化，仅用训练集拟合均值与标准差。
    scaler = StandardScaler().fit(split.x_train)
    # test_loader is not used below: evaluate() processes the test split in one pass.
    # test_loader 未在下文使用：evaluate() 一次性处理测试集。
    train_loader, val_loader, test_loader = _loaders(split, scaler, cfg)

    model = ZoneClassifier(
        n_features=split.n_features,
        n_classes=len(CLASS_NAMES),
        hidden_sizes=cfg.hidden_sizes,
        dropout=cfg.dropout,
    ).to(device)
    LOGGER.info("Model has %d trainable parameters", model.count_parameters())

    # Weighted, label-smoothed cross-entropy (applies log-softmax internally,
    # so the model outputs raw logits). 类别加权 + 标签平滑的交叉熵（内部含 log-softmax）。
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(split.y_train, len(CLASS_NAMES)).to(device),
        label_smoothing=cfg.label_smoothing,
    )
    # AdamW: Adam with decoupled weight decay. AdamW 优化器（解耦权重衰减）。
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                                  weight_decay=cfg.weight_decay)
    # Halve the LR when val loss plateaus ("min": lower is better).
    # 验证损失停滞时学习率减半。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=cfg.lr_scheduler_patience, min_lr=1e-6,
    )

    history = TrainingHistory()
    # Early-stopping state: best validation loss, a copy of the weights that
    # achieved it, and the number of epochs since the last improvement.
    # 早停状态：最佳验证损失、对应权重副本、自上次改进以来的轮数。
    best_val = float("inf")
    best_state: Optional[Dict] = None
    patience = 0
    t_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        # Training mode: Dropout active, BatchNorm uses batch statistics and
        # updates its running averages. 训练模式：启用 Dropout，BN 使用批统计量。
        model.train()
        running, seen, correct = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            # Clear the gradients of the previous step (set_to_none saves a
            # memset). 清除上一步梯度。
            optimiser.zero_grad(set_to_none=True)
            logits = model(xb)                       # (batch, 4)
            loss = criterion(logits, yb)
            loss.backward()                          # back-propagation 反向传播
            # Rescale gradients if their global L2 norm exceeds 5.0.
            # 梯度裁剪：全局 L2 范数超过 5.0 时按比例缩小。
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimiser.step()                         # parameter update 参数更新
            running += float(loss.item()) * len(yb)
            seen += len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())

        train_loss = running / max(seen, 1)
        train_acc = correct / max(seen, 1)
        val_loss, y_val, y_val_pred, _ = _evaluate_loader(model, val_loader, criterion, device)
        val_acc = float(accuracy_score(y_val, y_val_pred))
        val_f1 = float(f1_score(y_val, y_val_pred, average="macro"))
        # The scheduler watches the validation loss. 调度器依据验证损失调整学习率。
        scheduler.step(val_loss)

        record = EpochRecord(
            epoch=epoch, train_loss=train_loss, val_loss=val_loss,
            train_accuracy=train_acc, val_accuracy=val_acc, val_macro_f1=val_f1,
            learning_rate=float(optimiser.param_groups[0]["lr"]),
            seconds=time.perf_counter() - t0,
        )
        history.append(record)
        if progress_callback is not None:
            progress_callback(record)
        # Log the first epoch and every 5th to keep the console readable.
        # 仅记录第 1 轮和每 5 轮，保持日志简洁。
        if epoch % 5 == 0 or epoch == 1:
            LOGGER.info(
                "epoch %3d | train loss %.4f acc %.4f | val loss %.4f acc %.4f f1 %.4f",
                epoch, train_loss, train_acc, val_loss, val_acc, val_f1,
            )

        # Early stopping: an improvement must exceed 1e-5 to count, so that
        # numerical noise does not reset the counter.
        # 早停：改进幅度需大于 1e-5 才计入，避免数值噪声重置计数。
        if val_loss < best_val - 1e-5:
            best_val, patience = val_loss, 0
            # Deep copy of the weights (state_dict returns references that the
            # next optimiser step would overwrite). 深拷贝当前最佳权重。
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                LOGGER.info("Early stopping at epoch %d", epoch)
                break

    # Restore the best epoch, not the last one. 恢复最佳轮次的权重（而非最后一轮）。
    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - t_start

    metrics = evaluate(model, split, scaler, cfg)
    metrics["training"] = {
        "epochs_run": len(history.records),
        "seconds": round(train_seconds, 2),
        "best_val_loss": round(best_val, 6),
        "parameters": model.count_parameters(),
        "device": device,
    }

    # Persist the three artefacts used by the inference service.
    # The checkpoint stores the architecture next to the weights so that the
    # network can be rebuilt without knowing the training configuration.
    # 保存推理所需的三个文件；检查点中同时保存网络结构参数。
    Path(weights_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config_dict()}, weights_path)
    joblib.dump(scaler, scaler_path)
    metadata = {
        "version": "1.0.0",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "classes": CLASS_NAMES,
        "feature_names": list(FEATURE_NAMES),
        "model": model.config_dict(),
        "hyperparameters": {
            "hidden_sizes": list(cfg.hidden_sizes), "dropout": cfg.dropout,
            "batch_size": cfg.batch_size, "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate, "weight_decay": cfg.weight_decay,
            "label_smoothing": cfg.label_smoothing, "seed": cfg.random_seed,
        },
        "dataset": split.describe(),
        "metrics": metrics,
        "history": history.to_dict(),
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved model to %s", weights_path)
    return model, history, metrics


def evaluate(model: ZoneClassifier, split: SplitData, scaler: StandardScaler,
             cfg: TrainingConfig = TRAINING) -> Dict:
    """Compute the full metric set on the held-out test split.

    在测试集上计算准确率、宏平均F1、Kappa、每类指标、混淆矩阵与ROC-AUC。

    Metrics:

    * ``accuracy`` - share of correct predictions;
    * ``macro_f1`` - unweighted mean of per-class F1 (F1 = harmonic mean of
      precision and recall); every class counts equally;
    * ``weighted_f1`` - per-class F1 weighted by class support;
    * ``cohen_kappa`` - agreement corrected for the agreement expected by
      chance, ``(p_o - p_e) / (1 - p_e)``;
    * ``macro_roc_auc`` - one-vs-rest ROC AUC averaged over classes; measures
      ranking quality of the probabilities independently of the argmax
      threshold (NaN if a class is absent from the test set);
    * ``per_class`` - precision, recall, F1 and support of each class;
    * ``confusion_matrix`` - 4 x 4 counts, rows = true class, columns =
      predicted class, in :data:`~maritime_route.config.CLASS_NAMES` order;
    * ``report_text`` - scikit-learn's text report;
    * ``mean_confidence`` - mean of the highest class probability.

    Parameters
    ----------
    model:
        Trained network.
    split:
        Data; only ``x_test`` / ``y_test`` are used.
    scaler:
        The scaler fitted on the training split.
    cfg:
        Training configuration (device).

    Returns
    -------
    dict
        JSON-serialisable metrics as listed above.
    """
    device = cfg.device
    model.eval()
    # The whole test split in one tensor (n_test, 28); it is small enough.
    # 整个测试集一次性前向计算。
    x = torch.tensor(scaler.transform(split.x_test), dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()   # (n_test, 4)
    y_true = split.y_test
    y_pred = probs.argmax(axis=1)

    # labels=[0..3] fixes the class order and keeps absent classes as zeros;
    # zero_division=0 avoids warnings for classes that are never predicted.
    # 固定类别顺序；从未被预测的类别指标记为 0。
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro"))
    except ValueError:
        # Raised when a class is missing from y_true (e.g. on a tiny subset).
        # 当测试集中缺少某一类别时无法计算 AUC。
        auc = float("nan")

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "macro_roc_auc": auc,
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
        ).tolist(),
        "report_text": classification_report(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES, digits=4, zero_division=0,
        ),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }
