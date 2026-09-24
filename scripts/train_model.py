#!/usr/bin/env python3
"""Train the geographic-zone classifier.

用法: python scripts/train_model.py [--epochs 120]

Second step of the offline pipeline. Loads the matrices written by
``scripts/prepare_dataset.py``, trains the network with
:func:`maritime_route.model.trainer.train_model` and saves the weights, the
feature scaler and the metadata to ``models/``. Prints the per-class report
and the headline test metrics.

Options
-------
``--dataset``     processed ``.npz`` (default ``data/processed/zone_dataset.npz``)
``--epochs``      maximum number of epochs (default from ``TrainingConfig``)
``--batch-size``  mini-batch size
``--lr``          initial learning rate of AdamW

All other hyper-parameters come from :data:`maritime_route.config.TRAINING`.

离线流程第二步：读取预处理结果，训练网络并保存权重、标准化器与元数据。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

# Make ``src`` importable without installing the package. 将 src 加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maritime_route.config import TRAINING
from maritime_route.data.dataset import PROCESSED_NPZ, load_prepared
from maritime_route.model.trainer import train_model


def main() -> int:
    """Parse options, train the model and print the test metrics.

    Returns
    -------
    int
        Process exit code, 0 on success.

    解析参数、训练模型并输出测试集指标。
    """
    parser = argparse.ArgumentParser(description="Train the zone classifier")
    parser.add_argument("--dataset", default=str(PROCESSED_NPZ))
    parser.add_argument("--epochs", type=int, default=TRAINING.epochs)
    parser.add_argument("--batch-size", type=int, default=TRAINING.batch_size)
    parser.add_argument("--lr", type=float, default=TRAINING.learning_rate)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split = load_prepared(args.dataset)
    # TrainingConfig is frozen: replace() returns a copy with the overridden
    # fields and leaves the shared default untouched.
    # TrainingConfig 为冻结数据类：replace() 返回修改后的副本，不影响默认配置。
    cfg = replace(TRAINING, epochs=args.epochs, batch_size=args.batch_size,
                  learning_rate=args.lr)
    _, _, metrics = train_model(split, cfg)
    print(metrics["report_text"])
    # Headline metrics only; the full set is in models/model_metadata.json.
    # 仅输出主要指标，完整指标见 models/model_metadata.json。
    print(json.dumps({k: metrics[k] for k in
                      ("accuracy", "macro_f1", "weighted_f1", "cohen_kappa", "macro_roc_auc")},
                     indent=2))
    return 0


if __name__ == "__main__":
    # Exit code of the process = return value of main(). 进程退出码。
    raise SystemExit(main())
