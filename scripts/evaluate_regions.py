#!/usr/bin/env python3
"""Accuracy of the trained classifier in each of the ten sea regions.

The corpus was collected in ten separate sea regions (see
``maritime_route.config.REGIONS``). A single overall accuracy can hide a
region where the model is weak, so this script evaluates the held-out test
split region by region and reports, for every region:

* the number of test points,
* the accuracy,
* the number of *water <-> land* confusions - the only errors that matter for
  safety, because they can turn land into navigable water or the reverse.

The result is written to ``models/region_metrics.json`` and printed as a table.

Usage::

    python scripts/evaluate_regions.py

Prerequisites: ``scripts/prepare_dataset.py`` (which stores the region of
every test point as ``region_test``) and ``scripts/train_model.py``. The test
features are already computed, so the script only standardises them and runs
the network; it does not rebuild features.

按十个海域分别评估测试集：样本数、准确率，以及对航行安全最关键的"水域↔陆地"误判数。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make ``src`` importable without installing the package. 将 src 加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from maritime_route.config import CLASS_NAMES, LAND_CLASSES, MODEL_DIR, WATER_CLASSES
from maritime_route.data.dataset import PROCESSED_NPZ
from maritime_route.model.inference import ZoneClassifierService

#: Result file (JSON). 结果文件。
OUTPUT = MODEL_DIR / "region_metrics.json"


def main() -> int:
    """Evaluate the test split per region and save the table.

    A *water <-> land* error is a prediction on the other side of the
    shoreline from the true class (e.g. COASTAL_SEA predicted as NEAR_COAST).
    Confusions inside one side (OPEN_SEA vs COASTAL_SEA, NEAR_COAST vs
    COASTLINE) only change the traversal cost and are not counted.

    Returns
    -------
    int
        Exit code: 0 on success, 1 if the prepared dataset lacks the
        ``region_test`` array (dataset prepared by an older version).

    按海域评估测试集并保存结果；水陆两侧之间的误判才计为"水域↔陆地"错误。
    """
    # allow_pickle=True is needed to read the string arrays of the archive.
    # 读取字符串数组需要 allow_pickle=True。
    blob = np.load(PROCESSED_NPZ, allow_pickle=True)
    if "region_test" not in blob:
        print("The prepared dataset has no region column - rerun scripts/prepare_dataset.py")
        return 1
    x_test, y_test, regions = blob["x_test"], blob["y_test"], blob["region_test"]

    service = ZoneClassifierService()
    # _forward applies the saved scaler and the network to the unscaled
    # (n_test, 28) matrix and returns (n_test, 4) probabilities.
    # _forward 对未标准化的测试特征做标准化并前向计算，返回概率。
    y_pred = service._forward(x_test).argmax(axis=1)   # probabilities -> class index

    # Map every class index to "water" (True) or "land" (False).
    # 将类别索引映射为水域(True)或陆地(False)。
    is_water = np.array([name in WATER_CLASSES for name in CLASS_NAMES])
    # Fancy indexing: is_water[y] gives the water flag of every sample, shape
    # (n_test,). True where truth and prediction lie on different sides.
    # 按索引取出每个样本的水域标志；真实与预测位于水陆两侧时为 True。
    crossing = is_water[y_test] != is_water[y_pred]

    table = {}
    # Regions ordered by number of test points, largest first. 按样本数降序排列海域。
    for name in sorted(set(regions.tolist()), key=lambda r: -int((regions == r).sum())):
        m = regions == name   # boolean mask of the test points in this region 该海域的掩码
        table[name] = {
            "n_test": int(m.sum()),
            "accuracy": float((y_pred[m] == y_test[m]).mean()),
            "water_land_errors": int(crossing[m].sum()),
        }
    summary = {
        "overall_accuracy": float((y_pred == y_test).mean()),
        "overall_water_land_errors": int(crossing.sum()),
        "n_test": int(len(y_test)),
        "regions": table,
    }
    OUTPUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Fixed-width console table: region name 42 chars, then right-aligned numbers.
    # 固定宽度的控制台表格。
    print(f"{'Region':42s} {'n':>6s} {'accuracy':>9s} {'water<->land':>13s}")
    for name, row in table.items():
        print(f"{name:42s} {row['n_test']:6d} {100 * row['accuracy']:8.2f}% {row['water_land_errors']:13d}")
    print(f"{'All regions':42s} {summary['n_test']:6d} {100 * summary['overall_accuracy']:8.2f}% "
          f"{summary['overall_water_land_errors']:13d}")
    print(f"\nwritten {OUTPUT}")
    return 0


if __name__ == "__main__":
    # Exit code of the process = return value of main(). 进程退出码。
    raise SystemExit(main())
