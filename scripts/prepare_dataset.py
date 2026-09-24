#!/usr/bin/env python3
"""Command-line entry point for dataset preparation.

用法: python scripts/prepare_dataset.py --input data/raw/merged.geojson

First step of the offline pipeline (prepare -> train -> evaluate). It runs
:func:`maritime_route.data.dataset.prepare`, which loads and validates the
labelled GeoJSON corpus, removes mislabelled points, splits it 70/15/15,
builds the reference index and writes the feature matrices. Afterwards run
``scripts/train_model.py``.

Options
-------
``--input``     labelled GeoJSON corpus (default ``data/raw/merged.geojson``)
``--output``    processed ``.npz`` (default ``data/processed/zone_dataset.npz``)
``--index``     reference index ``.npz`` (default ``models/reference_index.npz``)
``--limit N``   read only the first N features, for quick experiments
``--no-clean``  keep points whose class contradicts their height/depth

Output on stdout: JSON with the split sizes, the class distribution and the
cleaning report.

离线流程第一步：加载、校验、清洗、划分数据并提取特征；随后运行 train_model.py。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Make ``src`` importable when the script is run directly from a checkout
# (the package does not need to be installed). parents[1] = project root.
# 直接运行脚本时将 src 加入导入路径（无需安装包）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maritime_route.config import BATHY_INDEX_PATH, DEFAULT_RAW_DATASET
from maritime_route.data.dataset import PROCESSED_NPZ, prepare


def main() -> int:
    """Parse the command line, run the preparation and print a summary.

    Returns
    -------
    int
        Process exit code, 0 on success (errors propagate as exceptions).

    解析命令行参数，执行预处理并输出摘要。
    """
    parser = argparse.ArgumentParser(description="Prepare the zone classification dataset")
    parser.add_argument("--input", default=str(DEFAULT_RAW_DATASET), help="raw GeoJSON file")
    parser.add_argument("--output", default=str(PROCESSED_NPZ), help="processed .npz file")
    parser.add_argument("--index", default=str(BATHY_INDEX_PATH), help="reference index .npz")
    parser.add_argument("--limit", type=int, default=None, help="only read N features")
    parser.add_argument("--no-clean", action="store_true",
                        help="keep the points whose class contradicts their height/depth")
    args = parser.parse_args()

    # INFO level so that the progress messages of prepare() are visible.
    # 使用 INFO 级别以显示 prepare() 的进度信息。
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    split, stats = prepare(args.input, args.output, args.index, limit=args.limit,
                           clean=not args.no_clean)
    print(json.dumps({"split": split.describe(),
                      "classes": stats.get("class_distribution", {}),
                      "cleaning": stats.get("cleaning")}, indent=2))
    return 0


if __name__ == "__main__":
    # SystemExit passes main()'s return value to the shell as the exit code.
    # 将 main() 的返回值作为进程退出码。
    raise SystemExit(main())
