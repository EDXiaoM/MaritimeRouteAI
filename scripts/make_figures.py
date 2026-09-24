#!/usr/bin/env python3
"""Generate every figure used in the explanatory note.

生成说明书所需的全部插图（均基于真实训练与测试结果）。
Figures are written to docs/figures/ as 200 dpi PNG.

Inputs
------
* ``config.MODEL_METADATA_PATH`` - metadata of the trained classifier
  (training history, confusion matrix, per-class metrics); written by
  ``scripts/train_model.py``.
* ``data.dataset.STATS_JSON`` - corpus statistics (class counts, depth
  percentiles); written by ``scripts/prepare_dataset.py``.
* ``docs/figures/routing_benchmark.json`` - optional; written by
  ``scripts/benchmark_routing.py``.
* The trained model itself, for the genetic-algorithm convergence figure,
  which runs the GA live.

Figures
-------
=============================  ================================================
File                           Content
=============================  ================================================
fig_class_distribution.png     points per class in the training corpus
fig_depth_by_class.png         5th-95th percentile of NOAA depth per class
fig_training_curves.png        loss and accuracy per epoch
fig_confusion_matrix.png       test-set confusion matrix (counts and shares)
fig_per_class_metrics.png      precision, recall, F1 per class
fig_cost_model.png             traversal weight of each zone
fig_routing_benchmark.png      A* vs Dijkstra effort, route length penalty
fig_genetic_convergence.png    GA best cost per generation vs A* optimum
=============================  ================================================

A single visual style (colours, grid, fonts) is set once through
``plt.rcParams`` so that all figures of the thesis look alike.

Usage::

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the package importable without installation (adds <project>/src).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
# Non-interactive backend: renders to files, works without a display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from maritime_route.config import (
    CLASS_COLOR, CLASS_NAMES, FIGURE_DIR, MODEL_METADATA_PATH, SERIES_COLOR,
)
from maritime_route.data.dataset import STATS_JSON

#: Output resolution, dots per inch (print quality for the thesis).
DPI = 200
#: Palette: background, primary text, secondary text and grid lines.
#: 配色：背景色、主文字色、次文字色、网格线颜色。
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = SERIES_COLOR["grid"]

# Global matplotlib style shared by every figure: light background, light
# grid behind the data, no top/right frame lines. 全局绘图样式。
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

#: Short axis labels for the four zone classes. 类别的简短标签。
SHORT = {"OPEN_SEA": "Open sea", "COASTAL_SEA": "Coastal sea",
         "NEAR_COAST": "Near coast", "COASTLINE": "Inland"}


def save(fig, name: str) -> None:
    """Save a figure into ``FIGURE_DIR`` and release its memory.

    ``bbox_inches="tight"`` crops the white margin around the drawing.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        File name, e.g. ``"fig_cost_model.png"``.
    """
    path = FIGURE_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}")


def load(path: Path) -> dict:
    """Read a UTF-8 JSON file.

    Parameters
    ----------
    path : Path
        JSON file.

    Returns
    -------
    dict
        Parsed content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist (e.g. the model has not been trained).
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
def fig_class_distribution(stats: dict) -> None:
    """Bar chart of the number of points per class in the corpus.

    Each bar is labelled with the count and its share of the total.

    Parameters
    ----------
    stats : dict
        Corpus statistics; uses ``class_distribution[class]["count"]``.
    """
    dist = stats["class_distribution"]
    counts = [dist[c]["count"] for c in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(6.2, 3.1))
    bars = ax.bar([SHORT[c] for c in CLASS_NAMES], counts,
                  color=[CLASS_COLOR[c] for c in CLASS_NAMES],
                  edgecolor=SURFACE, linewidth=2, width=0.66)
    total = sum(counts)
    for bar, n in zip(bars, counts):
        # Label just above the bar (offset 1.2 % of the total count).
        ax.text(bar.get_x() + bar.get_width() / 2, n + total * 0.012,
                f"{n:,}\n{100 * n / total:.1f}%", ha="center", va="bottom",
                fontsize=8.5, color=INK)
    ax.set_ylabel("Number of points")
    ax.set_ylim(0, max(counts) * 1.22)
    ax.set_title(f"Class distribution of the training corpus (n = {total:,})",
                 fontsize=10, color=INK, pad=10)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_class_distribution.png")


def fig_depth_by_class(stats: dict) -> None:
    """Two panels: the full bathymetric range and a zoom on the shoreline band.

    左图为完整水深范围，右图放大海岸带——两个"困难类别"正挤在此处。

    For every class a thick bar spans the 5th to 95th percentile of the NOAA
    relief value (metres, negative = below sea level) and a ring marks the
    median. The right panel is limited to -320 .. +900 m and annotates the
    medians that fall inside it. Nothing is drawn if the statistics contain
    no depth data.

    Parameters
    ----------
    stats : dict
        Corpus statistics; uses ``depth_by_class[class]`` with ``p05``,
        ``p50`` and ``p95``.
    """
    depth = stats.get("depth_by_class", {})
    if not depth:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    # (lo, hi) = x-axis limits of each panel; None = automatic (full range).
    for ax, (lo, hi), title in zip(
        axes, [(None, None), (-320, 900)],
        ["Full range", "Zoom on the shoreline band"],
    ):
        for i, code in enumerate(CLASS_NAMES):
            d = depth.get(code)
            if not d:
                continue
            ax.plot([d["p05"], d["p95"]], [i, i], color=CLASS_COLOR[code],
                    linewidth=7, solid_capstyle="round", zorder=2)
            ax.plot(d["p50"], i, "o", color=SURFACE, markersize=9,
                    markeredgecolor=CLASS_COLOR[code], markeredgewidth=2.2, zorder=3)
        # Dashed vertical line at 0 m = sea level.
        ax.axvline(0, color=INK_2, linewidth=1, linestyle="--", zorder=1)
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_yticklabels([SHORT[c] for c in CLASS_NAMES] if ax is axes[0] else [])
        ax.set_xlabel("NOAA elevation / depth, m")
        ax.set_title(title, fontsize=9.5, color=INK)
        ax.set_ylim(-0.6, 3.6)
        ax.grid(axis="y", visible=False)
        if lo is not None:
            ax.set_xlim(lo, hi)
            for i, code in enumerate(CLASS_NAMES):
                d = depth.get(code)
                if d and lo < d["p50"] < hi:
                    ax.annotate(f"median {d['p50']:,.0f} m", (d["p50"], i),
                                textcoords="offset points", xytext=(0, 13),
                                ha="center", fontsize=7.5, color=INK_2)
    axes[0].text(0, 3.45, "sea level", fontsize=7.5, color=INK_2, ha="center")
    fig.suptitle("Bathymetric signature of each class (5th-95th percentile)",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, "fig_depth_by_class.png")


def fig_training_curves(meta: dict) -> None:
    """Loss and accuracy per epoch for the training and validation sets.

    The epoch with the best validation accuracy is marked with a ring and
    annotated. (The trainer restores the weights of the epoch with the lowest
    validation *loss*, which may be a neighbouring epoch.)

    Parameters
    ----------
    meta : dict
        Model metadata; uses ``history`` with ``epoch``, ``train_loss``,
        ``val_loss``, ``train_accuracy`` and ``val_accuracy`` (fractions).
    """
    history = meta["history"]
    epochs = history["epoch"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.1))

    ax1.plot(epochs, history["train_loss"], color=SERIES_COLOR["primary"],
             linewidth=2, label="Training")
    ax1.plot(epochs, history["val_loss"], color=SERIES_COLOR["secondary"],
             linewidth=2, linestyle="--", label="Validation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Loss", fontsize=10, color=INK)
    ax1.legend(frameon=False, fontsize=8)

    ax2.plot(epochs, [100 * v for v in history["train_accuracy"]],
             color=SERIES_COLOR["primary"], linewidth=2, label="Training")
    ax2.plot(epochs, [100 * v for v in history["val_accuracy"]],
             color=SERIES_COLOR["secondary"], linewidth=2, linestyle="--",
             label="Validation")
    best = int(np.argmax(history["val_accuracy"]))
    ax2.plot(epochs[best], 100 * history["val_accuracy"][best], "o",
             color=SURFACE, markersize=8,
             markeredgecolor=SERIES_COLOR["secondary"], markeredgewidth=2)
    ax2.annotate(f"best {100 * history['val_accuracy'][best]:.2f}%",
                 (epochs[best], 100 * history["val_accuracy"][best]),
                 textcoords="offset points", xytext=(-14, -22), fontsize=8, color=INK_2)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy, %")
    ax2.set_title("Accuracy", fontsize=10, color=INK)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.suptitle("Training dynamics of the zone classifier", fontsize=10.5, color=INK)
    fig.tight_layout()
    save(fig, "fig_training_curves.png")


def fig_confusion(meta: dict) -> None:
    """Confusion matrix of the test set, as counts and row percentages.

    Rows are true classes, columns predicted classes. Each row is divided by
    its sum, so a cell shows the share of that true class predicted as the
    column class and the diagonal is the per-class recall.

    Parameters
    ----------
    meta : dict
        Model metadata; uses ``metrics.confusion_matrix`` and
        ``metrics.accuracy``.
    """
    matrix = np.array(meta["metrics"]["confusion_matrix"], dtype=float)
    # Row normalisation: share of each true class. 按行归一化。
    normalised = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    # Sequential single-hue ramp: magnitude, so one hue light -> dark.
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            # White text on dark cells (share > 0.5), dark text otherwise.
            ax.text(j, i, f"{int(matrix[i, j]):,}\n{100 * normalised[i, j]:.1f}%",
                    ha="center", va="center", fontsize=8,
                    color="#ffffff" if normalised[i, j] > 0.5 else INK)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([SHORT[c] for c in CLASS_NAMES], rotation=22, ha="right")
    ax.set_yticklabels([SHORT[c] for c in CLASS_NAMES])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"Confusion matrix, test set (accuracy "
                 f"{100 * meta['metrics']['accuracy']:.2f}%)", fontsize=10, color=INK, pad=10)
    ax.grid(False)
    fig.colorbar(image, ax=ax, fraction=0.045, label="Share of true class")
    save(fig, "fig_confusion_matrix.png")


def fig_per_class_metrics(meta: dict) -> None:
    """Grouped bars of precision, recall and F1 for each class.

    The y-axis starts at 0.85 to make the differences between classes
    visible; values are printed on the bars.

    Parameters
    ----------
    meta : dict
        Model metadata; uses ``metrics.per_class[class]`` with
        ``precision``, ``recall`` and ``f1``.
    """
    per = meta["metrics"]["per_class"]
    metrics = ["precision", "recall", "f1"]
    labels = ["Precision", "Recall", "F1"]
    colours = [SERIES_COLOR["primary"], SERIES_COLOR["secondary"], SERIES_COLOR["tertiary"]]
    x = np.arange(len(CLASS_NAMES)); width = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for k, (metric, label, colour) in enumerate(zip(metrics, labels, colours)):
        values = [per[c][metric] for c in CLASS_NAMES]
        # Offsets -width, 0, +width place the three bars side by side.
        bars = ax.bar(x + (k - 1) * width, values, width * 0.9, label=label,
                      color=colour, edgecolor=SURFACE, linewidth=1.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=6.8, color=INK_2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CLASS_NAMES])
    ax.set_ylim(0.85, 1.03); ax.set_ylabel("Score")
    ax.set_title("Per-class quality on the held-out test set", fontsize=10, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_per_class_metrics.png")


def fig_routing_benchmark(bench: dict) -> None:
    """Two panels summarising ``routing_benchmark.json``.

    Left: nodes expanded by A* and Dijkstra per voyage (log scale, because
    the counts differ by up to an order of magnitude). Right: length of the
    optimised route against the great-circle distance, labelled with the
    extra length in per cent.

    Parameters
    ----------
    bench : dict
        Content of ``routing_benchmark.json`` (``cases`` list).
    """
    cases = bench["cases"]
    # Drop the parenthesised sea name, e.g. "Kiel - Tallinn (Baltic)" -> "Kiel - Tallinn".
    names = [c["case"].split(" (")[0] for c in cases]
    astar = [c["astar"]["nodes_expanded"] for c in cases]
    dijkstra = [c["dijkstra"]["nodes_expanded"] for c in cases]
    y = np.arange(len(cases)); height = 0.36

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.9),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    ax1.barh(y + height / 2, dijkstra, height * 0.92, label="Dijkstra",
             color=SERIES_COLOR["secondary"], edgecolor=SURFACE, linewidth=1.5)
    ax1.barh(y - height / 2, astar, height * 0.92, label="A*",
             color=SERIES_COLOR["primary"], edgecolor=SURFACE, linewidth=1.5)
    for i, (a, d) in enumerate(zip(astar, dijkstra)):
        ax1.text(d * 1.03, i + height / 2, f"{d:,}", va="center", fontsize=7.5, color=INK_2)
        ax1.text(a * 1.03, i - height / 2, f"{a:,}", va="center", fontsize=7.5, color=INK_2)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=8)
    ax1.invert_yaxis(); ax1.set_xscale("log")
    ax1.set_xlabel("Nodes expanded (log scale)")
    ax1.set_title("Search effort", fontsize=10, color=INK, pad=8)
    ax1.set_xlim(right=max(dijkstra) * 2.4)
    # Legends sit outside the plot area so they cannot collide with the bars
    # or their value labels. 图例置于绘图区上方，避免与数据标签重叠。
    # (bbox_to_anchor y = -0.17 actually places them just *below* the axes.
    # 注：y = -0.17 实际上位于坐标轴下方。)
    ax1.legend(frameon=False, fontsize=8, ncol=2,
               loc="upper left", bbox_to_anchor=(0, -0.17))
    ax1.grid(axis="y", visible=False)

    optimal = [c["astar"]["distance_km"] for c in cases]
    great = [c["great_circle"]["distance_km"] for c in cases]
    ax2.barh(y + height / 2, great, height * 0.92, label="Great circle (crosses land)",
             color=CLASS_COLOR["NEAR_COAST"], edgecolor=SURFACE, linewidth=1.5)
    ax2.barh(y - height / 2, optimal, height * 0.92, label="Optimised (navigable)",
             color=CLASS_COLOR["OPEN_SEA"], edgecolor=SURFACE, linewidth=1.5)
    for i, (o, g) in enumerate(zip(optimal, great)):
        # Extra length of the safe route relative to the great circle, %.
        ax2.text(o * 1.02, i - height / 2, f"+{100 * (o / g - 1):.0f}%",
                 va="center", fontsize=7.5, color=INK_2)
    ax2.set_yticks(y); ax2.set_yticklabels([]); ax2.invert_yaxis()
    ax2.set_xlabel("Route length, km")
    ax2.set_title("Length penalty of a safe route", fontsize=10, color=INK, pad=8)
    ax2.set_xlim(right=max(optimal) * 1.18)
    ax2.legend(frameon=False, fontsize=7.5, ncol=2,
               loc="upper left", bbox_to_anchor=(0, -0.17))
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Routing benchmark over six voyages", fontsize=10.5, color=INK, y=1.0)
    fig.tight_layout()
    save(fig, "fig_routing_benchmark.png")


def fig_cost_model() -> None:
    """Bar chart of the traversal weight w(z) assigned to each zone.

    Impassable zones (infinite weight) are drawn as zero-height bars
    labelled "impassable". Values come from ``config.ZONE_COST``.
    """
    from maritime_route.config import ZONE_COST
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    values = [ZONE_COST[c] if np.isfinite(ZONE_COST[c]) else 0 for c in CLASS_NAMES]
    bars = ax.bar([SHORT[c] for c in CLASS_NAMES], values,
                  color=[CLASS_COLOR[c] for c in CLASS_NAMES],
                  edgecolor=SURFACE, linewidth=2, width=0.6)
    for bar, code, v in zip(bars, CLASS_NAMES, values):
        label = "impassable" if not np.isfinite(ZONE_COST[code]) else f"{v:.1f}"
        ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.06, label,
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_ylabel("Traversal weight w(z)")
    ax.set_ylim(0, 2.3)
    ax.set_title("Traversal cost assigned to each classified zone",
                 fontsize=10, color=INK, pad=10)
    ax.grid(axis="x", visible=False)
    save(fig, "fig_cost_model.png")


def fig_genetic_convergence() -> None:
    """Best route cost per generation of the genetic algorithm (three seeds).

    The dashed line is the optimum proven by A* on the same cost map, so the
    distance between the curves and the line is the optimality gap of the GA.
    遗传算法每代最优代价（三个随机种子），虚线为 A* 在同一代价图上的最优值。

    Runs A* once on the Kiel - Tallinn voyage to build the cost map and the
    reference optimum, then the GA three times (seeds 1, 2, 3) on the same
    map, snapping the ports exactly as the planner does. The y-axis is
    logarithmic because early generations still pay the land penalty and
    are orders of magnitude more expensive. A run whose best route still
    touches land is labelled as stuck in a local optimum.
    """
    import logging
    logging.getLogger().setLevel(logging.WARNING)     # silence planner INFO logs
    from maritime_route.model.inference import ZoneClassifierService
    from maritime_route.routing.cost_map import nearest_navigable
    from maritime_route.routing.genetic import genetic
    from maritime_route.routing.planner import RoutePlanner

    start, end = (54.3233, 10.1228), (59.4370, 24.7536)     # Kiel - Tallinn
    planner = RoutePlanner(ZoneClassifierService.instance())
    plan, cost_map = planner.plan(start, end, "astar")
    spec = cost_map.spec
    s = nearest_navigable(cost_map, *spec.coord_to_cell(*start))
    g = nearest_navigable(cost_map, *spec.coord_to_cell(*end))
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    colours = [SERIES_COLOR["primary"], SERIES_COLOR["secondary"], SERIES_COLOR["tertiary"]]
    for seed, colour in zip((1, 2, 3), colours):
        result = genetic(cost_map, s, g, seed=seed)
        ax.plot(range(len(result.history)), result.history, color=colour, lw=1.4,
                label=(f"seed {seed}: {result.total_cost:,.0f}" if result.found
                       else f"seed {seed}: stuck on land (local optimum)"))
    ax.axhline(plan.total_cost, color=INK, ls="--", lw=1.2,
               label=f"A* optimum: {plan.total_cost:,.0f}")
    ax.set_yscale("log")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best route cost (log scale)")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_title("Genetic algorithm on Kiel - Tallinn: best cost per generation",
                 fontsize=10, color=INK, pad=8)
    save(fig, "fig_genetic_convergence.png")


def fig_value_function() -> None:
    """The Bellman value function V of one voyage, with the optimal route.

    V(v) is the minimum cost of reaching the destination from cell v. It is
    drawn as a colour map with iso-cost contours; land cells have no value and
    are drawn grey. The optimal route follows the steepest descent of V, i.e.
    it crosses the contours at right angles wherever the zone weight is uniform.
    贝尔曼价值函数 V 的等值图及最优航线（沿 V 的最速下降方向）。
    """
    import logging
    logging.getLogger().setLevel(logging.WARNING)
    from maritime_route.model.inference import ZoneClassifierService
    from maritime_route.routing.planner import RoutePlanner

    start, end = (46.5209, 30.7727), (44.7252, 37.7930)     # Odesa - Novorossiysk
    planner = RoutePlanner(ZoneClassifierService.instance())
    _, cost_map = planner.plan(start, end, "astar")
    plan, _ = planner.plan(start, end, "dynamic", cost_map.spec.resolution_deg,
                           reuse_cost_map=cost_map, auto_refine=False)
    from maritime_route.routing.cost_map import nearest_navigable
    from maritime_route.routing.dynamic_programming import value_iteration
    spec = cost_map.spec
    goal = nearest_navigable(cost_map, *spec.coord_to_cell(*end))
    value, sweeps, _ = value_iteration(cost_map, goal)
    lon = spec.lon_min + np.arange(spec.n_cols) * spec.resolution_deg
    lat = spec.lat_min + np.arange(spec.n_rows) * spec.resolution_deg
    shown = np.where(np.isfinite(value), value, np.nan)

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.set_facecolor("#c9c7c1")                       # land / unreachable cells
    mesh = ax.pcolormesh(lon, lat, shown, cmap="viridis_r", shading="nearest")
    ax.contour(lon, lat, shown, levels=12, colors="white", linewidths=0.5, alpha=0.8)
    pts = np.array(plan.waypoints)
    ax.plot(pts[:, 1], pts[:, 0], color="#eb6834", lw=2.0, label="optimal route (greedy policy)")
    ax.plot(start[1], start[0], "o", color="#1baf7a", ms=6)
    ax.plot(end[1], end[0], "s", color="#eb6834", ms=6)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("V(v): cost to the destination", fontsize=8)
    ax.set_xlabel("Longitude, °")
    ax.set_ylabel("Latitude, °")
    ax.legend(fontsize=7.5, loc="upper right", frameon=True)
    ax.grid(False)
    ax.set_title(f"Bellman value function, Odesa - Novorossiysk ({sweeps} sweeps)",
                 fontsize=10, color=INK, pad=8)
    save(fig, "fig_value_function.png")


def main() -> int:
    """Generate all figures; the benchmark figure only if its JSON exists.

    Returns
    -------
    int
        Process exit code (0).
    """
    print("Generating figures:")
    meta = load(MODEL_METADATA_PATH)
    stats = load(STATS_JSON)
    # The two corpus figures describe the corpus as delivered, before cleaning.
    # 两幅语料图描述清洗前的原始语料。
    raw = stats.get("raw", stats)
    fig_class_distribution(raw)
    fig_depth_by_class(raw)
    fig_training_curves(meta)
    fig_confusion(meta)
    fig_per_class_metrics(meta)
    fig_cost_model()
    bench_path = FIGURE_DIR / "routing_benchmark.json"
    if bench_path.exists():
        fig_routing_benchmark(load(bench_path))
    fig_genetic_convergence()
    fig_value_function()
    print(f"\nFigures written to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
