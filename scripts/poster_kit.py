"""Drawing primitives for the A1 graphic material.

A1 图纸绘制工具：统一的标题栏、方框、箭头、表格与图片框。
Everything is drawn on a single normalised axes (0..1 in both directions) of an
A1 sheet (594 x 841 mm), so a poster script only places elements by fraction.

Place in the application
------------------------
Not part of the running application. It is a helper library for
``scripts/make_posters.py``, which produces the A1 sheets (graphic material)
presented with the diploma project. Every sheet shares the same frame, header
strip and title block, and every diagram uses the same boxes, arrows and
tables, so they are defined once here.

Coordinate system
-----------------
``new_sheet`` creates one Matplotlib axes that covers the whole figure with
limits 0..1 on both axes. A position such as ``(0.5, 0.5)`` is therefore the
centre of the sheet, and all sizes are fractions of the sheet width (x) or
height (y). Because the sheet is landscape (33.11 x 23.39 in), one unit in x is
physically longer than one unit in y; :func:`image` corrects for this when it
preserves an image's aspect ratio.
坐标系：整张图纸为 0..1 的归一化坐标；横向单位长度大于纵向单位长度。

Main objects
------------
Constants ``A1_W_IN``, ``A1_H_IN``, ``DPI`` and the colour palette (``INK``,
``PANEL``, ``ACCENT``...); functions ``new_sheet``, ``panel``, ``box``,
``diamond``, ``arrow``, ``bullets``, ``table``, ``image``, ``legend_chips``
and ``save``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Make the in-repository package importable without installing it
# (scripts/ is a sibling of src/). 将 src 目录加入导入路径。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
# Non-interactive backend: files are written without opening a window, which
# also works on a server without a display. 使用无界面后端。
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from maritime_route.config import APP_AUTHOR, APP_UNIVERSITY, CLASS_COLOR, SERIES_COLOR

# A1 in inches, landscape.
# 841 mm x 594 mm = 33.11 in x 23.39 in. A1 横向尺寸（英寸）。
A1_W_IN, A1_H_IN = 33.11, 23.39
#: Raster resolution of the PNG output (about 4 970 x 3 510 pixels).
DPI = 150

# Colour palette of the sheets: three ink greys for text hierarchy, white
# sheet, light-grey panels, grey borders, dark-blue accent (header strip,
# titles, table headers) and the primary series colour from config.
# 图纸配色：三级文字灰度、面板底色、边框色与强调蓝。
INK = "#0b0b0b"
INK_2 = "#3f4750"
INK_3 = "#6b7681"
SURFACE = "#ffffff"
PANEL = "#f4f6f8"
BORDER = "#b9c4ce"
ACCENT = "#14507f"
ACCENT_2 = SERIES_COLOR["primary"]

# DejaVu Sans ships with Matplotlib and contains Cyrillic and the symbols used
# on the sheets, so output does not depend on system fonts.
plt.rcParams.update({"font.family": "DejaVu Sans"})


def new_sheet(number: int, title: str, subtitle: str = "") -> Tuple[plt.Figure, plt.Axes]:
    """Create an A1 sheet with a frame and the standard title block.

    Draws a double frame (outer thick, inner thin line), a dark-blue header
    strip with the sheet title, subtitle and number, and a title block in the
    bottom-right corner with the project title, student and university.

    Parameters
    ----------
    number : int
        Sheet number shown in the header ("Sheet N").
    title : str
        Large title in the header strip.
    subtitle : str, optional
        Smaller line under the title; omitted when empty.

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
        The figure and its single full-page axes with limits 0..1 and hidden
        axis lines; all further drawing goes into this axes.
    """
    fig = plt.figure(figsize=(A1_W_IN, A1_H_IN), facecolor=SURFACE)
    # One axes covering the whole figure: [left, bottom, width, height] = [0, 0, 1, 1].
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Drawing frame (ESKD-style margins).
    # ESKD = the Russian/CIS unified system for design documentation.
    ax.add_patch(Rectangle((0.012, 0.014), 0.976, 0.972, fill=False,
                           edgecolor=INK, linewidth=2.2, zorder=1))
    ax.add_patch(Rectangle((0.018, 0.020), 0.964, 0.960, fill=False,
                           edgecolor=INK, linewidth=0.9, zorder=1))

    # Header strip.
    ax.add_patch(Rectangle((0.018, 0.918), 0.964, 0.062, facecolor=ACCENT,
                           edgecolor="none", zorder=2))
    ax.text(0.032, 0.957, title, fontsize=27, color="#ffffff",
            va="center", ha="left", fontweight="bold", zorder=3)
    if subtitle:
        ax.text(0.032, 0.932, subtitle, fontsize=13.5, color="#cfe2f2",
                va="center", ha="left", zorder=3)
    ax.text(0.968, 0.957, f"Sheet {number}", fontsize=17, color="#ffffff",
            va="center", ha="right", fontweight="bold", zorder=3)
    ax.text(0.968, 0.932, "Format A1", fontsize=11.5, color="#cfe2f2",
            va="center", ha="right", zorder=3)

    # Title block at the bottom right.
    ax.add_patch(Rectangle((0.678, 0.020), 0.304, 0.052, facecolor=PANEL,
                           edgecolor=INK, linewidth=1.0, zorder=2))
    ax.text(0.690, 0.058,
            "Application for constructing a maritime vessel's course based on "
            "neural-network classification of geographic zones",
            fontsize=8.0, color=INK, va="center", ha="left", zorder=3, wrap=True)
    ax.text(0.690, 0.038, f"Diploma project   ·   Student: {APP_AUTHOR}",
            fontsize=9.0, color=INK_2, va="center", ha="left", zorder=3)
    ax.text(0.690, 0.026, f"{APP_UNIVERSITY}   ·   Faculty EIS, Department IIT   ·   2026/27",
            fontsize=8.0, color=INK_3, va="center", ha="left", zorder=3)
    return fig, ax


def panel(ax, x, y, w, h, title: str = "", facecolor: str = PANEL,
          edgecolor: str = BORDER, title_size: float = 15) -> Tuple[float, float]:
    """Draw a titled panel; returns the content origin (x, y_top).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sheet axes from :func:`new_sheet`.
    x, y : float
        Lower-left corner in sheet fractions.
    w, h : float
        Width and height in sheet fractions.
    title : str, optional
        Heading drawn inside the top edge, underlined by a thin rule.
    facecolor, edgecolor : str, optional
        Panel fill and border colours.
    title_size : float, optional
        Heading font size in points.

    Returns
    -------
    tuple of float
        ``(x, y_top)``: left edge and top of the free content area, already
        inset by the padding and, if a title is drawn, placed below the rule.
    """
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.006",
                                facecolor=facecolor, edgecolor=edgecolor,
                                linewidth=1.3, zorder=2))
    if title:
        ax.text(x + 0.010, y + h - 0.020, title, fontsize=title_size, color=ACCENT,
                va="center", ha="left", fontweight="bold", zorder=3)
        ax.plot([x + 0.010, x + w - 0.010], [y + h - 0.033, y + h - 0.033],
                color=BORDER, linewidth=1.0, zorder=3)
        return x + 0.010, y + h - 0.045
    return x + 0.010, y + h - 0.012


def box(ax, x, y, w, h, text: str, facecolor: str = "#ffffff",
        edgecolor: str = ACCENT, textcolor: str = INK, fontsize: float = 11,
        style: str = "round", linewidth: float = 1.6, zorder: int = 4,
        bold: bool = False):
    """A labelled node of a block diagram or flowchart.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sheet axes.
    x, y, w, h : float
        Lower-left corner and size in sheet fractions.
    text : str
        Centred label; may contain ``\\n`` for several lines.
    facecolor, edgecolor, textcolor : str, optional
        Fill, border and text colours.
    fontsize : float, optional
        Label size in points.
    style : {"round", "sharp", "terminal"}, optional
        Corner style: rounded process box, square box, or strongly rounded
        start/end (terminal) node of a flowchart.
    linewidth : float, optional
        Border width.
    zorder : int, optional
        Drawing order of the box; the label is drawn at ``zorder + 1``.
    bold : bool, optional
        Bold label.

    Returns
    -------
    tuple of float
        Centre ``(cx, cy)`` of the box, convenient as an arrow end point.

    Raises
    ------
    KeyError
        If ``style`` is not one of the three names.
    """
    boxstyle = {
        "round": "round,pad=0.004,rounding_size=0.008",
        "sharp": "square,pad=0.004",
        "terminal": "round,pad=0.004,rounding_size=0.022",
    }[style]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=boxstyle, facecolor=facecolor,
                                edgecolor=edgecolor, linewidth=linewidth, zorder=zorder))
    ax.text(x + w / 2, y + h / 2, text, fontsize=fontsize, color=textcolor,
            ha="center", va="center", zorder=zorder + 1, linespacing=1.45,
            fontweight="bold" if bold else "normal")
    return (x + w / 2, y + h / 2)


def diamond(ax, cx, cy, w, h, text: str, fontsize: float = 10,
            facecolor: str = "#fff6e5", edgecolor: str = "#b3730a"):
    """Decision node of a flowchart. 判断节点。

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sheet axes.
    cx, cy : float
        Centre of the diamond.
    w, h : float
        Full width and height (distance between opposite corners).
    text : str
        Centred condition text.
    fontsize : float, optional
        Text size in points.
    facecolor, edgecolor : str, optional
        Fill and border colours (amber by default to set decisions apart).

    Returns
    -------
    dict of str to tuple
        The four corner points ``"top"``, ``"right"``, ``"bottom"``,
        ``"left"``, used as start points for the yes/no arrows.
    """
    # Corners in clockwise order starting at the top. 从顶点开始顺时针的四个角。
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor=facecolor,
                             edgecolor=edgecolor, linewidth=1.6, zorder=4))
    ax.text(cx, cy, text, fontsize=fontsize, color=INK, ha="center", va="center",
            zorder=5, linespacing=1.35)
    return {"top": (cx, cy + h / 2), "right": (cx + w / 2, cy),
            "bottom": (cx, cy - h / 2), "left": (cx - w / 2, cy)}


def arrow(ax, start: Tuple[float, float], end: Tuple[float, float],
          label: str = "", colour: str = INK_2, style: str = "-|>",
          linewidth: float = 1.7, connection: str = "arc3,rad=0",
          label_offset: Tuple[float, float] = (0.004, 0.004),
          fontsize: float = 9.5, linestyle: str = "-"):
    """Draw a connector with an optional label.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sheet axes.
    start, end : tuple of float
        End points in sheet fractions.
    label : str, optional
        Text placed at the midpoint of the straight line between the end
        points (plus ``label_offset``) on a white background, so it stays
        readable over other lines.
    colour : str, optional
        Line, arrow-head and label colour.
    style : str, optional
        Matplotlib arrow style, e.g. ``"-|>"`` (filled head) or ``"<|-|>"``.
    linewidth : float, optional
        Line width.
    connection : str, optional
        Matplotlib connection style; ``"arc3,rad=r"`` bends the line,
        ``"angle,..."`` gives right-angled connectors.
    label_offset : tuple of float, optional
        Shift of the label from the midpoint.
    fontsize : float, optional
        Label size.
    linestyle : str, optional
        ``"-"`` solid, ``"--"`` dashed (e.g. for optional flows).
    """
    # shrinkA/shrinkB leave a 2-point gap so the arrow does not touch the boxes.
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, color=colour,
                                 linewidth=linewidth, mutation_scale=22,
                                 connectionstyle=connection, zorder=3,
                                 linestyle=linestyle, shrinkA=2, shrinkB=2))
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, fontsize=fontsize, color=colour, ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.22", facecolor=SURFACE,
                          edgecolor="none", alpha=0.92))


def bullets(ax, x, y, lines: Sequence[str], fontsize: float = 11.5,
            leading: float = 0.0205, colour: str = INK, marker: str = "•",
            width: Optional[float] = None) -> float:
    """Render a bullet list downwards; returns the y of the last line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sheet axes.
    x, y : float
        Top-left position of the first line.
    lines : sequence of str
        List items. An item that starts with ``"~"`` is a bold sub-heading
        without a bullet (the ``~`` is removed). Items may contain ``\\n``.
    fontsize : float, optional
        Text size.
    leading : float, optional
        Vertical step per item in sheet fractions; an item with ``k``
        embedded line breaks takes ``1 + 0.85 k`` steps.
    colour : str, optional
        Text colour.
    marker : str, optional
        Bullet character.
    width : float, optional
        Currently not used by the function.

    Returns
    -------
    float
        The y coordinate just below the last item, where the next element
        can start.
    """
    for line in lines:
        # "~Heading" -> no bullet, bold; otherwise "• text". 以 ~ 开头为加粗小标题。
        prefix, text = (("", line[1:].strip()) if line.startswith("~") else (marker + "  ", line))
        ax.text(x, y, f"{prefix}{text}", fontsize=fontsize, color=colour,
                va="top", ha="left", zorder=4, linespacing=1.4,
                fontweight="bold" if line.startswith("~") else "normal")
        y -= leading * (1 + 0.85 * text.count("\n"))
    return y


def table(ax, x, y, w, rows: Sequence[Sequence[str]], col_w: Sequence[float],
          row_h: float = 0.0215, fontsize: float = 10.5, header: bool = True,
          align: Optional[Sequence[str]] = None):
    """Simple grid table. 表格绘制。

    Rows are drawn from the top down. The header row (if any) has a
    dark-blue background with white bold text; body rows alternate between
    white (even index) and the underlying panel colour (odd index), and a thin
    rule is drawn under every row.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sheet axes.
    x, y : float
        Top-left corner of the table.
    w : float
        Total table width.
    rows : sequence of sequence of str
        Cell texts; the first row is the header when ``header`` is true.
    col_w : sequence of float
        Relative column widths; they are normalised to sum to ``w``.
    row_h : float, optional
        Row height in sheet fractions.
    fontsize : float, optional
        Cell text size.
    header : bool, optional
        Style the first row as a header.
    align : sequence of {"left", "center", "right"}, optional
        Horizontal alignment per column; default all left.

    Returns
    -------
    float
        The y coordinate of the bottom edge of the table.
    """
    align = align or ["left"] * len(col_w)
    # Relative widths -> absolute widths that add up to w. 相对列宽换算为实际宽度。
    total = sum(col_w)
    widths = [c / total * w for c in col_w]
    top = y
    for r, row in enumerate(rows):
        yy = top - r * row_h
        if header and r == 0:
            ax.add_patch(Rectangle((x, yy - row_h), w, row_h, facecolor=ACCENT,
                                   edgecolor="none", zorder=3))
        elif r % 2 == 0:
            ax.add_patch(Rectangle((x, yy - row_h), w, row_h, facecolor="#ffffff",
                                   edgecolor="none", zorder=3))
        cx = x
        for c, cell in enumerate(row):
            colour = "#ffffff" if (header and r == 0) else INK
            weight = "bold" if (header and r == 0) else "normal"
            # 0.004 = small inner padding from the cell edge for left/right alignment.
            if align[c] == "center":
                ax.text(cx + widths[c] / 2, yy - row_h / 2, cell, fontsize=fontsize,
                        color=colour, ha="center", va="center", zorder=4, fontweight=weight)
            elif align[c] == "right":
                ax.text(cx + widths[c] - 0.004, yy - row_h / 2, cell, fontsize=fontsize,
                        color=colour, ha="right", va="center", zorder=4, fontweight=weight)
            else:
                ax.text(cx + 0.004, yy - row_h / 2, cell, fontsize=fontsize,
                        color=colour, ha="left", va="center", zorder=4, fontweight=weight)
            cx += widths[c]
        ax.plot([x, x + w], [yy - row_h, yy - row_h], color=BORDER,
                linewidth=0.8, zorder=4)
    return top - len(rows) * row_h


def image(ax, path: Path | str, x, y, w, h, caption: str = "",
          border: bool = True, fontsize: float = 10):
    """Place a PNG preserving aspect ratio inside the (x, y, w, h) slot.

    The image is scaled to the largest size that fits the slot without
    distortion and centred in it; when a caption is given, the image is
    raised by 0.014 to leave room for the caption below it. A missing file is
    not an error: a red "[missing: name]" placeholder is drawn instead, so a
    sheet can be generated before all figures exist.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sheet axes.
    path : pathlib.Path or str
        Image file (normally a PNG produced by ``scripts/make_figures.py``).
    x, y, w, h : float
        Slot: lower-left corner and size in sheet fractions.
    caption : str, optional
        Text centred below the image.
    border : bool, optional
        Draw a thin frame around the image.
    fontsize : float, optional
        Caption size.
    """
    path = Path(path)
    if not path.exists():
        ax.text(x + w / 2, y + h / 2, f"[missing: {path.name}]", fontsize=10,
                color="#aa0000", ha="center", va="center", zorder=4)
        return
    data = mpimg.imread(path)
    ih, iw = data.shape[0], data.shape[1]
    # Convert the slot into figure aspect terms (A1 is wider than tall).
    # Slot height/width in inches = (h * A1_H_IN) / (w * A1_W_IN); comparing it
    # with the image's height/width ratio decides which side limits the size.
    # 用英寸计算槽位的高宽比，与图片高宽比比较，决定按高度还是按宽度缩放。
    slot_ratio = (h * A1_H_IN) / (w * A1_W_IN)
    img_ratio = ih / iw
    if img_ratio > slot_ratio:            # image is relatively taller
        # Fill the height; width in x-fractions follows from the image ratio.
        draw_h = h
        draw_w = h * A1_H_IN / (img_ratio * A1_W_IN)
    else:
        # Fill the width; height in y-fractions follows from the image ratio.
        draw_w = w
        draw_h = w * A1_W_IN * img_ratio / A1_H_IN
    # Centre in the slot. 居中放置。
    ox = x + (w - draw_w) / 2
    oy = y + (h - draw_h) / 2 + (0.014 if caption else 0)
    # extent places the pixel array in data coordinates; aspect="auto" lets the
    # computed extent define the shape instead of Matplotlib's square pixels.
    ax.imshow(data, extent=(ox, ox + draw_w, oy, oy + draw_h),
              aspect="auto", zorder=4, interpolation="antialiased")
    if border:
        ax.add_patch(Rectangle((ox, oy), draw_w, draw_h, fill=False,
                               edgecolor=BORDER, linewidth=1.1, zorder=5))
    if caption:
        ax.text(x + w / 2, oy - 0.010, caption, fontsize=fontsize, color=INK_2,
                ha="center", va="top", zorder=5)


def legend_chips(ax, x, y, items: Sequence[Tuple[str, str]], fontsize: float = 10.5,
                 gap: float = 0.072, chip: float = 0.010):
    """Horizontal colour-chip legend.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Sheet axes.
    x, y : float
        Left end and vertical centre of the legend row.
    items : sequence of (label, colour)
        Legend entries, drawn left to right.
    fontsize : float, optional
        Label size.
    gap : float, optional
        Horizontal distance between the starts of consecutive entries.
    chip : float, optional
        Side of the colour square in sheet fractions.
    """
    for label, colour in items:
        ax.add_patch(Rectangle((x, y - chip / 2), chip, chip, facecolor=colour,
                               edgecolor=BORDER, linewidth=0.7, zorder=4))
        ax.text(x + chip + 0.005, y, label, fontsize=fontsize, color=INK,
                va="center", ha="left", zorder=4)
        x += gap


def save(fig, path: Path | str) -> None:
    """Write a sheet as PNG and PDF and close the figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Sheet created by :func:`new_sheet`.
    path : pathlib.Path or str
        PNG output path; the PDF is written next to it with the same stem.
        Missing parent directories are created.

    Notes
    -----
    The PNG is rendered at ``DPI``; the PDF is vector output for printing.
    The figure is closed to free memory, because each A1 figure is large.
    PNG 用于预览，PDF 为矢量格式用于打印；保存后关闭图形释放内存。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    fig.savefig(path.with_suffix(".pdf"), facecolor=SURFACE)
    plt.close(fig)
    print(f"  {path.name} (+ .pdf)")
