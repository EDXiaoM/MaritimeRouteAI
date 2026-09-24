/* ==========================================================================
 * charts.js - dependency-free canvas/SVG drawing helpers.
 * 轻量绘图工具：训练曲线与混淆矩阵，无第三方依赖。
 *
 * Place in the application: loaded by index.html before app.js; defines the
 * global `Charts` used by app.js for the "Model" tab (training curves,
 * confusion matrix) and the route summary (zone profile bar). No chart
 * library is bundled, so the page stays small and works offline.
 * Drawing uses a <canvas> 2D context (curves) and plain HTML <div> grids
 * styled by app.css (confusion matrix, zone bar).
 * 本文件定义全局对象 Charts，供 app.js 调用。
 * ========================================================================== */
'use strict';

/**
 * Small drawing helpers.
 * Built as an IIFE so that only the three public functions are exported.
 * @namespace Charts
 */
const Charts = (() => {

  /**
   * Draw the training/validation curves on a canvas. 绘制训练曲线。
   *
   * Loss (blue) and accuracy (teal) share the plot area but use separate
   * vertical scales: loss is scaled between its own min and max (labels on
   * the left axis) and accuracy between its own min and max (percent labels
   * on the right axis). Solid lines = training set, dashed = validation set.
   * 损失与准确率共用绘图区，但各自按最小/最大值缩放（左轴为损失，右轴为准确率）。
   *
   * @param {HTMLCanvasElement} canvas - Target canvas; its width/height
   *   attributes give the CSS size in pixels.
   * @param {Object} history - Per-epoch arrays saved by the trainer:
   *   epoch, train_loss, val_loss, train_accuracy, val_accuracy (equal length).
   *   Nothing is drawn if the history is missing or empty.
   */
  function trainingCurves(canvas, history) {
    if (!canvas || !history || !history.epoch || !history.epoch.length) return;
    const ctx = canvas.getContext('2d');
    // High-DPI support: enlarge the backing store by devicePixelRatio, keep the
    // CSS size, and scale the context so drawing code can use CSS pixels.
    // The canvas attributes are overwritten, so this function is meant to be
    // called once per canvas (app.js calls it once, at page load).
    // 高分屏适配：按像素比放大画布缓冲区，保持 CSS 尺寸不变。
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width, H = canvas.height;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    // Padding around the plot area (left/right leave room for axis labels);
    // iw/ih = inner width/height of the plot area.
    const pad = { l: 34, r: 34, t: 10, b: 20 };
    const iw = W - pad.l - pad.r, ih = H - pad.t - pad.b;
    const n = history.epoch.length;

    // Common range of both loss curves, and of both accuracy curves.
    const losses = history.train_loss.concat(history.val_loss);
    const lMin = Math.min(...losses), lMax = Math.max(...losses);
    const accs = history.train_accuracy.concat(history.val_accuracy);
    const aMin = Math.min(...accs), aMax = Math.max(...accs);

    // grid
    // Five horizontal grid lines (0 %, 25 %, 50 %, 75 %, 100 % of the height).
    ctx.strokeStyle = '#2a3c4f'; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const y = pad.t + (ih * i) / 4;
      ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + iw, y);
    }
    ctx.stroke();

    /**
     * Draw one series as a poly-line.
     * x: epochs evenly spread over the plot width (max(n-1, 1) avoids a
     * division by zero for a single epoch).
     * y: value mapped linearly from [min, max] to [bottom, top]; canvas y
     * grows downwards, hence `pad.t + ih - ...`. 1e-9 guards a flat series.
     * @param {number[]} values - Series values.
     * @param {number} min - Value drawn at the bottom edge.
     * @param {number} max - Value drawn at the top edge.
     * @param {string} colour - Stroke colour.
     * @param {number[]} [dash] - Canvas dash pattern; omitted = solid line.
     */
    const line = (values, min, max, colour, dash) => {
      ctx.strokeStyle = colour; ctx.lineWidth = 1.6;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      values.forEach((v, i) => {
        const x = pad.l + (iw * i) / Math.max(n - 1, 1);
        const y = pad.t + ih - (ih * (v - min)) / Math.max(max - min, 1e-9);
        // First point starts the path, the rest extend it.
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    };

    line(history.train_loss,     lMin, lMax, '#4da3ff');
    line(history.val_loss,       lMin, lMax, '#4da3ff', [4, 3]);
    line(history.train_accuracy, aMin, aMax, '#2a9d8f');
    line(history.val_accuracy,   aMin, aMax, '#2a9d8f', [4, 3]);

    // Axis labels: loss range on the left, accuracy range (in %) on the right,
    // first/last epoch below, and a two-entry colour key at the top.
    // 坐标标注：左侧损失范围，右侧准确率范围，下方为首末轮次。
    ctx.fillStyle = '#94a7bb'; ctx.font = '9px sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(lMax.toFixed(2), pad.l - 4, pad.t + 7);
    ctx.fillText(lMin.toFixed(2), pad.l - 4, pad.t + ih);
    ctx.textAlign = 'left';
    ctx.fillText((aMax * 100).toFixed(0) + '%', pad.l + iw + 4, pad.t + 7);
    ctx.fillText((aMin * 100).toFixed(0) + '%', pad.l + iw + 4, pad.t + ih);
    ctx.textAlign = 'center';
    ctx.fillText('epoch 1', pad.l + 16, H - 6);
    ctx.fillText('epoch ' + n, pad.l + iw - 16, H - 6);
    ctx.fillStyle = '#4da3ff'; ctx.textAlign = 'left';
    ctx.fillText('— loss', pad.l + 4, pad.t + 9);
    ctx.fillStyle = '#2a9d8f';
    ctx.fillText('— accuracy', pad.l + 48, pad.t + 9);
  }

  /**
   * Render the confusion matrix as a coloured grid. 绘制混淆矩阵。
   *
   * Rows are true classes, columns predicted classes; each cell shows the raw
   * count. Colour intensity is the row-normalised share (count / row total,
   * i.e. the recall contribution): diagonal cells (correct) are teal,
   * off-diagonal cells (errors) red, both more opaque for larger shares.
   * The share in percent is shown as the cell tooltip. Layout uses the
   * .cm-row / .cm-cell / .cm-head classes of app.css.
   * 行为真实类别，列为预测类别；颜色深浅按行归一化比例，对角线为绿色，其余为红色。
   *
   * @param {HTMLElement} container - Element that receives the grid.
   * @param {number[][]} matrix - Square count matrix from the model metrics.
   * @param {string[]} labels - Class codes in matrix order; shortened for
   *   display ('OPEN_SEA' -> 'OPEN SEA', 'NEAR_COAST' -> 'NEAR CST').
   */
  function confusionMatrix(container, matrix, labels) {
    if (!container || !matrix) return;
    const short = labels.map((l) => l.replace('_SEA', ' SEA').replace('_COAST', ' CST'));
    const rowTotals = matrix.map((r) => r.reduce((a, b) => a + b, 0));
    // Header row: corner cell + one column header per predicted class.
    let html = '<div class="cm-row"><div class="cm-cell cm-head">true \\ pred</div>' +
      short.map((s) => `<div class="cm-cell cm-head">${s}</div>`).join('') + '</div>';

    matrix.forEach((row, i) => {
      html += `<div class="cm-row"><div class="cm-cell cm-head">${short[i]}</div>`;
      row.forEach((value, j) => {
        const share = rowTotals[i] ? value / rowTotals[i] : 0;
        // Alpha ranges: diagonal 0.18..0.90, off-diagonal 0.10..0.90.
        const colour = i === j
          ? `rgba(42,157,143,${0.18 + share * 0.72})`
          : `rgba(239,100,97,${0.10 + share * 0.80})`;
        html += `<div class="cm-cell" style="background:${colour}" ` +
                `title="${(share * 100).toFixed(1)}%">${value}</div>`;
      });
      html += '</div>';
    });
    container.innerHTML = html;
  }

  /**
   * Stacked horizontal bar of the zone profile. 区域占比条。
   *
   * One <div> per zone with a non-zero count, its width set to the zone's
   * percentage of all waypoints, so the segments fill the bar (a flex row in
   * app.css). An optional legend lists the non-empty zones with percentages.
   *
   * @param {HTMLElement} container - The bar element (#zone-bar).
   * @param {?HTMLElement} legend - Legend element (#zone-legend) or null.
   * @param {Object<string, number>} profile - Waypoint count per zone code
   *   (RoutePlan.zone_profile).
   * @param {Object<string, string>} colours - Zone code -> CSS colour.
   */
  function zoneBar(container, legend, profile, colours) {
    if (!container) return;
    // `|| 1` avoids a division by zero for an empty profile.
    const total = Object.values(profile).reduce((a, b) => a + b, 0) || 1;
    container.innerHTML = Object.entries(profile)
      .map(([code, n]) => n
        ? `<div style="width:${(100 * n / total).toFixed(2)}%;background:${colours[code]}"
             title="${code}: ${n} (${(100 * n / total).toFixed(1)}%)"></div>`
        : '')
      .join('');
    if (legend) {
      legend.innerHTML = Object.entries(profile)
        .filter(([, n]) => n > 0)
        .map(([code, n]) =>
          `<span><i style="background:${colours[code]}"></i>${code} ${(100 * n / total).toFixed(1)}%</span>`)
        .join('');
    }
  }

  // Public interface of the Charts namespace.
  return { trainingCurves, confusionMatrix, zoneBar };
})();
