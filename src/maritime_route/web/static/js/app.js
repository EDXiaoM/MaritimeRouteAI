/* ==========================================================================
 * app.js - application controller: map interaction, live planning, exports.
 * 主控脚本：地图交互、实时规划、结果展示与导出。
 *
 * Place in the application: the main script of the single-page client
 * (index.html). It is loaded last, after vendor/leaflet/leaflet.js (global L),
 * js/api.js (globals Api, LiveChannel) and js/charts.js (global Charts), and
 * runs once when the page loads. It has no exports; everything it defines is
 * used inside this file.
 *
 * Sections (marked by the dashed banners below):
 *   map            Leaflet map, base layers, offline tile fallback, overlay groups
 *   elements       cached DOM references
 *   helpers        number formatting, status pills, marker icons
 *   map picking    departure/destination selection by click or drag
 *   cursor         live zone read-out under the mouse pointer (WebSocket)
 *   live channel   WebSocket frame handlers (progress, overlay, result, error)
 *   render result  drawing the route and the summary tables
 *   actions        buttons, sliders, preset chips
 *   tabs, upload, model panel, legend, bootstrap
 * 本文件是前端主控脚本，按上述分区组织。
 * ========================================================================== */
'use strict';

/* Diverging hypsometric ramp: water blue -> land ochre, lightness monotone
   within each pole (see config.py for the validated separation figures).
   发散型等深-等高配色，与服务端 CLASS_COLOR_DARK 保持一致。 */
/**
 * Zone class code -> CSS colour. Keys are the class codes returned by the
 * server (ClassificationResult.class_code, RouteLeg.class_code).
 * @type {Object<string, string>}
 */
const ZONE_COLOURS = {
  OPEN_SEA:    '#3d7fb8',
  COASTAL_SEA: '#7cc3e3',
  NEAR_COAST:  '#d9b877',
  COASTLINE:   '#a9763f',
};

/**
 * Mutable UI state shared by all handlers of this file.
 * 前端全局状态。
 *
 * @property {?number[]} start - Departure [lat, lon] or null.
 * @property {?number[]} end - Destination [lat, lon] or null.
 * @property {string} pickNext - Which endpoint the next map click sets: 'start' | 'end'.
 * @property {?Object} plan - Last rendered RoutePlan (as received from the server).
 * @property {?string} uploadToken - Token of the last upload ('upload-<session_id>').
 * @property {Object<string, L.LayerGroup>} layers - Overlay groups: overlay, points, route, markers.
 * @property {boolean} liveBusy - True while a cursor 'classify' request is in flight;
 *   prevents sending a new one before the answer arrives. Cleared only by a
 *   'classified' frame.
 */
const State = {
  start: null,
  end: null,
  pickNext: 'start',
  plan: null,
  uploadToken: null,
  layers: {},
  liveBusy: false,
};

/* ------------------------------------------------------------------ map */
/**
 * The Leaflet map, centred on the Baltic Sea at zoom 5.
 * worldCopyJump keeps markers on the visible world copy when the user pans
 * across the antimeridian. 地图对象，初始视图为波罗的海。
 * @type {L.Map}
 */
const map = L.map('map', { zoomControl: true, worldCopyJump: true })
  .setView([56.5, 17.0], 5);

/* Two base layers: the public OSM chart and an on-board layer rendered by the
   server from the neural network itself, so the tool also works offline.
   两种底图：在线 OSM 海图，以及由服务端神经网络渲染的离线随船底图。 */
/** Online OpenStreetMap tiles (require internet access). @type {L.TileLayer} */
const osmLayer = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 18, attribution: '&copy; OpenStreetMap contributors',
});
/**
 * Offline tiles from GET /api/tiles/zones/{z}/{x}/{y}.png (web/tiles.py).
 * maxZoom 10 matches the server-side zoom limit; minZoom 2 avoids rendering
 * the few whole-world tiles, which carry little information.
 * @type {L.TileLayer}
 */
const onboardLayer = L.tileLayer('/api/tiles/zones/{z}/{x}/{y}.png', {
  maxZoom: 10, minZoom: 2, attribution: 'On-board zone classification (this model)',
});
/* The dark backdrop behind the tiles is set in CSS on .leaflet-container.
   It must NOT be drawn as an L.rectangle: vector overlays live in the
   overlayPane (z-index 400), above the tilePane (z-index 200), so an opaque
   world-covering rectangle hides every basemap tile, online and offline alike.
   底图背景色由 CSS 的 .leaflet-container 提供，不能用矩形绘制：
   矢量图层位于 overlayPane(z-index 400)，会完全遮挡 tilePane(z-index 200) 的瓦片。 */
osmLayer.addTo(map);
// Base-layer switcher (radio buttons); no optional overlays are listed ({}).
// 底图切换控件（仅底图，无叠加层）。
L.control.layers(
  { 'OpenStreetMap (online)': osmLayer, 'On-board chart (offline)': onboardLayer },
  {}, { position: 'topleft' }
).addTo(map);

/* If the public tile service is unreachable, switch to the on-board chart.
   在线瓦片不可用时自动切换到离线底图。 */
/** Number of OSM tiles that failed to load since page start. @type {number} */
let tileErrors = 0;
/**
 * Offline fallback: on the 4th failed OSM tile (a single failure may be a
 * transient network error) the online layer is replaced by the on-board
 * chart. The strict `=== 4` makes the switch happen once; later errors do
 * nothing. The hasLayer check leaves the choice alone if the user has already
 * switched layers manually.
 * 第 4 次瓦片加载失败时切换一次；若用户已手动切换则不再处理。
 */
osmLayer.on('tileerror', () => {
  if (++tileErrors === 4 && map.hasLayer(osmLayer)) {
    map.removeLayer(osmLayer);
    onboardLayer.addTo(map);
    console.info('Tile service unreachable - switched to the on-board chart.');
  }
});

// Overlay groups, added in drawing order (later groups are drawn on top):
//   overlay - classified lattice from the 'overlay' WebSocket frame
//   points  - classified points of an uploaded AIS file
//   route   - the planned route, its casing and the great-circle reference
//   markers - draggable departure (A) and destination (B) markers
// Each group can be cleared independently with clearLayers().
// 图层组按绘制顺序添加，后添加的位于上方，可分别清空。
State.layers.overlay  = L.layerGroup().addTo(map);
State.layers.points   = L.layerGroup().addTo(map);
State.layers.route    = L.layerGroup().addTo(map);
State.layers.markers  = L.layerGroup().addTo(map);

/* ------------------------------------------------------------ elements */
/**
 * Shorthand for document.getElementById.
 * @param {string} id - Element id.
 * @returns {?HTMLElement}
 */
const $ = (id) => document.getElementById(id);
/**
 * DOM elements used by the handlers, looked up once at start-up.
 * Keys are camelCase versions of the element ids in index.html.
 * 启动时一次性缓存页面元素引用。
 * @type {Object<string, HTMLElement>}
 */
const els = {
  wsStatus: $('ws-status'), modelPill: $('model-pill'),
  startLat: $('start-lat'), startLon: $('start-lon'),
  endLat: $('end-lat'), endLon: $('end-lon'),
  algorithm: $('algorithm'), resolution: $('resolution'), resOut: $('res-out'),
  speed: $('speed'), speedOut: $('speed-out'),
  showOverlay: $('show-overlay'), liveClassify: $('live-classify'),
  btnPlan: $('btn-plan'), btnBenchmark: $('btn-benchmark'), btnClear: $('btn-clear'),
  progressBlock: $('progress-block'), progressStage: $('progress-stage'),
  progressPct: $('progress-pct'), progressFill: $('progress-fill'),
  progressDetail: $('progress-detail'),
  resultBlock: $('result-block'), metricGrid: $('metric-grid'),
  zoneBar: $('zone-bar'), zoneLegend: $('zone-legend'),
  compareBlock: $('compare-block'), benchmarkBlock: $('benchmark-block'),
  modelBlock: $('model-block'), confusion: $('confusion'), historyChart: $('history-chart'),
  historyList: $('history-list'), dbStats: $('db-stats'),
  uploadReport: $('upload-report'), pointsExport: $('points-export'),
  dropzone: $('dropzone'), fileInput: $('file-input'), dropzoneText: $('dropzone-text'),
  cursorReadout: $('cursor-readout'), mapLegend: $('map-legend'), pickHint: $('pick-hint'),
};

/* ------------------------------------------------------------- helpers */
/**
 * Format a number with fixed decimals; missing values become an em dash.
 * The server sends null for infinite/undefined metrics (e.g. an infeasible
 * route), which is why null/undefined/NaN are handled explicitly.
 * @param {?number} v - Value.
 * @param {number} [digits=1] - Decimal places.
 * @returns {string}
 */
const fmt = (v, digits = 1) =>
  (v === null || v === undefined || Number.isNaN(v)) ? '—' : Number(v).toFixed(digits);

/**
 * Format an integer with thousands separators ('12,345'); missing -> em dash.
 * @param {?number} v - Value.
 * @returns {string}
 */
const fmtInt = (v) =>
  (v === null || v === undefined) ? '—' : Number(v).toLocaleString('en-US');

/**
 * Update a status pill in the header (text and colour class).
 * @param {HTMLElement} element - The pill element.
 * @param {string} text - Text to show.
 * @param {string} cls - One of 'pill-ok', 'pill-wait', 'pill-bad' (see app.css).
 */
function setStatus(element, text, cls) {
  element.textContent = text;
  element.className = 'pill ' + cls;
}

/**
 * Build a round lettered marker icon (A = departure, B = destination).
 * An L.divIcon is used instead of the default image marker so no icon image
 * files are needed. iconAnchor [11, 11] puts the centre of the 22 px circle
 * on the coordinate.
 * @param {string} colour - Fill colour.
 * @param {string} label - Letter drawn in the circle.
 * @returns {L.DivIcon}
 */
function markerIcon(colour, label) {
  return L.divIcon({
    className: '',
    html: `<div style="background:${colour};color:#06202f;font:700 11px sans-serif;
           width:22px;height:22px;border-radius:50%;display:flex;align-items:center;
           justify-content:center;border:2px solid #e6edf5;box-shadow:0 2px 6px #0008">${label}</div>`,
    iconSize: [22, 22], iconAnchor: [11, 11],
  });
}

/* --------------------------------------------------------- map picking */
/**
 * Set the departure or destination, copy it into the form inputs (4 decimals,
 * about 11 m) and redraw the markers.
 * @param {'start'|'end'} which - Endpoint to set.
 * @param {number} lat - Latitude in degrees.
 * @param {number} lon - Longitude in degrees.
 */
function setEndpoint(which, lat, lon) {
  State[which] = [lat, lon];
  if (which === 'start') {
    els.startLat.value = lat.toFixed(4);
    els.startLon.value = lon.toFixed(4);
  } else {
    els.endLat.value = lat.toFixed(4);
    els.endLon.value = lon.toFixed(4);
  }
  redrawEndpoints();
}

/**
 * Redraw the A/B markers from State.start / State.end.
 * The markers are draggable; on 'dragend' the new position is written back
 * through setEndpoint, which redraws both markers again.
 */
function redrawEndpoints() {
  State.layers.markers.clearLayers();
  if (State.start) {
    L.marker(State.start, { icon: markerIcon('#7fe3b0', 'A'), draggable: true })
      .on('dragend', (e) => { const p = e.target.getLatLng(); setEndpoint('start', p.lat, p.lng); })
      .bindTooltip('Departure').addTo(State.layers.markers);
  }
  if (State.end) {
    L.marker(State.end, { icon: markerIcon('#ffb4b2', 'B'), draggable: true })
      .on('dragend', (e) => { const p = e.target.getLatLng(); setEndpoint('end', p.lat, p.lng); })
      .bindTooltip('Destination').addTo(State.layers.markers);
  }
}

/**
 * Take the endpoints from the four coordinate inputs (the user may have typed
 * them) and redraw the markers. Empty inputs give NaN; this is checked by
 * the Plan button handler before a request is sent.
 */
function readEndpointsFromInputs() {
  State.start = [parseFloat(els.startLat.value), parseFloat(els.startLon.value)];
  State.end   = [parseFloat(els.endLat.value),   parseFloat(els.endLon.value)];
  redrawEndpoints();
}

/**
 * Map click: set the endpoint named by State.pickNext, then alternate between
 * departure and destination and update the hint text below the inputs.
 * 点击地图交替设置起点与终点。
 */
map.on('click', (event) => {
  const { lat, lng } = event.latlng;
  setEndpoint(State.pickNext, lat, lng);
  State.pickNext = State.pickNext === 'start' ? 'end' : 'start';
  els.pickHint.innerHTML = State.pickNext === 'start'
    ? 'Next click sets the <b>departure</b>. Markers can also be dragged.'
    : 'Next click sets the <b>destination</b>. Markers can also be dragged.';
});

/* ---------------------------------------------- live cursor read-out */
/** Debounce timer id for the cursor classification. @type {?number} */
let liveTimer = null;
/**
 * Mouse move: always show the pointer coordinates; if the "Live zone read-out" checkbox is on,
 * ask the server for the zone under the pointer.
 *
 * Two throttles keep the load low: the request is debounced (sent only after
 * the pointer has rested for 160 ms) and State.liveBusy allows at most one
 * request in flight. The answer arrives as a 'classified' frame (handler
 * below). `channel` is declared further down; that is fine because this
 * handler only runs after the whole script has executed.
 * 鼠标停留 160 ms 后才发送分类请求，且同一时间只允许一个请求。
 */
map.on('mousemove', (event) => {
  const { lat, lng } = event.latlng;
  els.cursorReadout.querySelector('.coord').textContent =
    `${lat.toFixed(3)}, ${lng.toFixed(3)}`;
  if (!els.liveClassify.checked) return;
  clearTimeout(liveTimer);
  liveTimer = setTimeout(() => {
    if (State.liveBusy || !channel.isOpen) return;
    State.liveBusy = true;
    channel.send({ action: 'classify', points: [[lat, lng]] });
  }, 160);
});

/* ------------------------------------------------------- live channel */
/**
 * The single WebSocket connection to /ws/plan (see LiveChannel in api.js),
 * used for route planning with progress and for the cursor read-out.
 * @type {LiveChannel}
 */
const channel = new LiveChannel('/ws/plan');

/**
 * Connection state -> header pill. 'closed' is shown as "reconnecting"
 * because LiveChannel always schedules a reconnect after a close.
 */
channel.on('status', ({ state }) => {
  if (state === 'open')  setStatus(els.wsStatus, 'live connection', 'pill-ok');
  if (state === 'closed') setStatus(els.wsStatus, 'reconnecting…', 'pill-wait');
  if (state === 'error')  setStatus(els.wsStatus, 'connection error', 'pill-bad');
});

/**
 * 'classified' frame: answer to a cursor classification request.
 * Shows the zone colour swatch, class code, confidence and, if known, the
 * reference depth/elevation (depth_m > 0 means land elevation, < 0 water depth).
 * 显示指针处的区域类别、置信度与水深/高程。
 */
channel.on('classified', (frame) => {
  State.liveBusy = false;
  const r = frame.results && frame.results[0];
  if (!r) return;
  const depth = r.depth_m === null ? '' : ` · ${r.depth_m > 0 ? 'elev' : 'depth'} ${Math.abs(r.depth_m).toFixed(0)} m`;
  els.cursorReadout.querySelector('.zone').innerHTML =
    `<i style="display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
        background:${ZONE_COLOURS[r.class_code]}"></i>${r.class_code}
     <span style="opacity:.7">${(r.confidence * 100).toFixed(0)}%${depth}</span>`;
});

/**
 * 'started' frame: the server accepted a planning request. Shows and resets
 * the progress block and disables the Plan button until 'result' or 'error'.
 * The bar starts at 2 % so that it is visibly present.
 */
channel.on('started', (frame) => {
  els.progressBlock.hidden = false;
  els.btnPlan.disabled = true;
  els.progressStage.textContent = `starting ${frame.algorithm.toUpperCase()}…`;
  els.progressFill.style.width = '2%';
  els.progressPct.textContent = '0%';
  els.progressDetail.textContent = '';
});

/**
 * Human-readable labels for the `stage` field of 'progress' frames
 * (stage names are defined by the planner, cost-map builder and search modules).
 * Unknown stages are shown with their raw name.
 * @type {Object<string, string>}
 */
const STAGE_TEXT = {
  grid:     'Building geographic lattice',
  classify: 'Neural network classifying grid cells',
  cost_map: 'Cost map ready',
  search:   'Searching optimal path',
  astar:    'A* search in progress',
  dijkstra: 'Dijkstra search in progress',
  genetic:  'Genetic algorithm: evolving routes',
  dynamic:  'Dynamic programming: solving the Bellman equation',
  refine:   'Refining the lattice (narrow strait)',
  done:     'Route ready',
};

/**
 * 'progress' frame: {stage, progress (0..1 within the stage), detail}.
 * The progress fraction restarts for every stage, so the bar shows the
 * progress of the current stage, not of the whole job.
 * The detail line depends on which keys the stage supplies:
 *   cells_done/cells_total -> neural-network classification of the lattice;
 *   expanded/frontier      -> A* or Dijkstra search;
 *   generation/best_cost   -> genetic algorithm;
 *   zone_shares            -> cost-map statistics (share of each zone);
 *   stage                  -> plain text message (grid, search, refine, done).
 * 根据 detail 中出现的字段选择不同的进度说明文字。
 */
channel.on('progress', (frame) => {
  const pct = Math.round(frame.progress * 100);
  els.progressStage.textContent = STAGE_TEXT[frame.stage] || frame.stage;
  els.progressFill.style.width = Math.max(pct, 2) + '%';
  els.progressPct.textContent = pct + '%';
  const d = frame.detail || {};
  if (d.cells_done !== undefined) {
    els.progressDetail.textContent =
      `${fmtInt(d.cells_done)} / ${fmtInt(d.cells_total)} cells classified`;
  } else if (d.expanded !== undefined) {
    els.progressDetail.textContent =
      `${fmtInt(d.expanded)} nodes expanded · frontier ${fmtInt(d.frontier)}`;
  } else if (d.sweep !== undefined) {
    // Dynamic programming: value-iteration sweep and cells whose value changed.
    els.progressDetail.textContent =
      `sweep ${fmtInt(d.sweep)} · ${fmtInt(d.updated)} values improved`;
  } else if (d.generation !== undefined) {
    // Genetic algorithm: generation counter and best route cost so far.
    els.progressDetail.textContent =
      `generation ${fmtInt(d.generation)} · best cost ${fmt(d.best_cost, 0)}`;
  } else if (d.zone_shares) {
    els.progressDetail.textContent = Object.entries(d.zone_shares)
      .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`).join(' · ');
  } else if (d.stage) {
    els.progressDetail.textContent = d.stage;
  }
});

/**
 * 'overlay' frame: the classified lattice used for planning, down-sampled by
 * the server to about 6 000 cells {lat, lon, c (class index), p (confidence)}.
 * Each cell is drawn as a borderless, non-interactive rectangle centred on
 * (lat, lon) at 22 % opacity so the base map stays visible. The previous
 * overlay is always removed, even if drawing is switched off.
 * 绘制降采样后的分类网格，每个单元为以中心点为中心的半透明矩形。
 */
channel.on('overlay', (frame) => {
  State.layers.overlay.clearLayers();
  if (!els.showOverlay.checked) return;
  // cell_size_deg already accounts for the server-side down-sampling step, so
  // the rectangles tile the area seamlessly. 单元尺寸已含降采样步长，绘制无缝。
  const size = frame.cell_size_deg || frame.grid.resolution_deg;
  // Class index -> code, in the model's output order (config.CLASS_NAMES).
  const codes = ['OPEN_SEA', 'COASTAL_SEA', 'NEAR_COAST', 'COASTLINE'];
  const half = size / 2;
  frame.cells.forEach((cell) => {
    L.rectangle(
      [[cell.lat - half, cell.lon - half], [cell.lat + half, cell.lon + half]],
      { stroke: false, fillColor: ZONE_COLOURS[codes[cell.c]],
        fillOpacity: 0.22, interactive: false }
    ).addTo(State.layers.overlay);
  });
});

/**
 * 'result' frame: the finished RoutePlan. Completes the progress bar, hides
 * it after 0.9 s, renders the plan and refreshes the history list and the
 * database statistics (the route has just been stored).
 */
channel.on('result', (frame) => {
  els.btnPlan.disabled = false;
  els.progressFill.style.width = '100%';
  els.progressPct.textContent = '100%';
  els.progressStage.textContent = 'Route ready';
  setTimeout(() => { els.progressBlock.hidden = true; }, 900);
  renderPlan(frame.plan);
  loadHistory();
  loadDbStats();
});

/**
 * 'error' frame: show the server's message in the progress block and
 * re-enable the Plan button. The block stays visible so the text can be read.
 */
channel.on('error', (frame) => {
  // A failed cursor read-out must not block the next one. 光标实时分类失败时解除占用标志。
  State.liveBusy = false;
  els.btnPlan.disabled = false;
  els.progressStage.textContent = 'Error';
  els.progressDetail.textContent = frame.detail;
  els.progressFill.style.width = '100%';
});

// Open the socket; LiveChannel reconnects by itself after any disconnect.
// 建立 WebSocket 连接，断线后自动重连。
channel.connect();

/* ------------------------------------------------------ render result */
/**
 * Draw a planned route on the map and fill the summary panels.
 *
 * Layers drawn into State.layers.route, bottom to top:
 *   1. dark casing polyline (9 px) for contrast against any base map;
 *   2. one 5 px segment per leg, coloured by the zone of the leg's end point;
 *   3. a thin white centre line along the waypoints;
 *   4. the dashed red great-circle line from start to end (the naive
 *      reference route, which may cross land).
 * An infeasible plan only shows its message.
 * Also sets the export links (GeoJSON, CSV, PDF) and fills the comparison tab.
 * 在地图上绘制航线并填充结果摘要。
 *
 * @param {Object} plan - RoutePlan.to_dict() from the server (see app.py).
 */
function renderPlan(plan) {
  State.plan = plan;
  State.layers.route.clearLayers();

  if (!plan.feasible || !plan.waypoints.length) {
    els.resultBlock.hidden = false;
    els.metricGrid.innerHTML =
      `<div class="metric bad" style="grid-column:1/-1"><div class="label">Infeasible</div>
       <div class="value" style="font-size:13px">${plan.message || 'No navigable route'}</div></div>`;
    return;
  }

  // A dark casing is drawn first so that the route reads against any basemap.
  // The route is coloured by zone, and the on-board chart uses the same palette,
  // so without a casing a coastal leg would vanish into coastal water.
  // 先绘制深色描边：航线按区域着色，而离线底图用的是同一套配色，
  // 没有描边时沿岸航段会完全融进浅蓝色海面。
  L.polyline(plan.waypoints, {
    color: '#06121f', weight: 9, opacity: 0.55, lineJoin: 'round', lineCap: 'round',
  }).addTo(State.layers.route);

  // Optimised route, coloured per zone segment. 按区域分段着色。
  for (let i = 1; i < plan.legs.length; i++) {
    const a = plan.legs[i - 1], b = plan.legs[i];
    L.polyline([[a.latitude, a.longitude], [b.latitude, b.longitude]], {
      color: ZONE_COLOURS[b.class_code] || '#4da3ff', weight: 5, opacity: 1,
      lineJoin: 'round', lineCap: 'round',
    }).addTo(State.layers.route);
  }
  // A bright hairline down the middle keeps the track legible when zoomed out.
  L.polyline(plan.waypoints, { color: '#ffffff', weight: 1.6, opacity: 0.9 })
    .addTo(State.layers.route);

  // Great-circle reference. 大圆参考航线。
  L.polyline([plan.start, plan.end], {
    color: '#ff5d5a', weight: 2.6, opacity: 0.95, dashArray: '8,6',
  }).bindTooltip('Great-circle (shortest distance)').addTo(State.layers.route);

  // Zoom to the route with 18 % padding on each side. 缩放至航线范围。
  map.fitBounds(L.polyline(plan.waypoints).getBounds().pad(0.18));

  // Summary metric tiles. A detour ratio (route length / great-circle length)
  // above 1.5 is flagged with the 'warn' colour.
  // 结果指标；绕航系数大于 1.5 时以警示色显示。
  const detour = plan.detour_ratio;
  els.metricGrid.innerHTML = `
    ${metric('Route length', fmt(plan.distance_km, 1), 'km')}
    ${metric('Great circle', fmt(plan.great_circle_km, 1), 'km')}
    ${metric('Detour ratio', fmt(detour, 3), '×', detour > 1.5 ? 'warn' : 'good')}
    ${metric('Weighted cost', fmt(plan.total_cost, 0), '')}
    ${metric('Passage time', fmt(plan.estimated_hours, 1), 'h')}
    ${metric('Planning time', fmt(plan.runtime_s, 3), 's')}
    ${metric({ genetic: 'Routes evaluated', dynamic: 'Bellman updates' }[plan.algorithm] || 'Nodes expanded',
             fmtInt(plan.nodes_expanded), '')}
    ${metric('Waypoints', fmtInt(plan.waypoints.length), '')}`;

  Charts.zoneBar(els.zoneBar, els.zoneLegend, plan.zone_profile, ZONE_COLOURS);

  // Export links point to GET /api/export/{route_id}.{fmt}; the server keeps
  // the plan in memory (last 64 plans) for this purpose.
  // 导出链接；服务端在内存中保留最近 64 条航线用于导出。
  $('dl-geojson').href = `/api/export/${plan.route_id}.geojson`;
  $('dl-csv').href     = `/api/export/${plan.route_id}.csv`;
  $('dl-pdf').href     = `/api/export/${plan.route_id}.pdf`;
  els.resultBlock.hidden = false;

  renderComparison(plan);
}

/**
 * HTML for one metric tile of the result grid.
 * @param {string} label - Caption above the value.
 * @param {string} value - Already formatted value.
 * @param {string} unit - Unit shown after the value (may be '').
 * @param {string} [cls=''] - Extra class: 'good', 'warn' or 'bad' (colour).
 * @returns {string} HTML string.
 */
function metric(label, value, unit, cls = '') {
  return `<div class="metric ${cls}"><div class="label">${label}</div>
          <div class="value">${value} <small>${unit}</small></div></div>`;
}

/**
 * Fill the "Compare" tab: the optimised route versus the great-circle
 * baseline computed by the server (plan.baseline).
 *
 * The baseline samples points along the great circle and classifies them;
 * `land_samples / samples` and `land_share` show how much of the direct line
 * would cross land, and `verdict` is the server's one-line conclusion.
 * "Passage time saved/lost" is the optimised passage time minus the
 * great-circle time at the same speed (knots * 1.852 = km/h); a positive
 * value is the extra time the safe route costs.
 * 对比页：优化航线与大圆航线的距离、陆地占比和航行时间差。
 *
 * @param {Object} plan - RoutePlan.to_dict().
 */
function renderComparison(plan) {
  const b = plan.baseline || {};
  const navigable = b.is_navigable;
  els.compareBlock.innerHTML = `
    <table>
      <tr><td>Great-circle length</td><td>${fmt(b.distance_km, 1)} km</td></tr>
      <tr><td>Optimised length</td><td>${fmt(plan.distance_km, 1)} km</td></tr>
      <tr><td>Additional distance</td><td>${fmt(b.extra_distance_pct, 2)} %</td></tr>
      <tr><td>Samples over land</td><td>${b.land_samples} / ${b.samples}</td></tr>
      <tr><td>Share over land</td><td>${fmt((b.land_share || 0) * 100, 1)} %</td></tr>
      <tr><td>Optimised: intermediate waypoints over land</td>
          <td>${plan.interior_land_waypoints}</td></tr>
      <tr><td>Optimised: waypoints in total</td><td>${fmtInt(plan.waypoints.length)}</td></tr>
      <tr><td>Passage time saved/lost</td><td>${fmt(plan.estimated_hours - (b.distance_km / (plan.speed_knots * 1.852)), 1)} h</td></tr>
    </table>
    <div class="verdict ${navigable ? 'good' : 'bad'}">${b.verdict || ''}</div>`;
}

/* ------------------------------------------------------------ actions */
/**
 * Build a planning request from the form.
 *
 * The same object is used for the WebSocket ('plan' action) and for the REST
 * fallback / benchmark; the server ignores the keys it does not need
 * (action, include_overlay for REST). simplify_km is not sent, so the server
 * default (0 = no simplification) applies.
 * 构造规划请求，WebSocket 与 REST 共用同一对象。
 *
 * @returns {Object} Request body with action, coordinates, algorithm,
 *   resolution_deg, speed_knots, include_overlay and persist.
 */
function planRequest() {
  readEndpointsFromInputs();
  return {
    action: 'plan',
    start_lat: State.start[0], start_lon: State.start[1],
    end_lat: State.end[0],     end_lon: State.end[1],
    algorithm: els.algorithm.value,
    resolution_deg: parseFloat(els.resolution.value),
    speed_knots: parseFloat(els.speed.value),
    include_overlay: els.showOverlay.checked,
    persist: true,
  };
}

/**
 * "Build route" button (#btn-plan): validate that all four coordinates are numbers, then
 * send the request over the WebSocket (progress is shown by the frame
 * handlers above) or, if the socket is down, use the blocking REST endpoint
 * POST /api/route without progress.
 */
els.btnPlan.addEventListener('click', () => {
  const request = planRequest();
  if ([request.start_lat, request.start_lon, request.end_lat, request.end_lon].some(Number.isNaN)) {
    alert('Please set both the departure and the destination.');
    return;
  }
  if (channel.isOpen) {
    channel.send(request);
  } else {
    // Fall back to REST when the socket is down. WebSocket 不可用时退回 REST。
    els.btnPlan.disabled = true;
    Api.plan(request)
      .then(renderPlan)
      .catch((err) => alert('Planning failed: ' + err.message))
      .finally(() => { els.btnPlan.disabled = false; });
  }
});

/**
 * "Compare algorithms" button (#btn-benchmark): run all four methods on one shared cost map
 * (POST /api/benchmark), switch to the Compare tab and show a table with
 * distance, cost, expanded nodes and search time per algorithm, followed by
 * the derived ratios computed by the server. '⚠' marks an infeasible row
 * (e.g. a great circle crossing land). `??` shows '—' for null ratios.
 * 基准测试：同一代价地图上比较四种方法。
 */
els.btnBenchmark.addEventListener('click', async () => {
  const request = planRequest();
  els.btnBenchmark.disabled = true;
  els.benchmarkBlock.textContent = 'running five methods on one shared cost map…';
  // Programmatic click on the tab button reuses the tab-switch handler.
  document.querySelector('[data-tab="compare"]').click();
  try {
    const data = await Api.benchmark(request);
    els.benchmarkBlock.innerHTML = `
      <table>
        <tr><td><b>Algorithm</b></td><td><b>km / cost / nodes / search s</b></td></tr>
        ${data.results.map((r) => `<tr><td>${r.algorithm}${r.feasible ? '' : ' ⚠'}</td><td>
          ${fmt(r.distance_km, 0)} / ${r.total_cost === null ? '—' : fmt(r.total_cost, 0)} /
          ${fmtInt(r.nodes_expanded)} / ${fmt(r.runtime_s, 3)}</td></tr>`).join('')}
        <tr><td>A* node reduction vs Dijkstra</td><td>${data.astar_node_reduction ?? '—'}×</td></tr>
        <tr><td>A* speed-up (time)</td><td>${data.astar_time_ratio ?? '—'}×</td></tr>
        <tr><td>Cost difference A* vs Dijkstra</td><td>${data.optimality_gap_pct ?? '—'} %</td></tr>
        <tr><td>Genetic algorithm above the optimum</td><td>${data.genetic_gap_pct ?? '—'} %</td></tr>
        <tr><td>Cost difference DP vs A*</td><td>${data.dynamic_gap_pct ?? '—'} %</td></tr>
        <tr><td>Cost map build</td><td>${fmt(data.cost_map_build_s, 3)} s (shared)</td></tr>
        <tr><td>Grid</td><td>${data.grid.n_rows}×${data.grid.n_cols} @ ${data.grid.resolution_deg}°</td></tr>
      </table>
      <div class="muted" style="margin-top:6px">Runtimes exclude the shared cost-map
      construction, so they compare the search algorithms only.</div>`;
  } catch (err) {
    els.benchmarkBlock.textContent = 'Benchmark failed: ' + err.message;
  } finally {
    els.btnBenchmark.disabled = false;
  }
});

/**
 * "Clear" button (#btn-clear): remove every overlay (lattice, points, route, markers),
 * forget the endpoints and the plan and reset the result panels. The
 * coordinate inputs keep their values.
 */
els.btnClear.addEventListener('click', () => {
  Object.values(State.layers).forEach((layer) => layer.clearLayers());
  State.start = State.end = State.plan = null;
  State.pickNext = 'start';
  els.resultBlock.hidden = true;
  els.compareBlock.textContent = 'Build a route first.';
  els.cursorReadout.querySelector('.zone').textContent = 'move the pointer over the chart';
});

/**
 * "Show zone classification layer" checkbox (#show-overlay): unchecking removes the current overlay;
 * checking it has effect from the next planning run (the overlay is not
 * kept on the client).
 */
els.showOverlay.addEventListener('change', () => {
  if (!els.showOverlay.checked) State.layers.overlay.clearLayers();
});

/** Resolution slider: mirror the value next to the label (degrees). */
els.resolution.addEventListener('input', () => {
  els.resOut.textContent = parseFloat(els.resolution.value).toFixed(2) + '°';
});
/** Speed slider: mirror the value next to the label (knots). */
els.speed.addEventListener('input', () => {
  els.speedOut.textContent = parseFloat(els.speed.value).toFixed(1) + ' kn';
});

/**
 * Preset voyage chips: data-preset="startLat,startLon,endLat,endLon".
 * Fills the inputs, redraws the markers and zooms to both ports with 40 %
 * padding. Planning is not started automatically.
 * 预设航线按钮：填入坐标并缩放地图，不自动规划。
 */
document.querySelectorAll('.chip[data-preset]').forEach((chip) => {
  chip.addEventListener('click', () => {
    const [a, b, c, d] = chip.dataset.preset.split(',').map(Number);
    els.startLat.value = a; els.startLon.value = b;
    els.endLat.value = c;   els.endLon.value = d;
    readEndpointsFromInputs();
    map.fitBounds(L.latLngBounds([[a, b], [c, d]]).pad(0.4));
  });
});

/* ---------------------------------------------------------------- tabs */
/**
 * Tab bar of the side panel: a button with data-tab="X" activates the page
 * with id "tab-X"; exactly one button and one page carry the 'active' class.
 * 选项卡切换：data-tab="X" 对应 id 为 "tab-X" 的页面。
 */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.tab-page').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $('tab-' + tab.dataset.tab).classList.add('active');
  });
});

/* -------------------------------------------------------------- upload */
// Clicking the drop zone opens the hidden <input type="file">.
els.dropzone.addEventListener('click', () => els.fileInput.click());
// preventDefault on dragenter/dragover is required, otherwise the browser
// does not allow a drop and would open the file instead. The 'drag' class
// highlights the zone while a file is held over it.
// 必须阻止 dragenter/dragover 的默认行为，否则浏览器会直接打开文件。
['dragenter', 'dragover'].forEach((type) =>
  els.dropzone.addEventListener(type, (e) => {
    e.preventDefault(); els.dropzone.classList.add('drag');
  }));
['dragleave', 'drop'].forEach((type) =>
  els.dropzone.addEventListener(type, (e) => {
    e.preventDefault(); els.dropzone.classList.remove('drag');
  }));
// Second 'drop' listener: hand the first dropped file to the upload routine.
els.dropzone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) handleUpload(e.dataTransfer.files[0]);
});
// File chosen through the browse dialog.
els.fileInput.addEventListener('change', () => {
  if (els.fileInput.files.length) handleUpload(els.fileInput.files[0]);
});

/**
 * Upload an AIS file, show the loading/validation report and draw the
 * classified points.
 *
 * The server returns at most about 6 000 thinned points; each is drawn as a
 * small borderless circle in its zone colour with a tooltip showing class
 * and confidence. The map is zoomed to the points, and the points export
 * links (GeoJSON/CSV of all stored points) are enabled.
 * 上传 AIS 文件，显示校验报告并在地图上绘制分类后的点。
 *
 * @param {File} file - The file to upload.
 * @returns {Promise<void>} Errors are shown in the report panel, not thrown.
 */
async function handleUpload(file) {
  els.dropzoneText.textContent = `uploading ${file.name}…`;
  els.uploadReport.hidden = false;
  els.uploadReport.textContent = 'classifying points with the neural network…';
  try {
    const data = await Api.upload(file);
    State.uploadToken = data.token;
    // `|| 1` avoids division by zero in the percentages of an empty histogram.
    const total = Object.values(data.histogram).reduce((a, b) => a + b, 0) || 1;
    // Report table; the label agreement row appears only if the file had labels.
    // 报告表；仅当文件带标签时才显示一致率。
    els.uploadReport.innerHTML = `
      <table>
        <tr><td>File</td><td>${data.report.source}</td></tr>
        <tr><td>Features read</td><td>${fmtInt(data.report.n_features)}</td></tr>
        <tr><td>Valid points</td><td>${fmtInt(data.report.n_valid)}</td></tr>
        <tr><td>Dropped</td><td>${fmtInt(data.report.n_dropped)}</td></tr>
        <tr><td>Classified</td><td>${fmtInt(data.classified)}</td></tr>
        ${data.accuracy_vs_labels ? `<tr><td>Agreement with labels</td>
          <td>${(data.accuracy_vs_labels.agreement * 100).toFixed(2)} %
          (n=${fmtInt(data.accuracy_vs_labels.n_labelled)})</td></tr>` : ''}
        ${Object.entries(data.histogram).map(([k, v]) =>
          `<tr><td>${k}</td><td>${fmtInt(v)} (${(100 * v / total).toFixed(1)}%)</td></tr>`).join('')}
      </table>`;
    els.dropzoneText.textContent = `${file.name} — ${fmtInt(data.classified)} points classified`;

    State.layers.points.clearLayers();
    data.points.forEach((p) => {
      L.circleMarker([p.latitude, p.longitude], {
        radius: 2.6, stroke: false,
        fillColor: ZONE_COLOURS[p.class_code], fillOpacity: 0.75,
      }).bindTooltip(`${p.class_code} · ${(p.confidence * 100).toFixed(0)}%`)
        .addTo(State.layers.points);
    });
    if (data.points.length) {
      map.fitBounds(L.latLngBounds(data.points.map((p) => [p.latitude, p.longitude])).pad(0.15));
    }
    $('dl-points-geojson').href = `/api/export/points/${data.token}.geojson`;
    $('dl-points-csv').href     = `/api/export/points/${data.token}.csv`;
    els.pointsExport.hidden = false;
    loadDbStats();
  } catch (err) {
    els.uploadReport.textContent = 'Upload failed: ' + err.message;
    els.dropzoneText.textContent = 'Drop a file here or click to browse';
  }
}

/* --------------------------------------------------------- model panel */
/**
 * Load GET /api/model and fill the "Model" tab: the header pill with the test
 * accuracy, a table of architecture and test metrics (accuracy, macro F1,
 * Cohen's kappa, macro ROC-AUC), per-class F1, the confusion matrix and the
 * training curves. If the model is missing (HTTP 503) the pill turns red and
 * the error text is shown instead.
 * 加载模型信息并填充“模型”页。
 * @returns {Promise<void>}
 */
async function loadModel() {
  try {
    const data = await Api.model();
    const s = data.summary, m = data.metrics || {};
    setStatus(els.modelPill,
      `model ${(s.accuracy * 100).toFixed(1)}% acc`, 'pill-ok');
    els.modelBlock.innerHTML = `
      <table>
        <tr><td>Architecture</td><td>${s.architecture}</td></tr>
        <tr><td>Input features</td><td>${s.n_features}</td></tr>
        <tr><td>Parameters</td><td>${fmtInt(s.parameters)}</td></tr>
        <tr><td>Reference samples</td><td>${fmtInt(s.reference_samples)}</td></tr>
        <tr><td>Test accuracy</td><td>${fmt(m.accuracy * 100, 2)} %</td></tr>
        <tr><td>Macro F1</td><td>${fmt(m.macro_f1, 4)}</td></tr>
        <tr><td>Cohen κ</td><td>${fmt(m.cohen_kappa, 4)}</td></tr>
        <tr><td>Macro ROC-AUC</td><td>${fmt(m.macro_roc_auc, 4)}</td></tr>
        <tr><td>Trained</td><td>${s.trained_utc || '—'}</td></tr>
      </table>
      ${m.per_class ? '<h4>Per class F1</h4><table>' + Object.entries(m.per_class)
        .map(([k, v]) => `<tr><td>${k}</td><td>${fmt(v.f1, 4)}</td></tr>`).join('') + '</table>' : ''}`;
    Charts.confusionMatrix(els.confusion, m.confusion_matrix,
      ['OPEN_SEA', 'COASTAL_SEA', 'NEAR_COAST', 'COASTLINE']);
    Charts.trainingCurves(els.historyChart, data.history);
  } catch (err) {
    setStatus(els.modelPill, 'model unavailable', 'pill-bad');
    els.modelBlock.textContent = 'Model not available: ' + err.message;
  }
}

/**
 * Load GET /api/statistics and show the database counters and file size in
 * the "Data" tab. Called at start-up, after each route and each upload.
 * @returns {Promise<void>}
 */
async function loadDbStats() {
  try {
    const s = await Api.statistics();
    els.dbStats.innerHTML = `<table>
      <tr><td>Sessions</td><td>${fmtInt(s.sessions)}</td></tr>
      <tr><td>Stored points</td><td>${fmtInt(s.points)}</td></tr>
      <tr><td>Stored routes</td><td>${fmtInt(s.routes)}</td></tr>
      <tr><td>Waypoints</td><td>${fmtInt(s.waypoints)}</td></tr>
      <tr><td>Database size</td><td>${fmt(s.database_size_kb, 1)} kB</td></tr>
    </table>`;
  } catch (err) { els.dbStats.textContent = 'unavailable'; }
}

/**
 * Load the 25 most recent stored routes (GET /api/routes?limit=25) into the
 * history list. Clicking an item fetches that route (GET /api/routes/{uid})
 * and draws its waypoints as a plain blue line; stored routes are shown
 * without zone colouring and without updating the result panels.
 * 加载历史航线列表，点击条目在地图上显示该航线。
 * @returns {Promise<void>}
 */
async function loadHistory() {
  try {
    const data = await Api.routes(25);
    els.historyList.innerHTML = data.routes.map((r) => `
      <div class="history-item" data-uid="${r.route_uid}">
        <div class="row1"><span>${r.algorithm.toUpperCase()}</span>
          <span>${fmt(r.distance_km, 0)} km</span></div>
        <div class="row2">${r.created_utc} · ${fmtInt(r.n_waypoints)} wp ·
          ${fmt(r.runtime_s, 3)} s · detour ${fmt(r.detour_ratio, 3)}×</div>
      </div>`).join('') || '<div class="muted">No routes stored yet.</div>';
    // Handlers are attached after every refresh because innerHTML replaced the items.
    // 每次刷新后重新绑定点击事件（innerHTML 已替换元素）。
    els.historyList.querySelectorAll('.history-item').forEach((item) => {
      item.addEventListener('click', async () => {
        const route = await Api.route(item.dataset.uid);
        State.layers.route.clearLayers();
        const coords = route.waypoints.map((w) => [w.latitude, w.longitude]);
        if (coords.length) {
          L.polyline(coords, { color: '#4da3ff', weight: 4 }).addTo(State.layers.route);
          map.fitBounds(L.polyline(coords).getBounds().pad(0.2));
        }
      });
    });
  } catch (err) { els.historyList.textContent = 'unavailable'; }
}

// Manual refresh button of the history list.
$('btn-refresh-history').addEventListener('click', loadHistory);

/* ------------------------------------------------------------- legend */
// Map legend built from GET /api/zones: colour swatch, class code and the
// traversal cost (or "impassable"), plus the great-circle line colour.
// 由 /api/zones 生成地图图例。
Api.zones().then((data) => {
  els.mapLegend.innerHTML = data.classes.map((z) =>
    `<div><i style="background:${z.colour}"></i>${z.code}
      ${z.navigable ? `(cost ${z.cost})` : '(impassable)'}</div>`).join('') +
    `<div style="margin-top:5px"><i style="background:#ef6461"></i>great circle</div>`;
}).catch(() => { els.mapLegend.textContent = 'legend unavailable'; });

/* ---------------------------------------------------------- bootstrap */
// Initial load: draw markers for the default coordinates in the form and
// fetch the model panel, database statistics and route history in parallel.
// 页面启动：绘制默认起终点并加载模型、统计与历史数据。
readEndpointsFromInputs();
loadModel();
loadDbStats();
loadHistory();
