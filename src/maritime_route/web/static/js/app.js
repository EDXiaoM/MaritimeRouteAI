/* ==========================================================================
 * app.js - application controller: map interaction, live planning, exports.
 * 主控脚本：地图交互、实时规划、结果展示与导出。
 *
 * Place in the application: the main script of the single-page client
 * (index.html). It is loaded last, after vendor/leaflet/leaflet.js (global L),
 * js/i18n.js (globals I18N, t: interface texts in English, Russian and
 * Chinese), js/api.js (globals Api, LiveChannel) and js/charts.js (global Charts), and
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
 *   tabs, upload, model panel, legend, language, bootstrap
 * Every text shown to the user goes through t(key) or I18N.server(text)
 * (version 1.3); switching the language re-renders what is on the screen.
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
 * @property {number[][]} via - Intermediate points [[lat, lon], ...] in order (1.3).
 * @property {boolean} viaMode - True while every map click adds a via point.
 * @property {?Object} paramSpec - Parameter description from GET /api/parameters.
 * @property {Object} params - Parameter values that differ from the defaults.
 * @property {?string} job - Kind of the running search: 'plan' | 'benchmark' | null.
 * @property {?number} jobStart - performance.now() when the search started (timer).
 */
const State = {
  start: null,
  end: null,
  pickNext: 'start',
  plan: null,
  uploadToken: null,
  layers: {},
  liveBusy: false,
  via: [],
  viaMode: false,
  paramSpec: null,
  params: {},
  job: null,
  jobStart: null,
  // Kept to redraw the screen when the interface language changes (1.3).
  wsState: null, lastFrame: null, lastDone: null, started: null,
  bench: null, model: null, zones: null, upload: null,
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
// Rebuilt when the interface language changes (the names are translated).
let layerControl = null;
function buildLayerControl() {
  if (layerControl) layerControl.remove();
  // Attribution of the on-board chart in the interface language.
  const before = onboardLayer.options.attribution;
  onboardLayer.options.attribution = t('layer.onboardAttr');
  if (map.hasLayer(onboardLayer) && map.attributionControl && before !== onboardLayer.options.attribution) {
    map.attributionControl.removeAttribution(before).addAttribution(onboardLayer.options.attribution);
  }
  layerControl = L.control.layers(
    { [t('layer.osm')]: osmLayer, [t('layer.onboard')]: onboardLayer },
    {}, { position: 'topleft' }
  ).addTo(map);
}
buildLayerControl();

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
//   bench   - routes of all methods of the last comparison (version 1.3)
State.layers.bench    = L.layerGroup().addTo(map);

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
  resultBlock: $('result-block'), metricGrid: $('metric-grid'), coastCheck: $('coast-check'),
  zoneBar: $('zone-bar'), zoneLegend: $('zone-legend'),
  compareBlock: $('compare-block'), benchmarkBlock: $('benchmark-block'),
  modelBlock: $('model-block'), confusion: $('confusion'), historyChart: $('history-chart'),
  historyList: $('history-list'), dbStats: $('db-stats'),
  uploadReport: $('upload-report'), pointsExport: $('points-export'),
  dropzone: $('dropzone'), fileInput: $('file-input'), dropzoneText: $('dropzone-text'),
  cursorReadout: $('cursor-readout'), mapLegend: $('map-legend'), pickHint: $('pick-hint'),
  // version 1.3
  viaBlock: $('via-block'), viaList: $('via-list'),
  btnPickStart: $('btn-pick-start'), btnPickEnd: $('btn-pick-end'), btnPickVia: $('btn-pick-via'),
  btnSwap: $('btn-swap'), btnClearVia: $('btn-clear-via'),
  paramsBody: $('params-body'), paramsChanged: $('params-changed'), btnParamsReset: $('btn-params-reset'),
  timer: $('timer'), timerLimit: $('timer-limit'), btnStop: $('btn-stop'),
  resultFile: $('result-file'), resultPath: $('result-path'), btnCopyPath: $('btn-copy-path'),
  dlResult: $('dl-result'), statusBanner: $('status-banner'),
  attempts: $('attempts'), attemptsBody: $('attempts-body'),
  showBenchRoutes: $('show-bench-routes'),
  resultsDir: $('results-dir'), resultsList: $('results-list'),
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
  (v === null || v === undefined) ? '—' : Number(v).toLocaleString(I18N.locale);

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
  // Intermediate points: numbered yellow markers, draggable; a right click
  // (or the × in the list) removes one. 途经点：可拖动的编号标记，右键删除。
  State.via.forEach((p, i) => {
    L.marker(p, { icon: markerIcon('#ffd166', String(i + 1)), draggable: true })
      .on('dragend', (e) => { const q = e.target.getLatLng(); State.via[i] = [q.lat, q.lng]; redrawEndpoints(); })
      .on('contextmenu', () => removeVia(i))
      .bindTooltip(t('via.tooltip', { n: i + 1 })).addTo(State.layers.markers);
  });
  renderViaList();
  if (State.start) {
    L.marker(State.start, { icon: markerIcon('#7fe3b0', 'A'), draggable: true })
      .on('dragend', (e) => { const p = e.target.getLatLng(); setEndpoint('start', p.lat, p.lng); })
      .bindTooltip(t('marker.start')).addTo(State.layers.markers);
  }
  if (State.end) {
    L.marker(State.end, { icon: markerIcon('#ffb4b2', 'B'), draggable: true })
      .on('dragend', (e) => { const p = e.target.getLatLng(); setEndpoint('end', p.lat, p.lng); })
      .bindTooltip(t('marker.end')).addTo(State.layers.markers);
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
  // Shift + click, or the "Add via point" mode, adds an intermediate point.
  // Shift+点击或“添加途经点”模式下，点击添加途经点。
  if (State.viaMode || (event.originalEvent && event.originalEvent.shiftKey)) {
    State.via.push([lat, lng]);
    redrawEndpoints();
    updatePickHint();
    return;
  }
  setEndpoint(State.pickNext, lat, lng);
  State.pickNext = State.pickNext === 'start' ? 'end' : 'start';
  updatePickHint();
});

/** Text of the hint line: what the next click on the chart will do. */
function updatePickHint() {
  els.pickHint.innerHTML = State.viaMode ? t('pick.via')
    : t(State.pickNext === 'start' ? 'pick.start' : 'pick.end');
  els.btnPickVia.classList.toggle('active', State.viaMode);
}

/** List of the via points under the coordinate inputs, with remove buttons. */
function renderViaList() {
  els.viaBlock.hidden = State.via.length === 0;
  els.viaList.innerHTML = State.via.map((p, i) =>
    `<li>${p[0].toFixed(4)}, ${p[1].toFixed(4)}<button data-i="${i}" title="${t('via.remove')}">×</button></li>`).join('');
  els.viaList.querySelectorAll('button').forEach((b) =>
    b.addEventListener('click', () => removeVia(Number(b.dataset.i))));
}

/** Remove via point i. @param {number} i - Index in State.via. */
function removeVia(i) {
  State.via.splice(i, 1);
  redrawEndpoints();
}

els.btnPickStart.addEventListener('click', () => {
  State.viaMode = false; State.pickNext = 'start'; updatePickHint();
});
els.btnPickEnd.addEventListener('click', () => {
  State.viaMode = false; State.pickNext = 'end'; updatePickHint();
});
els.btnPickVia.addEventListener('click', () => {
  State.viaMode = !State.viaMode; updatePickHint();
});
els.btnSwap.addEventListener('click', () => {
  readEndpointsFromInputs();
  const a = State.start, b = State.end;
  setEndpoint('start', b[0], b[1]);
  setEndpoint('end', a[0], a[1]);
  State.via.reverse();
  redrawEndpoints();
});
els.btnClearVia.addEventListener('click', () => { State.via = []; redrawEndpoints(); });

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
  State.wsState = state;
  if (state === 'open')  setStatus(els.wsStatus, t('ws.open'), 'pill-ok');
  if (state === 'closed') setStatus(els.wsStatus, t('ws.closed'), 'pill-wait');
  if (state === 'error')  setStatus(els.wsStatus, t('ws.error'), 'pill-bad');
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
  const depth = r.depth_m === null ? '' : ` · ${t(r.depth_m > 0 ? 'read.elev' : 'read.depth')} ${Math.abs(r.depth_m).toFixed(0)} m`;
  els.cursorReadout.querySelector('.zone').innerHTML =
    `<i style="display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
        background:${ZONE_COLOURS[r.class_code]}"></i>${I18N.zone(r.class_code)}
     <span style="opacity:.7">${(r.confidence * 100).toFixed(0)}%${depth}</span>`;
});

/**
 * 'started' frame: the server accepted a planning request. Shows and resets
 * the progress block and disables the Plan button until 'result' or 'error'.
 * The bar starts at 2 % so that it is visibly present.
 */
channel.on('started', (frame) => {
  els.progressBlock.hidden = false;
  els.resultFile.hidden = true;
  setBusy(frame.kind || 'plan');
  els.progressStage.textContent = frame.kind === 'benchmark'
    ? t('started.bench')
    : t('started.plan', { alg: t('meth.' + frame.algorithm) });
  els.progressFill.style.width = '2%';
  els.progressPct.textContent = '0%';
  els.progressDetail.textContent = '';
  State.started = frame;
  showTimerLimit();
  startTimer();
});

/** Time limit of the running (or last) search next to the timer. */
function showTimerLimit() {
  const frame = State.started;
  if (!frame) return;
  els.timerLimit.textContent = frame.time_limit_s
    ? t('timer.limit', { s: frame.time_limit_s }) + (frame.kind === 'benchmark' ? t('timer.perMethod') : '')
    : t('timer.noLimit');
}

/* ------------------------------------------------ timer and Stop (1.3) */
/** setInterval id of the running timer. @type {?number} */
let timerId = null;

/** Format seconds as m:ss.s for the timer. @param {number} t @returns {string} */
const fmtClock = (t) => `${Math.floor(t / 60)}:${(t % 60).toFixed(1).padStart(4, '0')}`;

/** Start the timer of the progress block (updated 10 times per second). */
function startTimer() {
  State.jobStart = performance.now();
  clearInterval(timerId);
  timerId = setInterval(() => {
    els.timer.textContent = '⏱ ' + fmtClock((performance.now() - State.jobStart) / 1000);
  }, 100);
}

/** Stop the timer and show the final time (the server's own measurement if given). */
function stopTimer(seconds) {
  clearInterval(timerId);
  timerId = null;
  const t = seconds ?? (State.jobStart ? (performance.now() - State.jobStart) / 1000 : 0);
  els.timer.textContent = '⏱ ' + fmtClock(t);
}

/**
 * Mark a search as running (or finished): the Plan and Compare buttons are
 * disabled and the Stop button enabled while a search runs.
 * @param {?string} kind - 'plan', 'benchmark' or null when finished.
 */
function setBusy(kind) {
  State.job = kind;
  els.btnPlan.disabled = !!kind;
  els.btnBenchmark.disabled = !!kind;
  els.btnStop.disabled = !kind;
  els.btnStop.textContent = t('btn.stop');
}

/** Stop button: ask the server to stop the running search ('cancel' action). */
els.btnStop.addEventListener('click', () => {
  if (!State.job || !channel.isOpen) return;
  channel.send({ action: 'cancel' });
  els.btnStop.disabled = true;
  els.btnStop.textContent = t('btn.stopping');
});

/** 'cancelling' frame: the server has set the stop flag. */
channel.on('cancelling', () => {
  els.progressStage.textContent = t('stopping');
});

/**
 * Human-readable label of the `stage` field of 'progress' frames (stage names
 * are defined by the planner, cost-map builder and search modules): the key
 * 'stage.<name>' of i18n.js (grid, classify, cost_map, search, astar,
 * dijkstra, genetic, dynamic, refine, coast, leg, benchmark, stopped, done).
 * Unknown stages are shown with their raw name.
 * @param {string} stage - Stage name.
 * @returns {string}
 */
const stageText = (stage) => {
  const s = t('stage.' + stage);
  return s === 'stage.' + stage ? stage : s;
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
  const d0 = frame.detail || {};
  // Multi-leg route: the stage text names the leg. 多段航线：显示当前航段。
  const legText = d0.legs > 1 ? t('prog.leg', { i: d0.leg, n: d0.legs }) : '';
  els.progressStage.textContent = legText + stageText(frame.stage);
  els.progressFill.style.width = Math.max(pct, 2) + '%';
  els.progressPct.textContent = pct + '%';
  const d = frame.detail || {};
  if (d.cells_done !== undefined) {
    els.progressDetail.textContent = t('prog.cells', { a: fmtInt(d.cells_done), b: fmtInt(d.cells_total) });
  } else if (d.expanded !== undefined) {
    els.progressDetail.textContent = t('prog.nodes', { a: fmtInt(d.expanded), b: fmtInt(d.frontier) });
  } else if (d.sweep !== undefined) {
    // Dynamic programming: value-iteration sweep and cells whose value changed.
    els.progressDetail.textContent = t('prog.sweep', { a: fmtInt(d.sweep), b: fmtInt(d.updated) });
  } else if (d.method !== undefined) {
    els.progressDetail.textContent = I18N.server(d.stage);
  } else if (d.generation !== undefined) {
    // Genetic algorithm: generation counter and best route cost so far.
    els.progressDetail.textContent = t('prog.gen', { a: fmtInt(d.generation), b: fmt(d.best_cost, 0) });
  } else if (d.zone_shares) {
    els.progressDetail.textContent = Object.entries(d.zone_shares)
      .map(([k, v]) => `${I18N.zone(k)} ${(v * 100).toFixed(0)}%`).join(' · ');
  } else if (d.stage) {
    els.progressDetail.textContent = I18N.server(d.stage);
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
  setBusy(null);
  stopTimer(frame.plan.runtime_s);
  els.progressFill.style.width = '100%';
  els.progressPct.textContent = '100%';
  State.lastFrame = frame; State.lastDone = 'plan';
  showFinalProgress(frame.plan);
  els.btnStop.disabled = true;
  renderPlan(frame.plan);
  showResultFile(frame.result_file);
  loadHistory();
  loadDbStats();
  loadResults();
});

/** Final line of the progress block (stage = outcome, detail = steps, lattices, time). */
function showFinalProgress(plan) {
  els.progressStage.textContent = statusText(plan.status) || t('stage.done');
  els.progressDetail.textContent = t('prog.final', {
    steps: fmtInt(plan.nodes_expanded), lat: (plan.attempts || []).length, s: fmt(plan.runtime_s, 2) });
}

/** Text of a search outcome (found, no_path, step_limit, cancelled, time_limit). */
const statusText = (status) => (status ? t('status.' + status) : '');

/**
 * Show where the result file was written: absolute path (selectable), a Copy
 * button and a download link (GET /api/results/{name}).
 * 显示结果文件的位置，并提供复制路径与下载。
 * @param {?string} path - Absolute path on the server's disk.
 */
function showResultFile(path) {
  if (!path) { els.resultFile.hidden = true; return; }
  const name = path.split(/[\\/]/).pop();
  els.resultPath.textContent = path;
  els.dlResult.href = `/api/results/${encodeURIComponent(name)}`;
  els.dlResult.setAttribute('download', name);
  els.resultFile.hidden = false;
}

els.btnCopyPath.addEventListener('click', async () => {
  const text = els.resultPath.textContent;
  try {
    await navigator.clipboard.writeText(text);
    els.btnCopyPath.textContent = t('rf.copied');
  } catch (err) {
    // Clipboard API unavailable (plain http on another host): select the text.
    const range = document.createRange();
    range.selectNodeContents(els.resultPath);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    els.btnCopyPath.textContent = t('rf.ctrlc');
  }
  setTimeout(() => { els.btnCopyPath.textContent = t('rf.copy'); }, 1600);
});

/**
 * 'error' frame: show the server's message in the progress block and
 * re-enable the Plan button. The block stays visible so the text can be read.
 */
channel.on('error', (frame) => {
  // A failed cursor read-out must not block the next one. 光标实时分类失败时解除占用标志。
  State.liveBusy = false;
  if (State.job) { setBusy(null); stopTimer(); els.progressBlock.hidden = false; }
  els.btnStop.disabled = true;
  State.lastFrame = null;
  els.progressStage.textContent = t('error');
  els.progressDetail.textContent = I18N.server(frame.detail);
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
 * @param {boolean} [keepView=false] - Do not zoom the map (language switch).
 */
function renderPlan(plan, keepView = false) {
  State.plan = plan;
  State.layers.route.clearLayers();

  renderCoastCheck(plan);
  renderStatus(plan);
  renderAttempts(plan);
  if (!plan.feasible || !plan.waypoints.length) {
    els.resultBlock.hidden = false;
    els.metricGrid.innerHTML =
      `<div class="metric bad" style="grid-column:1/-1"><div class="label">${statusText(plan.status) || t('res.infeasible')}</div>
       <div class="value" style="font-size:13px">${I18N.server(plan.message) || t('res.noRoute')}</div></div>
      ${metric(t('m.searchTime'), fmt(plan.runtime_s, 2), t('u.s'))}
      ${metric(t('m.steps'), fmtInt(plan.nodes_expanded), '')}`;
    $('dl-geojson').removeAttribute('href');
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
  }).bindTooltip(t('tip.gc')).addTo(State.layers.route);

  // Points where the route still crosses land on the independent coastline
  // (only when the automatic correction could not remove them).
  // 经自动修正后仍位于陆地上的航线采样点，以红色圆点标出。
  const cc = plan.coast_check || {};
  if (cc.verdict === 'crossing' && plan.algorithm !== 'great_circle') {
    (cc.land_points || []).forEach((p) => {
      L.circleMarker(p, { radius: 4, color: '#ff2d55', weight: 2, fillOpacity: 0.85 })
        .bindTooltip(t('tip.land')).addTo(State.layers.route);
    });
  }

  // Zoom to the route with 18 % padding on each side (not when the page is
  // only redrawn in another language). 缩放至航线范围（切换语言重绘时不缩放）。
  if (!keepView) map.fitBounds(L.polyline(plan.waypoints).getBounds().pad(0.18));

  // Summary metric tiles. A detour ratio (route length / great-circle length)
  // above 1.5 is flagged with the 'warn' colour.
  // 结果指标；绕航系数大于 1.5 时以警示色显示。
  const detour = plan.detour_ratio;
  els.metricGrid.innerHTML = `
    ${metric(t('m.length'), fmt(plan.distance_km, 1), t('u.km'))}
    ${metric(t('m.gc'), fmt(plan.great_circle_km, 1), t('u.km'))}
    ${metric(t('m.detour'), fmt(detour, 3), '×', detour > 1.5 ? 'warn' : 'good')}
    ${metric(t('m.cost'), fmt(plan.total_cost, 0), '')}
    ${metric(t('m.passage'), fmt(plan.estimated_hours, 1), t('u.h'))}
    ${metric(t('m.planning'), fmt(plan.runtime_s, 3), t('u.s'))}
    ${metric(t({ genetic: 'm.evaluated', dynamic: 'm.updates' }[plan.algorithm] || 'm.nodes'),
             fmtInt(plan.nodes_expanded), '')}
    ${metric(t('m.waypoints'), fmtInt(plan.waypoints.length), '')}`;

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
 * Banner with the independent coastline check of the route (plan.coast_check):
 *   clear / grazing        -> green, "passed";
 *   corrected              -> amber, how the first route was corrected;
 *   crossing               -> red, the server's warning text;
 *   unchecked / absent     -> hidden (land mask not installed).
 * 显示独立海岸线检查结果的横幅。
 *
 * @param {Object} plan - RoutePlan.to_dict().
 */
function renderCoastCheck(plan) {
  const c = plan.coast_check || {};
  const box = els.coastCheck;
  if (!c.verdict || c.verdict === 'unchecked' || !plan.feasible) { box.hidden = true; return; }
  box.hidden = false;
  if (c.verdict === 'crossing') {
    box.className = 'verdict bad';
    box.textContent = plan.algorithm === 'great_circle'
      ? t('coast.gc', { km: fmt(c.max_land_run_km, 0) })
      : t('coast.crossing', { km: fmt(c.max_land_run_km, 1) });
    return;
  }
  const corrected = c.initial_verdict === 'crossing';
  box.className = corrected ? 'verdict fixed' : 'verdict good';
  const how = [];
  if (c.corrections) {
    how.push(t('coast.how.re', { n: c.corrections, c: c.cells_blocked }));
  }
  if (c.refined) how.push(t('coast.how.fine', { r: plan.grid.resolution_deg }));
  box.textContent = corrected
    ? t('coast.fixed', { km: fmt(c.initial_max_land_run_km, 0), how: how.join('; ') })
    : t('coast.passed', {
        what: c.verdict === 'clear' ? t('coast.clear') : t('coast.touch', { km: fmt(c.max_land_run_km, 0) }),
        n: fmtInt(c.samples) });
}

/**
 * Banner with the outcome of a search that did not find a route (version 1.3):
 * step limit, time limit, stopped by the user, no path - with a hint on
 * which parameter to change. Hidden for a found route.
 * 未找到航线时的状态说明及参数调整提示。
 * @param {Object} plan - RoutePlan.to_dict().
 */
function renderStatus(plan) {
  const box = els.statusBanner;
  // Which parameter to change: per method for the step limit (keys hint.<method>),
  // one hint for the other outcomes (hint.time_limit, hint.cancelled, hint.no_path).
  if (plan.feasible || plan.status === 'found') { box.hidden = true; return; }
  const key = plan.status === 'step_limit' ? 'hint.' + plan.algorithm : 'hint.' + plan.status;
  const hint = t(key) === key ? '' : t(key);
  box.className = 'verdict ' + (plan.status === 'no_path' ? 'bad' : 'fixed');
  box.innerHTML = t('status.after', { st: statusText(plan.status) || plan.status, s: fmt(plan.runtime_s, 1), hint });
  box.hidden = false;
}

/**
 * "How the route was found": one row per lattice tried by the planner -
 * cell size, cells, steps, time and outcome - so that the work on a complex
 * course (refinement, wider area, corridor, legs) is visible.
 * 规划过程表：每一次尝试的网格、格子数、步数、耗时与结果。
 * @param {Object} plan - RoutePlan.to_dict().
 */
function renderAttempts(plan) {
  const rows = plan.attempts || [];
  els.attempts.hidden = rows.length === 0;
  if (!rows.length) return;
  els.attemptsBody.innerHTML = `<table class="bench-table">
    <tr><th>#</th><th>${t('att.leg')}</th><th>${t('att.cell')}</th><th>${t('att.cells')}</th>
      <th>${t('att.steps')}</th><th>${t('att.s')}</th><th>${t('att.outcome')}</th></tr>
    ${rows.map((a) => `<tr><td>${a.attempt}${a.corridor_km ? ' ⟂' : ''}</td><td>${a.leg || 1}</td>
      <td>${a.resolution_deg}</td><td>${fmtInt(a.cells)}</td><td>${fmtInt(a.steps)}</td>
      <td>${fmt(a.seconds, 2)}</td>
      <td><span class="status-tag status-${a.status}">${t('tag.' + a.status)}</span>${a.coast_verdict === 'crossing' ? t('att.land') : ''}</td></tr>
      ${a.next ? `<tr><td></td><td colspan="6" style="text-align:left;color:var(--muted)">→ ${I18N.server(a.next)}</td></tr>` : ''}`).join('')}
  </table>
  <div class="muted" style="margin-top:4px">${t('att.note')}</div>`;
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
  const cv = (plan.coast_check || {}).verdict;
  const km = t('u.km');
  els.compareBlock.innerHTML = `
    <table>
      <tr><td>${t('c.gcLen')}</td><td>${fmt(b.distance_km, 1)} ${km}</td></tr>
      <tr><td>${t('c.optLen')}</td><td>${fmt(plan.distance_km, 1)} ${km}</td></tr>
      <tr><td>${t('c.extra')}</td><td>${fmt(b.extra_distance_pct, 2)} %</td></tr>
      <tr><td>${t('c.samples')}</td><td>${b.land_samples} / ${b.samples}</td></tr>
      <tr><td>${t('c.share')}</td><td>${fmt((b.land_share || 0) * 100, 1)} %</td></tr>
      <tr><td>${t('c.interior')}</td>
          <td>${plan.interior_land_waypoints}</td></tr>
      <tr><td>${t('c.coast')}</td>
          <td>${cv ? t('verdict.' + cv) : '-'}${(plan.coast_check || {}).max_land_run_km
               ? t('c.longest', { km: fmt(plan.coast_check.max_land_run_km, 1) }) : ''}</td></tr>
      <tr><td>${t('c.wps')}</td><td>${fmtInt(plan.waypoints.length)}</td></tr>
      <tr><td>${t('c.time')}</td><td>${fmt(plan.estimated_hours - (b.distance_km / (plan.speed_knots * 1.852)), 1)} ${t('u.h')}</td></tr>
    </table>
    <div class="verdict ${navigable ? 'good' : 'bad'}">${baselineVerdict(b)}</div>`;
}

/**
 * Conclusion about the great circle in the current language. The server
 * sends it in English; for a route with via points it lists the legs
 * ("Great circles of the individual legs: navigable, crosses land").
 * @param {Object} b - plan.baseline.
 * @returns {string}
 */
function baselineVerdict(b) {
  const v = b.verdict || '';
  const legs = v.match(/^Great circles of the individual legs: (.*)$/);
  if (legs) {
    return t('c.legs', { list: legs[1].split(', ')
      .map((x) => t(x === 'navigable' ? 'c.nav' : 'c.crossesLand')).join(', ') });
  }
  if (!v) return '';
  return t(b.is_navigable ? 'c.gcNav' : 'c.gcLand');
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
    via: State.via,                                  // intermediate points (1.3)
    parameters: { ...State.params },                 // values changed in the panels (1.3)
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
    alert(t('alert.points'));
    return;
  }
  State.layers.bench.clearLayers();
  if (channel.isOpen) {
    channel.send(request);
  } else {
    // Fall back to REST when the socket is down (no progress, no Stop).
    // WebSocket 不可用时退回 REST（无进度、不能停止）。
    els.btnPlan.disabled = true;
    Api.plan(request)
      .then((plan) => { renderPlan(plan); showResultFile(plan.result_file); })
      .catch((err) => alert(t('alert.planFailed') + err.message))
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
  State.bench = null;
  els.benchmarkBlock.textContent = t('bench.running');
  // Programmatic click on the tab button reuses the tab-switch handler.
  document.querySelector('[data-tab="compare"]').click();
  if (channel.isOpen) {
    // Over the WebSocket: progress, timer and Stop work as for one route (1.3).
    channel.send({ ...request, action: 'benchmark' });
    return;
  }
  els.btnBenchmark.disabled = true;
  try {
    renderBenchmark(await Api.benchmark(request));
  } catch (err) {
    els.benchmarkBlock.textContent = t('bench.failed') + err.message;
  } finally {
    els.btnBenchmark.disabled = false;
  }
});

/** Colours of the methods in the comparison table and on the chart. */
const METHOD_COLOURS = {
  astar: '#4da3ff', dijkstra: '#c77dff', dynamic: '#2a9d8f', genetic: '#ffb703', great_circle: '#ef6461',
};
/** Display name of a method in the current language (keys meth.<method>). */
const methodName = (a) => t('meth.' + a);

/** 'benchmark' frame (WebSocket comparison finished, possibly stopped). */
channel.on('benchmark', (frame) => {
  setBusy(null);
  stopTimer(frame.data.total_seconds);
  State.lastDone = 'bench';
  els.progressStage.textContent = t(frame.data.status === 'cancelled' ? 'bench.stopped' : 'bench.finished');
  els.progressFill.style.width = '100%';
  els.progressPct.textContent = '100%';
  renderBenchmark(frame.data);
  showResultFile(frame.result_file);
  loadResults();
});

/**
 * Fill the comparison table: per method the status, length, cost, gap to the
 * optimum, steps (and the step limit), search time, and the theoretical
 * complexity with its worst-case bound; then the derived ratios. The routes
 * are drawn on the chart in the colours of the table.
 * 算法对比表：状态、长度、代价、与最优值差距、步数与上限、耗时及理论复杂度。
 * @param {Object} data - Output of planner.benchmark (see app.py).
 */
function renderBenchmark(data) {
  State.bench = data;                     // kept to redraw the table in another language
  State.layers.bench.clearLayers();
  const rows = data.results || [];
  if (els.showBenchRoutes.checked) {
    rows.forEach((r) => {
      if (r.waypoints && r.waypoints.length > 1) {
        L.polyline(r.waypoints, {
          color: METHOD_COLOURS[r.algorithm], weight: r.algorithm === 'great_circle' ? 2 : 3.2,
          opacity: 0.9, dashArray: r.algorithm === 'great_circle' ? '7,6' : null,
        }).bindTooltip(methodName(r.algorithm)).addTo(State.layers.bench);
      }
    });
  }
  const sw = (a) => `<i class="bench-swatch" style="background:${METHOD_COLOURS[a]}"></i>`;
  els.benchmarkBlock.innerHTML = `
    <table class="bench-table">
      <tr><th>${t('b.method')}</th><th>${t('b.status')}</th><th>${t('b.km')}</th><th>${t('b.cost')}</th>
        <th>${t('b.opt')}</th><th>${t('b.steps')}</th><th>${t('b.limit')}</th><th>${t('b.s')}</th></tr>
      ${rows.map((r) => `<tr><td>${sw(r.algorithm)}${methodName(r.algorithm)}</td>
        <td><span class="status-tag status-${r.status}">${t('tag.' + (r.feasible || r.status !== 'found' ? r.status : 'land'))}</span></td>
        <td>${fmt(r.distance_km, 0)}</td><td>${r.total_cost === null || r.total_cost === undefined ? '—' : fmt(r.total_cost, 0)}</td>
        <td>${r.gap_to_optimum_pct === null || r.gap_to_optimum_pct === undefined ? '—' : fmt(r.gap_to_optimum_pct, 2)}</td>
        <td>${fmtInt(r.nodes_expanded)}</td><td>${r.step_limit ? fmtInt(r.step_limit) : '—'}</td>
        <td>${fmt(r.runtime_s, 3)}</td></tr>`).join('')}
    </table>
    <h4>${t('b.complexity')}</h4>
    <table class="bench-table">
      <tr><th>${t('b.method')}</th><th>${t('b.time')}</th><th>${t('b.ops')}</th><th>${t('b.bound')}</th><th>${t('b.share')}</th></tr>
      ${rows.map((r) => `<tr><td>${methodName(r.algorithm)}</td>
        <td><code>${(r.complexity || {}).time || ''}</code></td>
        <td title="${(r.complexity || {}).work || ''}">${fmtInt(r.measured_ops ?? r.nodes_expanded)}</td>
        <td>${r.theoretical_bound_ops ? fmtInt(r.theoretical_bound_ops) : '—'}</td>
        <td>${r.theoretical_bound_ops && r.measured_ops ? fmt(100 * r.measured_ops / r.theoretical_bound_ops, 1) + ' %' : '—'}</td></tr>`).join('')}
    </table>
    <div class="muted" style="margin-top:4px">${t('b.note', { v: fmtInt(data.navigable_cells) })}</div>
    <table style="margin-top:8px">
      <tr><td>${t('b.reduction')}</td><td>${data.astar_node_reduction ?? '—'}×</td></tr>
      <tr><td>${t('b.speedup')}</td><td>${data.astar_time_ratio ?? '—'}×</td></tr>
      <tr><td>${t('b.diffAD')}</td><td>${data.optimality_gap_pct ?? '—'} %</td></tr>
      <tr><td>${t('b.gaGap')}</td><td>${data.genetic_gap_pct ?? '—'} %</td></tr>
      <tr><td>${t('b.diffDP')}</td><td>${data.dynamic_gap_pct ?? '—'} %</td></tr>
      <tr><td>${t('b.build')}</td><td>${fmt(data.cost_map_build_s, 3)} ${t('b.shared')}</td></tr>
      <tr><td>${t('b.grid')}</td><td>${data.grid && data.grid.n_rows ? `${data.grid.n_rows}×${data.grid.n_cols} @ ${data.grid.resolution_deg}°` : '—'}</td></tr>
      <tr><td>${t('b.whole')}</td><td>${fmt(data.total_seconds, 2)} ${t('u.s')}</td></tr>
    </table>
    <div class="muted" style="margin-top:6px">${t('b.runtimes')}</div>`;
}

els.showBenchRoutes.addEventListener('change', () => {
  if (!els.showBenchRoutes.checked) State.layers.bench.clearLayers();
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
  State.via = []; State.viaMode = false;
  renderViaList(); updatePickHint();
  els.resultBlock.hidden = true;
  els.resultFile.hidden = true;
  els.progressBlock.hidden = true;
  State.lastFrame = null;
  els.compareBlock.textContent = t('cmp.first');
  els.cursorReadout.querySelector('.zone').textContent = t('cursor.hint');
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
  els.speedOut.textContent = parseFloat(els.speed.value).toFixed(1) + ' ' + t('unit.kn');
});

/**
 * Preset voyage chips: data-preset="startLat,startLon,endLat,endLon".
 * Fills the inputs, redraws the markers and zooms to both ports with 40 %
 * padding. Planning is not started automatically. An optional
 * data-resolution sets the lattice step recommended for that voyage (the
 * island-dense Aegean Sea needs 0.1°, see subsection 3.5.5 of the thesis);
 * the other presets restore the default of 0.25°.
 * 预设航线按钮：填入坐标并缩放地图，不自动规划；data-resolution 给出该航线推荐的
 * 网格分辨率（岛屿密集的爱琴海需要 0.1°），其余预设恢复默认 0.25°。
 */
document.querySelectorAll('.chip[data-preset]').forEach((chip) => {
  chip.addEventListener('click', () => {
    const [a, b, c, d] = chip.dataset.preset.split(',').map(Number);
    els.startLat.value = a; els.startLon.value = b;
    els.endLat.value = c;   els.endLon.value = d;
    els.resolution.value = chip.dataset.resolution || '0.25';
    els.resolution.dispatchEvent(new Event('input'));     // refresh the label
    State.via = [];                                       // a preset has no via points
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
  els.dropzoneText.textContent = t('up.uploading', { f: file.name });
  els.uploadReport.hidden = false;
  els.uploadReport.textContent = t('up.classifying');
  try {
    const data = await Api.upload(file);
    State.uploadToken = data.token;
    State.upload = { data, name: file.name };      // kept to redraw in another language
    renderUploadReport();

    State.layers.points.clearLayers();
    data.points.forEach((p) => {
      L.circleMarker([p.latitude, p.longitude], {
        radius: 2.6, stroke: false,
        fillColor: ZONE_COLOURS[p.class_code], fillOpacity: 0.75,
      }).bindTooltip(() => `${I18N.zone(p.class_code)} · ${(p.confidence * 100).toFixed(0)}%`)
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
    State.upload = null;
    els.uploadReport.textContent = t('up.failed') + err.message;
    els.dropzoneText.textContent = t('drop.text');
  }
}

/**
 * Loading/validation report of the last upload (State.upload) and the class
 * histogram; the label agreement row appears only if the file had labels.
 * 上传报告表；仅当文件带标签时才显示一致率。
 */
function renderUploadReport() {
  if (!State.upload) return;
  const { data, name } = State.upload;
  // `|| 1` avoids division by zero in the percentages of an empty histogram.
  const total = Object.values(data.histogram).reduce((a, b) => a + b, 0) || 1;
  els.uploadReport.innerHTML = `
    <table>
      <tr><td>${t('up.file')}</td><td>${data.report.source}</td></tr>
      <tr><td>${t('up.features')}</td><td>${fmtInt(data.report.n_features)}</td></tr>
      <tr><td>${t('up.valid')}</td><td>${fmtInt(data.report.n_valid)}</td></tr>
      <tr><td>${t('up.dropped')}</td><td>${fmtInt(data.report.n_dropped)}</td></tr>
      <tr><td>${t('up.classified')}</td><td>${fmtInt(data.classified)}</td></tr>
      ${data.accuracy_vs_labels ? `<tr><td>${t('up.agreement')}</td>
        <td>${(data.accuracy_vs_labels.agreement * 100).toFixed(2)} %
        (n=${fmtInt(data.accuracy_vs_labels.n_labelled)})</td></tr>` : ''}
      ${Object.entries(data.histogram).map(([k, v]) =>
        `<tr><td>${I18N.zone(k)}</td><td>${fmtInt(v)} (${(100 * v / total).toFixed(1)}%)</td></tr>`).join('')}
    </table>`;
  els.dropzoneText.textContent = t('up.done', { f: name, n: fmtInt(data.classified) });
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
    State.model = { data: await Api.model() };
  } catch (err) {
    State.model = { error: err.message };
  }
  renderModel();
}

/** Fill the "Model" tab and the header pill from State.model (any language). */
function renderModel() {
  if (!State.model) return;
  if (State.model.error !== undefined) {
    setStatus(els.modelPill, t('model.unavailable'), 'pill-bad');
    els.modelBlock.textContent = t('mo.unavailable') + State.model.error;
    return;
  }
  const data = State.model.data;
  const s = data.summary, m = data.metrics || {};
  setStatus(els.modelPill, t('model.pill', { acc: (s.accuracy * 100).toFixed(1) }), 'pill-ok');
  els.modelBlock.innerHTML = `
    <table>
      <tr><td>${t('mo.arch')}</td><td>${s.architecture}</td></tr>
      <tr><td>${t('mo.features')}</td><td>${s.n_features}</td></tr>
      <tr><td>${t('mo.params')}</td><td>${fmtInt(s.parameters)}</td></tr>
      <tr><td>${t('mo.ref')}</td><td>${fmtInt(s.reference_samples)}</td></tr>
      <tr><td>${t('mo.acc')}</td><td>${fmt(m.accuracy * 100, 2)} %</td></tr>
      <tr><td>${t('mo.f1')}</td><td>${fmt(m.macro_f1, 4)}</td></tr>
      <tr><td>${t('mo.kappa')}</td><td>${fmt(m.cohen_kappa, 4)}</td></tr>
      <tr><td>${t('mo.auc')}</td><td>${fmt(m.macro_roc_auc, 4)}</td></tr>
      <tr><td>${t('mo.trained')}</td><td>${s.trained_utc || '—'}</td></tr>
    </table>
    ${m.per_class ? `<h4>${t('mo.perClass')}</h4><table>` + Object.entries(m.per_class)
      .map(([k, v]) => `<tr><td>${I18N.zone(k)}</td><td>${fmt(v.f1, 4)}</td></tr>`).join('') + '</table>' : ''}`;
  Charts.confusionMatrix(els.confusion, m.confusion_matrix,
    ['OPEN_SEA', 'COASTAL_SEA', 'NEAR_COAST', 'COASTLINE']);
  Charts.trainingCurves(els.historyChart, data.history);
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
      <tr><td>${t('db.sessions')}</td><td>${fmtInt(s.sessions)}</td></tr>
      <tr><td>${t('db.points')}</td><td>${fmtInt(s.points)}</td></tr>
      <tr><td>${t('db.routes')}</td><td>${fmtInt(s.routes)}</td></tr>
      <tr><td>${t('db.waypoints')}</td><td>${fmtInt(s.waypoints)}</td></tr>
      <tr><td>${t('db.size')}</td><td>${fmt(s.database_size_kb, 1)} kB</td></tr>
    </table>`;
  } catch (err) { els.dbStats.textContent = t('unavailable'); }
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
        <div class="row1"><span>${methodName(r.algorithm)}</span>
          <span>${fmt(r.distance_km, 0)} ${t('u.km')}</span></div>
        <div class="row2">${r.created_utc} · ${fmtInt(r.n_waypoints)} ${t('hist.wp')} ·
          ${fmt(r.runtime_s, 3)} ${t('u.s')} · ${t('hist.detour')} ${fmt(r.detour_ratio, 3)}×</div>
      </div>`).join('') || `<div class="muted">${t('hist.none')}</div>`;
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
  } catch (err) { els.historyList.textContent = t('unavailable'); }
}

// Manual refresh button of the history list.
$('btn-refresh-history').addEventListener('click', loadHistory);

/* ------------------------------------------------------------- legend */
// Map legend built from GET /api/zones: colour swatch, class code and the
// traversal cost (or "impassable"), plus the great-circle line colour.
// 由 /api/zones 生成地图图例。
Api.zones().then((data) => { State.zones = data; renderLegend(); })
  .catch(() => { State.zones = null; els.mapLegend.textContent = t('legend.unavailable'); });

/** Draw the map legend from State.zones in the current language. */
function renderLegend() {
  if (!State.zones) return;
  els.mapLegend.innerHTML = State.zones.classes.map((z) =>
    `<div><i style="background:${z.colour}"></i>${t('legend.item', { z: I18N.zone(z.code),
      c: z.navigable ? t('legend.cost', { c: z.cost }) : t('legend.impassable') })}</div>`).join('') +
    `<div style="margin-top:5px"><i style="background:#ef6461"></i>${t('legend.gc')}</div>`;
}

/* -------------------------------------------- parameter panels (1.3) */
/**
 * Key under which the changed parameter values are remembered in this
 * browser (a convenience only: the page works without browser storage).
 */
const PARAMS_KEY = 'maritime-route.parameters.v1';

/* The parameter panel is closed by default; whether the user opened it is
   remembered in this browser. 参数面板默认收起；用户展开后在本浏览器中记住。 */
const PARAMS_OPEN_KEY = 'maritime-route.params-open';
try {
  if (localStorage.getItem(PARAMS_OPEN_KEY) === '1') $('params-panel').open = true;
} catch (err) { /* storage blocked: stays closed */ }
$('params-panel').addEventListener('toggle', () => {
  try { localStorage.setItem(PARAMS_OPEN_KEY, $('params-panel').open ? '1' : '0'); } catch (err) { /* not remembered */ }
});

/**
 * Load the parameter description (GET /api/parameters) and build the panels.
 * Values changed earlier in this browser are restored.
 * 加载参数说明并生成参数面板；恢复此前在本浏览器中修改过的值。
 */
async function loadParameters() {
  try {
    State.paramSpec = await Api.parameters();
    try {
      const saved = JSON.parse(localStorage.getItem(PARAMS_KEY) || '{}');
      const known = new Set(State.paramSpec.groups.flatMap((g) => g.fields.map((f) => f.key)));
      State.params = Object.fromEntries(Object.entries(saved).filter(([k]) => known.has(k)));
    } catch (err) { State.params = {}; }
    if (els.resultsDir) els.resultsDir.textContent = State.paramSpec.results_dir;
    renderParameters();
  } catch (err) {
    els.paramsBody.textContent = t('params.unavailable') + err.message;
  }
}

/** Remember the changed values in this browser (ignored if storage is blocked). */
function saveParameters() {
  try { localStorage.setItem(PARAMS_KEY, JSON.stringify(State.params)); } catch (err) { /* private mode */ }
}

/**
 * Render the "Lattice and run" group and the group of the selected method.
 * A field whose value differs from the default is highlighted; the summary
 * line counts the changed values of the groups shown.
 * 显示“网格与运行”分组以及当前算法的参数分组；修改过的参数高亮显示。
 */
function renderParameters() {
  const spec = State.paramSpec;
  if (!spec) return;
  const algorithm = els.algorithm.value;
  const groups = spec.groups.filter((g) => g.id === 'common' || g.id === algorithm);
  const value = (f) => (f.key in State.params ? State.params[f.key] : f.default);
  // Group titles, labels, help and units in the interface language (I18N.param);
  // the server's English texts are the fallback. 参数名称与说明随界面语言变化。
  const groupTitle = (g) => { const k = 'g.' + g.id; return t(k) === k ? g.title : t(k); };
  els.paramsBody.innerHTML = groups.map((g) => `
    <div class="params-group"><h4>${groupTitle(g)}</h4>
      ${g.fields.map((f) => {
        const changed = f.key in State.params;
        const tr = I18N.param(f);
        const def = f.type === 'bool' ? t(f.default ? 'params.on' : 'params.off') : fmtParam(f.default);
        const help = escapeAttr(`${tr.help} (${t('params.default')} ${def})`);
        if (f.type === 'bool') {
          return `<div class="param param-bool ${changed ? 'changed' : ''}">
            <label for="p-${f.key}" title="${help}">${tr.label}</label>
            <input type="checkbox" id="p-${f.key}" data-key="${f.key}" ${value(f) ? 'checked' : ''}></div>`;
        }
        return `<div class="param ${changed ? 'changed' : ''}">
          <label for="p-${f.key}" title="${help}">${tr.label}${tr.unit ? `<span class="unit">${tr.unit}</span>` : ''}</label>
          <input type="number" id="p-${f.key}" data-key="${f.key}" value="${value(f)}"
                 min="${f.min ?? ''}" max="${f.max ?? ''}" step="${f.step ?? 'any'}"></div>`;
      }).join('')}
    </div>`).join('') +
    (algorithm === 'great_circle' ? `<div class="muted">${t('params.noneGC')}</div>` : '');
  els.paramsBody.querySelectorAll('input[data-key]').forEach((input) => {
    input.addEventListener('change', () => {
      const field = spec.groups.flatMap((g) => g.fields).find((f) => f.key === input.dataset.key);
      let v = field.type === 'bool' ? input.checked : Number(input.value);
      if (field.type !== 'bool') {
        if (Number.isNaN(v)) v = field.default;
        if (field.min !== null && v < field.min) v = field.min;      // clamp like the server
        if (field.max !== null && v > field.max) v = field.max;
        if (field.type === 'int') v = Math.round(v);
        input.value = v;
      }
      if (v === field.default) delete State.params[field.key]; else State.params[field.key] = v;
      saveParameters();
      renderParameters();
    });
  });
  const shown = groups.flatMap((g) => g.fields).filter((f) => f.key in State.params).length;
  els.paramsChanged.textContent = shown ? t('params.changed', { n: shown }) : t('params.defaults');
}

/** Escape a text for an HTML attribute value (the help texts contain quotes). */
const escapeAttr = (s) => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

/** Number formatting of a default value in the tooltip. */
const fmtParam = (v) => (Number.isInteger(v) ? fmtInt(v) : String(v));

els.algorithm.addEventListener('change', renderParameters);
els.btnParamsReset.addEventListener('click', () => {
  // Reset only the groups shown (common + the selected method).
  const algorithm = els.algorithm.value;
  State.paramSpec.groups.filter((g) => g.id === 'common' || g.id === algorithm)
    .forEach((g) => g.fields.forEach((f) => delete State.params[f.key]));
  saveParameters();
  renderParameters();
});

/* ------------------------------------------------ result files (1.3) */
/**
 * List of the most recent result files on the Data tab (GET /api/results).
 * Each item shows the file name and its absolute path and downloads the file.
 * 数据页的结果文件列表。
 * @returns {Promise<void>}
 */
async function loadResults() {
  try {
    const data = await Api.results(15);
    els.resultsDir.textContent = data.directory;
    els.resultsList.innerHTML = data.files.map((f) => `
      <a class="history-item" href="/api/results/${encodeURIComponent(f.name)}" download="${f.name}"
         style="text-decoration:none">
        <div class="row1"><span style="word-break:break-all">${f.name}</span><span>${fmt(f.size_kb, 1)} kB</span></div>
        <div class="row2">${f.modified_utc}</div>
      </a>`).join('') || `<div class="muted">${t('results.none')}</div>`;
  } catch (err) { els.resultsList.textContent = t('unavailable'); }
}
$('btn-refresh-results').addEventListener('click', loadResults);

/* ------------------------------------------- interface language (1.3) */
/**
 * Language switch in the top bar (EN / RU / 中文). I18N.setLang translates the
 * marked static texts of index.html; this listener redraws everything that
 * app.js generated: hints, pills, parameter panels, the last route and its
 * summary, the comparison table, the model, database, history and result
 * lists, the legend and the layer names.
 * 顶栏语言切换：静态文字由 I18N.apply 翻译，动态生成的内容在此重新绘制。
 */
document.querySelectorAll('.lang-switch button').forEach((b) =>
  b.addEventListener('click', () => I18N.setLang(b.dataset.lang)));

I18N.onChange(() => {
  buildLayerControl();
  updatePickHint();
  redrawEndpoints();
  els.speed.dispatchEvent(new Event('input'));
  const ws = { open: ['ws.open', 'pill-ok'], closed: ['ws.closed', 'pill-wait'],
               error: ['ws.error', 'pill-bad'] }[State.wsState] || ['ws.connecting', 'pill-wait'];
  setStatus(els.wsStatus, t(ws[0]), ws[1]);
  if (!State.job) els.btnStop.textContent = t('btn.stop');
  showTimerLimit();
  renderParameters();
  renderModel();
  renderLegend();
  renderUploadReport();
  if (State.plan) renderPlan(State.plan, true);
  else els.compareBlock.textContent = t('cmp.first');
  if (State.bench) renderBenchmark(State.bench);
  else if (!State.job) els.benchmarkBlock.textContent = t('bench.run');
  if (!State.job && State.lastDone === 'plan' && State.lastFrame) showFinalProgress(State.lastFrame.plan);
  if (!State.job && State.lastDone === 'bench' && State.bench) {
    els.progressStage.textContent = t(State.bench.status === 'cancelled' ? 'bench.stopped' : 'bench.finished');
  }
  if (!els.cursorReadout.querySelector('.zone i')) {
    els.cursorReadout.querySelector('.zone').textContent = t('cursor.hint');
  }
  loadDbStats();
  loadHistory();
  loadResults();
});

/* ---------------------------------------------------------- bootstrap */
// Initial load: translate the page, draw markers for the default coordinates
// in the form and fetch the model panel, database statistics and route
// history in parallel.
// 页面启动：翻译页面，绘制默认起终点并加载模型、统计与历史数据。
I18N.apply();
setStatus(els.wsStatus, t('ws.connecting'), 'pill-wait');
if (!State.model) setStatus(els.modelPill, t('model.none'), 'pill-muted');
els.modelBlock.textContent = t('model.loading');
els.paramsBody.textContent = t('params.loading');
els.compareBlock.textContent = t('cmp.first');
els.benchmarkBlock.textContent = t('bench.run');
els.progressStage.textContent = t('progress.working');
els.cursorReadout.querySelector('.zone').textContent = t('cursor.hint');
buildLayerControl();
updatePickHint();
els.speed.dispatchEvent(new Event('input'));
readEndpointsFromInputs();
setBusy(null);
loadModel();
loadDbStats();
loadHistory();
loadParameters();
loadResults();
