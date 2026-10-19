/* ==========================================================================
 * i18n.js - interface languages: English, Russian, Chinese (version 1.3).
 * 界面多语言：英语、俄语、中文。
 *
 * Place in the application: loaded by index.html before app.js. It defines
 *   I18N.t(key, params)   the text of `key` in the current language, with
 *                         {name} placeholders replaced from `params`;
 *   I18N.apply(root)      translates every element of the page marked with
 *                         data-i18n (text), data-i18n-html (markup),
 *                         data-i18n-title (tooltip);
 *   I18N.server(text)     translates the known phrases that the server sends
 *                         (planning stages, outcomes, the lattices tried);
 *   I18N.zone(code)       name of a zone class (the code itself in English);
 *   I18N.param(field)     label, help and unit of an algorithm parameter;
 *   I18N.setLang(lang)    switches the language, remembers it in this browser
 *                         and calls the listeners registered with onChange.
 * English is the language of the code and of the server; a key missing in
 * Russian or Chinese falls back to English, and a server phrase that is not
 * recognised is shown as it is.
 * 英语是代码和服务端的语言；俄语或中文缺少某个键时显示英文，未识别的服务端文本原样显示。
 * ========================================================================== */
'use strict';

const I18N = (() => {
  const LANGS = ['en', 'ru', 'zh'];
  const STORE_KEY = 'maritime-route.lang';

  /* ------------------------------------------------------- dictionaries */
  const D = {
    en: {
      'app.title': 'Maritime Route Planner',
      'app.subtitle': 'Neural-network classification of geographic zones · A* · Dijkstra · genetic · dynamic programming',
      'ws.connecting': 'connecting…', 'ws.open': 'live connection', 'ws.closed': 'reconnecting…',
      'ws.error': 'connection error',
      'model.pill': 'model {acc}% acc', 'version.title': 'Version of the application', 'model.none': 'model —', 'model.unavailable': 'model unavailable',
      'tab.plan': 'Plan', 'tab.data': 'Data', 'tab.compare': 'Compare', 'tab.model': 'Model', 'tab.history': 'History',
      'own.html': '<b>Your own route:</b> click the chart — the 1st click sets the departure ' +
        '<span class="mk mk-a">A</span>, the 2nd the destination <span class="mk mk-b">B</span>; drag a marker ' +
        'to move it, or type the coordinates below. <b>Shift + click</b> (or “Add via point”) adds an ' +
        'intermediate point the route must pass through — useful for complex courses.',
      'pick.start': 'Next click sets the <b>departure</b> (A). Markers can also be dragged.',
      'pick.end': 'Next click sets the <b>destination</b> (B). Markers can also be dragged.',
      'pick.via': 'Every click adds a <b>via point</b>; press “Add via point” again to finish.',
      'btn.pickA': 'Set A on chart', 'btn.pickB': 'Set B on chart', 'btn.pickVia': 'Add via point',
      'btn.swap': 'Swap A ⇄ B',
      'title.pickA': 'The next click on the chart sets the departure',
      'title.pickB': 'The next click on the chart sets the destination',
      'title.pickVia': 'Every click adds an intermediate point until you press it again',
      'title.swap': 'Exchange departure and destination',
      'f.startLat': 'Departure lat', 'f.startLon': 'Departure lon',
      'f.endLat': 'Destination lat', 'f.endLon': 'Destination lon',
      'via.head': 'Via points (in order)', 'via.clear': 'Remove all', 'via.remove': 'Remove',
      'via.tooltip': 'Via point {n} (right click to remove)',
      'marker.start': 'Departure', 'marker.end': 'Destination',
      'preset.label': 'Presets', 'preset.complex': 'Complex',
      'city.Kiel': 'Kiel', 'city.Tallinn': 'Tallinn', 'city.Rotterdam': 'Rotterdam', 'city.Lisbon': 'Lisbon',
      'city.Helsinki': 'Helsinki', 'city.Copenhagen': 'Copenhagen', 'city.Odesa': 'Odesa',
      'city.Novorossiysk': 'Novorossiysk', 'city.Piraeus': 'Piraeus', 'city.Thessaloniki': 'Thessaloniki',
      'city.Gdansk': 'Gdańsk', 'city.Bergen': 'Bergen', 'city.Jeddah': 'Jeddah', 'city.Dubai': 'Dubai',
      'city.Rhodes': 'Rhodes', 'city.Singapore': 'Singapore', 'city.HongKong': 'Hong Kong',
      'alg.label': 'Search algorithm',
      'alg.astar': 'A* (heuristic, fastest)', 'alg.dijkstra': 'Dijkstra (uniform cost)',
      'alg.genetic': 'Genetic algorithm (evolutionary)', 'alg.dynamic': 'Dynamic programming (Bellman equation)',
      'alg.great_circle': 'Great circle (shortest distance)',
      'res.label': 'Grid resolution', 'speed.label': 'Service speed', 'unit.kn': 'kn',
      'params.title': 'Algorithm parameters', 'params.changed': '{n} changed', 'params.defaults': '(defaults)',
      'params.reset': 'Reset to defaults', 'params.loading': 'loading…',
      'params.noneGC': 'The great circle has no parameters of its own.',
      'params.default': 'default', 'params.on': 'on', 'params.off': 'off',
      'params.unavailable': 'Parameters unavailable: ',
      'opt.overlay': 'Show zone classification layer', 'opt.live': 'Live zone read-out while pointing at the chart',
      'btn.plan': 'Build route', 'btn.bench': 'Compare algorithms', 'btn.clear': 'Clear',
      'timer.title': 'Time since the search started', 'btn.stop': '■ Stop', 'btn.stopping': 'stopping…',
      'progress.working': 'working…', 'timer.limit': 'time limit {s} s', 'timer.perMethod': ' per method',
      'timer.noLimit': 'no time limit',
      'rf.head': 'Result saved to file (JSON)', 'rf.copy': 'Copy path', 'rf.copied': 'Copied',
      'rf.ctrlc': 'Press Ctrl+C', 'rf.download': 'Download JSON',
      'summary.title': 'Route summary', 'attempts.title': 'How the route was found', 'zone.title': 'Zone profile',
      'dl.pdf': 'PDF report',
      'data.title': 'Load AIS data',
      'data.text': 'Upload a GeoJSON, CSV or JSON file of route points. The points are classified by the network and stored in the database.',
      'drop.text': 'Drop a file here or click to browse', 'points.geojson': 'Points GeoJSON', 'points.csv': 'Points CSV',
      'results.title': 'Result files', 'results.folder': 'Folder:', 'btn.refresh': 'Refresh',
      'results.none': 'No result files yet.', 'db.title': 'Database', 'unavailable': 'unavailable',
      'cmp.title': 'Optimised vs shortest distance', 'cmp.first': 'Build a route first.',
      'bench.title': 'Algorithm benchmark',
      'bench.text': 'All methods on one shared cost map of the same course, with the parameters set on the Plan tab (the time limit applies to each method). The routes are drawn on the chart in the colours of the table.',
      'bench.show': 'Show the routes of all methods on the chart', 'bench.run': 'Run “Compare algorithms”.',
      'model.title': 'Classification model', 'model.loading': 'loading…', 'model.cm': 'Confusion matrix (test set)',
      'model.curves': 'Training curves', 'hist.title': 'Stored routes',
      'cursor.hint': 'move the pointer over the chart',
      // map
      'layer.osm': 'OpenStreetMap (online)', 'layer.onboard': 'On-board chart (offline)',
      'layer.onboardAttr': 'On-board zone classification (this model)', 'legend.item': '{z} ({c})',
      'legend.cost': 'cost {c}', 'legend.impassable': 'impassable', 'legend.gc': 'great circle',
      'legend.unavailable': 'legend unavailable',
      'zone.OPEN_SEA': 'open sea', 'zone.COASTAL_SEA': 'coastal sea', 'zone.NEAR_COAST': 'near coast',
      'zone.COASTLINE': 'coastline (land)',
      'read.elev': 'elev', 'read.depth': 'depth',
      'tip.gc': 'Great-circle (shortest distance)', 'tip.land': 'Over land on the independent coastline',
      // progress
      'started.bench': 'starting the comparison of all methods…', 'started.plan': 'starting {alg}…',
      'stopping': 'Stopping the search…',
      'stage.grid': 'Building geographic lattice', 'stage.classify': 'Neural network classifying grid cells',
      'stage.cost_map': 'Cost map ready', 'stage.search': 'Searching optimal path',
      'stage.astar': 'A* search in progress', 'stage.dijkstra': 'Dijkstra search in progress',
      'stage.genetic': 'Genetic algorithm: evolving routes',
      'stage.dynamic': 'Dynamic programming: solving the Bellman equation',
      'stage.refine': 'Adapting the lattice (narrow strait or wider area)',
      'stage.coast': 'Coastline check: correcting the route', 'stage.leg': 'Leg finished',
      'stage.benchmark': 'Comparing the methods on one cost map', 'stage.stopped': 'Search stopped',
      'stage.done': 'Route ready',
      'prog.leg': 'Leg {i}/{n} · ', 'prog.cells': '{a} / {b} cells classified',
      'prog.nodes': '{a} nodes expanded · frontier {b}', 'prog.sweep': 'sweep {a} · {b} values improved',
      'prog.gen': 'generation {a} · best cost {b}',
      'prog.final': '{steps} steps · {lat} lattice(s) tried · search time {s} s',
      'error': 'Error', 'bench.stopped': 'Comparison stopped', 'bench.finished': 'Comparison finished',
      // statuses
      'status.found': 'Route found', 'status.no_path': 'No navigable route between the points',
      'status.step_limit': 'Step limit reached before the goal', 'status.cancelled': 'Stopped by the user',
      'status.time_limit': 'Time limit reached', 'status.running': 'running',
      'tag.found': 'found', 'tag.no_path': 'no path', 'tag.step_limit': 'step limit', 'tag.cancelled': 'stopped',
      'tag.time_limit': 'time limit', 'tag.land': 'land', 'tag.running': 'running',
      // result
      'res.infeasible': 'Infeasible', 'res.noRoute': 'No navigable route',
      'm.searchTime': 'Search time', 'm.steps': 'Steps made', 'm.length': 'Route length', 'm.gc': 'Great circle',
      'm.detour': 'Detour ratio', 'm.cost': 'Weighted cost', 'm.passage': 'Passage time', 'm.planning': 'Planning time',
      'm.nodes': 'Nodes expanded', 'm.evaluated': 'Routes evaluated', 'm.updates': 'Bellman updates',
      'm.waypoints': 'Waypoints', 'u.km': 'km', 'u.h': 'h', 'u.s': 's',
      'coast.gc': 'Coastline check: the great circle crosses land for {km} km.',
      'coast.crossing': '⚠ The route crosses land for {km} km according to the independent 1 km coastline (GLOBE land mask), and the automatic correction could not remove it. Do not follow this course without checking the passage on an official nautical chart.',
      'coast.how.re': '{n} re-search(es), {c} cell(s) blocked', 'coast.how.fine': 'finer lattice {r}°',
      'coast.fixed': 'Coastline check: the first route crossed land for {km} km and was corrected automatically ({how}). The final route passes the check.',
      'coast.passed': 'Coastline check passed: {what} (independent 1 km land mask, {n} samples).',
      'coast.clear': 'no land on the route', 'coast.touch': 'touches the coast for at most {km} km',
      'hint.astar': 'Increase “Maximum number of steps” in the A* parameters (for example ×10) or raise the heuristic weight.',
      'hint.dijkstra': 'Increase “Maximum number of steps” in the Dijkstra parameters (for example from 1,000 to 100,000).',
      'hint.dynamic': 'Increase “Maximum number of sweeps” (0 = until convergence).',
      'hint.genetic': 'Increase “Generations” or “Maximum waypoints”, or add via points.',
      'hint.time_limit': 'Increase the time limit in the “Lattice and run” parameters (0 = no limit) or use a coarser lattice.',
      'hint.cancelled': 'The search was stopped with the Stop button; the partial result was saved.',
      'hint.no_path': 'Try a finer lattice, a larger margin, via points in the passage, or switch the land mask off.',
      'status.after': '<b>{st}</b> after {s} s. {hint}',
      'att.leg': 'leg', 'att.cell': 'cell °', 'att.cells': 'cells', 'att.steps': 'steps', 'att.s': 's',
      'att.outcome': 'outcome', 'att.land': ' land',
      'att.note': '⟂ = fine lattice classified only in a corridor around the coarse route (two-level search).',
      // compare tab
      'c.gcLen': 'Great-circle length', 'c.optLen': 'Optimised length', 'c.extra': 'Additional distance',
      'c.samples': 'Samples over land', 'c.share': 'Share over land',
      'c.interior': 'Optimised: intermediate waypoints over land',
      'c.coast': 'Optimised: coastline check (1 km land mask)', 'c.longest': ', longest land stretch {km} km',
      'c.wps': 'Optimised: waypoints in total', 'c.time': 'Passage time saved/lost',
      'c.gcNav': 'Great-circle route is navigable but ignores zone risk',
      'c.gcLand': 'Great-circle route crosses land and is not navigable',
      'c.legs': 'Great circles of the individual legs: {list}', 'c.nav': 'navigable', 'c.crossesLand': 'crosses land',
      'verdict.clear': 'clear', 'verdict.grazing': 'grazing', 'verdict.crossing': 'crossing', 'verdict.unchecked': 'unchecked',
      'alert.points': 'Please set both the departure and the destination.', 'alert.planFailed': 'Planning failed: ',
      'bench.running': 'running all methods on one shared cost map…', 'bench.failed': 'Benchmark failed: ',
      'meth.astar': 'A*', 'meth.dijkstra': 'Dijkstra', 'meth.dynamic': 'Dynamic prog.', 'meth.genetic': 'Genetic',
      'meth.great_circle': 'Great circle',
      'b.method': 'Method', 'b.status': 'status', 'b.km': 'km', 'b.cost': 'cost', 'b.opt': '+opt %', 'b.steps': 'steps',
      'b.limit': 'limit', 'b.s': 's', 'b.complexity': 'Complexity: theory and measurement', 'b.time': 'time',
      'b.ops': 'operations made', 'b.bound': 'worst-case bound', 'b.share': 'share',
      'b.note': 'V = {v} navigable cells, E ≤ 8V edges; operations = moves relaxed (8 per cell expanded or Bellman update) or cells of the routes evaluated (GA); bounds: (V + E) log₂ V, V · E (at most V sweeps), ((G + 1) · P + 1) · L.',
      'b.reduction': 'A* node reduction vs Dijkstra', 'b.speedup': 'A* speed-up (time)',
      'b.diffAD': 'Cost difference A* vs Dijkstra', 'b.gaGap': 'Genetic algorithm above the optimum',
      'b.diffDP': 'Cost difference DP vs A*', 'b.build': 'Cost map build', 'b.shared': 's (shared)', 'b.grid': 'Grid',
      'b.whole': 'Whole comparison',
      'b.runtimes': 'Runtimes exclude the shared cost-map construction, so they compare the search algorithms only.',
      // data / model / history
      'up.uploading': 'uploading {f}…', 'up.classifying': 'classifying points with the neural network…',
      'up.file': 'File', 'up.features': 'Features read', 'up.valid': 'Valid points', 'up.dropped': 'Dropped',
      'up.classified': 'Classified', 'up.agreement': 'Agreement with labels', 'up.done': '{f} — {n} points classified',
      'up.failed': 'Upload failed: ',
      'mo.arch': 'Architecture', 'mo.features': 'Input features', 'mo.params': 'Parameters',
      'mo.ref': 'Reference samples', 'mo.acc': 'Test accuracy', 'mo.f1': 'Macro F1', 'mo.kappa': 'Cohen κ',
      'mo.auc': 'Macro ROC-AUC', 'mo.trained': 'Trained', 'mo.perClass': 'Per class F1',
      'mo.unavailable': 'Model not available: ',
      'db.sessions': 'Sessions', 'db.points': 'Stored points', 'db.routes': 'Stored routes', 'db.waypoints': 'Waypoints',
      'db.size': 'Database size',
      'hist.wp': 'wp', 'hist.detour': 'detour', 'hist.none': 'No routes stored yet.',
      'chart.epoch': 'epoch', 'chart.loss': '— loss', 'chart.acc': '— accuracy', 'chart.truePred': 'true \\ pred',
      // parameter groups
      'g.common': 'Lattice and run', 'g.astar': 'A* search', 'g.dijkstra': 'Dijkstra search',
      'g.dynamic': 'Dynamic programming', 'g.genetic': 'Genetic algorithm',
      'lang.title': 'Interface language', 'lang.label': 'Language',
    },

    ru: {
      'app.title': 'Планировщик морских маршрутов',
      'app.subtitle': 'Нейросетевая классификация географических зон · A* · Дейкстра · генетический алгоритм · динамическое программирование',
      'ws.connecting': 'подключение…', 'ws.open': 'соединение активно', 'ws.closed': 'переподключение…',
      'ws.error': 'ошибка соединения',
      'model.pill': 'модель: точность {acc}%', 'version.title': 'Версия приложения', 'model.none': 'модель —', 'model.unavailable': 'модель недоступна',
      'tab.plan': 'Маршрут', 'tab.data': 'Данные', 'tab.compare': 'Сравнение', 'tab.model': 'Модель',
      'tab.history': 'История',
      'own.html': '<b>Собственный маршрут:</b> щёлкните по карте — первый щелчок задаёт пункт отправления ' +
        '<span class="mk mk-a">A</span>, второй — пункт назначения <span class="mk mk-b">B</span>; маркер можно ' +
        'перетащить или ввести координаты ниже. <b>Shift + щелчок</b> (или «Добавить точку») добавляет ' +
        'промежуточную точку, через которую должен пройти маршрут, — удобно для сложных курсов.',
      'pick.start': 'Следующий щелчок задаёт <b>пункт отправления</b> (A). Маркеры можно перетаскивать.',
      'pick.end': 'Следующий щелчок задаёт <b>пункт назначения</b> (B). Маркеры можно перетаскивать.',
      'pick.via': 'Каждый щелчок добавляет <b>промежуточную точку</b>; чтобы закончить, нажмите «Добавить точку» ещё раз.',
      'btn.pickA': 'Задать A на карте', 'btn.pickB': 'Задать B на карте', 'btn.pickVia': 'Добавить точку',
      'btn.swap': 'Поменять A ⇄ B',
      'title.pickA': 'Следующий щелчок по карте задаёт пункт отправления',
      'title.pickB': 'Следующий щелчок по карте задаёт пункт назначения',
      'title.pickVia': 'Каждый щелчок добавляет промежуточную точку, пока кнопка не нажата снова',
      'title.swap': 'Поменять местами пункты отправления и назначения',
      'f.startLat': 'Широта отправления', 'f.startLon': 'Долгота отправления',
      'f.endLat': 'Широта назначения', 'f.endLon': 'Долгота назначения',
      'via.head': 'Промежуточные точки (по порядку)', 'via.clear': 'Удалить все', 'via.remove': 'Удалить',
      'via.tooltip': 'Промежуточная точка {n} (правый щелчок — удалить)',
      'marker.start': 'Отправление', 'marker.end': 'Назначение',
      'preset.label': 'Примеры', 'preset.complex': 'Сложные',
      'city.Kiel': 'Киль', 'city.Tallinn': 'Таллин', 'city.Rotterdam': 'Роттердам', 'city.Lisbon': 'Лиссабон',
      'city.Helsinki': 'Хельсинки', 'city.Copenhagen': 'Копенгаген', 'city.Odesa': 'Одесса',
      'city.Novorossiysk': 'Новороссийск', 'city.Piraeus': 'Пирей', 'city.Thessaloniki': 'Салоники',
      'city.Gdansk': 'Гданьск', 'city.Bergen': 'Берген', 'city.Jeddah': 'Джидда', 'city.Dubai': 'Дубай',
      'city.Rhodes': 'Родос', 'city.Singapore': 'Сингапур', 'city.HongKong': 'Гонконг',
      'alg.label': 'Алгоритм поиска',
      'alg.astar': 'A* (эвристический, самый быстрый)', 'alg.dijkstra': 'Дейкстра (равномерная стоимость)',
      'alg.genetic': 'Генетический алгоритм (эволюционный)',
      'alg.dynamic': 'Динамическое программирование (уравнение Беллмана)',
      'alg.great_circle': 'Ортодромия (кратчайшее расстояние)',
      'res.label': 'Шаг сетки', 'speed.label': 'Скорость судна', 'unit.kn': 'уз',
      'params.title': 'Параметры алгоритма', 'params.changed': 'изменено: {n}', 'params.defaults': '(по умолчанию)',
      'params.reset': 'Сбросить к значениям по умолчанию', 'params.loading': 'загрузка…',
      'params.noneGC': 'У ортодромии нет собственных параметров.',
      'params.default': 'по умолчанию', 'params.on': 'вкл.', 'params.off': 'выкл.',
      'params.unavailable': 'Параметры недоступны: ',
      'opt.overlay': 'Показывать слой классификации зон', 'opt.live': 'Показывать зону под указателем мыши',
      'btn.plan': 'Построить маршрут', 'btn.bench': 'Сравнить алгоритмы', 'btn.clear': 'Очистить',
      'timer.title': 'Время с начала поиска', 'btn.stop': '■ Стоп', 'btn.stopping': 'остановка…',
      'progress.working': 'выполняется…', 'timer.limit': 'лимит времени {s} с', 'timer.perMethod': ' на каждый метод',
      'timer.noLimit': 'без ограничения времени',
      'rf.head': 'Результат сохранён в файл (JSON)', 'rf.copy': 'Копировать путь', 'rf.copied': 'Скопировано',
      'rf.ctrlc': 'Нажмите Ctrl+C', 'rf.download': 'Скачать JSON',
      'summary.title': 'Сводка маршрута', 'attempts.title': 'Как был найден маршрут', 'zone.title': 'Профиль зон',
      'dl.pdf': 'Отчёт PDF',
      'data.title': 'Загрузка данных AIS',
      'data.text': 'Загрузите файл GeoJSON, CSV или JSON с точками маршрута. Точки классифицируются нейросетью и сохраняются в базе данных.',
      'drop.text': 'Перетащите файл сюда или щёлкните, чтобы выбрать', 'points.geojson': 'Точки GeoJSON',
      'points.csv': 'Точки CSV',
      'results.title': 'Файлы результатов', 'results.folder': 'Папка:', 'btn.refresh': 'Обновить',
      'results.none': 'Файлов результатов пока нет.', 'db.title': 'База данных', 'unavailable': 'недоступно',
      'cmp.title': 'Оптимальный маршрут и кратчайшее расстояние', 'cmp.first': 'Сначала постройте маршрут.',
      'bench.title': 'Сравнение алгоритмов',
      'bench.text': 'Все методы на одной общей карте стоимости одного курса, с параметрами со вкладки «Маршрут» (лимит времени действует для каждого метода). Маршруты рисуются на карте цветами таблицы.',
      'bench.show': 'Показывать маршруты всех методов на карте', 'bench.run': 'Нажмите «Сравнить алгоритмы».',
      'model.title': 'Модель классификации', 'model.loading': 'загрузка…',
      'model.cm': 'Матрица ошибок (тестовая выборка)', 'model.curves': 'Кривые обучения',
      'hist.title': 'Сохранённые маршруты', 'cursor.hint': 'наведите указатель на карту',
      'layer.osm': 'OpenStreetMap (онлайн)', 'layer.onboard': 'Бортовая карта (офлайн)',
      'layer.onboardAttr': 'Бортовая классификация зон (эта модель)', 'legend.item': '{z} ({c})',
      'legend.cost': 'стоимость {c}', 'legend.impassable': 'непроходимо', 'legend.gc': 'ортодромия',
      'legend.unavailable': 'легенда недоступна',
      'zone.OPEN_SEA': 'открытое море', 'zone.COASTAL_SEA': 'прибрежное море', 'zone.NEAR_COAST': 'у берега (≤ 5 км)',
      'zone.COASTLINE': 'суша',
      'read.elev': 'высота', 'read.depth': 'глубина',
      'tip.gc': 'Ортодромия (кратчайшее расстояние)', 'tip.land': 'Суша по независимой береговой линии',
      'started.bench': 'запуск сравнения всех методов…', 'started.plan': 'запуск {alg}…',
      'stopping': 'Поиск останавливается…',
      'stage.grid': 'Построение географической сетки', 'stage.classify': 'Нейросеть классифицирует ячейки сетки',
      'stage.cost_map': 'Карта стоимости готова', 'stage.search': 'Поиск оптимального пути',
      'stage.astar': 'Выполняется поиск A*', 'stage.dijkstra': 'Выполняется поиск Дейкстры',
      'stage.genetic': 'Генетический алгоритм: эволюция маршрутов',
      'stage.dynamic': 'Динамическое программирование: решение уравнения Беллмана',
      'stage.refine': 'Адаптация сетки (узкий пролив или расширение области)',
      'stage.coast': 'Проверка по береговой линии: исправление маршрута', 'stage.leg': 'Участок завершён',
      'stage.benchmark': 'Сравнение методов на одной карте стоимости', 'stage.stopped': 'Поиск остановлен',
      'stage.done': 'Маршрут готов',
      'prog.leg': 'Участок {i}/{n} · ', 'prog.cells': 'классифицировано ячеек: {a} / {b}',
      'prog.nodes': 'раскрыто вершин: {a} · фронт {b}', 'prog.sweep': 'итерация {a} · улучшено значений: {b}',
      'prog.gen': 'поколение {a} · лучшая стоимость {b}',
      'prog.final': 'шагов: {steps} · сеток: {lat} · время поиска {s} с',
      'error': 'Ошибка', 'bench.stopped': 'Сравнение остановлено', 'bench.finished': 'Сравнение завершено',
      'status.found': 'Маршрут найден', 'status.no_path': 'Между точками нет судоходного маршрута',
      'status.step_limit': 'Достигнут лимит шагов до цели', 'status.cancelled': 'Остановлено пользователем',
      'status.time_limit': 'Достигнут лимит времени', 'status.running': 'выполняется',
      'tag.found': 'найден', 'tag.no_path': 'нет пути', 'tag.step_limit': 'лимит шагов', 'tag.cancelled': 'остановлен',
      'tag.time_limit': 'лимит времени', 'tag.land': 'суша', 'tag.running': 'выполняется',
      'res.infeasible': 'Маршрут не построен', 'res.noRoute': 'Нет судоходного маршрута',
      'm.searchTime': 'Время поиска', 'm.steps': 'Сделано шагов', 'm.length': 'Длина маршрута', 'm.gc': 'Ортодромия',
      'm.detour': 'Коэффициент удлинения', 'm.cost': 'Взвешенная стоимость', 'm.passage': 'Время перехода',
      'm.planning': 'Время расчёта', 'm.nodes': 'Раскрыто вершин', 'm.evaluated': 'Оценено маршрутов',
      'm.updates': 'Обновлений Беллмана', 'm.waypoints': 'Путевых точек', 'u.km': 'км', 'u.h': 'ч', 'u.s': 'с',
      'coast.gc': 'Проверка по береговой линии: ортодромия пересекает сушу на {km} км.',
      'coast.crossing': '⚠ По независимой береговой линии (маска суши GLOBE, 1 км) маршрут пересекает сушу на {km} км, и автоматическое исправление не смогло это устранить. Не следуйте этим курсом, не проверив проход по официальной навигационной карте.',
      'coast.how.re': 'повторных поисков: {n}, заблокировано ячеек: {c}', 'coast.how.fine': 'более мелкая сетка {r}°',
      'coast.fixed': 'Проверка по береговой линии: первый маршрут пересекал сушу на {km} км и был исправлен автоматически ({how}). Итоговый маршрут проходит проверку.',
      'coast.passed': 'Проверка по береговой линии пройдена: {what} (независимая маска суши 1 км, отсчётов: {n}).',
      'coast.clear': 'суши на маршруте нет', 'coast.touch': 'маршрут касается берега не более чем на {km} км',
      'hint.astar': 'Увеличьте «Максимальное число шагов» в параметрах A* (например, в 10 раз) или вес эвристики.',
      'hint.dijkstra': 'Увеличьте «Максимальное число шагов» в параметрах алгоритма Дейкстры (например, с 1 000 до 100 000).',
      'hint.dynamic': 'Увеличьте «Максимальное число итераций» (0 — до сходимости).',
      'hint.genetic': 'Увеличьте число поколений или максимальное число точек либо добавьте промежуточные точки.',
      'hint.time_limit': 'Увеличьте лимит времени в группе «Сетка и запуск» (0 — без ограничения) или используйте более крупную сетку.',
      'hint.cancelled': 'Поиск остановлен кнопкой «Стоп»; результат сохранён.',
      'hint.no_path': 'Попробуйте более мелкую сетку, больший запас, промежуточные точки в проливе или отключите маску суши.',
      'status.after': '<b>{st}</b> через {s} с. {hint}',
      'att.leg': 'участок', 'att.cell': 'ячейка °', 'att.cells': 'ячеек', 'att.steps': 'шагов', 'att.s': 'с',
      'att.outcome': 'итог', 'att.land': ' суша',
      'att.note': '⟂ — мелкая сетка, классифицированная только в коридоре вокруг грубого маршрута (двухуровневый поиск).',
      'c.gcLen': 'Длина ортодромии', 'c.optLen': 'Длина оптимального маршрута', 'c.extra': 'Дополнительное расстояние',
      'c.samples': 'Отсчётов на суше', 'c.share': 'Доля над сушей',
      'c.interior': 'Оптимальный: промежуточных точек на суше',
      'c.coast': 'Оптимальный: проверка по береговой линии (маска 1 км)', 'c.longest': ', наибольший участок суши {km} км',
      'c.wps': 'Оптимальный: всего путевых точек', 'c.time': 'Выигрыш/потеря времени перехода',
      'c.gcNav': 'Ортодромия судоходна, но не учитывает риск зон',
      'c.gcLand': 'Ортодромия пересекает сушу и не судоходна',
      'c.legs': 'Ортодромии отдельных участков: {list}', 'c.nav': 'судоходна', 'c.crossesLand': 'пересекает сушу',
      'verdict.clear': 'чисто', 'verdict.grazing': 'касается берега', 'verdict.crossing': 'пересекает сушу',
      'verdict.unchecked': 'не проверялось',
      'alert.points': 'Задайте пункты отправления и назначения.', 'alert.planFailed': 'Ошибка расчёта: ',
      'bench.running': 'все методы выполняются на одной общей карте стоимости…', 'bench.failed': 'Ошибка сравнения: ',
      'meth.astar': 'A*', 'meth.dijkstra': 'Дейкстра', 'meth.dynamic': 'ДП', 'meth.genetic': 'ГА',
      'meth.great_circle': 'Ортодромия',
      'b.method': 'Метод', 'b.status': 'итог', 'b.km': 'км', 'b.cost': 'стоим.', 'b.opt': '+опт. %', 'b.steps': 'шагов',
      'b.limit': 'лимит', 'b.s': 'с', 'b.complexity': 'Сложность: теория и измерение', 'b.time': 'время',
      'b.ops': 'операций', 'b.bound': 'верхняя оценка', 'b.share': 'доля',
      'b.note': 'V = {v} судоходных ячеек, E ≤ 8V рёбер; операции — релаксации переходов (8 на раскрытую ячейку или обновление Беллмана) либо ячейки оцениваемых маршрутов (ГА); оценки: (V + E) log₂ V, V · E (не более V итераций), ((G + 1) · P + 1) · L.',
      'b.reduction': 'Сокращение числа вершин A* по сравнению с Дейкстрой', 'b.speedup': 'Ускорение A* (по времени)',
      'b.diffAD': 'Разница стоимостей A* и Дейкстры', 'b.gaGap': 'Генетический алгоритм выше оптимума',
      'b.diffDP': 'Разница стоимостей ДП и A*', 'b.build': 'Построение карты стоимости', 'b.shared': 'с (общая)',
      'b.grid': 'Сетка', 'b.whole': 'Всё сравнение',
      'b.runtimes': 'Время не включает построение общей карты стоимости, поэтому сравниваются только алгоритмы поиска.',
      'up.uploading': 'загрузка {f}…', 'up.classifying': 'нейросеть классифицирует точки…',
      'up.file': 'Файл', 'up.features': 'Прочитано объектов', 'up.valid': 'Корректных точек', 'up.dropped': 'Отброшено',
      'up.classified': 'Классифицировано', 'up.agreement': 'Совпадение с метками',
      'up.done': '{f} — классифицировано точек: {n}', 'up.failed': 'Ошибка загрузки: ',
      'mo.arch': 'Архитектура', 'mo.features': 'Входных признаков', 'mo.params': 'Параметров',
      'mo.ref': 'Опорных образцов', 'mo.acc': 'Точность на тесте', 'mo.f1': 'Макро-F1', 'mo.kappa': 'Каппа Коэна',
      'mo.auc': 'Макро ROC-AUC', 'mo.trained': 'Обучена', 'mo.perClass': 'F1 по классам',
      'mo.unavailable': 'Модель недоступна: ',
      'db.sessions': 'Сеансов', 'db.points': 'Сохранено точек', 'db.routes': 'Сохранено маршрутов',
      'db.waypoints': 'Путевых точек', 'db.size': 'Размер базы данных',
      'hist.wp': 'точек', 'hist.detour': 'удлинение', 'hist.none': 'Сохранённых маршрутов пока нет.',
      'chart.epoch': 'эпоха', 'chart.loss': '— потери', 'chart.acc': '— точность', 'chart.truePred': 'истина \\ прогноз',
      'g.common': 'Сетка и запуск', 'g.astar': 'Поиск A*', 'g.dijkstra': 'Поиск Дейкстры',
      'g.dynamic': 'Динамическое программирование', 'g.genetic': 'Генетический алгоритм',
      'lang.title': 'Язык интерфейса', 'lang.label': 'Язык',
    },

    zh: {
      'app.title': '海上航线规划系统',
      'app.subtitle': '基于神经网络的地理区域分类 · A* · Dijkstra · 遗传算法 · 动态规划',
      'ws.connecting': '正在连接…', 'ws.open': '实时连接', 'ws.closed': '正在重连…', 'ws.error': '连接错误',
      'model.pill': '模型准确率 {acc}%', 'version.title': '程序版本', 'model.none': '模型 —', 'model.unavailable': '模型不可用',
      'tab.plan': '规划', 'tab.data': '数据', 'tab.compare': '对比', 'tab.model': '模型', 'tab.history': '历史',
      'own.html': '<b>自定义航线：</b>在地图上点击——第一次点击设置出发点 <span class="mk mk-a">A</span>，' +
        '第二次设置目的地 <span class="mk mk-b">B</span>；可以拖动标记，或在下方输入坐标。' +
        '<b>Shift + 点击</b>（或“添加途经点”）添加航线必须经过的中间点，适合复杂航线。',
      'pick.start': '下一次点击设置<b>出发点</b>（A）。标记也可以拖动。',
      'pick.end': '下一次点击设置<b>目的地</b>（B）。标记也可以拖动。',
      'pick.via': '每次点击添加一个<b>途经点</b>；再次点击“添加途经点”结束。',
      'btn.pickA': '在地图上设置 A', 'btn.pickB': '在地图上设置 B', 'btn.pickVia': '添加途经点',
      'btn.swap': '交换 A ⇄ B',
      'title.pickA': '下一次点击地图设置出发点', 'title.pickB': '下一次点击地图设置目的地',
      'title.pickVia': '再次按下之前，每次点击都会添加一个途经点', 'title.swap': '交换出发点和目的地',
      'f.startLat': '出发点纬度', 'f.startLon': '出发点经度', 'f.endLat': '目的地纬度', 'f.endLon': '目的地经度',
      'via.head': '途经点（按顺序）', 'via.clear': '全部删除', 'via.remove': '删除',
      'via.tooltip': '途经点 {n}（右键删除）',
      'marker.start': '出发点', 'marker.end': '目的地',
      'preset.label': '预设', 'preset.complex': '复杂航线',
      'city.Kiel': '基尔', 'city.Tallinn': '塔林', 'city.Rotterdam': '鹿特丹', 'city.Lisbon': '里斯本',
      'city.Helsinki': '赫尔辛基', 'city.Copenhagen': '哥本哈根', 'city.Odesa': '敖德萨',
      'city.Novorossiysk': '新罗西斯克', 'city.Piraeus': '比雷埃夫斯', 'city.Thessaloniki': '塞萨洛尼基',
      'city.Gdansk': '格但斯克', 'city.Bergen': '卑尔根', 'city.Jeddah': '吉达', 'city.Dubai': '迪拜',
      'city.Rhodes': '罗得岛', 'city.Singapore': '新加坡', 'city.HongKong': '香港',
      'alg.label': '搜索算法',
      'alg.astar': 'A*（启发式，最快）', 'alg.dijkstra': 'Dijkstra（一致代价）', 'alg.genetic': '遗传算法（进化搜索）',
      'alg.dynamic': '动态规划（贝尔曼方程）', 'alg.great_circle': '大圆航线（最短距离）',
      'res.label': '网格分辨率', 'speed.label': '航速', 'unit.kn': '节',
      'params.title': '算法参数', 'params.changed': '已修改 {n} 项', 'params.defaults': '（默认值）',
      'params.reset': '恢复默认值', 'params.loading': '加载中…', 'params.noneGC': '大圆航线没有自己的参数。',
      'params.default': '默认', 'params.on': '开', 'params.off': '关', 'params.unavailable': '参数不可用：',
      'opt.overlay': '显示区域分类图层', 'opt.live': '鼠标指向地图时实时显示区域',
      'btn.plan': '规划航线', 'btn.bench': '比较算法', 'btn.clear': '清除',
      'timer.title': '搜索开始后经过的时间', 'btn.stop': '■ 停止', 'btn.stopping': '正在停止…',
      'progress.working': '处理中…', 'timer.limit': '时间限制 {s} 秒', 'timer.perMethod': '（每种方法）',
      'timer.noLimit': '无时间限制',
      'rf.head': '结果已保存到文件（JSON）', 'rf.copy': '复制路径', 'rf.copied': '已复制', 'rf.ctrlc': '请按 Ctrl+C',
      'rf.download': '下载 JSON',
      'summary.title': '航线摘要', 'attempts.title': '航线是如何找到的', 'zone.title': '区域分布', 'dl.pdf': 'PDF 报告',
      'data.title': '加载 AIS 数据',
      'data.text': '上传包含航线点的 GeoJSON、CSV 或 JSON 文件。这些点会由神经网络分类并保存到数据库。',
      'drop.text': '将文件拖到这里，或点击选择文件', 'points.geojson': '点 GeoJSON', 'points.csv': '点 CSV',
      'results.title': '结果文件', 'results.folder': '文件夹：', 'btn.refresh': '刷新', 'results.none': '还没有结果文件。',
      'db.title': '数据库', 'unavailable': '不可用',
      'cmp.title': '优化航线与最短距离对比', 'cmp.first': '请先规划一条航线。',
      'bench.title': '算法对比',
      'bench.text': '在同一航线的同一张代价地图上运行所有方法，使用“规划”页设置的参数（时间限制对每种方法分别生效）。各方法的航线按表格中的颜色画在地图上。',
      'bench.show': '在地图上显示所有方法的航线', 'bench.run': '请点击“比较算法”。',
      'model.title': '分类模型', 'model.loading': '加载中…', 'model.cm': '混淆矩阵（测试集）', 'model.curves': '训练曲线',
      'hist.title': '已保存的航线', 'cursor.hint': '将鼠标移到地图上',
      'layer.osm': 'OpenStreetMap（在线）', 'layer.onboard': '船载海图（离线）',
      'layer.onboardAttr': '船载区域分类（本模型）', 'legend.item': '{z}（{c}）',
      'legend.cost': '代价 {c}', 'legend.impassable': '不可通行', 'legend.gc': '大圆航线', 'legend.unavailable': '图例不可用',
      'zone.OPEN_SEA': '开阔海域', 'zone.COASTAL_SEA': '沿海海域', 'zone.NEAR_COAST': '近岸带 ≤5 公里',
      'zone.COASTLINE': '陆地',
      'read.elev': '海拔', 'read.depth': '水深',
      'tip.gc': '大圆航线（最短距离）', 'tip.land': '按独立海岸线位于陆地上',
      'started.bench': '开始比较所有方法…', 'started.plan': '开始 {alg}…', 'stopping': '正在停止搜索…',
      'stage.grid': '正在构建地理网格', 'stage.classify': '神经网络正在对网格单元分类', 'stage.cost_map': '代价地图已就绪',
      'stage.search': '正在搜索最优路径', 'stage.astar': 'A* 搜索中', 'stage.dijkstra': 'Dijkstra 搜索中',
      'stage.genetic': '遗传算法：航线进化中', 'stage.dynamic': '动态规划：求解贝尔曼方程',
      'stage.refine': '调整网格（狭窄海峡或扩大区域）', 'stage.coast': '海岸线检查：修正航线',
      'stage.leg': '本段完成', 'stage.benchmark': '在同一代价地图上比较各方法', 'stage.stopped': '搜索已停止',
      'stage.done': '航线已完成',
      'prog.leg': '第 {i}/{n} 段 · ', 'prog.cells': '已分类单元 {a} / {b}', 'prog.nodes': '已扩展节点 {a} · 前沿 {b}',
      'prog.sweep': '第 {a} 次迭代 · 改进值 {b} 个', 'prog.gen': '第 {a} 代 · 最优代价 {b}',
      'prog.final': '{steps} 步 · 尝试 {lat} 个网格 · 搜索时间 {s} 秒',
      'error': '错误', 'bench.stopped': '对比已停止', 'bench.finished': '对比已完成',
      'status.found': '已找到航线', 'status.no_path': '两点之间没有可航行的航线', 'status.step_limit': '到达目标前达到步数上限',
      'status.cancelled': '已被用户停止', 'status.time_limit': '达到时间限制', 'status.running': '运行中',
      'tag.found': '找到', 'tag.no_path': '无路径', 'tag.step_limit': '步数上限', 'tag.cancelled': '已停止',
      'tag.time_limit': '超时', 'tag.land': '穿陆', 'tag.running': '运行中',
      'res.infeasible': '未能规划', 'res.noRoute': '没有可航行的航线',
      'm.searchTime': '搜索时间', 'm.steps': '已执行步数', 'm.length': '航线长度', 'm.gc': '大圆距离', 'm.detour': '绕航系数',
      'm.cost': '加权代价', 'm.passage': '航行时间', 'm.planning': '规划耗时', 'm.nodes': '扩展节点数',
      'm.evaluated': '评估的航线数', 'm.updates': '贝尔曼更新次数', 'm.waypoints': '航路点数', 'u.km': '公里',
      'u.h': '小时', 'u.s': '秒',
      'coast.gc': '海岸线检查：大圆航线穿越陆地 {km} 公里。',
      'coast.crossing': '⚠ 根据独立的 1 公里海岸线（GLOBE 陆地掩膜），航线穿越陆地 {km} 公里，自动修正未能消除。未在官方海图上核实通道之前，请勿按此航线航行。',
      'coast.how.re': '重新搜索 {n} 次，封锁 {c} 个单元', 'coast.how.fine': '更细的网格 {r}°',
      'coast.fixed': '海岸线检查：第一条航线穿越陆地 {km} 公里，已自动修正（{how}）。最终航线通过检查。',
      'coast.passed': '海岸线检查通过：{what}（独立的 1 公里陆地掩膜，{n} 个采样点）。',
      'coast.clear': '航线上没有陆地', 'coast.touch': '最多擦岸 {km} 公里',
      'hint.astar': '在 A* 参数中增大“最大步数”（例如 10 倍），或增大启发权重。',
      'hint.dijkstra': '在 Dijkstra 参数中增大“最大步数”（例如从 1,000 增加到 100,000）。',
      'hint.dynamic': '增大“最大迭代次数”（0 表示直到收敛）。',
      'hint.genetic': '增大“代数”或“最大航点数”，或添加途经点。',
      'hint.time_limit': '在“网格与运行”参数中增大时间限制（0 表示不限），或使用更粗的网格。',
      'hint.cancelled': '搜索已通过“停止”按钮中止，结果已保存。',
      'hint.no_path': '请尝试更细的网格、更大的边距、在海峡处设置途经点，或关闭陆地掩膜。',
      'status.after': '<b>{st}</b>，用时 {s} 秒。{hint}',
      'att.leg': '段', 'att.cell': '单元 °', 'att.cells': '单元数', 'att.steps': '步数', 'att.s': '秒', 'att.outcome': '结果',
      'att.land': ' 穿陆', 'att.note': '⟂ = 只在粗网格航线周围的走廊内分类的细网格（两级搜索）。',
      'c.gcLen': '大圆航线长度', 'c.optLen': '优化航线长度', 'c.extra': '额外距离', 'c.samples': '陆地上的采样点',
      'c.share': '陆地比例', 'c.interior': '优化航线：位于陆地的中间航路点', 'c.coast': '优化航线：海岸线检查（1 公里掩膜）',
      'c.longest': '，最长穿陆段 {km} 公里', 'c.wps': '优化航线：航路点总数', 'c.time': '航行时间增减',
      'c.gcNav': '大圆航线可航行，但没有考虑区域风险', 'c.gcLand': '大圆航线穿越陆地，不可航行',
      'c.legs': '各航段的大圆航线：{list}', 'c.nav': '可航行', 'c.crossesLand': '穿越陆地',
      'verdict.clear': '无陆地', 'verdict.grazing': '擦岸', 'verdict.crossing': '穿陆', 'verdict.unchecked': '未检查',
      'alert.points': '请设置出发点和目的地。', 'alert.planFailed': '规划失败：',
      'bench.running': '正在同一张代价地图上运行所有方法…', 'bench.failed': '对比失败：',
      'meth.astar': 'A*', 'meth.dijkstra': 'Dijkstra', 'meth.dynamic': '动态规划', 'meth.genetic': '遗传算法',
      'meth.great_circle': '大圆航线',
      'b.method': '方法', 'b.status': '结果', 'b.km': '公里', 'b.cost': '代价', 'b.opt': '差距 %', 'b.steps': '步数',
      'b.limit': '上限', 'b.s': '秒', 'b.complexity': '复杂度：理论与实测', 'b.time': '时间复杂度',
      'b.ops': '实际运算量', 'b.bound': '最坏情况上界', 'b.share': '占比',
      'b.note': 'V = {v} 个可航行单元，E ≤ 8V 条边；运算量 = 松弛的移动次数（每次扩展或贝尔曼更新 8 次）或所评估航线的单元数（遗传算法）；上界：(V + E) log₂ V、V · E（最多 V 次迭代）、((G + 1) · P + 1) · L。',
      'b.reduction': 'A* 相比 Dijkstra 的节点减少倍数', 'b.speedup': 'A* 加速比（时间）',
      'b.diffAD': 'A* 与 Dijkstra 的代价差', 'b.gaGap': '遗传算法高出最优值', 'b.diffDP': '动态规划与 A* 的代价差',
      'b.build': '代价地图构建', 'b.shared': '秒（共享）', 'b.grid': '网格', 'b.whole': '整个对比',
      'b.runtimes': '耗时不包括共享代价地图的构建，因此只比较搜索算法本身。',
      'up.uploading': '正在上传 {f}…', 'up.classifying': '神经网络正在对点进行分类…',
      'up.file': '文件', 'up.features': '读取的要素', 'up.valid': '有效点', 'up.dropped': '丢弃', 'up.classified': '已分类',
      'up.agreement': '与标签一致率', 'up.done': '{f} — 已分类 {n} 个点', 'up.failed': '上传失败：',
      'mo.arch': '网络结构', 'mo.features': '输入特征', 'mo.params': '参数数量', 'mo.ref': '参考样本',
      'mo.acc': '测试准确率', 'mo.f1': '宏平均 F1', 'mo.kappa': 'Cohen κ', 'mo.auc': '宏平均 ROC-AUC',
      'mo.trained': '训练时间', 'mo.perClass': '各类别 F1', 'mo.unavailable': '模型不可用：',
      'db.sessions': '会话', 'db.points': '保存的点', 'db.routes': '保存的航线', 'db.waypoints': '航路点',
      'db.size': '数据库大小',
      'hist.wp': '个航点', 'hist.detour': '绕航', 'hist.none': '还没有保存的航线。',
      'chart.epoch': '轮次', 'chart.loss': '— 损失', 'chart.acc': '— 准确率', 'chart.truePred': '真实 \\ 预测',
      'g.common': '网格与运行', 'g.astar': 'A* 搜索', 'g.dijkstra': 'Dijkstra 搜索', 'g.dynamic': '动态规划',
      'g.genetic': '遗传算法',
      'lang.title': '界面语言', 'lang.label': '语言',
    },
  };

  /* --------------------------------------- parameters: label and help */
  // English labels and help come from the server (GET /api/parameters).
  const P = {
    ru: {
      max_grid_cells: ['Бюджет ячеек сетки', 'Наибольшая сетка, которая классифицируется. Длинному рейсу на мелкой сетке нужно больше ячеек; если их не хватает, шаг сетки увеличивается в 1,5 раза. Время классификации растёт линейно с числом ячеек.'],
      margin_deg: ['Запас вокруг точек', 'Дополнительное пространство вокруг прямоугольника точек, чтобы маршрут мог обогнуть мыс или пройти через пролив вне прямого прямоугольника.'],
      diagonal_moves: ['Диагональные переходы (8 соседей)', 'Вкл.: ячейка связана с 8 соседями; выкл.: только с 4 (С, Ю, З, В).'],
      safety_margin_cells: ['Прибрежная полоса', 'Ширина полосы воды у берега, стоимость которой умножается на прибрежный штраф.'],
      proximity_penalty: ['Прибрежный штраф', 'Множитель стоимости ячейки воды в прибрежной полосе; 1 — без штрафа.'],
      auto_refine: ['Автоматическое измельчение сетки', 'Если на сетке нет маршрута, повторить поиск на более мелкой и широкой сетке.'],
      max_refinements: ['Максимум измельчений', 'Число повторов на более мелкой и широкой сетке (шаг × 0,55, запас + 2° каждый раз).'],
      coast_guard: ['Проверка по береговой линии', 'Проверить маршрут по независимой маске суши 1 км и исправить его, если он пересекает сушу.'],
      use_land_mask: ['Маска суши при исправлении', 'Если маршрут пересекает сушу, сделать непроходимыми ячейки, которые маска суши 1 км показывает как преимущественно сушу (без повторной классификации), и повторить поиск; мелкая сетка коридора тоже использует маску.'],
      land_mask_always: ['Маска суши с первой сетки', 'Использовать маску суши на каждой сетке с самого начала, а не только для исправления. Надёжнее у незнакомых берегов, но узкие проливы закрываются раньше и сетка чаще измельчается.'],
      land_mask_share: ['Доля суши, блокирующая ячейку', 'Ячейка блокируется, если не меньше этой доли её 3 × 3 отсчётов — суша по маске. Меньше — строже; 1 — только ячейки целиком на суше.'],
      adaptive_area: ['Адаптивная область поиска', 'Если маршрута нет, а вода вокруг обеих точек доходит до края сетки, сначала расширить область поиска (проход лежит вне её); иначе измельчить сетку (пролив уже ячейки).'],
      corridor_search: ['Двухуровневый поиск (коридор)', 'Если маршрут на сетке всё ещё пересекает сушу, искать заново на более мелкой сетке только в коридоре вокруг него, а не измельчать всю область.'],
      corridor_km: ['Полуширина коридора', 'Расстояние от грубого маршрута, в пределах которого строится мелкая сетка.'],
      time_limit_s: ['Лимит времени', 'Поиск останавливается через это время (0 — без ограничения). Кнопка «Стоп» работает всегда.'],
      astar_max_steps: ['Максимальное число шагов', 'Один шаг — одна ячейка, взятая из очереди с приоритетом и раскрытая. Поиск останавливается с итогом «лимит шагов», если цель не достигнута; 0 — без ограничения. Сложному курсу на мелкой сетке нужно 10⁵–10⁶ шагов.'],
      astar_heuristic_weight: ['Вес эвристики', '1 — классический A*, маршрут оптимален. Больше 1 — взвешенный A*: меньше шагов, стоимость маршрута не больше этого коэффициента, умноженного на оптимум.'],
      dijkstra_max_steps: ['Максимальное число шагов', 'Один шаг — одна ячейка, взятая из очереди. Алгоритм Дейкстры раскрывает все ячейки ближе к старту, чем цель, поэтому ему нужно в несколько раз больше шагов, чем A*; 0 — без ограничения.'],
      dp_max_sweeps: ['Максимальное число итераций', 'Одна итерация обновляет уравнение Беллмана во всех ячейках. После k итераций значения точны для маршрутов не более чем из k переходов; 0 — до сходимости.'],
      ga_population: ['Размер популяции', 'Число маршрутов в каждом поколении.'],
      ga_generations: ['Число поколений (эпох)', 'Верхний предел числа поколений — лимит шагов генетического алгоритма.'],
      ga_stagnation: ['Ранняя остановка через', 'Остановиться, если лучший маршрут не улучшался столько поколений; 0 — не останавливаться досрочно.'],
      ga_tournament: ['Размер турнира', 'Отбор: лучший из стольких случайных особей становится родителем.'],
      ga_crossover: ['Вероятность скрещивания', 'Вероятность того, что потомок получается из двух родителей (одноточечное скрещивание).'],
      ga_mutation: ['Вероятность мутации', 'Вероятность мутации потомка (точка сдвигается, добавляется, удаляется или исправляется).'],
      ga_elite: ['Элита', 'Лучшие маршруты, переходящие в следующее поколение без изменений.'],
      ga_max_waypoints: ['Максимум путевых точек', 'Наибольшая длина хромосомы. Извилистому курсу (много мысов, островов) нужно больше точек.'],
      ga_land_penalty: ['Штраф за сушу', 'Вес одной ячейки суши, пересекаемой маршрутом, в функции приспособленности.'],
      ga_seed: ['Начальное значение ГСЧ', 'Одинаковые seed и параметры дают одинаковый маршрут.'],
    },
    zh: {
      max_grid_cells: ['网格单元上限', '参与分类的最大网格。精细网格上的长航线需要更多单元；超出上限时，网格步长放大 1.5 倍直到满足。分类时间与单元数成正比。'],
      margin_deg: ['点周围的边距', '在各点外包矩形周围额外留出的空间，使航线能绕过矩形之外的岬角或海峡。'],
      diagonal_moves: ['允许斜向移动（8 邻域）', '开：每个单元与 8 个相邻单元相连；关：只与 4 个（北、南、西、东）相连。'],
      safety_margin_cells: ['近岸带宽度', '靠近陆地的水域带宽度，该带内单元的代价乘以近岸惩罚系数。'],
      proximity_penalty: ['近岸惩罚系数', '近岸带内水域单元的代价倍数；1 表示不惩罚。'],
      auto_refine: ['自动细化网格', '网格上找不到航线时，在更细、更大的网格上重试。'],
      max_refinements: ['最大细化次数', '在更细、更大网格上重试的次数（每次步长 × 0.55，边距 + 2°）。'],
      coast_guard: ['海岸线检查', '用独立的 1 公里陆地掩膜检查航线，穿越陆地时自动修正。'],
      use_land_mask: ['修正时使用陆地掩膜', '航线在海岸线检查中穿越陆地时，把 1 公里掩膜显示为大部分是陆地的单元设为不可通行（无需重新分类）并重新搜索；走廊细网格也使用掩膜。'],
      land_mask_always: ['从第一个网格起使用掩膜', '从一开始就在每个网格上使用陆地掩膜，而不仅用于修正。对陌生海岸更安全，但狭窄海峡更早被封闭，网格需要更频繁地细化。'],
      land_mask_share: ['封锁单元的陆地比例', '单元的 3 × 3 个采样点中至少有这一比例为陆地时封锁该单元。越小越严格；1 表示只封锁完全是陆地的单元。'],
      adaptive_area: ['自适应搜索区域', '没有航线且两点周围的水域都延伸到网格边缘时，先扩大搜索区域（通道在区域外）；否则细化网格（海峡比单元窄）。'],
      corridor_search: ['两级搜索（走廊）', '网格上的航线仍穿越陆地时，只在其周围的走廊内用更细的网格重新搜索，而不是细化整个区域。'],
      corridor_km: ['走廊半宽', '在粗网格航线两侧多远的范围内构建细网格。'],
      time_limit_s: ['时间限制', '超过该时间搜索自动停止（0 表示不限）。“停止”按钮随时可用。'],
      astar_max_steps: ['最大步数', '一步 = 从优先队列取出并扩展一个单元。到达目标前达到该数目时，搜索以“步数上限”结束；0 表示不限。精细网格上的复杂航线需要 10⁵–10⁶ 步。'],
      astar_heuristic_weight: ['启发权重', '1 = 经典 A*，航线最优。大于 1 = 加权 A*：步数更少，航线代价不超过该倍数乘以最优值。'],
      dijkstra_max_steps: ['最大步数', '一步 = 从队列取出一个单元。Dijkstra 会扩展所有比目标更靠近起点的单元，因此所需步数是 A* 的数倍；0 表示不限。'],
      dp_max_sweeps: ['最大迭代次数', '一次迭代在所有单元上更新一次贝尔曼方程。k 次迭代后，对最多 k 步的航线取值精确；0 表示直到收敛。'],
      ga_population: ['种群规模', '每一代中的航线数量。'],
      ga_generations: ['代数（epoch）', '代数上限，即遗传算法的步数上限。'],
      ga_stagnation: ['早停代数', '最优航线连续这么多代没有改进时停止；0 表示不提前停止。'],
      ga_tournament: ['锦标赛规模', '选择：从这么多个随机个体中选最好的作为父代。'],
      ga_crossover: ['交叉概率', '子代由两个父代（单点交叉）产生的概率。'],
      ga_mutation: ['变异概率', '子代发生变异（航点移动、增加、删除或修复）的概率。'],
      ga_elite: ['精英数', '原样进入下一代的最优航线数量。'],
      ga_max_waypoints: ['最大航点数', '染色体的最大长度。曲折的航线（多岬角、多岛屿）需要更多航点。'],
      ga_land_penalty: ['陆地惩罚', '适应度中航线穿越的每个陆地单元的权重。'],
      ga_seed: ['随机种子', '相同的种子和参数得到相同的航线。'],
    },
  };
  const UNITS = {
    ru: { cells: 'ячеек', steps: 'шагов', s: 'с', km: 'км', individuals: 'особей', generations: 'поколений',
          sweeps: 'итераций' },
    zh: { cells: '个单元', steps: '步', s: '秒', km: '公里', individuals: '个个体', generations: '代', sweeps: '次' },
  };

  /* ------------------------------------- phrases sent by the server */
  // [regular expression, {ru, zh}] - $1, $2... are replaced by the groups.
  const SERVER = [
    [/^Optimal path found$/, { ru: 'Оптимальный путь найден', zh: '已找到最优路径' }],
    [/^Path found \(weighted A\*, at most (.+) x the optimum\)$/,
     { ru: 'Путь найден (взвешенный A*, не более чем в $1 раза дороже оптимума)', zh: '已找到路径（加权 A*，代价至多为最优值的 $1 倍）' }],
    [/^coastline check passed$/, { ru: 'проверка по береговой линии пройдена', zh: '海岸线检查通过' }],
    [/^coastline check: the first route crossed land for (\d+) km and was corrected automatically \((.+)\)$/,
     { ru: 'проверка по береговой линии: первый маршрут пересекал сушу на $1 км и был исправлен автоматически ($2)',
       zh: '海岸线检查：第一条航线穿越陆地 $1 公里，已自动修正（$2）' }],
    [/^(\d+) re-search(?:es)? with (\d+) cells blocked$/,
     { ru: 'повторных поисков: $1, заблокировано ячеек: $2', zh: '重新搜索 $1 次，封锁 $2 个单元' }],
    [/^a finer lattice \((.+)°\)$/, { ru: 'более мелкая сетка ($1°)', zh: '更细的网格（$1°）' }],
    [/^Bellman equation solved in (\d+) sweeps$/, { ru: 'Уравнение Беллмана решено за $1 итераций', zh: '贝尔曼方程经 $1 次迭代求解' }],
    [/^Best route after (\d+) generations \((\d+) waypoints\)$/,
     { ru: 'Лучший маршрут после $1 поколений ($2 точек)', zh: '经过 $1 代得到最优航线（$2 个航点）' }],
    [/^No navigable route evolved in (\d+) generations(.*)$/,
     { ru: 'За $1 поколений не получено судоходного маршрута$2', zh: '$1 代内未进化出可航行航线$2' }],
    [/^No navigable path exists between the given points(.*)$/,
     { ru: 'Между заданными точками нет судоходного пути$1', zh: '给定的两点之间不存在可航行路径$1' }],
    [/^ at the finest lattice tried \((.+)°\)$/, { ru: ' на самой мелкой опробованной сетке ($1°)', zh: '（已尝试的最细网格 $1°）' }],
    [/^Start or destination lies on land and no navigable cell was found nearby$/,
     { ru: 'Пункт отправления или назначения находится на суше, и рядом нет судоходной ячейки', zh: '出发点或目的地位于陆地，附近找不到可航行单元' }],
    [/^Step limit of ([\d,]+) reached before the goal \((.+)\) - increase the maximum number of steps$/,
     { ru: 'Лимит в $1 шагов достигнут до цели ($2) — увеличьте максимальное число шагов',
       zh: '到达目标前已达到 $1 步的上限（$2）——请增大最大步数' }],
    [/^search radius ([\d,]+) weighted km$/, { ru: 'радиус поиска $1 взвешенных км', zh: '搜索半径 $1 加权公里' }],
    [/^([\d,]+) cells still in the queue$/, { ru: 'в очереди ещё $1 ячеек', zh: '队列中还有 $1 个单元' }],
    [/^Sweep limit of ([\d,]+) reached before the value reached the start - increase the maximum number of sweeps$/,
     { ru: 'Лимит в $1 итераций достигнут раньше, чем значение дошло до старта, — увеличьте максимальное число итераций',
       zh: '数值传到起点之前已达到 $1 次迭代上限——请增大最大迭代次数' }],
    [/^Stopped by the user after (.+) s$/, { ru: 'Остановлено пользователем через $1 с', zh: '用户在 $1 秒后停止' }],
    [/^Time limit reached after (.+) s \(time limit (.+) s\)$/,
     { ru: 'Лимит времени достигнут через $1 с (лимит $2 с)', zh: '$1 秒后达到时间限制（限制 $2 秒）' }],
    [/^Route through (\d+) intermediate points? \((\d+) legs\): (.*)$/,
     { ru: 'Маршрут через промежуточные точки: $1 (участков: $2): $3', zh: '经过 $1 个途经点的航线（$2 段）：$3' }],
    [/^leg (\d+): (.*)$/, { ru: 'участок $1: $2', zh: '第 $1 段：$2' }],
    [/^Leg (\d+) of (\d+) \((.+)\): (.*)$/, { ru: 'Участок $1 из $2 ($3): $4', zh: '第 $1 段（共 $2 段，$3）：$4' }],
    // lattices tried
    [/^lattice refined to (.+)° \(coastline check\)$/, { ru: 'сетка измельчена до $1° (проверка по береговой линии)', zh: '网格细化到 $1°（海岸线检查）' }],
    [/^lattice refined to (.+)°$/, { ru: 'сетка измельчена до $1°', zh: '网格细化到 $1°' }],
    [/^search area widened to a (.+)° margin$/, { ru: 'область поиска расширена до запаса $1°', zh: '搜索区域扩大到 $1° 边距' }],
    [/^two-level search: (.+)° lattice in a ±(.+) km corridor$/,
     { ru: 'двухуровневый поиск: сетка $1° в коридоре ±$2 км', zh: '两级搜索：±$2 公里走廊内的 $1° 网格' }],
    // progress stages
    [/^Building geographic lattice$/, { ru: 'Построение географической сетки', zh: '正在构建地理网格' }],
    [/^Running (\w+) search$/, { ru: 'Выполняется поиск $1', zh: '正在运行 $1 搜索' }],
    [/^Route ready$/, { ru: 'Маршрут готов', zh: '航线已完成' }],
    [/^No passage at (.+)° - refining the lattice to (.+)°$/,
     { ru: 'Нет прохода при $1° — сетка измельчается до $2°', zh: '$1° 网格上没有通道——网格细化到 $2°' }],
    [/^No passage inside the search area - widening it \(margin (.+)° -> (.+)°\)$/,
     { ru: 'Внутри области поиска нет прохода — область расширяется (запас $1° → $2°)', zh: '搜索区域内没有通道——扩大区域（边距 $1° → $2°）' }],
    [/^Route crosses land for (\d+) km on the coastline check - blocking (\d+) cells and searching again$/,
     { ru: 'Маршрут пересекает сушу на $1 км — блокируется $2 ячеек, повторный поиск', zh: '航线穿越陆地 $1 公里——封锁 $2 个单元后重新搜索' }],
    [/^Route crosses land for (\d+) km - fusing the 1 km land mask \(([\d,]+) cells\) and searching again$/,
     { ru: 'Маршрут пересекает сушу на $1 км — объединение с маской суши 1 км ($2 ячеек), повторный поиск', zh: '航线穿越陆地 $1 公里——融合 1 公里陆地掩膜（$2 个单元）后重新搜索' }],
    [/^Route still crosses land at (.+)° - planning again on a (.+)° lattice$/,
     { ru: 'При $1° маршрут всё ещё пересекает сушу — расчёт на сетке $2°', zh: '$1° 时航线仍穿越陆地——在 $2° 网格上重新规划' }],
    [/^Route still crosses land at (.+)° - two-level search: (.+)° lattice in a ±(.+) km corridor \((.+) of (.+) cells\)$/,
     { ru: 'При $1° маршрут всё ещё пересекает сушу — двухуровневый поиск: сетка $2° в коридоре ±$3 км ($4 из $5 ячеек)', zh: '$1° 时航线仍穿越陆地——两级搜索：±$3 公里走廊内的 $2° 网格（$5 个单元中的 $4 个）' }],
    [/^Leg (\d+)\/(\d+): (.*)$/, { ru: 'Участок $1/$2: $3', zh: '第 $1/$2 段：$3' }],
    [/^Building the shared cost map$/, { ru: 'Построение общей карты стоимости', zh: '正在构建共享代价地图' }],
    [/^Comparison: running (\w+) \((\d+)\/(\d+)\)$/, { ru: 'Сравнение: выполняется $1 ($2/$3)', zh: '对比：正在运行 $1（$2/$3）' }],
    [/^Comparison finished$/, { ru: 'Сравнение завершено', zh: '对比已完成' }],
    [/^crosses land - not navigable$/, { ru: 'пересекает сушу — не судоходна', zh: '穿越陆地——不可航行' }],
    [/^Great-circle reference route$/, { ru: 'Опорная ортодромия', zh: '大圆参考航线' }],
    [/^skipped - the comparison was stopped$/, { ru: 'пропущено — сравнение остановлено', zh: '已跳过——对比已停止' }],
    [/^Sweep limit of ([\d,]+) reached: route over at most (\d+) moves, may be longer than the optimum$/,
     { ru: 'Достигнут лимит в $1 итераций: маршрут не более чем из $2 переходов, он может быть длиннее оптимального',
       zh: '已达到 $1 次迭代上限：航线最多 $2 步，可能比最优航线更长' }],
    [/^Start or goal cell is not navigable$/, { ru: 'Ячейка старта или цели несудоходна', zh: '起点或终点单元不可航行' }],
    [/^no navigable cell near a point$/, { ru: 'рядом с точкой нет судоходной ячейки', zh: '点附近没有可航行单元' }],
    [/^Great-circle route crosses land and is not navigable$/,
     { ru: 'Ортодромия пересекает сушу и не судоходна', zh: '大圆航线穿越陆地，不可航行' }],
    [/^Great-circle route is navigable but ignores zone risk$/,
     { ru: 'Ортодромия судоходна, но не учитывает риск зон', zh: '大圆航线可航行，但没有考虑区域风险' }],
  ];

  /* ----------------------------------------------------------- state */
  let lang = 'en';
  try {
    const saved = localStorage.getItem(STORE_KEY);
    if (LANGS.includes(saved)) lang = saved;
    else {
      const nav = (navigator.language || 'en').slice(0, 2).toLowerCase();
      if (LANGS.includes(nav)) lang = nav;
    }
  } catch (err) { /* storage blocked: browser language or English */ }
  const listeners = [];
  const defaults = new WeakMap();          // English text of the marked elements, read once

  /** Text of `key` in the current language with {name} placeholders filled. */
  function t(key, params) {
    let s = (D[lang] && D[lang][key]) ?? D.en[key] ?? key;
    if (params) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in params ? params[k] : m));
    return s;
  }

  /** Translate one server phrase (recursively for the parts it contains). */
  function server(text) {
    if (lang === 'en' || text === null || text === undefined) return text;
    const parts = String(text).split('; ');
    if (parts.length > 1) return parts.map(server).join('; ');
    for (const [rx, tr] of SERVER) {
      const m = String(text).match(rx);
      if (m) {
        return tr[lang].replace(/\$(\d)/g, (_, i) => {
          const g = m[Number(i)] || '';
          // A group may itself be a phrase (a nested message or a parenthesis).
          if (!/[a-z]{3}/.test(g)) return g;
          const whole = server(g);
          // Several corrections joined by " and " (coastline check message).
          if (whole === g && g.includes(' and ')) {
            return g.split(' and ').map(server).join(lang === 'ru' ? ' и ' : '，');
          }
          return whole;
        });
      }
    }
    return text;
  }

  /** Display name of a zone class: the code in English (as in the thesis), a name otherwise. */
  function zone(code) {
    return lang === 'en' ? code : t('zone.' + code);
  }

  /** Label and help of a parameter field (server English as the fallback). */
  function param(field) {
    const tr = P[lang] && P[lang][field.key];
    return { label: tr ? tr[0] : field.label, help: tr ? tr[1] : field.help,
             unit: (UNITS[lang] && UNITS[lang][field.unit]) || field.unit };
  }

  /** Translate the marked elements under `root` (the whole page by default). */
  function apply(root = document) {
    root.querySelectorAll('[data-i18n]').forEach((el) => {
      if (!defaults.has(el)) defaults.set(el, el.textContent);
      const key = el.dataset.i18n;
      el.textContent = (D[lang][key] ?? D.en[key]) ?? defaults.get(el);
    });
    root.querySelectorAll('[data-i18n-html]').forEach((el) => {
      const key = el.dataset.i18nHtml;
      el.innerHTML = (D[lang][key] ?? D.en[key]) ?? el.innerHTML;
    });
    root.querySelectorAll('[data-i18n-title]').forEach((el) => {
      const key = el.dataset.i18nTitle;
      el.title = (D[lang][key] ?? D.en[key]) ?? el.title;
    });
    root.querySelectorAll('[data-route]').forEach((el) => {
      // Preset chips: data-route="Kiel,Tallinn".
      el.textContent = el.dataset.route.split(',').map((c) => t('city.' + c)).join(' → ');
    });
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : lang;
    document.title = t('app.title');
    document.querySelectorAll('.lang-switch button').forEach((b) =>
      b.classList.toggle('active', b.dataset.lang === lang));
  }

  /** Switch the language, remember it and notify the listeners. */
  function setLang(next) {
    if (!LANGS.includes(next)) return;
    lang = next;
    try { localStorage.setItem(STORE_KEY, next); } catch (err) { /* not remembered */ }
    apply();
    listeners.forEach((fn) => { try { fn(lang); } catch (err) { console.error(err); } });
  }

  return {
    t, server, param, zone, apply, setLang, LANGS,
    get lang() { return lang; },
    onChange(fn) { listeners.push(fn); },
    /** Locale for number formatting. */
    get locale() { return { en: 'en-US', ru: 'ru-RU', zh: 'zh-CN' }[lang]; },
  };
})();

/** Short alias used by app.js and charts.js. */
const t = I18N.t;
