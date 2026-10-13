# Maritime Route Planner — diploma project

**Русская версия — [README_ru.md](README_ru.md)** · 中文 — [README_zh.md](README_zh.md)

Li Haoran · Brest State Technical University, Department of Intelligent
Information Technologies · supervisor А.А. Козинский · 2026/27

**Diploma project:** *application for constructing a maritime vessel's course based
on neural-network classification of geographic zones.*

> The **course project** of the author («Sea Course GA», genetic algorithm) is a
> separate repository: <https://github.com/EDXiaoM/SeaCourseGA>.

## Materials of the diploma project

| Material | Where |
|---|---|
| Explanatory note, **first version** (Word and PDF) | [`docs/diploma/`](docs/diploma) |
| Assignment for the diploma project (Russian) | [`docs/diploma/Assignment_diploma_ru.pdf`](docs/diploma/Assignment_diploma_ru.pdf) |
| Graphic material, 5 sheets A1 (PDF and PNG) | [`docs/posters/`](docs/posters) |
| **User guide** (installation and work, PDF) | [`docs/User_Guide_MaritimeRouteAI.pdf`](docs/User_Guide_MaritimeRouteAI.pdf) |
| User guide and installation guide (Markdown) | [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md), [`docs/INSTALL.md`](docs/INSTALL.md) |
| Program code | [`src/`](src), [`scripts/`](scripts), [`tests/`](tests) |
| Trained model | [`models/`](models) |
| Figures and screenshots of the thesis | [`docs/figures/`](docs/figures), [`docs/screenshots/`](docs/screenshots) |
| Labelled dataset (second release, 156 838 points, zip 7 MB) | `merged_geojson_v2.zip` in [release v1.0](https://github.com/EDXiaoM/MaritimeRouteAI/releases/tag/v1.0) |

---

## 1. What the application does

The application plans a safe sea route between two points and compares it with
the shortest possible route.

1. **Zone classification.** A neural network (a residual multilayer perceptron
   written in PyTorch) classifies every cell of a geographic lattice into one of
   four zones:

   | Zone | Meaning | Traversal weight |
   |---|---|---|
   | `OPEN_SEA` | deep water far from the shore | 1.0 |
   | `COASTAL_SEA` | navigable water close to the shore | 1.8 |
   | `NEAR_COAST` | land within 5 km of the shoreline | impassable |
   | `COASTLINE` | land far from the sea | impassable |

2. **Cost map.** The zones are turned into a traversal cost map. Water cells that
   touch land are penalised by a further factor of 1.6 so that routes keep a
   safety distance from the shore.
3. **Route search.** Four search methods are available on the same cost map,
   all built on the same cost of one discrete step
   `c(u, v) = d_gc(u, v) · (w(u) + w(v)) / 2`:
   * **A\*** — exact, uses an admissible great-circle heuristic (default);
   * **Dijkstra** — exact, no heuristic (reference for A\*);
   * **dynamic programming** — exact; solves the Bellman equation
     `V(v) = min_u [c(v, u) + V(u)]`, `V(goal) = 0` by value iteration and follows
     the value function from the departure;
   * **genetic algorithm** — evolutionary search over waypoint sequences, guided by
     a fitness function; compared with the exact optimum.
4. **Comparison and export.** The route is compared with the great circle (the
   shortest distance on the sphere, which usually crosses land) and exported to
   **GeoJSON, CSV and PDF**. Every route is stored in an **SQLite** database.

The browser client shows the planning process in **real time** over a WebSocket:
lattice construction, cell-by-cell classification progress, the number of nodes
expanded by A\*, the value-iteration sweep of dynamic programming or the
generation counter of the genetic algorithm.

## 2. Measured results

All numbers below are produced by the scripts in `scripts/` and can be reproduced
(section 6).

**Corpus.** The second release of the labelled corpus, supplied by the
supervisor, contains 156 838 points in eleven sea regions; it adds 4 000 points
of the Aegean Sea (1 000 per class) to the first release.

**Data cleaning.** 5 427 of the 156 838 labelled points (3.46 %) carry a class that
contradicts their measured height or depth — for example a "coastal sea" point
lying several metres above sea level. Both measurements of a point (`noaa_depth`
and `elevation`) are checked, with a tolerance of ±5 m, and the conflicting
points are removed before training (`src/maritime_route/data/cleaning.py`).
151 411 points remain — exactly the number of clean points stated by the
supervisor for this release.

**Classifier** (22 712 held-out test points of the cleaned corpus):

| Metric | Value |
|---|---|
| Accuracy | **96.95 %** |
| Macro F1 / Cohen κ / macro ROC-AUC | 0.9695 / 0.9582 / 0.9985 |
| Water ↔ land confusions (safety-critical errors) | 67 (0.29 %) |
| Parameters / training time | 98 364 / 97 epochs, 4.3 min on a CPU |

Accuracy in each of the eleven sea regions of the corpus is written to
`models/region_metrics.json` (from 93.5 % in the South China Sea to 99.5 % in the
Red Sea; 97.9 % in the newly added Aegean Sea).

**Routing** (6 voyages in European waters + 11 voyages, one in each sea region):

| Result | Value |
|---|---|
| Great-circle routes that cross land | 17 of 17 |
| Optimised routes that are navigable (as judged by the classifier) | 17 of 17 |
| A\* vs Dijkstra vs dynamic programming | identical optimal cost on all 17 voyages; A\* expands 2.5× fewer nodes than Dijkstra on average |
| Genetic algorithm | navigable route in 15 of 17 voyages; on average 1.1 % (regional) and 3.8 % (European) above the optimum |
| Extra distance of a safe route | +17 % (European) and +28 % (regional) on average |

**Test on 100 random voyages** (`scripts/route_audit.py`; every route checked
kilometre by kilometre against an independent 1 km land mask, GLOBE / global-land-mask):

| Verdict | Voyages |
|---|---|
| Correct course | 66 |
| Grazing the coast (land runs under 3 km) | 5 |
| Crossing land | 19 — 16 because the classifier takes sparsely sampled land for water, 3 because of a diagonal step of the lattice |
| No route | 10 — 7 without an open sea connection (Caribbean ↔ Pacific, Bosporus), 3 with the passage closed on the classified lattice |

Every voyage is listed in `docs/figures/route_audit.csv` and drawn in appendix C of
the thesis. The errors show where the corpus needs denser coastal samples.

**Automatic coastline check (version 1.2).** Since the test showed that the classifier
sometimes takes sparsely sampled land for water, the planner now checks *every* route
in the same way (`src/maritime_route/routing/coast_check.py`): the route is sampled every
kilometre against the independent land mask; if it crosses land for 3 km or more, the
offending lattice cells are blocked and the search is repeated (if necessary on a finer
lattice), and a route that still crosses land is shown with a red warning. The result
appears as a banner above the route summary and in the PDF report, the GeoJSON export
and the database. Repeating the 100-voyage test with the check:

| Verdict | Without the check | With the check |
|---|---|---|
| Correct course | 66 | 75 |
| Grazing the coast | 5 | 15 |
| Crossing land | 19 | 0 |
| No route | 10 | 10 |

19 of the 19 incorrect courses are corrected automatically;
the voyages that were already correct keep their routes, and the mean request time
grows by about 1 %.
Four of the five presets of the interface also crossed land without the check
(Helsinki → Copenhagen by 108 km) and now pass it. The check needs the package
`global-land-mask` (in `requirements.txt`); without it the planner works as before.

The genetic algorithm fails on Rotterdam → Gdańsk, where it converges on the
Øresund (which the lattice closes by one cell) and cannot jump to the Great Belt
that A\* finds, and on Piraeus → Thessaloniki, where no individual finds a
land-free passage among the Aegean islands. This is the known weakness of
evolutionary search in maze-like terrain and is discussed in the thesis.

The Aegean voyage is planned at 0.1° (the preset button selects it): at the
default 0.25° a diagonal step of the lattice cuts across Attica, which is
narrower than one cell. Some Aegean islands (Kea, Naxos, Skiathos) have no land
sample in the corpus and are therefore classified as sea.

## 3. Requirements

| Item | Minimum | Recommended |
|---|---|---|
| Operating system | Windows 10, Ubuntu 20.04, macOS 12 | any current 64-bit system |
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB |
| Free disk space | 3 GB (PyTorch is large) | 5 GB |
| GPU | not needed | not needed |
| Browser | any with WebSocket support | current Chrome, Firefox or Edge |

## 4. Installation and start

```bash
git clone https://github.com/EDXiaoM/MaritimeRouteAI.git
cd MaritimeRouteAI

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python scripts/run_server.py
```

When the console prints `Zone classifier ready`, open <http://127.0.0.1:8000>.
A trained model is included in `models/`, so **no training is needed** to run the
application. Detailed instructions for each operating system, including the
installation of the CPU build of PyTorch, are in [docs/INSTALL.md](docs/INSTALL.md).

## 5. Using the application

1. Click the chart twice (departure, then destination), drag the markers, type
   coordinates, or press a **preset** (Kiel → Tallinn, Rotterdam → Lisbon,
   Helsinki → Copenhagen, Odesa → Novorossiysk, Piraeus → Thessaloniki).
2. Choose the **search algorithm** (A\*, Dijkstra, genetic algorithm, dynamic
   programming or great circle), the lattice resolution and the service speed.
3. Press **Build route** and watch the progress bar.
4. Read the result in *Route summary*; open **Compare** for the comparison with the
   great circle; press **Compare algorithms** to run all five methods on one cost
   map.
5. Export with **GeoJSON**, **CSV** or **PDF report**.
6. Load your own AIS track in **Data** (`.geojson`, `.csv`, `.json`); a sample is in
   `data/samples/sample_ais_track.geojson`.

Every control is explained in [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

## 6. Reproducing the results

The labelled corpus `merged.geojson` (119 MB, second release with the Aegean
Sea) is too large to be stored in the repository itself; it is attached to
[release v1.0](https://github.com/EDXiaoM/MaritimeRouteAI/releases/tag/v1.0) as
`merged_geojson_v2.zip` (7 MB). Download it, unpack it to `data/raw/merged.geojson`
and run:

```bash
python scripts/prepare_dataset.py     # ~16 s: cleaning, split, features  -> data/processed/
python scripts/train_model.py         # ~5 min on a CPU                   -> models/
python scripts/evaluate_regions.py    # accuracy per sea region           -> models/region_metrics.json
python scripts/benchmark_routing.py   # 6 European voyages                -> docs/figures/routing_benchmark.json
python scripts/regional_examples.py   # 11 regional voyages + map figure  -> docs/figures/regional_routes.json
python scripts/make_figures.py        # all figures of the thesis         -> docs/figures/
python scripts/route_audit.py         # 100 random voyages, land check    -> docs/figures/route_audit.*  (needs global-land-mask)
```

`prepare_dataset.py --no-clean` reproduces the dataset without the cleaning step.

## 7. Repository layout

```
MaritimeRouteAI/
├── src/maritime_route/            application package
│   ├── config.py                  paths, zones, weights, regions, all hyper-parameters
│   ├── data/
│   │   ├── ais_loader.py          GeoJSON / CSV / JSON readers and validation
│   │   ├── cleaning.py            removal of mislabelled points, region assignment
│   │   ├── features.py            28 features + label-free reference index (BallTree)
│   │   └── dataset.py             preparation pipeline and stratified split
│   ├── model/
│   │   ├── network.py             residual MLP (PyTorch)
│   │   ├── trainer.py             training loop, early stopping, metrics
│   │   └── inference.py           thread-safe inference service
│   ├── routing/
│   │   ├── geodesy.py             great-circle distance, bearing, Douglas–Peucker
│   │   ├── cost_map.py            lattice, classification, traversal costs
│   │   ├── astar.py               A* search
│   │   ├── dijkstra.py            Dijkstra search
│   │   ├── genetic.py             genetic-algorithm search (fitness function)
│   │   ├── dynamic_programming.py value iteration on the Bellman equation (value function)
│   │   ├── coast_check.py         independent coastline check of every route
│   │   └── planner.py             orchestration, refinement, coastline correction, benchmark
│   ├── storage/                   SQLite schema and repository
│   ├── export/                    GeoJSON / CSV / PDF writers
│   └── web/                       FastAPI service, WebSocket, tiles, browser client
├── scripts/                       command-line tools (section 6) and run_server.py
├── tests/                         101 automated tests (pytest)
├── models/                        trained weights, scaler, reference index, metadata
├── data/samples/                  sample AIS track
└── docs/
    ├── diploma/                   explanatory note, first version (Word, PDF), assignment
    ├── posters/                   graphic material, 5 sheets A1
    ├── User_Guide_MaritimeRouteAI.pdf   user guide (installation and work)
    ├── INSTALL.md, USER_GUIDE.md  the same guides in Markdown
    └── figures/, screenshots/     figures and screenshots of the thesis
```

Every module, class and function carries an English docstring describing its
purpose, parameters and return values; non-obvious steps are commented inline.
Chinese comments are secondary notes of the author.

## 8. Tests

```bash
pip install -r requirements-dev.txt
pytest
```

All 101 tests must pass. Tests that need the trained model are skipped when
`models/zone_classifier.pt` is missing.

## 9. Limitations

* This is a decision-support tool, not a certified navigation system; it does not
  use official electronic navigational charts.
* The cost model ignores weather, currents, tides, ice, traffic separation schemes
  and the draught of a particular vessel.
* The corpus covers eleven sea regions (North Europe, Aegean Sea, Black Sea and
  Eastern Mediterranean, Red Sea, Persian Gulf and Arabian Sea, Bay of Bengal,
  Malacca Strait, South China Sea, Japan, Southern Africa, Central America). Outside them
  the classification is unreliable — for example the sea off Durban is classified
  as land.
* A strait narrower than one lattice cell is found only after automatic
  refinement, and only down to 0.04°.
