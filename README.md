# Maritime Route Planner

**Application for constructing a maritime vessel's course based on neural-network
classification of geographic zones.**

Diploma project · Li Haoran · Brest State Technical University, Department of
Intelligent Information Technologies · supervisor А.А. Козинский · 2026/27

> Chinese version of this document: [README_zh.md](README_zh.md).
> Step-by-step installation: [docs/INSTALL.md](docs/INSTALL.md).
> Complete user guide: [docs/USER_GUIDE.md](docs/USER_GUIDE.md).

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

**Data cleaning.** 5 075 of the 152 838 labelled points (3.32 %) carry a class that
contradicts their measured height or depth — for example a "coastal sea" point
lying several metres above sea level. They are removed before training
(`src/maritime_route/data/cleaning.py`, tolerance ±5 m).

**Classifier** (22 165 held-out test points of the cleaned corpus):

| Metric | Value |
|---|---|
| Accuracy | **96.62 %** |
| Macro F1 / Cohen κ / macro ROC-AUC | 0.9660 / 0.9536 / 0.9982 |
| Water ↔ land confusions (safety-critical errors) | 74 (0.33 %) |
| Parameters / training time | 98 364 / 105 epochs, 4.5 min on a CPU |

Accuracy in each of the ten sea regions of the corpus is written to
`models/region_metrics.json` (from 92.3 % in the South China Sea to 99.9 % in the
Red Sea).

**Routing** (6 voyages in European waters + 10 voyages, one in each sea region):

| Result | Value |
|---|---|
| Great-circle routes that cross land | 16 of 16 |
| Optimised routes that are navigable | 16 of 16 |
| A\* vs Dijkstra vs dynamic programming | identical optimal cost on all 16 voyages; A\* expands 2.43× fewer nodes than Dijkstra on average |
| Genetic algorithm | navigable route in 15 of 16 voyages; on average 1.1 % (regional) and 5.5 % (European) above the optimum |
| Extra distance of a safe route | +16 % (European) and +22 % (regional) on average |

The genetic algorithm fails on Rotterdam → Gdańsk: it converges on the Øresund,
which the lattice closes by one cell, and cannot jump to the Great Belt that A\*
finds. This is the known weakness of evolutionary search in maze-like terrain
and is discussed in the thesis.

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
   Helsinki → Copenhagen, Odesa → Novorossiysk).
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

The labelled corpus `merged.geojson` (116 MB) is too large for GitHub. Put it at
`data/raw/merged.geojson` and run:

```bash
python scripts/prepare_dataset.py     # ~15 s: cleaning, split, features  -> data/processed/
python scripts/train_model.py         # ~5 min on a CPU                   -> models/
python scripts/evaluate_regions.py    # accuracy per sea region           -> models/region_metrics.json
python scripts/benchmark_routing.py   # 6 European voyages                -> docs/figures/routing_benchmark.json
python scripts/regional_examples.py   # 10 regional voyages + map figure  -> docs/figures/regional_routes.json
python scripts/make_figures.py        # all figures of the thesis         -> docs/figures/
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
│   │   └── planner.py             orchestration, refinement, benchmark
│   ├── storage/                   SQLite schema and repository
│   ├── export/                    GeoJSON / CSV / PDF writers
│   └── web/                       FastAPI service, WebSocket, tiles, browser client
├── scripts/                       command-line tools (section 6) and run_server.py
├── tests/                         79 automated tests (pytest)
├── models/                        trained weights, scaler, reference index, metadata
├── data/samples/                  sample AIS track
└── docs/                          INSTALL.md, USER_GUIDE.md, figures, posters, screenshots
```

Every module, class and function carries an English docstring describing its
purpose, parameters and return values; non-obvious steps are commented inline.
Chinese comments are secondary notes of the author.

## 8. Tests

```bash
pip install -r requirements-dev.txt
pytest
```

All 79 tests must pass. Tests that need the trained model are skipped when
`models/zone_classifier.pt` is missing.

## 9. Limitations

* This is a decision-support tool, not a certified navigation system; it does not
  use official electronic navigational charts.
* The cost model ignores weather, currents, tides, ice, traffic separation schemes
  and the draught of a particular vessel.
* The corpus covers ten sea regions (North Europe, Black Sea and Eastern
  Mediterranean, Red Sea, Persian Gulf and Arabian Sea, Bay of Bengal, Malacca
  Strait, South China Sea, Japan, Southern Africa, Central America). Outside them
  the classification is unreliable — for example the sea off Durban is classified
  as land.
* A strait narrower than one lattice cell is found only after automatic
  refinement, and only down to 0.04°.
