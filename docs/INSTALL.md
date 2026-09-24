# Installation and configuration guide (Appendix B of the explanatory note)

## 1. System requirements

| Item | Minimum | Recommended |
|---|---|---|
| Operating system | Windows 10 / Ubuntu 20.04 / macOS 12 | any 64-bit, current |
| Python | 3.10 | 3.11 |
| RAM | 4 GB | 8 GB |
| Free disk space | 3 GB | 5 GB |
| CPU | 2 cores | 4 cores or more |
| GPU | not required | not required (CPU inference is sufficient) |
| Browser | any with WebSocket support | Chrome / Firefox / Edge, current |

Training the model from scratch takes about five minutes on four CPU cores; no
GPU is needed at any stage.

## 2. Installation

### 2.1 Obtain the sources

Clone the repository from GitHub (or unpack the delivered archive):

```bash
git clone https://github.com/<your-account>/MaritimeRouteAI.git
cd MaritimeRouteAI
```

On Windows without Git, use *Code → Download ZIP* on the GitHub page, unpack the
archive and open a command prompt in the unpacked folder.

### 2.2 Create an isolated environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 2.3 Install the dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `pip` cannot reach the default PyTorch wheels, install the CPU build
explicitly:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2.4 Verify the installation

```bash
pip install -r requirements-dev.txt
pytest
```

All tests must pass. Tests that need the trained model are skipped
automatically when `models/zone_classifier.pt` is absent.

## 3. Preparing the data and the model

The archive already contains a trained model (`models/`), so this section is
needed only to retrain or to use a different corpus.

```bash
# 1. put the labelled corpus here:
#    data/raw/merged.geojson
python scripts/prepare_dataset.py            # ~15 s, cleaning + features, writes data/processed/
python scripts/train_model.py                # ~5 min on CPU, writes models/
python scripts/evaluate_regions.py           # optional: accuracy per sea region
python scripts/benchmark_routing.py          # optional: 6 European voyages, all methods
python scripts/regional_examples.py          # optional: 10 regional voyages + map figure
python scripts/make_figures.py               # optional: regenerate the figures
```

`prepare_dataset.py` removes the points whose class contradicts their measured
height or depth (5 075 points at the default tolerance of 5 m, set by
`LABEL_CHECK_TOLERANCE_M` in `config.py`); pass `--no-clean` to keep them.

`prepare_dataset.py` writes:

| File | Content |
|---|---|
| `data/processed/zone_dataset.npz` | feature matrices and labels for the three splits |
| `data/processed/dataset_statistics.json` | corpus statistics, cleaning report and per-region counts |
| `models/reference_index.npz` | label-free bathymetry / OSM index |

`train_model.py` writes:

| File | Content |
|---|---|
| `models/zone_classifier.pt` | network weights and architecture |
| `models/feature_scaler.joblib` | fitted standardiser |
| `models/model_metadata.json` | hyper-parameters, metrics and training history |

## 4. Running the application

```bash
python scripts/run_server.py
```

Then open <http://127.0.0.1:8000>.

Options:

```bash
python scripts/run_server.py --host 0.0.0.0 --port 8080
python scripts/run_server.py --reload          # development: auto-reload
```

## 5. Configuration

All tunable values live in `src/maritime_route/config.py`. The most useful ones:

| Setting | Default | Meaning |
|---|---|---|
| `ZONE_COST` | 1.0 / 1.8 / ∞ / ∞ | traversal weight of each zone |
| `ROUTING.grid_resolution_deg` | 0.25 | default lattice step, degrees |
| `ROUTING.max_grid_cells` | 400 000 | lattice budget; larger areas are coarsened |
| `ROUTING.safety_margin_cells` | 1 | width of the penalised belt along the shore |
| `ROUTING.proximity_penalty` | 1.6 | extra cost of a water cell next to land |
| `GENETIC.*` | see file | population size, generations, rates and seed of the genetic algorithm |
| `LABEL_CHECK_TOLERANCE_M` | 5.0 | tolerance of the data-cleaning check, metres |
| `REGIONS` | ten boxes | the sea regions used for per-region evaluation |
| `TRAINING.*` | see file | network size, epochs, learning rate, seed |
| `SERVER.progress_interval_s` | 0.15 | minimum interval between progress frames |

Several settings can also be overridden with environment variables, which is
convenient for deployment:

```bash
export MR_HOST=0.0.0.0
export MR_PORT=8080
export MR_DB_PATH=/var/lib/maritime/routes.db
export MR_MODEL_DIR=/opt/maritime/models
export MR_OUTPUT_DIR=/var/lib/maritime/exports
```

## 6. The database

SQLite is created automatically at `data/maritime_route.db` on first start; the
schema is in `src/maritime_route/storage/schema.sql`. No database server has to
be installed. To reset the stored history, stop the application and delete the
`data/maritime_route.db*` files.

## 7. Offline operation

The client requests OpenStreetMap tiles when the machine is online. If the tile
service is unreachable, after four failed tiles the client switches by itself to
the **on-board chart** — tiles rendered by the server from the model's own
classification — and the application stays fully usable. The layer can also be
selected manually in the layer control at the top-left corner of the map.

## 8. Troubleshooting

| Symptom | Cause and remedy |
|---|---|
| `503 Model weights not found` | the model has not been trained — run `scripts/train_model.py` |
| The status pill shows *reconnecting…* | the server was restarted; the client reconnects by itself with exponential back-off |
| `Address already in use` | another process holds port 8000 — start with `--port 8081` |
| Planning reports *no navigable path* | departure or destination is landlocked; the lattice is already refined and widened three times automatically |
| Planning reports *lies on land* | the port is more than 60 km from any navigable cell, often because it lies outside the ten sea regions covered by the training data |
| The genetic algorithm reports *no navigable route evolved* | the evolutionary search converged on a closed strait; use A\* |
| The map is blank | no internet access — switch to *On-board chart (offline)* in the layer control |
| `ModuleNotFoundError: maritime_route` | run the scripts from the project root, or set `PYTHONPATH=src` |
