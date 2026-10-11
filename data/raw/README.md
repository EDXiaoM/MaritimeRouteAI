# Raw data

Place the labelled AIS corpus here as `merged.geojson`.

The current corpus is the second release supplied by the supervisor: 156 838
points in eleven sea regions (the first release plus 4 000 points of the Aegean
Sea). After cleaning, 151 411 points remain.

It is not distributed inside the delivery archive because of its size (119 MB).
The application does **not** need it to run — a trained model ships in
`models/`. The corpus is required only to rebuild the dataset and retrain:

```bash
python scripts/prepare_dataset.py
python scripts/train_model.py
```

Expected format: a GeoJSON `FeatureCollection` of points, each feature carrying
`latitude`, `longitude`, `class_code` (one of `OPEN_SEA`, `COASTAL_SEA`,
`NEAR_COAST`, `COASTLINE`), `noaa_depth`, `noaa_type`, `elevation` and the nine
`osm_*` counters. See section 1.2.4 of the explanatory note.
