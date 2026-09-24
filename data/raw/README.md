# Raw data

Place the labelled AIS corpus here as `merged.geojson`.

It is not distributed inside the delivery archive because of its size (116 MB).
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
