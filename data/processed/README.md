# Processed data

`prepare_dataset.py` writes two files here:

- `zone_dataset.npz` — feature matrices and labels for the train/validation/test splits
- `dataset_statistics.json` — corpus statistics quoted in the explanatory note

Both are regenerated from `data/raw/merged.geojson`; `zone_dataset.npz` is not
shipped in the delivery archive because it is derived data.
