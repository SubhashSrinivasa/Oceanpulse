# OceanPulse Model 2 — Species Distribution Model (SDM)

Probabilistic species distribution model trained on
`Model2DataEngineering/outputs/training_data.csv`. For each of ~270 fish
species, fits a binary classifier that predicts `P(species present |
environment, location, time-of-year)`, then projects those probabilities
onto the 0.25° master grid to produce a probability map per species.

Does not touch `Model1DataEngineering/` at all.

## Inputs

`training_data.csv` (wide multi-label): each row is one independent
`(time, lat, lon)` observation with 8 environmental features followed by
369 binary species-presence columns (0/1). See the Model 2 README for
column semantics.

## Features used

- Environmental: `sst`, `sst_anomaly`, `chlorophyll`, `salinity`,
  `dissolved_oxygen`, `ssh`
- Spatial: `lat`, `lon`
- Temporal (cyclical): `sin/cos` of `day_of_year` and `month`

NaNs in any feature are filled with that feature's median.

## Models

- **`logreg`** (default) — `StandardScaler` + `LogisticRegression`
  (`class_weight="balanced"`, L2, `max_iter=2000`). Fast, interpretable,
  well-calibrated.
- **`gbm`** — `HistGradientBoostingClassifier` (depth 6, 200 iterations).
  Captures non-linear environmental responses.

One classifier is trained per species. Species with fewer than
`--min-positives` occurrences (default 20) are skipped.

## Usage

All commands run from inside `Model2SDM/` using the Model 2 venv:

```bash
# one-time dependencies (already installed in Model2DataEngineering/.venv)
python -m pip install scikit-learn matplotlib joblib

# 1. train
python train.py                        # logreg, min_positives=20
python train.py --model gbm            # gradient boosting variant

# 2. score every grid cell (produces a long-form parquet)
python predict_map.py                  # yearly-average environment
python predict_map.py --month 7        # July-average environment

# 3. render heatmaps
python visualize.py --top 20           # 20 best-AUC species
python visualize.py --species sebastes_mystinus
python visualize.py --model gbm --month 7 --top 30
```

## Outputs

```
artifacts/
  sdm_logreg.joblib          # {model_kind, feature_names, species, estimators{species: pipeline}}
  metrics_logreg.csv         # per-species ROC AUC, average precision, positive counts
  summary_logreg.json        # aggregate stats

outputs/
  probability_maps_logreg_yearly.parquet
    columns: lat, lon, species, probability, n_obs_in_cell
  maps/
    logreg_yearly/
      <species>.png          # viridis heatmap on the 30–48°N, -124–-116°W grid
```

## Current results (logreg, min_positives=20)

- **270** species modeled (of 369 total — the rest had <20 presences).
- **Mean ROC AUC: 0.832**, median 0.852 on a held-out 20% split.
- Top species exceed AUC 0.99 (mostly mesopelagic myctophids and
  bathypelagic species with tight thermal/depth niches, e.g.
  `chiasmodon_subniger`, `ceratoscopelus_townsendi`,
  `symbolophorus_californiensis`).

## Design notes

- **Per-species independent classifiers** rather than a joint multilabel
  model. Simple, parallelisable, and each species gets its own
  `class_weight="balanced"` — important because positive rates range
  across 5+ orders of magnitude.
- **Background grid from training data**: the prediction grid's
  environmental features are the cell-wise mean of observed conditions
  in `training_data.csv`. Cells never observed are filled with the
  dataset-wide feature median and flagged via `n_obs_in_cell=0`.
- **No land mask**: the Model 2 pipeline drops land implicitly (observations
  only exist where species were sampled). For visualization, zero-obs
  cells still get a prediction; they are simply extrapolated.
- **No data leakage into test split**: train/test is a random 80/20 on
  rows. A spatial or temporal block-holdout would be more conservative
  if geographic generalization matters; easy to swap in by changing the
  split in `train.py`.

## File map

| File | Role |
|------|------|
| `config.py` | Paths, grid constants, feature list |
| `data.py` | CSV load, NaN fill, cyclical time encoding |
| `train.py` | Per-species fit, metrics, bundle save |
| `predict_map.py` | Score every grid cell → long parquet |
| `visualize.py` | Parquet → per-species PNG heatmaps |
