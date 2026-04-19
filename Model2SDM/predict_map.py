"""Generate probability maps for each species on the master lat/lon grid.

Builds a background of oceanographic features by averaging training data
within each grid cell (optionally filtered by month), then scores every
trained species classifier on that background.

Output: outputs/probability_maps.parquet with columns
    (lat, lon, species, probability, n_obs_in_cell)

Usage:
    python predict_map.py                     # yearly average
    python predict_map.py --month 7           # July only
    python predict_map.py --model gbm
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (
    ARTIFACTS_DIR,
    FEATURE_COLUMNS,
    LAT_MAX,
    LAT_MIN,
    LAT_STEP,
    LON_MAX,
    LON_MIN,
    LON_STEP,
    OUTPUTS_DIR,
)
from data import build_feature_matrix, load_dataset


def build_grid_background(df, month=None):
    if month is not None:
        df = df[df["month"] == month].copy()
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_STEP / 2, LAT_STEP)
    lon_bins = np.arange(LON_MIN, LON_MAX + LON_STEP / 2, LON_STEP)

    df = df.copy()
    df["lat_snap"] = np.round(df["lat"] / LAT_STEP) * LAT_STEP
    df["lon_snap"] = np.round(df["lon"] / LON_STEP) * LON_STEP

    env_cols = ["sst", "sst_anomaly", "chlorophyll", "salinity",
                "dissolved_oxygen", "ssh"]
    agg = df.groupby(["lat_snap", "lon_snap"]).agg(
        **{c: (c, "mean") for c in env_cols},
        n_obs=("time", "size"),
    ).reset_index().rename(columns={"lat_snap": "lat", "lon_snap": "lon"})

    lat_grid, lon_grid = np.meshgrid(lat_bins, lon_bins, indexing="ij")
    full_grid = pd.DataFrame({"lat": lat_grid.ravel(), "lon": lon_grid.ravel()})
    full_grid["lat"] = np.round(full_grid["lat"] / LAT_STEP) * LAT_STEP
    full_grid["lon"] = np.round(full_grid["lon"] / LON_STEP) * LON_STEP

    merged = full_grid.merge(agg, on=["lat", "lon"], how="left")

    for c in env_cols:
        merged[c] = merged[c].fillna(df[c].median())
    merged["n_obs"] = merged["n_obs"].fillna(0).astype(int)

    if month is None:
        merged["month"] = 6
        merged["day_of_year"] = 172
    else:
        merged["month"] = month
        month_doy = {1:15, 2:46, 3:75, 4:106, 5:136, 6:167,
                     7:197, 8:228, 9:259, 10:289, 11:320, 12:350}
        merged["day_of_year"] = month_doy[month]

    return merged


def predict(model_kind="logreg", month=None):
    bundle_path = ARTIFACTS_DIR / f"sdm_{model_kind}.joblib"
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"No model bundle found at {bundle_path}. Run train.py first."
        )
    bundle = joblib.load(bundle_path)
    estimators = bundle["estimators"]
    feature_names = bundle["feature_names"]

    print(f"[predict] loaded {len(estimators)} species models from {bundle_path.name}")

    df = load_dataset()
    grid = build_grid_background(df, month=month)
    print(f"[predict] grid cells: {len(grid)}  (month={month})")

    X = build_feature_matrix(grid[FEATURE_COLUMNS])
    X = X[feature_names].values

    records = []
    for species, est in estimators.items():
        proba = est.predict_proba(X)[:, 1]
        block = pd.DataFrame({
            "lat": grid["lat"].values,
            "lon": grid["lon"].values,
            "species": species,
            "probability": proba.astype("float32"),
            "n_obs_in_cell": grid["n_obs"].values.astype("int32"),
        })
        records.append(block)

    result = pd.concat(records, ignore_index=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_m{month:02d}" if month else "_yearly"
    out_path = OUTPUTS_DIR / f"probability_maps_{model_kind}{suffix}.parquet"
    try:
        result.to_parquet(out_path, index=False)
    except Exception:
        out_path = out_path.with_suffix(".csv")
        result.to_csv(out_path, index=False)
    print(f"[predict] wrote {len(result):,} rows -> {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    p.add_argument("--month", type=int, default=None,
                   help="1-12 for a specific month; omit for yearly average")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict(model_kind=args.model, month=args.month)
