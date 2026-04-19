"""Reconstruct dense (T, C, H, W) grid tensor from ocean_cube_sequences.csv.

The sequences CSV stores one row per surviving ocean (lat, lon) with each
variable as a JSON list of 1826 daily values. This module parses that file
once and saves a cached .npz so downstream training doesn't pay the JSON
tax every epoch.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    GRID_CACHE,
    GRID_H,
    GRID_T,
    GRID_W,
    MASTER_LAT,
    MASTER_LON,
    MASTER_TIME,
    SEQUENCE_VARS,
    SEQUENCES_CSV,
)


def _latlon_to_index(lat: float, lon: float) -> tuple[int, int]:
    i = int(round((lat - MASTER_LAT[0]) / (MASTER_LAT[1] - MASTER_LAT[0])))
    j = int(round((lon - MASTER_LON[0]) / (MASTER_LON[1] - MASTER_LON[0])))
    return i, j


def build_grid(csv_path: Path = SEQUENCES_CSV) -> dict:
    t0 = time.time()
    print(f"[grid] loading {csv_path}")
    df = pd.read_csv(csv_path, comment="#")
    print(f"[grid]   {len(df)} ocean rows, columns={list(df.columns)}")

    for v in SEQUENCE_VARS:
        if v not in df.columns:
            raise KeyError(f"sequences CSV missing expected column: {v}")

    data = np.full((GRID_T, len(SEQUENCE_VARS), GRID_H, GRID_W), np.nan, dtype=np.float32)
    ocean_mask = np.zeros((GRID_H, GRID_W), dtype=bool)

    for ridx, row in df.iterrows():
        i, j = _latlon_to_index(row["lat"], row["lon"])
        if not (0 <= i < GRID_H and 0 <= j < GRID_W):
            raise ValueError(f"row {ridx} lat={row['lat']} lon={row['lon']} maps outside grid")
        ocean_mask[i, j] = True
        for ci, v in enumerate(SEQUENCE_VARS):
            arr = np.asarray(json.loads(row[v]), dtype=np.float32)
            if arr.shape[0] != GRID_T:
                raise ValueError(
                    f"row {ridx} var={v} has {arr.shape[0]} timesteps, expected {GRID_T}"
                )
            data[:, ci, i, j] = arr

    t = time.time() - t0
    print(
        f"[grid] built tensor shape={data.shape} ocean_cells={ocean_mask.sum()} "
        f"in {t:.1f}s"
    )
    return {
        "data": data,
        "ocean_mask": ocean_mask,
        "time": np.asarray(MASTER_TIME, dtype="datetime64[ns]"),
        "lat": np.asarray(MASTER_LAT, dtype=np.float32),
        "lon": np.asarray(MASTER_LON, dtype=np.float32),
        "var_names": np.asarray(SEQUENCE_VARS),
    }


def save_grid(bundle: dict, path: Path = GRID_CACHE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **bundle)
    mb = path.stat().st_size / 1e6
    print(f"[grid] saved {path} ({mb:.1f} MB)")


def load_grid(path: Path = GRID_CACHE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/prepare_tensors.py"
        )
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}
