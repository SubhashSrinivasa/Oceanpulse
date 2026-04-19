"""Quick analysis of ocean_cube_long.csv and ocean_cube_sequences.csv.

Also reports land vs sea breakdown of the master grid (the writers drop
land cells; for sequences, sea cells can also be dropped by the DROP rule).
"""
from __future__ import annotations

from ensure_deps import ensure_scientific_stack

ensure_scientific_stack()

import json
import numpy as np
import pandas as pd

from export_csv import DROP_THRESHOLD
from pipeline.config import MASTER_LAT, MASTER_LON, MASTER_TIME, OUTPUT_DIR

LONG_PATH = OUTPUT_DIR / "ocean_cube_long.csv"
SEQ_PATH = OUTPUT_DIR / "ocean_cube_sequences.csv"
VAR_COLS = ["sst", "sst_anomaly", "ssh", "chlorophyll", "chlorophyll_log",
            "mld", "mld_source", "salinity"]

N_LAT = len(MASTER_LAT)
N_LON = len(MASTER_LON)
N_TIME = len(MASTER_TIME)
TOTAL_CELLS = N_LAT * N_LON
TOTAL_GRID_VALUES = TOTAL_CELLS * N_TIME


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{100.0 * num / den:.2f}%"


def _land_sea_long(df: pd.DataFrame) -> set[tuple[float, float]]:
    """Report land/sea breakdown for the long CSV.

    The long writer drops rows with sst == NaN (land). So unique (lat, lon)
    in the file is exactly the ocean mask from stage 6 (cells where SST is
    not all-NaN across time).
    """
    sea_pairs = set(map(tuple, df[["lat", "lon"]].drop_duplicates().to_numpy()))
    n_sea_cells = len(sea_pairs)
    n_land_cells = TOTAL_CELLS - n_sea_cells

    n_sea_rows = len(df)
    n_land_rows = TOTAL_GRID_VALUES - n_sea_rows
    print("Land / sea breakdown (long CSV):")
    print(
        f"  master grid: {N_LAT} lat x {N_LON} lon = {TOTAL_CELLS:,} cells "
        f"x {N_TIME} days = {TOTAL_GRID_VALUES:,} total grid values"
    )
    print(
        f"  sea cells kept:  {n_sea_cells:>8,}  ({_pct(n_sea_cells, TOTAL_CELLS)} of grid)"
    )
    print(
        f"  land cells dropped: {n_land_cells:>5,}  ({_pct(n_land_cells, TOTAL_CELLS)} of grid)"
    )
    print(
        f"  sea rows (CSV):  {n_sea_rows:>8,}  ({_pct(n_sea_rows, TOTAL_GRID_VALUES)} of grid values)"
    )
    print(
        f"  land rows dropped: {n_land_rows:>7,}  ({_pct(n_land_rows, TOTAL_GRID_VALUES)} of grid values)"
    )
    print()
    return sea_pairs


def analyse_long() -> set[tuple[float, float]] | None:
    if not LONG_PATH.exists():
        print(f"(skipping long analysis — {LONG_PATH} not found)")
        return None
    print("\n" + "=" * 60)
    print("LONG FORMAT  —  ocean_cube_long.csv")
    print("=" * 60)
    df = pd.read_csv(LONG_PATH, parse_dates=["time"])
    print(f"Rows: {len(df):,}   Columns: {len(df.columns)}")
    print(f"Date range: {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"Unique locations (lat,lon): {df[['lat','lon']].drop_duplicates().shape[0]}")
    print()

    sea_pairs = _land_sea_long(df)

    stats = []
    for col in VAR_COLS:
        if col not in df.columns:
            continue
        s = df[col]
        null_n = s.isna().sum()
        stats.append({
            "variable": col,
            "null_count": null_n,
            "null_%": round(100 * null_n / len(df), 2),
            "mean": round(s.mean(), 4),
            "std": round(s.std(), 4),
            "min": round(s.min(), 4),
            "max": round(s.max(), 4),
        })
    print(pd.DataFrame(stats).to_string(index=False))
    return sea_pairs


def analyse_sequences(long_sea_pairs: set[tuple[float, float]] | None) -> None:
    if not SEQ_PATH.exists():
        print(f"(skipping sequence analysis — {SEQ_PATH} not found)")
        return
    print("\n" + "=" * 60)
    print("SEQUENCE FORMAT  —  ocean_cube_sequences.csv")
    print("=" * 60)
    df = pd.read_csv(SEQ_PATH, comment="#")
    n_locations = len(df)
    print(f"Locations (rows): {n_locations}")

    # Read header comment for metadata
    with open(SEQ_PATH) as f:
        header = f.readline().strip()
    print(f"Metadata: {header.lstrip('# ')}")
    print()

    # ---- Land / sea breakdown for sequences ----
    seq_pairs = set(map(tuple, df[["lat", "lon"]].drop_duplicates().to_numpy()))
    print("Land / sea breakdown (sequences CSV):")
    print(
        f"  master grid: {TOTAL_CELLS:,} cells "
        f"({N_LAT} lat x {N_LON} lon)"
    )
    thresh_pct = int(round(DROP_THRESHOLD * 100))
    print(f"  kept (passed {thresh_pct}% rule): {len(seq_pairs):>6,}  "
          f"({_pct(len(seq_pairs), TOTAL_CELLS)} of grid)")

    if long_sea_pairs is not None:
        n_land = TOTAL_CELLS - len(long_sea_pairs)
        dropped_by_thresh = len(long_sea_pairs - seq_pairs)
        print(
            f"  land cells (from long):  {n_land:>6,}  "
            f"({_pct(n_land, TOTAL_CELLS)} of grid)"
        )
        print(
            f"  sea cells dropped by {thresh_pct}% rule: {dropped_by_thresh:>4,}  "
            f"({_pct(dropped_by_thresh, len(long_sea_pairs))} of sea cells)"
        )
    else:
        print(f"  (long CSV not available; cannot split land vs {thresh_pct}%-dropped)")
    print()

    stats = []
    for col in VAR_COLS:
        if col not in df.columns:
            continue
        arrays = df[col].apply(json.loads).tolist()
        seq_len = len(arrays[0]) if arrays else 0

        null_arrays = sum(1 for a in arrays if all(v is None for v in a))
        null_values = sum(v is None for a in arrays for v in a)
        total_values = n_locations * seq_len
        avg_nulls_per_array = null_values / n_locations if n_locations else 0

        flat = [v for a in arrays for v in a if v is not None]
        stats.append({
            "variable": col,
            "all-null arrays": null_arrays,
            "total null values": null_values,
            "null_%": round(100 * null_values / total_values, 2),
            "avg nulls/array": round(avg_nulls_per_array, 1),
            "mean": round(np.mean(flat), 4) if flat else float("nan"),
            "std": round(np.std(flat), 4) if flat else float("nan"),
            "min": round(np.min(flat), 4) if flat else float("nan"),
            "max": round(np.max(flat), 4) if flat else float("nan"),
        })
    print(pd.DataFrame(stats).to_string(index=False))
    print()


if __name__ == "__main__":
    sea_pairs = analyse_long()
    analyse_sequences(sea_pairs)
    print()
