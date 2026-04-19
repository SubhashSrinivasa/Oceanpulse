"""Quick analysis of ocean_cube_long.csv and ocean_cube_sequences.csv."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from pipeline.config import OUTPUT_DIR

LONG_PATH = OUTPUT_DIR / "ocean_cube_long.csv"
SEQ_PATH = OUTPUT_DIR / "ocean_cube_sequences.csv"
VAR_COLS = ["sst", "sst_anomaly", "ssh", "chlorophyll", "chlorophyll_log",
            "mld", "mld_source", "salinity"]


def analyse_long() -> None:
    print("\n" + "=" * 60)
    print("LONG FORMAT  —  ocean_cube_long.csv")
    print("=" * 60)
    df = pd.read_csv(LONG_PATH, parse_dates=["time"])
    print(f"Rows: {len(df):,}   Columns: {len(df.columns)}")
    print(f"Date range: {df['time'].min().date()} → {df['time'].max().date()}")
    print(f"Unique locations (lat,lon): {df[['lat','lon']].drop_duplicates().shape[0]}")
    print()

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


def analyse_sequences() -> None:
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
    analyse_long()
    analyse_sequences()
    print()
