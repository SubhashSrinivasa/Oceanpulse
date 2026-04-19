"""Write the final wide-format training dataset as CSV.

Output: outputs/training_data.csv

Column order:
  time, lat, lon,
  sst, sst_anomaly, chlorophyll, salinity, dissolved_oxygen, ssh,
  day_of_year, month,
  <species_1>, <species_2>, ..., <species_N>   (alphabetical, binary 0/1)

Written in chunks to avoid peak memory usage on large DataFrames.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FIXED_COLS, TRAINING_CSV_PATH
from .utils import get_logger

log = get_logger("export_dataset")

CHUNK_SIZE = 500_000  # rows written per chunk


def export_training_csv(df: pd.DataFrame, path: Path = TRAINING_CSV_PATH) -> Path:
    """Write wide-format *df* to *path* as CSV only. Returns the output path."""
    # Fixed columns first, then any remaining species columns (already alphabetical)
    fixed_present = [c for c in FIXED_COLS if c in df.columns]
    species_cols = [c for c in df.columns if c not in FIXED_COLS]
    ordered_cols = fixed_present + sorted(species_cols)
    df = df[ordered_cols].copy()

    if pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = df["time"].dt.strftime("%Y-%m-%d")

    log.info("Writing %d rows × %d columns to %s ...", len(df), len(df.columns), path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_chunks = max(1, len(df) // CHUNK_SIZE + (1 if len(df) % CHUNK_SIZE else 0))
    for i in range(n_chunks):
        chunk = df.iloc[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
        chunk.to_csv(
            path,
            mode="w" if i == 0 else "a",
            header=(i == 0),
            index=False,
            float_format="%.6g",
        )
        if n_chunks > 1:
            log.info("  chunk %d/%d written (%d rows)", i + 1, n_chunks, len(chunk))

    size_mb = path.stat().st_size / 1e6
    size_gb = size_mb / 1e3

    log.info("=== Export complete ===")
    log.info("  File:          %s", path)
    log.info("  Size:          %.1f MB (%.3f GB)", size_mb, size_gb)
    log.info("  Rows:          %d  (unique time×lat×lon observations)", len(df))
    log.info("  Fixed cols:    %d", len(fixed_present))
    log.info("  Species cols:  %d", len(species_cols))
    log.info("  Total cols:    %d", len(df.columns))

    for col in fixed_present:
        nan_pct = 100 * df[col].isna().mean()
        if nan_pct > 0:
            log.info("  NaN %s: %.2f%%", col, nan_pct)

    if len(species_cols):
        avg_presence = df[species_cols].mean().mean() * 100
        log.info("  Avg presence rate across species: %.2f%%", avg_presence)

    if size_gb > 5.0:
        log.error(
            "WARNING: output is %.3f GB — exceeds 5 GB limit. "
            "Re-run with stricter size_control settings.",
            size_gb,
        )
    elif size_gb < 1.0:
        log.info(
            "Note: output is %.3f GB (below 1 GB target). "
            "Normal if species observation density is low for this region/period.",
            size_gb,
        )

    return path
