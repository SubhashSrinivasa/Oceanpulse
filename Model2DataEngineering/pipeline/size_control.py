"""Dataset size estimation and automatic downscaling.

Target: 1 GB – 5 GB final CSV.
Hard cap: 5 GB (MUST NOT exceed).

Downscaling cascade (applied in order until size ≤ MAX_SIZE_GB):

  Step 1 — Temporal downsampling
    Keep observations where day-of-year % stride == 0.
    Increases stride from 2 → 3 → 5 → 7 days.

  Step 2 — Species filtering
    Keep only the top-N species by total observation count.
    Reduces N from 50 → 40 → 30 → 20.

  Step 3 — Spatial coarsening (last resort)
    Round lat/lon to 0.5 deg resolution, deduplicate.

At each step the estimated size is re-checked before proceeding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MAX_SIZE_GB, MAX_SPECIES_TOP_N, TEMPORAL_STRIDE_DAYS
from .utils import get_logger

log = get_logger("size_control")

# Bytes-per-character estimate for CSV: includes quotes, commas, newlines
_BYTES_PER_CELL = 15


# ---------------------------------------------------------------------------
# Size estimation
# ---------------------------------------------------------------------------

def estimate_csv_size_gb(df: pd.DataFrame) -> float:
    """Conservative estimate of the CSV file size in GB.

    Uses 15 bytes per cell (covers floats with decimals, strings, delimiters).
    Actual size may be 10-20% lower after pandas writes optimised CSV.
    """
    n_rows, n_cols = df.shape
    return (n_rows * n_cols * _BYTES_PER_CELL) / 1e9


def log_dataset_stats(df: pd.DataFrame, label: str = "") -> float:
    size_gb = estimate_csv_size_gb(df)
    prefix = f"[{label}] " if label else ""
    log.info(
        "%sDataset stats: %d rows | %d species | %.3f GB estimated",
        prefix,
        len(df),
        df["species"].nunique() if "species" in df.columns else -1,
        size_gb,
    )
    return size_gb


# ---------------------------------------------------------------------------
# Downscaling steps
# ---------------------------------------------------------------------------

def _temporal_downsample(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    """Keep rows where day_of_year % stride == 0."""
    if "day_of_year" not in df.columns:
        df = df.copy()
        df["day_of_year"] = pd.to_datetime(df["time"]).dt.dayofyear
    return df[df["day_of_year"] % stride == 0].copy()


def _filter_top_species(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Keep only the top-n species by observation count."""
    counts = df["species"].value_counts()
    top = counts.head(n).index
    return df[df["species"].isin(top)].copy()


def _coarsen_spatial(df: pd.DataFrame, res: float = 0.5) -> pd.DataFrame:
    """Round lat/lon to coarser resolution and deduplicate."""
    df = df.copy()
    df["lat"] = (df["lat"] / res).round() * res
    df["lon"] = (df["lon"] / res).round() * res
    # After coarsening, keep max-presence for any collision
    if "presence" in df.columns:
        df = (
            df.sort_values("presence", ascending=False)
            .drop_duplicates(subset=["time", "lat", "lon", "species"])
            .reset_index(drop=True)
        )
    else:
        df = df.drop_duplicates(subset=["time", "lat", "lon", "species"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Main controller
# ---------------------------------------------------------------------------

def apply_size_control(df: pd.DataFrame) -> pd.DataFrame:
    """Apply downscaling until the dataset fits within MAX_SIZE_GB.

    Returns the (possibly reduced) DataFrame, logging each step taken.
    Never modifies the input; always returns a copy.
    """
    df = df.copy()
    size_gb = log_dataset_stats(df, "initial")

    if size_gb <= MAX_SIZE_GB:
        log.info("Dataset size %.3f GB is within the %.1f GB limit — no downscaling needed",
                 size_gb, MAX_SIZE_GB)
        return df

    log.warning(
        "Dataset size %.3f GB exceeds %.1f GB limit — starting downscaling cascade",
        size_gb, MAX_SIZE_GB,
    )

    # -----------------------------------------------------------------------
    # Step 1: Temporal downsampling
    # -----------------------------------------------------------------------
    for stride in [2, 3, 5, 7]:
        sampled = _temporal_downsample(df, stride)
        new_size = estimate_csv_size_gb(sampled)
        log.info("Temporal stride=%d: %d rows, %.3f GB", stride, len(sampled), new_size)
        if new_size <= MAX_SIZE_GB:
            log.info("Step 1 resolved: keeping every %d days", stride)
            return log_and_return(sampled, "after_temporal_downsample")
        df = sampled  # carry forward the smallest temporal stride

    # -----------------------------------------------------------------------
    # Step 2: Species filtering
    # -----------------------------------------------------------------------
    for top_n in [50, 40, 30, 20]:
        filtered = _filter_top_species(df, top_n)
        new_size = estimate_csv_size_gb(filtered)
        log.info("Top-%d species: %d rows, %.3f GB", top_n, len(filtered), new_size)
        if new_size <= MAX_SIZE_GB:
            log.info("Step 2 resolved: keeping top %d species", top_n)
            return log_and_return(filtered, f"after_top{top_n}_species")
        df = filtered

    # -----------------------------------------------------------------------
    # Step 3: Spatial coarsening (last resort)
    # -----------------------------------------------------------------------
    log.warning("Applying spatial coarsening to 0.5 deg resolution")
    coarsened = _coarsen_spatial(df, res=0.5)
    new_size = estimate_csv_size_gb(coarsened)
    log.info("After spatial coarsening: %d rows, %.3f GB", len(coarsened), new_size)

    if new_size > MAX_SIZE_GB:
        log.error(
            "Dataset still %.3f GB after all downscaling steps — "
            "proceeding but size constraint not fully met. "
            "Consider reducing YEARS range in config.py.",
            new_size,
        )
    return log_and_return(coarsened, "after_spatial_coarsen")


def log_and_return(df: pd.DataFrame, label: str) -> pd.DataFrame:
    log_dataset_stats(df, label)
    return df
