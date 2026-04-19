"""Join ocean features onto species observations, patch features, and pivot to wide format.

JOIN STRATEGY:
  - Species observations are the PRIMARY (left) table.
  - Ocean features are LEFT JOINed using (time, lat, lon) as the key.
  - Missing feature values are filled from CalCOFI in-situ measurements.

WIDE OUTPUT FORMAT:
  Each row = one unique (time, lat, lon) observation.
  One binary column per species: 1 = present, 0 = not observed.
  Fixed columns come first, species columns follow alphabetically.

  time, lat, lon, sst, sst_anomaly, chlorophyll, salinity, dissolved_oxygen,
  ssh, day_of_year, month, <species_1>, <species_2>, ..., <species_N>
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLS, FIXED_COLS, LONG_COLS
from .feature_store import build_feature_store
from .load_species_data import load_all_species
from .process_calcofi import fetch_calcofi_insitu
from .utils import get_logger

log = get_logger("join_species")


# ---------------------------------------------------------------------------
# Feature patching from CalCOFI in-situ
# ---------------------------------------------------------------------------

def _patch_from_insitu(df: pd.DataFrame, insitu: pd.DataFrame) -> pd.DataFrame:
    """Fill missing feature values using CalCOFI in-situ measurements.

    For each row with NaN features, searches insitu for measurements at the
    same snapped grid point within ±3 days. Takes the mean of any matches.
    """
    if insitu.empty:
        return df

    patchable = ["sst", "salinity", "dissolved_oxygen"]
    insitu_cols = [c for c in patchable if c in insitu.columns]
    if not insitu_cols:
        return df

    df = df.copy()
    insitu = insitu.copy()
    insitu["time"] = pd.to_datetime(insitu["time"])

    # Build lookup dict: (time, lat, lon) → {feature: value}
    insitu_idx = insitu.set_index(["time", "lat", "lon"])

    window = pd.Timedelta(days=3)
    n_patched = 0

    for col in insitu_cols:
        missing_mask = df[col].isna()
        if not missing_mask.any():
            continue

        missing_rows = df[missing_mask]
        for idx, row in missing_rows.iterrows():
            t, lat, lon = row["time"], row["lat"], row["lon"]
            nearby = insitu[
                (np.abs(insitu["time"] - t) <= window)
                & (insitu["lat"] == lat)
                & (insitu["lon"] == lon)
                & insitu[col].notna()
            ]
            if len(nearby) > 0:
                df.at[idx, col] = float(nearby[col].mean())
                n_patched += 1

    log.info("Feature patching: %d values filled from CalCOFI in-situ", n_patched)
    return df


# ---------------------------------------------------------------------------
# Main join
# ---------------------------------------------------------------------------

def build_training_table() -> pd.DataFrame:
    """Build the full training table: species × ocean features, flat rows.

    Each row is one independent observation:
      (time, lat, lon, features..., metadata..., species, presence)

    No sequences, no aggregation, no time-series windows.
    """
    # 1. Load species observations (primary table)
    log.info("Loading species observations...")
    species_df = load_all_species()
    log.info("Species table: %d rows", len(species_df))

    # 2. Build/load feature store (LEFT JOIN ocean features onto species obs)
    log.info("Building feature store (left-joining ocean features)...")
    joined = build_feature_store(species_df)
    log.info("After feature join: %d rows", len(joined))

    # 3. Feature patching from CalCOFI in-situ (runs BEFORE median imputation
    #    so real in-situ measurements take priority over synthetic medians).
    log.info("Loading CalCOFI in-situ for feature patching...")
    insitu = fetch_calcofi_insitu()
    if len(insitu) > 0:
        joined = _patch_from_insitu(joined, insitu)
    else:
        log.info("No CalCOFI in-situ data available — skipping patching")

    # 3b. Guarantee zero NaN: median-impute any remaining per-feature NaN
    #     (cloud cover in chlorophyll, coastal gaps in WOA18 DO, etc.).
    for col in FEATURE_COLS:
        if col not in joined.columns:
            continue
        joined[col] = pd.to_numeric(joined[col], errors="coerce")
        n_nan = int(joined[col].isna().sum())
        if n_nan:
            med = float(joined[col].median())
            if not np.isfinite(med):
                med = 0.0
            joined[col] = joined[col].fillna(med)
            log.info("Filled %d NaN in %s with median=%.3f", n_nan, col, med)

    # 4. Add metadata columns
    joined["time"] = pd.to_datetime(joined["time"])
    joined["day_of_year"] = joined["time"].dt.dayofyear.astype("int16")
    joined["month"] = joined["time"].dt.month.astype("int8")

    # 5. Select and order intermediate long-format columns
    for c in LONG_COLS:
        if c not in joined.columns:
            log.warning("Column missing (will be NaN): %s", c)
            joined[c] = np.nan

    result = joined[LONG_COLS].copy()

    # 6. Type enforcement
    result["time"] = result["time"].dt.date.astype(str)
    for col in FEATURE_COLS:
        result[col] = pd.to_numeric(result[col], errors="coerce").astype("float32")
    result["day_of_year"] = result["day_of_year"].astype("int16")
    result["month"] = result["month"].astype("int8")
    result["presence"] = result["presence"].astype("int8")
    result["species"] = result["species"].astype(str)

    # 7. Log long-format stats
    log.info("=== Long-format table (pre-pivot) ===")
    log.info("  Rows:    %d", len(result))
    log.info("  Species: %d unique", result["species"].nunique())
    log.info("  Presence rate: %.1f%%", 100 * (result["presence"] == 1).mean())
    log.info("  Time range: %s — %s", result["time"].min(), result["time"].max())
    for col in FEATURE_COLS:
        nan_pct = 100 * result[col].isna().mean()
        if nan_pct:
            log.info("    %s NaN: %.1f%%", col, nan_pct)

    return result


# ---------------------------------------------------------------------------
# Pivot long → wide
# ---------------------------------------------------------------------------

def pivot_to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long-format training table to one row per (time, lat, lon).

    Each species becomes a binary column: 1 = present/caught, 0 = not observed.
    The fixed feature columns are deduplicated and merged back onto the pivot.

    Column order:
      time, lat, lon, sst, sst_anomaly, chlorophyll, salinity,
      dissolved_oxygen, ssh, day_of_year, month,
      <species_1>, <species_2>, ..., <species_N>   (alphabetical)
    """
    log.info("Pivoting to wide format (%d long rows, %d species)...",
             len(df), df["species"].nunique())

    # Pivot: (time, lat, lon) × species → presence value
    # aggfunc='max' so that presence=1 from any source wins over 0
    pivot = df.pivot_table(
        index=["time", "lat", "lon"],
        columns="species",
        values="presence",
        aggfunc="max",
        fill_value=0,
    ).reset_index()
    pivot.columns.name = None  # drop the "species" axis label

    # Extract one feature row per (time, lat, lon)
    feat_cols = [c for c in FIXED_COLS if c in df.columns]
    features = (
        df[feat_cols]
        .drop_duplicates(subset=["time", "lat", "lon"])
        .reset_index(drop=True)
    )

    # Merge features onto pivot
    result = features.merge(pivot, on=["time", "lat", "lon"], how="inner")

    # Reorder: fixed cols first, species cols alphabetically
    species_cols = sorted(c for c in result.columns if c not in FIXED_COLS)
    result = result[feat_cols + species_cols]

    # Cast species columns to int8 (0/1)
    result[species_cols] = result[species_cols].astype("int8")

    log.info("=== Wide-format table ===")
    log.info("  Rows:          %d  (unique time×lat×lon observations)", len(result))
    log.info("  Species cols:  %d", len(species_cols))
    log.info("  Total columns: %d", len(result.columns))

    return result
