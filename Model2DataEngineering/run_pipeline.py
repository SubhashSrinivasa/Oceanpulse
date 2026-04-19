#!/usr/bin/env python
"""OceanPulse — Species Classification Training Dataset Pipeline (Model 2).

Orchestrates the full pipeline:

  Stage 1  — Download oceanographic data (ERDDAP SST + chlorophyll)
  Stage 2  — Download SSH (CMEMS DUACS)
  Stage 3  — Download / build Argo salinity + GLORYS12 fallback
  Stage 4  — Download WOA18 dissolved oxygen climatology
  Stage 5  — Download CalCOFI species (larvae) + in-situ bottle features
  Stage 6  — Download iNaturalist species (GBIF)
  Stage 7  — Download NOAA WCGBT trawl species (presences + absences)
  Stage 8  — Build feature store: LEFT JOIN ocean features onto species obs
  Stage 9  — Apply feature patching (CalCOFI in-situ fills gridded gaps)
  Stage 10 — Size control: estimate CSV size, downsample if > 5 GB
  Stage 11 — Export training_data.csv

Run:
  python run_pipeline.py [--skip-download] [--force-rebuild]

Flags:
  --skip-download   Skip download stages; use only cached files.
  --force-rebuild   Ignore caches and rebuild feature store and species tables.

CMEMS authentication:
  Set CMEMS_USERNAME and CMEMS_PASSWORD environment variables before running,
  or run ``copernicusmarine login`` once to store credentials persistently.

Output:
  outputs/training_data.csv  — flat (time, lat, lon, features, species, presence)
  logs/pipeline.log          — full pipeline log
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path so pipeline package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.config import (
    CALCOFI_INSITU_CACHE,
    FEATURE_JOINED_CACHE,
    TRAINING_CSV_PATH,
    YEARS,
)
from pipeline.download_cmems import fetch_ssh, fetch_woa18_dissolved_oxygen
from pipeline.download_erddap import fetch_chlorophyll, fetch_sst
from pipeline.export_dataset import export_training_csv
from pipeline.join_species import build_training_table, pivot_to_wide_format
from pipeline.process_calcofi import SPECIES_CACHE as CALCOFI_SP_CACHE
from pipeline.process_inaturalist import INAT_CACHE
from pipeline.process_noaa_trawl import TRAWL_CACHE
from pipeline.size_control import apply_size_control, estimate_csv_size_gb
from pipeline.utils import get_logger

log = get_logger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OceanPulse Model 2 — species dataset pipeline")
    p.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download stages; only use cached files.",
    )
    p.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Delete intermediate caches and rebuild from scratch.",
    )
    return p.parse_args()


def clear_caches() -> None:
    """Remove intermediate caches so the pipeline rebuilds from raw data."""
    caches = [
        FEATURE_JOINED_CACHE,
        CALCOFI_SP_CACHE,
        CALCOFI_INSITU_CACHE,
        INAT_CACHE,
        TRAWL_CACHE,
    ]
    for c in caches:
        if c.exists():
            c.unlink()
            log.info("Cleared cache: %s", c.name)


def run(skip_download: bool = False, force_rebuild: bool = False) -> None:
    t0 = time.time()
    log.info("=" * 60)
    log.info("OceanPulse Model 2 — Species Classification Dataset Pipeline")
    log.info("Years:  %d – %d", YEARS[0], YEARS[-1])
    log.info("Output: %s", TRAINING_CSV_PATH)
    log.info("=" * 60)

    if force_rebuild:
        log.info("--force-rebuild: clearing intermediate caches")
        clear_caches()

    # ------------------------------------------------------------------
    # Stages 1-4: Ocean data downloads — all run concurrently
    # ------------------------------------------------------------------
    if not skip_download:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        ocean_tasks = {
            "SST (OISST)":          fetch_sst,
            "Chlorophyll (MODIS)":  fetch_chlorophyll,
            "SSH (CMEMS)":          fetch_ssh,
            "Dissolved O2 (WOA18)": fetch_woa18_dissolved_oxygen,
        }

        log.info("--- Stages 1-4: Downloading all ocean datasets in parallel ---")
        with ThreadPoolExecutor(max_workers=len(ocean_tasks)) as executor:
            futures = {executor.submit(fn): name for name, fn in ocean_tasks.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    future.result()
                    log.info("%s download complete", name)
                except Exception as exc:  # noqa: BLE001
                    log.error("%s download failed: %s — continuing with cached/NaN data", name, exc)
    else:
        log.info("--skip-download: skipping ocean data downloads")

    # ------------------------------------------------------------------
    # Stages 5-7: Species data — all three sources run concurrently
    # ------------------------------------------------------------------
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pipeline.process_calcofi import fetch_calcofi_species
    from pipeline.process_inaturalist import fetch_inaturalist_species
    from pipeline.process_noaa_trawl import fetch_noaa_trawl_species

    species_tasks = {
        "CalCOFI":      fetch_calcofi_species,
        "iNaturalist":  fetch_inaturalist_species,
        "NOAA trawl":   fetch_noaa_trawl_species,
    }

    log.info("--- Stages 5-7: Downloading all species datasets in parallel ---")
    with ThreadPoolExecutor(max_workers=len(species_tasks)) as executor:
        futures = {executor.submit(fn): name for name, fn in species_tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                df = future.result()
                log.info("%s complete: %d rows", name, len(df))
            except Exception as exc:  # noqa: BLE001
                log.error("%s failed: %s", name, exc)

    # ------------------------------------------------------------------
    # Stage 8-9: Build training table (feature store + join + patching)
    # ------------------------------------------------------------------
    log.info("--- Stage 8-9: Feature store + species join + patching ---")
    try:
        long_df = build_training_table()
    except Exception as exc:  # noqa: BLE001
        log.error("Training table build failed: %s", exc)
        raise

    log.info("Long-format table: %d rows, %d columns", len(long_df), len(long_df.columns))

    # ------------------------------------------------------------------
    # Stage 10: Size control (applied on long format before pivot)
    # ------------------------------------------------------------------
    log.info("--- Stage 10: Size control ---")
    est_gb = estimate_csv_size_gb(long_df)
    log.info("Estimated size before control: %.3f GB", est_gb)
    long_df = apply_size_control(long_df)

    # ------------------------------------------------------------------
    # Stage 10b: Pivot to wide format
    # ------------------------------------------------------------------
    log.info("--- Stage 10b: Pivot long -> wide (one row per observation, one col per species) ---")
    training_df = pivot_to_wide_format(long_df)
    del long_df

    # ------------------------------------------------------------------
    # Stage 11: Export
    # ------------------------------------------------------------------
    log.info("--- Stage 11: Export CSV ---")
    out_path = export_training_csv(training_df)

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("Pipeline complete in %.1f s", elapsed)
    log.info("Output: %s", out_path)
    log.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run(skip_download=args.skip_download, force_rebuild=args.force_rebuild)
