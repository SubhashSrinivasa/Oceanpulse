"""Dispatcher: load, combine, and quality-gate all species observation datasets.

Calls:
  1. process_calcofi.fetch_calcofi_species()      – CalCOFI larvae (presence=1)
  2. process_inaturalist.fetch_inaturalist_species() – iNaturalist (presence=1)
  3. process_noaa_trawl.fetch_noaa_trawl_species()   – NOAA WCGBT (presence 0/1)

Returns a single DataFrame with columns:
  time, lat, lon, species, presence, source

Rules:
  - CalCOFI + iNaturalist: presence = 1 only (no confirmed absences recorded)
  - NOAA trawl:            presence = 1 (caught) or 0 (confirmed not caught)
  - Species with fewer than MIN_SPECIES_OBS total observations are dropped
  - All coordinates already snapped to master grid by each processor
"""
from __future__ import annotations

import pandas as pd

from .config import MIN_SPECIES_OBS
from .process_calcofi import fetch_calcofi_species
from .process_inaturalist import fetch_inaturalist_species
from .process_noaa_trawl import fetch_noaa_trawl_species
from .utils import get_logger

log = get_logger("load_species")


def load_all_species() -> pd.DataFrame:
    """Return combined species observation table from all three sources."""
    frames = []

    log.info("=== Loading CalCOFI species ===")
    calcofi = fetch_calcofi_species()
    if len(calcofi) > 0:
        frames.append(calcofi)
        log.info("CalCOFI: %d rows, %d species", len(calcofi), calcofi["species"].nunique())
    else:
        log.warning("CalCOFI returned no data")

    log.info("=== Loading iNaturalist species ===")
    inat = fetch_inaturalist_species()
    if len(inat) > 0:
        frames.append(inat)
        log.info("iNaturalist: %d rows, %d species", len(inat), inat["species"].nunique())
    else:
        log.warning("iNaturalist returned no data")

    log.info("=== Loading NOAA WCGBT trawl species ===")
    trawl = fetch_noaa_trawl_species()
    if len(trawl) > 0:
        frames.append(trawl)
        log.info("NOAA trawl: %d rows, %d species", len(trawl), trawl["species"].nunique())
    else:
        log.warning("NOAA trawl returned no data")

    if not frames:
        raise RuntimeError(
            "All three species datasets returned empty — "
            "cannot build training dataset without species labels."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"]).dt.normalize()
    combined["lat"] = combined["lat"].astype("float32")
    combined["lon"] = combined["lon"].astype("float32")
    combined["presence"] = combined["presence"].astype("int8")

    log.info(
        "Combined species: %d rows | %d unique species | %d sources",
        len(combined),
        combined["species"].nunique(),
        combined["source"].nunique(),
    )

    # Drop rarely-observed species (too few samples for reliable ML training)
    species_counts = combined.groupby("species")["presence"].count()
    valid_species = species_counts[species_counts >= MIN_SPECIES_OBS].index
    n_before = combined["species"].nunique()
    combined = combined[combined["species"].isin(valid_species)].copy()
    n_after = combined["species"].nunique()
    log.info(
        "Species filter (min %d obs): %d -> %d species kept",
        MIN_SPECIES_OBS, n_before, n_after,
    )

    # Deduplicate: if same (time, lat, lon, species) appears in multiple sources,
    # keep presence=1 over presence=0 (positive observation wins)
    combined = (
        combined.sort_values("presence", ascending=False)
        .drop_duplicates(subset=["time", "lat", "lon", "species"])
        .reset_index(drop=True)
    )

    log.info(
        "Final species table: %d rows | %d species | presence rate %.1f%%",
        len(combined),
        combined["species"].nunique(),
        100 * (combined["presence"] == 1).mean(),
    )

    _log_source_breakdown(combined)
    return combined


def _log_source_breakdown(df: pd.DataFrame) -> None:
    log.info("Source breakdown:")
    for src, grp in df.groupby("source"):
        pres_pct = 100 * (grp["presence"] == 1).mean()
        log.info(
            "  %s: %d rows, %d species, %.1f%% presences",
            src, len(grp), grp["species"].nunique(), pres_pct,
        )
