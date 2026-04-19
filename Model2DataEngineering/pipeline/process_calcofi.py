"""Process CalCOFI datasets from ERDDAP.

Two products are used:

1. **CalCOFI Larval Fish Counts** (siocalcofiLarvalCounts or fallback IDs)
   - Species observations: presence = 1
   - Provides long-term (since 1949) ichthyoplankton data

2. **CalCOFI Hydro Bottle** (siocalcofiHydroBottle)
   - In-situ oceanographic measurements at survey stations
   - Used for *feature patching*: fills SST, salinity, DO where gridded
     products are missing at species observation locations

Returns:
  - species_df: (time, lat, lon, species, presence, source)
  - insitu_df:  (time, lat, lon, sst, salinity, dissolved_oxygen)
"""
from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd
import requests

from .config import (
    CACHE_DIR,
    CALCOFI_INSITU_CACHE,
    ERDDAP_BASE_TABLEDAP,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    TIME_END,
    TIME_START,
)
from .utils import get_logger, snap_to_grid, standardize_species_name

log = get_logger("process_calcofi")

SPECIES_CACHE = CACHE_DIR / "calcofi_species.parquet"

# ERDDAP servers to try, in order
ERDDAP_SERVERS = [
    "https://coastwatch.pfeg.noaa.gov/erddap/tabledap",
    "https://erddap.calcofi.io/erddap/tabledap",
]

# Known larvae dataset IDs to try on coastwatch
LARVAE_DATASET_IDS = [
    "siocalcofiLarvalCounts",
    "erdCalCOFIlrvcntpos",
    "erdCalCOFIlrvcnt",
    "CalCOFI_larvae_counts",
    "siocalcofiLarvaeCounts",
]
BOTTLE_DATASET_ID = "siocalcofiHydroBottle"

# Possible column name variants
LAT_COLS = ["latitude", "lat", "Latitude"]
LON_COLS = ["longitude", "lon", "Longitude"]
TIME_COLS = ["time", "Time", "date"]
SPECIES_COLS = [
    "scientific_name", "scientificname", "ScientificName",
    "fish_species", "species", "taxa", "taxon",
]
COUNT_COLS = [
    "total_larvae_100m2", "larvae_count", "count", "total_count",
    "larvae_10m2", "tot_larvae",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _erddap_csv(
    dataset_id: str, query: str, timeout: int = 300, base_url: str | None = None
) -> pd.DataFrame | None:
    """Query ERDDAP tabledap, skip the units row, return DataFrame or None."""
    base = base_url or ERDDAP_BASE_TABLEDAP
    url = f"{base}/{dataset_id}.csv{query}"
    log.info("CalCOFI ERDDAP query: %s", url[:120])
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            log.warning("HTTP %s from %s/%s", r.status_code, base.split("/")[-2], dataset_id)
            return None
        df = pd.read_csv(StringIO(r.text), skiprows=[1], low_memory=False)
        log.info("  -> %d rows from %s", len(df), dataset_id)
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("ERDDAP query error for %s: %s", dataset_id, exc)
        return None


def _discover_erddap_larvae(server_tabledap: str) -> list[str]:
    """Search an ERDDAP server for larvae/CalCOFI tabledap datasets."""
    server_base = server_tabledap.replace("/tabledap", "")
    try:
        r = requests.get(
            f"{server_base}/search/index.json",
            params={"searchFor": "calcofi larvae", "page": 1, "itemsPerPage": 20},
            timeout=20,
        )
        if r.status_code == 200:
            rows = r.json().get("table", {}).get("rows", [])
            return [row[0] for row in rows if row and "larv" in str(row).lower()]
    except Exception:  # noqa: BLE001
        pass
    return []


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first matching column name from a list of candidates."""
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols:
            return cols[c.lower()]
    return None


def _bbox_constraints() -> str:
    return (
        f"&time>={TIME_START}T00:00:00Z"
        f"&time<={TIME_END}T23:59:59Z"
        f"&latitude>={LAT_MIN}&latitude<={LAT_MAX}"
        f"&longitude>={LON_MIN}&longitude<={LON_MAX}"
    )


# ---------------------------------------------------------------------------
# Larval fish species observations
# ---------------------------------------------------------------------------

def _fetch_larvae(dataset_id: str, base_url: str) -> pd.DataFrame | None:
    """Try fetching larvae from a specific dataset ID on a specific ERDDAP server."""
    server_base = base_url.replace("/tabledap", "")
    info_url = f"{server_base}/info/{dataset_id}/index.json"
    available_vars: list[str] = []
    try:
        r = requests.get(info_url, timeout=30)
        if r.status_code != 200:
            return None
        rows = r.json().get("table", {}).get("rows", [])
        # rows format: [row_type, variable_name, attribute_name, data_type, value]
        available_vars = [row[1] for row in rows if row and row[0] == "variable"]
    except Exception:  # noqa: BLE001
        available_vars = []

    # Build query with available columns
    want = ["time", "latitude", "longitude"]
    for cand in SPECIES_COLS:
        if cand in available_vars:
            want.append(cand)
            break
    for cand in COUNT_COLS:
        if cand in available_vars:
            want.append(cand)
            break

    if len(want) < 4:
        query = "?" + _bbox_constraints()
    else:
        query = "?" + ",".join(want) + _bbox_constraints()

    return _erddap_csv(dataset_id, query, base_url=base_url)


def fetch_calcofi_species() -> pd.DataFrame:
    """Download and standardise CalCOFI larval fish species observations."""
    if SPECIES_CACHE.exists():
        log.info("CalCOFI species cache hit")
        return pd.read_parquet(SPECIES_CACHE)

    # Build candidate list: known IDs on all servers, then discover extras
    candidates: list[tuple[str, str]] = [
        (did, srv) for did in LARVAE_DATASET_IDS for srv in ERDDAP_SERVERS
    ]
    for srv in ERDDAP_SERVERS:
        for did in _discover_erddap_larvae(srv):
            if (did, srv) not in candidates:
                candidates.append((did, srv))

    raw = None
    for dataset_id, server in candidates:
        log.info("Trying CalCOFI larvae dataset: %s @ %s", dataset_id, server.split("/")[2])
        raw = _fetch_larvae(dataset_id, server)
        if raw is not None and len(raw) > 0:
            log.info("Using CalCOFI dataset: %s (%d rows)", dataset_id, len(raw))
            break

    if raw is None or len(raw) == 0:
        log.error("No CalCOFI larvae data retrieved — returning empty DataFrame")
        return pd.DataFrame(columns=["time", "lat", "lon", "species", "presence", "source"])

    # Normalise column names
    lat_col = _find_col(raw, LAT_COLS)
    lon_col = _find_col(raw, LON_COLS)
    time_col = _find_col(raw, TIME_COLS)
    sp_col = _find_col(raw, SPECIES_COLS)
    cnt_col = _find_col(raw, COUNT_COLS)

    if not all([lat_col, lon_col, time_col, sp_col]):
        log.error("Could not identify required columns in CalCOFI data: %s", list(raw.columns))
        return pd.DataFrame(columns=["time", "lat", "lon", "species", "presence", "source"])

    df = raw[[time_col, lat_col, lon_col, sp_col]].copy()
    if cnt_col:
        df[cnt_col] = raw[cnt_col]
    df = df.rename(columns={time_col: "time", lat_col: "lat", lon_col: "lon", sp_col: "species"})

    # Parse and normalise
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["time", "lat", "lon", "species"])

    # Filter to domain
    df = df[
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)
        & (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    ].copy()

    # CalCOFI + iNaturalist: presence = 1, no confirmed absences
    df["presence"] = 1
    df["source"] = "calcofi"
    df["species"] = df["species"].map(standardize_species_name)
    df = df[df["species"] != "unknown"].copy()

    # Snap to master grid
    df = snap_to_grid(df)

    # Keep only unique (time, lat, lon, species) — deduplicate same station/species
    df = df.drop_duplicates(subset=["time", "lat", "lon", "species"]).reset_index(drop=True)

    df[["time", "lat", "lon", "species", "presence", "source"]].to_parquet(
        SPECIES_CACHE, index=False
    )
    log.info("CalCOFI species: %d rows, %d unique species saved", len(df), df["species"].nunique())
    return df[["time", "lat", "lon", "species", "presence", "source"]]


# ---------------------------------------------------------------------------
# CalCOFI in-situ features for feature patching
# ---------------------------------------------------------------------------

def fetch_calcofi_insitu() -> pd.DataFrame:
    """Download CalCOFI bottle data for in-situ feature patching.

    Returns surface (≤10 m) measurements: time, lat, lon, sst, salinity,
    dissolved_oxygen. These are used to fill missing gridded feature values
    at species observation locations.
    """
    if CALCOFI_INSITU_CACHE.exists():
        log.info("CalCOFI in-situ cache hit")
        return pd.read_parquet(CALCOFI_INSITU_CACHE)

    query = (
        "?time,latitude,longitude,depthm,t_degc,salinity,o2ml_l"
        + _bbox_constraints()
        + "&depthm<=10"
        + "&salinity!=NaN"
    )
    raw = _erddap_csv(BOTTLE_DATASET_ID, query)

    if raw is None or len(raw) == 0:
        log.warning("No CalCOFI bottle data — in-situ patching will be skipped")
        empty = pd.DataFrame(
            columns=["time", "lat", "lon", "sst", "salinity", "dissolved_oxygen"]
        )
        empty.to_parquet(CALCOFI_INSITU_CACHE, index=False)
        return empty

    # Map column names
    col_map = {}
    for src, dst in [
        ("latitude", "lat"), ("longitude", "lon"),
        ("t_degc", "sst"), ("o2ml_l", "dissolved_oxygen"),
    ]:
        actual = _find_col(raw, [src])
        if actual:
            col_map[actual] = dst
    raw = raw.rename(columns=col_map)

    raw["time"] = pd.to_datetime(raw["time"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    for col in ["lat", "lon", "sst", "salinity", "dissolved_oxygen"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    keep = ["time", "lat", "lon"]
    for c in ["sst", "salinity", "dissolved_oxygen"]:
        if c in raw.columns:
            keep.append(c)

    df = raw[keep].dropna(subset=["time", "lat", "lon"]).copy()
    df = df[
        (df["lat"] >= LAT_MIN) & (df["lat"] <= LAT_MAX)
        & (df["lon"] >= LON_MIN) & (df["lon"] <= LON_MAX)
    ]
    df = snap_to_grid(df)

    # Average measurements at same (time, lat, lon) to get one row per station-day
    df = df.groupby(["time", "lat", "lon"]).mean(numeric_only=True).reset_index()

    df.to_parquet(CALCOFI_INSITU_CACHE, index=False)
    log.info("CalCOFI in-situ: %d rows saved -> %s", len(df), CALCOFI_INSITU_CACHE.name)
    return df
