"""Central configuration for the OceanPulse ingestion pipeline.

Budget-locked subset: Central California Upwelling Zone, 2018-2022, 0.25 deg daily.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# --- Master grid -------------------------------------------------------------
# Expanded Pacific bbox (vs original 34-42 N / -124 to -118 W) to produce a
# ~3-4 GB long-format CSV from the same 0.25 deg daily 2018-2022 window.
# Cache filenames in data/cache/ do NOT encode the bbox -- wipe that directory
# (and outputs/ocean_cube.zarr) whenever these bounds change.
LAT_MIN, LAT_MAX = 20.0, 55.0
LON_MIN, LON_MAX = -150.0, -110.0
TIME_START, TIME_END = "2018-01-01", "2022-12-31"

MASTER_LAT = np.arange(LAT_MIN, LAT_MAX + 0.125, 0.25)
MASTER_LON = np.arange(LON_MIN, LON_MAX + 0.125, 0.25)
MASTER_TIME = pd.date_range(TIME_START, TIME_END, freq="1D")
YEARS = list(range(2018, 2023))

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

for _p in (RAW_DIR, CACHE_DIR, OUTPUT_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

ZARR_PATH = OUTPUT_DIR / "ocean_cube.zarr"
QC_REPORT_PATH = OUTPUT_DIR / "data_quality_report.txt"
MISSING_MAP_PATH = OUTPUT_DIR / "missing_data_map.png"

# --- HTTP downloads (ERDDAP, WOA, etc.) --------------------------------------
# Parallel independent file fetches (OISST/MODIS per-year, WOA monthly).
# Set to 1 to force sequential. CMEMS copernicusmarine subset is still one server job.
DOWNLOAD_MAX_WORKERS = max(1, int(os.environ.get("OCEANPULSE_DOWNLOAD_WORKERS", "4")))
HTTP_CHUNK_BYTES = int(os.environ.get("OCEANPULSE_HTTP_CHUNK_BYTES", str(4 * 1024 * 1024)))

# --- Physical sanity bounds for QC ------------------------------------------
SANITY_BOUNDS = {
    "sst": (-2.0, 35.0),
    "sst_anomaly": (-10.0, 10.0),
    "ssh": (-2.0, 2.0),
    # MODIS L3 can hit ~100 mg m-3 in coastal blooms.
    "chlorophyll": (0.0, 100.0),
    "chlorophyll_log": (-3.0, 2.5),
    # Coastal waters under river plumes dip below 20 PSU; keep floor permissive.
    "salinity": (15.0, 40.0),
    "mld": (1.0, 1000.0),
}
