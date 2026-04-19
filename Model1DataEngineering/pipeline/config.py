"""Central configuration for the OceanPulse ingestion pipeline.

Budget-locked subset: Central California Upwelling Zone, 2018-2022, 0.25 deg daily.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- Master grid -------------------------------------------------------------
LAT_MIN, LAT_MAX = 34.0, 42.0
LON_MIN, LON_MAX = -124.0, -118.0
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
