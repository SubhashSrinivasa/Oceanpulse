"""Central configuration for the OceanPulse species classification dataset pipeline.

California Current system, 2018-2022, 0.25 deg daily grid.
Same time window as Model 1; expanded spatial domain (30-48 N, -124 to -116 W).
Target output: flat (time, lat, lon, features, species, presence) CSV, ~0.7 GB.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Master grid — California Current region (expanded vs. Model 1)
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 30.0, 48.0
LON_MIN, LON_MAX = -124.0, -116.0
GRID_RES = 0.25
TIME_START = "2018-01-01"       # same 5-year window as Model 1
TIME_END = "2022-12-31"

MASTER_LAT = np.round(np.arange(LAT_MIN, LAT_MAX + GRID_RES / 2, GRID_RES), 4)
MASTER_LON = np.round(np.arange(LON_MIN, LON_MAX + GRID_RES / 2, GRID_RES), 4)
MASTER_TIME = pd.date_range(TIME_START, TIME_END, freq="1D")
YEARS = list(range(2018, 2023))

# ---------------------------------------------------------------------------
# Size-control parameters
# ---------------------------------------------------------------------------
MAX_SIZE_GB = 5.0
MIN_SIZE_GB = 1.0          # target floor (informational only — no inflation)
MAX_SPECIES_TOP_N = 50     # keep only the N most-observed species when trimming
MIN_SPECIES_OBS = 10       # drop species with fewer observations than this
TEMPORAL_STRIDE_DAYS = 2   # initial downsampling stride (days) when oversized

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = PROJECT_ROOT / "logs"

for _p in (CACHE_DIR, RAW_DIR, OUTPUT_DIR, LOG_DIR):
    _p.mkdir(parents=True, exist_ok=True)

TRAINING_CSV_PATH = OUTPUT_DIR / "training_data.csv"

# Intermediate caches (parquet — efficient for columnar data)
SST_CLIM_CACHE = CACHE_DIR / "sst_doy_climatology.nc"
SPECIES_CACHE = CACHE_DIR / "species_observations.parquet"
CALCOFI_INSITU_CACHE = CACHE_DIR / "calcofi_insitu_features.parquet"
FEATURE_JOINED_CACHE = CACHE_DIR / "features_joined.parquet"

# ---------------------------------------------------------------------------
# External service endpoints
# ---------------------------------------------------------------------------
ERDDAP_BASE_GRIDDAP = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
ERDDAP_BASE_TABLEDAP = "https://coastwatch.pfeg.noaa.gov/erddap/tabledap"
GBIF_API = "https://api.gbif.org/v1/occurrence/search"
INATURALIST_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
NWFSC_API = "https://www.webapps.nwfsc.noaa.gov/data/api/v1/source/trawl.catch_fact/selection.json"

# CMEMS products
CMEMS_SSH_DATASET = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
CMEMS_SSH_VARIABLE = "sla"

# WOA18 dissolved oxygen
WOA18_DO_URL_TMPL = (
    "https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA/"
    "oxygen/netcdf/all/1.00/woa18_all_o{mm:02d}_01.nc"
)

# ---------------------------------------------------------------------------
# Physical sanity bounds for QC
# ---------------------------------------------------------------------------
SANITY_BOUNDS = {
    "sst": (-2.0, 35.0),
    "sst_anomaly": (-10.0, 10.0),
    "ssh": (-2.0, 2.0),
    "chlorophyll": (0.0, 100.0),
    "salinity": (15.0, 40.0),
    "dissolved_oxygen": (0.0, 12.0),  # ml/L
}

# ---------------------------------------------------------------------------
# Final output column schema
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "sst",
    "sst_anomaly",
    "chlorophyll",
    "salinity",
    "dissolved_oxygen",
    "ssh",
]
META_COLS = ["day_of_year", "month"]

# Columns that are always present and in a fixed order.
# Species columns are appended dynamically after pivot (one binary column per species).
FIXED_COLS = ["time", "lat", "lon"] + FEATURE_COLS + META_COLS

# Long-format intermediate columns (used internally before pivot)
LONG_COLS = FIXED_COLS + ["species", "presence"]
