"""Central configuration for Model1Forecasting.

Standalone by default: grid constants are defined here directly.
When running inside the full OceanPulse repo (Model1DataEngineering is a sibling
folder), this file will try to import from there first so the two stay in sync.
Override the dataset path with the env var OCEANPULSE_SEQUENCES_CSV, e.g.:
    export OCEANPULSE_SEQUENCES_CSV=/data/ocean_cube_sequences.csv
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
ENG_ROOT = REPO_ROOT / "Model1DataEngineering"

# Try importing from the sibling DataEngineering module; fall back to inline constants.
try:
    if str(ENG_ROOT) not in sys.path:
        sys.path.insert(0, str(ENG_ROOT))
    from pipeline.config import (  # noqa: E402
        MASTER_LAT,
        MASTER_LON,
        MASTER_TIME,
        OUTPUT_DIR as _ENG_OUTPUT_DIR,
    )
    _DEFAULT_CSV = _ENG_OUTPUT_DIR / "ocean_cube_sequences.csv"
except Exception:
    # Standalone mode — grid constants inlined (must match Model1DataEngineering/pipeline/config.py).
    MASTER_LAT = np.arange(20.0, 55.0 + 0.125, 0.25)
    MASTER_LON = np.arange(-150.0, -110.0 + 0.125, 0.25)
    MASTER_TIME = pd.date_range("2018-01-01", "2022-12-31", freq="1D")
    _DEFAULT_CSV = PROJECT_ROOT / "data" / "ocean_cube_sequences.csv"

# Override dataset path via env var (useful when moving to a GPU box).
SEQUENCES_CSV = Path(os.environ.get("OCEANPULSE_SEQUENCES_CSV", str(_DEFAULT_CSV)))

# --- Paths -------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
FORECAST_DIR = OUTPUT_DIR / "forecasts"

for _p in (CACHE_DIR, CHECKPOINT_DIR, METRICS_DIR, FORECAST_DIR):
    _p.mkdir(parents=True, exist_ok=True)

GRID_CACHE = CACHE_DIR / "grid.npz"
STATS_JSON = CACHE_DIR / "stats.json"

# --- Variables ---------------------------------------------------------------
# Order must match the columns produced by prepare_tensors.py. Input channels
# append sin_doy, cos_doy, ocean_mask before the model sees the tensor.
SEQUENCE_VARS = (
    "sst",
    "sst_anomaly",
    "ssh",
    "chlorophyll_log",
    "mld",
    "mld_source",
    "salinity",
)

INPUT_VARS = SEQUENCE_VARS                              # 7 channels from CSV
AUX_CHANNELS = ("sin_doy", "cos_doy", "ocean_mask")      # 3 added in dataset.py
TARGET_VARS = ("sst", "ssh", "chlorophyll_log", "mld", "salinity")

N_INPUT_CHANNELS = len(INPUT_VARS) + len(AUX_CHANNELS)   # 10
N_TARGET_CHANNELS = len(TARGET_VARS)                     # 5
TARGET_INDICES_IN_INPUT = tuple(INPUT_VARS.index(v) for v in TARGET_VARS)

# --- Windowing ---------------------------------------------------------------
INPUT_LEN = 30
OUTPUT_LEN = 7
STRIDE = 1

# --- Splits (by date, inclusive ends) ---------------------------------------
TRAIN_END = "2020-12-31"
VAL_END = "2021-12-31"
# test: 2022-01-01 .. 2022-12-31

# --- Training ----------------------------------------------------------------
TILE_SIZE = 96
TILE_OCEAN_MIN = 1152          # ~12.5% of 96²=9216; scales with tile area
BATCH_SIZE = 4                 # 128 hidden doubles conv memory vs 64; halve batch to stay in 16 GB
NUM_WORKERS = 0  # data is in RAM — workers only add fork+copy overhead for in-memory datasets
EPOCHS = 50
WARMUP_EPOCHS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 12

SCHEDULED_SAMPLING_MAX = 0.5

# Per-variable loss weights (order matches TARGET_VARS).
# salinity=0.0: data is 71% WOA18 climatology with no real temporal dynamics;
# supervising on it degrades the loss signal for the other variables.
LOSS_WEIGHTS = (3.0, 2.0, 1.0, 0.5, 0.0)  # sst, ssh, chlorophyll_log, mld, salinity

# --- Model -------------------------------------------------------------------
HIDDEN_CHANNELS = (128, 128)
KERNEL_SIZE = 3

# --- Grid sizes (re-exported for convenience) --------------------------------
GRID_H = len(MASTER_LAT)
GRID_W = len(MASTER_LON)
GRID_T = len(MASTER_TIME)
