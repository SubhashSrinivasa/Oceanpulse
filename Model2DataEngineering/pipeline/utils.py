"""Shared utilities: logging, grid snapping, QC helpers."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .config import GRID_RES, LOG_DIR, MASTER_LAT, MASTER_LON, SANITY_BOUNDS


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "pipeline.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Grid snapping
# ---------------------------------------------------------------------------

def snap_lat(lat: np.ndarray | float) -> np.ndarray | float:
    """Snap latitude(s) to nearest master grid point."""
    idx = np.round((np.asarray(lat, dtype=float) - MASTER_LAT[0]) / GRID_RES).astype(int)
    idx = np.clip(idx, 0, len(MASTER_LAT) - 1)
    result = MASTER_LAT[idx]
    return float(result) if np.ndim(lat) == 0 else result


def snap_lon(lon: np.ndarray | float) -> np.ndarray | float:
    """Snap longitude(s) to nearest master grid point."""
    idx = np.round((np.asarray(lon, dtype=float) - MASTER_LON[0]) / GRID_RES).astype(int)
    idx = np.clip(idx, 0, len(MASTER_LON) - 1)
    result = MASTER_LON[idx]
    return float(result) if np.ndim(lon) == 0 else result


def snap_to_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Snap lat/lon columns of a DataFrame to master grid in-place (copy)."""
    df = df.copy()
    df["lat"] = snap_lat(df["lat"].to_numpy())
    df["lon"] = snap_lon(df["lon"].to_numpy())
    return df


def normalize_lon(lon: np.ndarray | float) -> np.ndarray | float:
    """Convert 0-360 longitude to -180 to 180."""
    arr = np.asarray(lon, dtype=float)
    result = np.where(arr > 180, arr - 360, arr)
    return float(result) if np.ndim(lon) == 0 else result


def in_domain(lat: float, lon: float) -> bool:
    """Return True if the point falls within the master grid bounding box."""
    from .config import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN
    return LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX


# ---------------------------------------------------------------------------
# Species name normalisation
# ---------------------------------------------------------------------------

def standardize_species_name(name: str | None) -> str:
    """Lowercase, strip, collapse whitespace, replace spaces with underscores."""
    if not isinstance(name, str) or not name.strip():
        return "unknown"
    return "_".join(name.strip().lower().split())


# ---------------------------------------------------------------------------
# QC helpers
# ---------------------------------------------------------------------------

QC_RECORDS: list[Dict] = []


def qc_variable(name: str, values: np.ndarray, extra: Dict | None = None) -> Dict:
    total = values.size
    nan_mask = np.isnan(values)
    nan_pct = 100.0 * nan_mask.sum() / total if total else float("nan")
    if nan_pct < 100.0:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    else:
        vmin = vmax = float("nan")
    lo, hi = SANITY_BOUNDS.get(name, (float("-inf"), float("inf")))
    in_range = True
    if np.isfinite(vmin) and np.isfinite(vmax):
        in_range = (vmin >= lo - 1e-6) and (vmax <= hi + 1e-6)
    record = {
        "variable": name,
        "nan_pct": round(nan_pct, 3),
        "min": round(vmin, 4) if np.isfinite(vmin) else None,
        "max": round(vmax, 4) if np.isfinite(vmax) else None,
        "in_range": bool(in_range),
    }
    if extra:
        record.update(extra)
    QC_RECORDS.append(record)
    return record


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def http_get_with_retry(
    url: str,
    *,
    params: dict | None = None,
    timeout: int = 180,
    retries: int = 3,
    backoff: float = 5.0,
    stream: bool = False,
    logger=None,
):
    """GET with exponential back-off; returns requests.Response or None."""
    import time

    import requests

    log = logger or get_logger("http")
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout, stream=stream)
            if r.status_code == 200:
                return r
            log.warning("HTTP %s for %s (attempt %d)", r.status_code, url, attempt)
        except Exception as exc:  # noqa: BLE001
            log.warning("Request error (attempt %d): %s", attempt, exc)
        if attempt < retries:
            time.sleep(backoff * attempt)
    return None


def download_to_file(url: str, dest: Path, *, logger=None, **kwargs) -> bool:
    """Stream-download URL to *dest*; return True on success."""
    log = logger or get_logger("http")
    r = http_get_with_retry(url, stream=True, logger=log, **kwargs)
    if r is None:
        return False
    try:
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    fh.write(chunk)
        log.info("Downloaded %s (%.2f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    except Exception as exc:  # noqa: BLE001
        log.error("Write failed for %s: %s", dest, exc)
        dest.unlink(missing_ok=True)
        return False
