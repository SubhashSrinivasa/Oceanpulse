"""Logging and QC helpers shared across every stage."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import xarray as xr

from .config import LOG_DIR, SANITY_BOUNDS


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "pipeline.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


QC_RECORDS: List[Dict] = []


def qc_variable(name: str, da: xr.DataArray, extra: Dict | None = None) -> Dict:
    """Compute standard QC stats for a variable and append to QC_RECORDS."""
    values = da.values
    total = values.size
    nan_mask = np.isnan(values)
    nan_pct = 100.0 * nan_mask.sum() / total if total else float("nan")

    if nan_pct < 100.0:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    else:
        vmin = float("nan")
        vmax = float("nan")

    lo, hi = SANITY_BOUNDS.get(name, (float("-inf"), float("inf")))
    in_range = True
    if np.isfinite(vmin) and np.isfinite(vmax):
        in_range = (vmin >= lo - 1e-6) and (vmax <= hi + 1e-6)

    record = {
        "variable": name,
        "nan_pct": round(nan_pct, 3),
        "min": vmin,
        "max": vmax,
        "in_range": bool(in_range),
    }
    if extra:
        record.update(extra)
    QC_RECORDS.append(record)
    return record


def save_tempfile_note(path: Path, note: str) -> None:
    """Write a tiny marker so we remember why a temp file is on disk."""
    (path.parent / (path.name + ".note.txt")).write_text(note)
