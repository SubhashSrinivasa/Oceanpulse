"""Download Argo float profiles for surface salinity.

Uses argopy to query the Ifremer ERDDAP backend.
Extracts near-surface (0-50 m) temperature and salinity per profile,
snaps to master grid, and caches as parquet.

Fallback: GLORYS12 surface salinity via CMEMS if Argo yields no data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CACHE_DIR,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    TIME_END,
    TIME_START,
    YEARS,
)
from .utils import get_logger, snap_to_grid

log = get_logger("download_argo")

ARGO_CACHE = CACHE_DIR / "argo_surface_salinity.parquet"
SURFACE_DEPTH_MAX = 50.0  # metres


# ---------------------------------------------------------------------------
# Primary: argopy
# ---------------------------------------------------------------------------

def _fetch_argo_year(year: int) -> pd.DataFrame | None:
    """Fetch one year of Argo profiles for the domain; return surface rows."""
    import os
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

    region = [
        LON_MIN - 1, LON_MAX + 1,
        LAT_MIN - 1, LAT_MAX + 1,
        0, SURFACE_DEPTH_MAX,
        f"{year}-01-01", f"{year}-12-31",
    ]

    # Try argovis first (avoids Ifremer SSL issues on Windows), then erddap
    for src in ("argovis", "erddap"):
        try:
            from argopy import DataFetcher as ArgoDataFetcher
            fetcher = ArgoDataFetcher(src=src, progress=False)
            ds = fetcher.region(region).to_xarray()
            if ds is None or "TEMP" not in ds:
                continue
            df = ds[["TEMP", "PSAL", "LATITUDE", "LONGITUDE", "TIME", "PRES"]].to_dataframe()
            df = df.rename(columns={
                "TEMP": "temperature",
                "PSAL": "salinity",
                "LATITUDE": "lat",
                "LONGITUDE": "lon",
                "TIME": "time",
                "PRES": "depth",
            })
            df = df[df["depth"] <= SURFACE_DEPTH_MAX].dropna(subset=["salinity"])
            df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None).dt.normalize()
            df = (
                df.groupby(["time", "lat", "lon"])
                .agg(salinity=("salinity", "mean"), temperature=("temperature", "mean"))
                .reset_index()
            )
            log.info("Argo year %s (src=%s): %d surface profiles", year, src, len(df))
            return df
        except Exception as exc:  # noqa: BLE001
            log.warning("Argo fetch failed for year %s (src=%s): %s", year, src, exc)

    return None


def fetch_argo_salinity() -> pd.DataFrame:
    """Return cached or freshly-downloaded Argo surface salinity DataFrame."""
    if ARGO_CACHE.exists():
        log.info("Argo cache hit: %s", ARGO_CACHE.name)
        return pd.read_parquet(ARGO_CACHE)

    frames = []
    for year in YEARS:
        df = _fetch_argo_year(year)
        if df is not None and len(df) > 0:
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = snap_to_grid(combined)
        # Keep domain bounds
        mask = (
            (combined["lat"] >= LAT_MIN) & (combined["lat"] <= LAT_MAX)
            & (combined["lon"] >= LON_MIN) & (combined["lon"] <= LON_MAX)
        )
        combined = combined[mask].copy()
        combined.to_parquet(ARGO_CACHE, index=False)
        log.info("Argo cache saved: %d rows -> %s", len(combined), ARGO_CACHE.name)
        return combined

    log.warning("No Argo data retrieved — returning empty DataFrame")
    return pd.DataFrame(columns=["time", "lat", "lon", "salinity", "temperature"])


# ---------------------------------------------------------------------------
# GLORYS12 salinity (fallback for gridded coverage)
# ---------------------------------------------------------------------------

def fetch_glorys12_salinity_gridded() -> "xr.Dataset | None":
    """Server-side CMEMS subset of GLORYS12 surface salinity for the full domain."""
    import os

    dest = CACHE_DIR / "glorys12_salinity.nc"
    if dest.exists():
        log.info("GLORYS12 salinity cache hit")
        try:
            import xarray as xr
            return xr.open_dataset(dest)
        except Exception:  # noqa: BLE001
            pass

    username = os.getenv("CMEMS_USERNAME")
    password = os.getenv("CMEMS_PASSWORD")

    try:
        import copernicusmarine as cm

        kwargs = dict(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["so"],
            minimum_longitude=LON_MIN - 0.5,
            maximum_longitude=LON_MAX + 0.5,
            minimum_latitude=LAT_MIN - 0.5,
            maximum_latitude=LAT_MAX + 0.5,
            start_datetime=f"{TIME_START}T00:00:00",
            end_datetime=f"{TIME_END}T23:59:59",
            minimum_depth=0.0,
            maximum_depth=1.0,
            output_filename=dest.name,
            output_directory=str(dest.parent),
            overwrite=True,
        )
        if username and password:
            kwargs["username"] = username
            kwargs["password"] = password

        log.info("Requesting GLORYS12 salinity from CMEMS...")
        cm.subset(**kwargs)

        if dest.exists():
            import xarray as xr
            return xr.open_dataset(dest)
    except Exception as exc:  # noqa: BLE001
        log.warning("GLORYS12 salinity fetch failed: %s", exc)
    return None
