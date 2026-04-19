"""Build the ocean feature store for year-by-year point lookups.

Architecture (memory-efficient):
  - All gridded ocean data lives on disk as annual NetCDF files (already
    downloaded and cached by download_*.py modules).
  - This module provides build() which, for a given set of species observation
    points, loads ONE YEAR of gridded data at a time and extracts feature
    values at the exact (time, lat, lon) of each observation.
  - The extracted feature values are concatenated across all years and
    returned as a DataFrame joined onto the species table.

Features extracted:
  sst, sst_anomaly, chlorophyll, salinity, dissolved_oxygen, ssh

SST anomaly uses a pre-computed DOY climatology (saved by download_erddap).
Dissolved oxygen uses WOA18 monthly climatology broadcast to daily.
Salinity uses Argo sparse-to-grid interpolation (gridded GLORYS12 if Argo
is thin) with a 33.5 PSU constant fallback.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    CACHE_DIR,
    FEATURE_JOINED_CACHE,
    MASTER_LAT,
    MASTER_LON,
    SST_CLIM_CACHE,
    YEARS,
)
from .download_argo import fetch_argo_salinity, fetch_glorys12_salinity_gridded
from .download_cmems import fetch_woa18_dissolved_oxygen
from .grid_align import monthly_clim_to_daily, sparse_to_daily_grid
from .utils import get_logger

log = get_logger("feature_store")

CONST_SAL = 33.5  # PSU fallback


# ---------------------------------------------------------------------------
# SST DOY climatology loader
# ---------------------------------------------------------------------------

def _load_sst_climatology() -> xr.DataArray | None:
    if SST_CLIM_CACHE.exists():
        try:
            ds = xr.open_dataset(SST_CLIM_CACHE)
            var = list(ds.data_vars)[0]
            return ds[var]
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load SST climatology: %s", exc)
    return None


# ---------------------------------------------------------------------------
# Per-year gridded data loaders
# ---------------------------------------------------------------------------

def _load_sst_year(year: int) -> xr.DataArray | None:
    p = CACHE_DIR / f"oisst_{year}.nc"
    if not p.exists():
        return None
    try:
        ds = xr.open_dataset(p)
        da = ds["sst"] if "sst" in ds else list(ds.data_vars)[0]
        if "zlev" in da.dims:
            da = da.isel(zlev=0, drop=True)
        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)
        da = da.assign_coords(time=da["time"].dt.floor("1D"))
        da = da.reindex(lat=MASTER_LAT, lon=MASTER_LON, method="nearest", tolerance=0.13)
        da.name = "sst"
        ds.close()
        return da
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not open SST year %s: %s", year, exc)
        return None


def _load_chl_year(year: int) -> xr.DataArray | None:
    p = CACHE_DIR / f"modis_chl_{year}.nc"
    if not p.exists():
        return None
    try:
        ds = xr.open_dataset(p)
        da = ds["chlorophyll"] if "chlorophyll" in ds else list(ds.data_vars)[0]
        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)
        da = da.assign_coords(time=da["time"].dt.floor("1D"))
        da = da.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")
        # Resample 8-day to daily
        da = da.resample(time="1D").interpolate("linear")
        da.name = "chlorophyll"
        ds.close()
        return da
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not open chlorophyll year %s: %s", year, exc)
        return None


def _load_ssh_year(year: int, ssh_full: xr.DataArray | None) -> xr.DataArray | None:
    """Slice the full SSH dataset to the given year."""
    if ssh_full is None:
        return None
    try:
        return ssh_full.sel(time=str(year))
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Point lookup helper
# ---------------------------------------------------------------------------

def _lookup_at_points(
    da: xr.DataArray | None,
    times: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Extract scalar values from a gridded DataArray at (time, lat, lon) points.

    Uses label-based nearest selection. Returns float32 array with NaN for
    out-of-range or missing data.
    """
    n = len(times)
    result = np.full(n, np.nan, dtype="float32")
    if da is None:
        return result

    for i in range(n):
        try:
            val = da.sel(
                time=times[i],
                lat=lats[i],
                lon=lons[i],
                method="nearest",
                tolerance=0.13,
            ).values
            if np.isfinite(val):
                result[i] = float(val)
        except Exception:  # noqa: BLE001
            pass
    return result


def _lookup_at_points_fast(
    da: xr.DataArray | None,
    times: pd.DatetimeIndex,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """Vectorised lookup with neighbour-search. If the primary grid cell is NaN
    (e.g. coastal/land pixel in OISST), searches outward up to ±2 deg (~220 km)
    to find the nearest valid ocean value."""
    n = len(times)
    result = np.full(n, np.nan, dtype="float32")
    if da is None or n == 0:
        return result

    try:
        t_idx = da.indexes["time"]
        lat_arr = np.asarray(da.indexes["lat"])
        lon_arr = np.asarray(da.indexes["lon"])
        data = da.values  # (time, lat, lon)
        n_y, n_x = len(lat_arr), len(lon_arr)

        # Vectorised time lookup (pandas-2 compatible)
        ti_arr = t_idx.get_indexer(pd.DatetimeIndex(times), method="nearest")

        # Uniform-grid primary indices
        lat_step = (lat_arr[-1] - lat_arr[0]) / max(n_y - 1, 1)
        lon_step = (lon_arr[-1] - lon_arr[0]) / max(n_x - 1, 1)
        primary_yi = np.clip(
            np.round((lats - lat_arr[0]) / lat_step).astype(int), 0, n_y - 1
        )
        primary_xi = np.clip(
            np.round((lons - lon_arr[0]) / lon_step).astype(int), 0, n_x - 1
        )

        # Cell offsets sorted by squared distance, up to ±8 cells (±2 deg)
        offsets = sorted(
            [(dy, dx) for dy in range(-8, 9) for dx in range(-8, 9)],
            key=lambda p: p[0] * p[0] + p[1] * p[1],
        )

        for i in range(n):
            ti = int(ti_arr[i])
            if ti < 0:
                continue
            yi_p = int(primary_yi[i])
            xi_p = int(primary_xi[i])
            for dy, dx in offsets:
                yi = yi_p + dy
                xi = xi_p + dx
                if 0 <= yi < n_y and 0 <= xi < n_x:
                    val = data[ti, yi, xi]
                    if np.isfinite(val):
                        result[i] = float(val)
                        break
    except Exception as exc:  # noqa: BLE001
        log.debug("Fast lookup failed, falling back: %s", exc)
        return _lookup_at_points(da, times, lats, lons, name="")
    return result


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_feature_store(species_df: pd.DataFrame) -> pd.DataFrame:
    """Add ocean feature columns to the species observation DataFrame.

    Processes one year at a time to avoid loading the full time series
    into memory. The species_df must already have time, lat, lon columns
    snapped to the master grid.

    Returns species_df with additional columns:
      sst, sst_anomaly, chlorophyll, salinity, dissolved_oxygen, ssh
    """
    if FEATURE_JOINED_CACHE.exists():
        log.info("Feature store cache hit: %s", FEATURE_JOINED_CACHE.name)
        return pd.read_parquet(FEATURE_JOINED_CACHE)

    log.info("Building feature store for %d species observations...", len(species_df))

    # -----------------------------------------------------------------------
    # Preload data structures that cover all years
    # -----------------------------------------------------------------------
    sst_clim = _load_sst_climatology()
    if sst_clim is not None:
        log.info("SST DOY climatology loaded (%d DOY entries)", len(sst_clim["dayofyear"]))
    else:
        log.warning("SST climatology not available — sst_anomaly will be NaN")

    # WOA18 DO: (12, n_lat, n_lon) broadcast to daily
    log.info("Loading WOA18 dissolved oxygen climatology...")
    do_clim_monthly = fetch_woa18_dissolved_oxygen()
    do_daily = monthly_clim_to_daily(do_clim_monthly)
    log.info("DO daily broadcast ready")

    # Argo salinity: sparse parquet → gridded DataArray
    log.info("Loading Argo salinity...")
    argo_df = fetch_argo_salinity()
    if len(argo_df) > 0:
        log.info("Gridding Argo salinity (%d profiles) ...", len(argo_df))
        sal_da = sparse_to_daily_grid(argo_df, "salinity", time_window_days=7)
        log.info(
            "Argo salinity grid coverage: %.1f%%",
            100 * np.isfinite(sal_da.values).mean(),
        )
    else:
        log.warning("No Argo data — trying GLORYS12 salinity as fallback...")
        glorys_ds = fetch_glorys12_salinity_gridded()
        if glorys_ds is not None:
            sal_raw = glorys_ds["so"] if "so" in glorys_ds else list(glorys_ds.data_vars)[0]
            rename = {}
            if "latitude" in sal_raw.dims:
                rename["latitude"] = "lat"
            if "longitude" in sal_raw.dims:
                rename["longitude"] = "lon"
            if rename:
                sal_raw = sal_raw.rename(rename)
            if "depth" in sal_raw.dims:
                sal_raw = sal_raw.isel(depth=0, drop=True)
            sal_da = sal_raw.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")
            sal_da.name = "salinity"
            glorys_ds.close()
        else:
            log.warning("GLORYS12 unavailable — salinity will use constant %.1f PSU fallback", CONST_SAL)
            sal_da = None

    # SSH: loaded once across all years
    log.info("Loading SSH dataset...")
    ssh_dest = CACHE_DIR / "cmems_ssh.nc"
    if ssh_dest.exists():
        try:
            ssh_ds = xr.open_dataset(ssh_dest)
            ssh_var = "sla" if "sla" in ssh_ds else list(ssh_ds.data_vars)[0]
            ssh_full = ssh_ds[ssh_var]
            rename = {}
            if "latitude" in ssh_full.dims:
                rename["latitude"] = "lat"
            if "longitude" in ssh_full.dims:
                rename["longitude"] = "lon"
            if rename:
                ssh_full = ssh_full.rename(rename)
            ssh_full = ssh_full.assign_coords(time=ssh_full["time"].dt.floor("1D"))
            ssh_full = ssh_full.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")
            ssh_full.name = "ssh"
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load SSH: %s", exc)
            ssh_full = None
    else:
        log.warning("SSH file not found — ssh column will be NaN")
        ssh_full = None

    # -----------------------------------------------------------------------
    # Year-by-year point extraction
    # -----------------------------------------------------------------------
    year_frames = []
    species_df = species_df.copy()
    species_df["time"] = pd.to_datetime(species_df["time"])

    for year in YEARS:
        yr_mask = species_df["time"].dt.year == year
        if not yr_mask.any():
            continue

        yr_df = species_df[yr_mask].copy()
        log.info("Year %s: %d observations", year, len(yr_df))

        times = pd.DatetimeIndex(yr_df["time"])
        lats = yr_df["lat"].to_numpy()
        lons = yr_df["lon"].to_numpy()

        # SST
        sst_yr = _load_sst_year(year)
        yr_df["sst"] = _lookup_at_points_fast(sst_yr, times, lats, lons)

        # SST anomaly from DOY climatology
        if sst_clim is not None and yr_df["sst"].notna().any():
            doys = times.dayofyear
            anom = np.full(len(yr_df), np.nan, dtype="float32")
            for i, (sst_val, doy) in enumerate(zip(yr_df["sst"], doys)):
                if np.isfinite(sst_val):
                    try:
                        clim_val = float(
                            sst_clim.sel(dayofyear=doy, lat=lats[i], lon=lons[i],
                                         method="nearest").values
                        )
                        if np.isfinite(clim_val):
                            anom[i] = sst_val - clim_val
                    except Exception:  # noqa: BLE001
                        pass
            yr_df["sst_anomaly"] = anom
        else:
            yr_df["sst_anomaly"] = np.nan

        if sst_yr is not None:
            sst_yr.close()

        # Chlorophyll
        chl_yr = _load_chl_year(year)
        yr_df["chlorophyll"] = _lookup_at_points_fast(chl_yr, times, lats, lons)
        if chl_yr is not None:
            chl_yr.close()

        # Salinity
        yr_df["salinity"] = _lookup_at_points_fast(sal_da, times, lats, lons)
        # Constant fallback for any remaining NaN
        sal_arr = yr_df["salinity"].to_numpy()
        sal_arr[~np.isfinite(sal_arr)] = CONST_SAL
        yr_df["salinity"] = sal_arr

        # Dissolved oxygen from WOA18 daily broadcast
        yr_df["dissolved_oxygen"] = _lookup_at_points_fast(do_daily, times, lats, lons)

        # SSH
        ssh_yr = _load_ssh_year(year, ssh_full)
        yr_df["ssh"] = _lookup_at_points_fast(ssh_yr, times, lats, lons)

        year_frames.append(yr_df)
        log.info(
            "Year %s features — SST NaN %.1f%%, CHL NaN %.1f%%, "
            "SAL NaN %.1f%%, DO NaN %.1f%%",
            year,
            100 * yr_df["sst"].isna().mean(),
            100 * yr_df["chlorophyll"].isna().mean(),
            100 * yr_df["salinity"].isna().mean(),
            100 * yr_df["dissolved_oxygen"].isna().mean(),
        )

    if ssh_full is not None:
        ssh_full.close()

    if not year_frames:
        raise RuntimeError("No data extracted — check that ocean NetCDF files exist in cache/")

    result = pd.concat(year_frames, ignore_index=True)

    # Drop observations that could not be matched to any valid ocean
    # pixel within ±2 deg for SST (the most strict ocean-only feature).
    # Remaining per-feature NaN is handled downstream (CalCOFI patching,
    # then median imputation) so that in-situ measurements take priority
    # over imputed medians.
    before = len(result)
    result = result.dropna(subset=["sst"]).reset_index(drop=True)
    dropped = before - len(result)
    if dropped:
        log.info(
            "Dropped %d / %d observations (%.1f%%) with no valid ocean pixel "
            "within 2 deg — these were on land / in masked coastal areas",
            dropped, before, 100 * dropped / before,
        )

    result.to_parquet(FEATURE_JOINED_CACHE, index=False)
    log.info(
        "Feature store complete: %d rows -> %s",
        len(result), FEATURE_JOINED_CACHE.name,
    )
    return result
