"""Download SST (OISST v2.1) and chlorophyll (MODIS 8-day) from NOAA ERDDAP.

Downloads are cached as annual NetCDF files under data/cache/.
- SST:         ncdcOisst21Agg_LonPM180  — 0.25 deg daily
- Chlorophyll: erdMH1chla8day           — 0.05 deg 8-day composite, resampled to daily

Returns xr.Dataset keyed by variable name aligned to master grid.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .config import (
    CACHE_DIR,
    ERDDAP_BASE_GRIDDAP,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MASTER_LAT,
    MASTER_LON,
    MASTER_TIME,
    SST_CLIM_CACHE,
    YEARS,
)
from .utils import download_to_file, get_logger, qc_variable

log = get_logger("download_erddap")

# ---------------------------------------------------------------------------
# SST — OISST v2.1
# ---------------------------------------------------------------------------
SST_DATASET = "ncdcOisst21Agg_LonPM180"
CHL_DATASET = "erdMH1chla8day"


def _sst_url(year: int) -> str:
    start = f"{year}-01-01T12:00:00Z"
    end = f"{year}-12-31T12:00:00Z"
    return (
        f"{ERDDAP_BASE_GRIDDAP}/{SST_DATASET}.nc?sst"
        f"[({start}):1:({end})]"
        f"[(0.0):1:(0.0)]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )


def _download_sst_year(year: int) -> Path | None:
    dest = CACHE_DIR / f"oisst_{year}.nc"
    if dest.exists():
        log.info("SST cache hit: %s", dest.name)
        return dest
    ok = download_to_file(_sst_url(year), dest, logger=log, retries=3, backoff=6.0)
    return dest if ok else None


def _open_sst_file(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    da = ds["sst"]
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
    da = da.drop_duplicates("time").sortby("time")
    return da


def fetch_sst() -> xr.Dataset:
    """Download all SST years in parallel, stitch, compute DOY climatology + anomaly."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    annual_files = {}
    with ThreadPoolExecutor(max_workers=len(YEARS)) as executor:
        futures = {executor.submit(_download_sst_year, year): year for year in YEARS}
        for future in as_completed(futures):
            year = futures[future]
            p = future.result()
            if p:
                annual_files[year] = p
            else:
                log.warning("No SST data for %s — year will be NaN-filled", year)

    annual = [_open_sst_file(annual_files[y]) for y in sorted(annual_files)]

    if not annual:
        log.error("No SST files available — emitting full NaN SST dataset")
        nan = np.full(
            (len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)), np.nan, dtype="float32"
        )
        sst = xr.DataArray(
            nan,
            dims=("time", "lat", "lon"),
            coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
            name="sst",
        )
    else:
        sst_raw = xr.concat(annual, dim="time")
        for da in annual:
            da.close()
        sst = sst_raw.reindex(lat=MASTER_LAT, lon=MASTER_LON, method="nearest", tolerance=0.13)
        sst = sst.reindex(time=MASTER_TIME)
        sst.name = "sst"

    # Bridge short gaps (≤3 days) before climatology
    sst = sst.interpolate_na(dim="time", method="linear", max_gap=pd.Timedelta(days=3))

    # DOY climatology and anomaly
    doy = sst["time"].dt.dayofyear
    clim = sst.groupby(doy).mean("time", skipna=True)
    anom = sst.groupby(doy) - clim
    anom.name = "sst_anomaly"

    # Cache climatology for use in per-year feature lookups
    if not SST_CLIM_CACHE.exists():
        clim.to_netcdf(SST_CLIM_CACHE)
        log.info("Saved SST DOY climatology -> %s", SST_CLIM_CACHE.name)

    sst.attrs.update(units="degree_C", source="NOAA OISST v2.1 AVHRR via ERDDAP")
    anom.attrs.update(units="degree_C", long_name="SST anomaly from full-period DOY climatology")

    qc_variable("sst", sst.values)
    qc_variable("sst_anomaly", anom.values)

    return xr.Dataset({"sst": sst, "sst_anomaly": anom})


# ---------------------------------------------------------------------------
# Chlorophyll — MODIS Aqua 8-day L3
# ---------------------------------------------------------------------------

def _chl_url(year: int) -> str:
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T00:00:00Z"
    return (
        f"{ERDDAP_BASE_GRIDDAP}/{CHL_DATASET}.nc?chlorophyll"
        f"[({start}):1:({end})]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )


def _probe_chl_end_date() -> str:
    """Query the dataset's actual time coverage to cap the end request."""
    import requests

    info_url = f"https://coastwatch.pfeg.noaa.gov/erddap/info/{CHL_DATASET}/index.json"
    try:
        r = requests.get(info_url, timeout=30)
        if r.status_code == 200:
            rows = r.json().get("table", {}).get("rows", [])
            for row in rows:
                if "time_coverage_end" in str(row):
                    for item in row:
                        if isinstance(item, str) and "T" in item and item[:4].isdigit():
                            return item[:10]
    except Exception:  # noqa: BLE001
        pass
    return "2024-12-31"


def _download_chl_year(year: int, capped_end: str) -> Path | None:
    dest = CACHE_DIR / f"modis_chl_{year}.nc"
    if dest.exists():
        log.info("Chlorophyll cache hit: %s", dest.name)
        return dest
    year_end = min(f"{year}-12-31", capped_end)
    if f"{year}-01-01" > year_end:
        log.warning("Chlorophyll dataset ends before %s — skipping", year)
        return None
    start = f"{year}-01-01T00:00:00Z"
    end = year_end + "T00:00:00Z"
    url = (
        f"{ERDDAP_BASE_GRIDDAP}/{CHL_DATASET}.nc?chlorophyll"
        f"[({start}):1:({end})]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )
    ok = download_to_file(url, dest, logger=log, retries=3, backoff=6.0)
    return dest if ok else None


def fetch_chlorophyll() -> xr.Dataset:
    """Download MODIS 8-day chlorophyll, resample to daily, align to master grid."""
    capped_end = _probe_chl_end_date()
    log.info("MODIS chlorophyll data available through %s", capped_end)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    annual_files = {}
    with ThreadPoolExecutor(max_workers=len(YEARS)) as executor:
        futures = {
            executor.submit(_download_chl_year, year, capped_end): year
            for year in YEARS
        }
        for future in as_completed(futures):
            year = futures[future]
            p = future.result()
            if p:
                annual_files[year] = p

    annual = []
    for year in sorted(annual_files):
        p = annual_files[year]
        try:
            ds = xr.open_dataset(p)
            da = ds["chlorophyll"]
            rename = {}
            if "latitude" in da.dims:
                rename["latitude"] = "lat"
            if "longitude" in da.dims:
                rename["longitude"] = "lon"
            if rename:
                da = da.rename(rename)
            da = da.assign_coords(time=da["time"].dt.floor("1D"))
            annual.append(da)
            ds.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not open chlorophyll file %s: %s", p.name, exc)

    if not annual:
        log.error("No chlorophyll files available — emitting NaN chlorophyll")
        nan = np.full(
            (len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)), np.nan, dtype="float32"
        )
        chl = xr.DataArray(
            nan,
            dims=("time", "lat", "lon"),
            coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
            name="chlorophyll",
        )
    else:
        chl_raw = xr.concat(annual, dim="time").sortby("time")
        for da in annual:
            da.close()

        # Drop duplicate timestamps (MODIS 8-day composites can overlap year boundaries)
        _, unique_idx = np.unique(chl_raw["time"].values, return_index=True)
        chl_raw = chl_raw.isel(time=unique_idx)

        # Regrid to master (MODIS is 0.05 deg native)
        chl_grid = chl_raw.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")

        # 8-day composites → daily: resample fills each composite to all 8 days
        chl_daily = chl_grid.resample(time="1D").interpolate("linear")
        chl = chl_daily.reindex(time=MASTER_TIME)
        chl.name = "chlorophyll"

    # Three-pass imputation (same logic as Model 1 export_csv.py)
    log.info("Imputing chlorophyll gaps...")
    chl = _impute_chlorophyll(chl)

    chl.attrs.update(units="mg m-3", source="MODIS Aqua 8-day L3 via ERDDAP + imputation")
    qc_variable("chlorophyll", chl.values)
    return xr.Dataset({"chlorophyll": chl})


def _impute_chlorophyll(chl: xr.DataArray) -> xr.DataArray:
    """Three-pass imputation in log10 space: temporal → spatial → DOY climatology."""
    import warnings

    from scipy.interpolate import griddata

    log_chl = np.log10(np.where(chl.values > 0, chl.values, np.nan))
    n_t, n_y, n_x = log_chl.shape

    # Pass 1: temporal interpolation per cell, gap ≤ 30 days
    log_chl_da = xr.DataArray(
        log_chl, dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
    )
    log_chl_da = log_chl_da.interpolate_na(
        dim="time", method="linear", max_gap=pd.Timedelta(days=30)
    )
    log_chl = log_chl_da.values

    # Pass 2: spatial griddata per timestep for remaining NaNs
    Xg, Yg = np.meshgrid(MASTER_LON, MASTER_LAT)
    target = np.column_stack([Yg.ravel(), Xg.ravel()])
    for ti in range(n_t):
        slice_ = log_chl[ti]
        valid = np.isfinite(slice_)
        if valid.sum() < 4 or (~valid).sum() == 0:
            continue
        pts = np.column_stack([MASTER_LAT[np.where(valid)[0]], MASTER_LON[np.where(valid)[1]]])
        vals = slice_[valid]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                filled = griddata(pts, vals, target, method="linear")
            nan_mask = np.isnan(filled)
            if nan_mask.any():
                filled[nan_mask] = griddata(pts, vals, target[nan_mask], method="nearest")
            result = filled.reshape(n_y, n_x)
            log_chl[ti] = np.where(np.isfinite(slice_), slice_, result)
        except Exception:  # noqa: BLE001
            continue

    # Pass 3: DOY climatology for remaining NaNs
    log_chl_da2 = xr.DataArray(
        log_chl, dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
    )
    doy = log_chl_da2["time"].dt.dayofyear
    clim = log_chl_da2.groupby(doy).mean("time", skipna=True)
    filled = log_chl_da2.groupby(doy).map(
        lambda x, mean=clim: x.fillna(mean.sel(dayofyear=x["time"].dt.dayofyear))
        if hasattr(mean, "sel") else x
    )
    log_chl_final = filled.values
    # Back-transform
    result = np.power(10.0, log_chl_final)
    return xr.DataArray(
        result.astype("float32"), dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
    )
