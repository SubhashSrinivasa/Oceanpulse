"""Stage 1 - OISST sea surface temperature via NOAA ERDDAP.

Strategy: pull one year at a time, cache as NetCDF, stitch, compute
day-of-year climatology + anomaly on the master grid.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .config import (
    CACHE_DIR,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MASTER_LAT,
    MASTER_LON,
    MASTER_TIME,
    YEARS,
)
from .utils import get_logger, qc_variable

log = get_logger("stage1_sst")

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
# NOAA NCEI OISST v2.1 AVHRR, 0.25 deg daily
DATASET_ID = "ncdcOisst21Agg_LonPM180"
VARIABLE = "sst"


def _year_url(year: int) -> str:
    start = f"{year}-01-01T12:00:00Z"
    end = f"{year}-12-31T12:00:00Z"
    # Dataset has a zlev=0 depth axis.
    return (
        f"{ERDDAP_BASE}/{DATASET_ID}.nc?{VARIABLE}"
        f"[({start}):1:({end})]"
        f"[(0.0):1:(0.0)]"
        f"[({LAT_MIN}):1:({LAT_MAX})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )


def _download_year(year: int, dest: Path, retries: int = 3) -> bool:
    url = _year_url(year)
    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading OISST year %s (attempt %d)", year, attempt)
            r = requests.get(url, timeout=180, stream=True)
            if r.status_code != 200:
                log.warning("OISST %s HTTP %s: %s", year, r.status_code, r.text[:200])
                time.sleep(3 * attempt)
                continue
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
            log.info("Saved %s (%.2f MB)", dest.name, dest.stat().st_size / 1e6)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("OISST %s attempt %d failed: %s", year, attempt, exc)
            time.sleep(5 * attempt)
    return False


def fetch_sst() -> xr.Dataset:
    annual_files = []
    for year in YEARS:
        dest = CACHE_DIR / f"oisst_{year}.nc"
        if not dest.exists():
            ok = _download_year(year, dest)
            if not ok:
                log.error("Failed OISST year %s — will NaN-fill", year)
                continue
        annual_files.append(dest)

    if not annual_files:
        log.error("No OISST files available — emitting full NaN SST")
        nan = np.full((len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)), np.nan, dtype="float32")
        sst = xr.DataArray(
            nan,
            dims=("time", "lat", "lon"),
            coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
            name="sst",
        )
    else:
        ds_list = []
        for f in annual_files:
            ds = xr.open_dataset(f)
            if "zlev" in ds.dims:
                ds = ds.isel(zlev=0, drop=True)
            ds_list.append(ds[[VARIABLE]])
        ds_all = xr.concat(ds_list, dim="time")
        for d in ds_list:
            d.close()

        # Normalize axis names.
        rename_map = {}
        if "latitude" in ds_all.dims:
            rename_map["latitude"] = "lat"
        if "longitude" in ds_all.dims:
            rename_map["longitude"] = "lon"
        if rename_map:
            ds_all = ds_all.rename(rename_map)

        # Round and align to master grid (OISST already at 0.25 deg).
        ds_all = ds_all.assign_coords(
            lat=np.round(ds_all["lat"].values, 3),
            lon=np.round(ds_all["lon"].values, 3),
        )
        # Take noon-UTC -> date
        ds_all = ds_all.assign_coords(time=ds_all["time"].dt.floor("1D"))
        ds_all = ds_all.drop_duplicates("time").sortby("time")

        # Reindex to master (nearest for small grid snap).
        sst = ds_all[VARIABLE].reindex(
            lat=MASTER_LAT, lon=MASTER_LON, method="nearest", tolerance=0.13
        )
        sst = sst.reindex(time=MASTER_TIME)
        sst.name = "sst"

    # Bridge short (<=3 day) time gaps before climatology so DOY groups stay aligned.
    sst = sst.interpolate_na(dim="time", method="linear", max_gap=pd.Timedelta(days=3))

    # Climatology by day-of-year and anomaly
    doy = sst["time"].dt.dayofyear
    clim = sst.groupby(doy).mean("time", skipna=True)
    anom = sst.groupby(doy) - clim
    anom.name = "sst_anomaly"

    sst.attrs.update(units="degree_C", source="NOAA OISST v2.1 AVHRR via ERDDAP")
    anom.attrs.update(
        units="degree_C",
        long_name="SST anomaly from 2018-2022 day-of-year climatology",
    )

    qc_variable("sst", sst)
    qc_variable("sst_anomaly", anom)

    return xr.Dataset({"sst": sst, "sst_anomaly": anom})


if __name__ == "__main__":
    ds = fetch_sst()
    print(ds)
