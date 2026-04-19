"""Stage 3 - MODIS Aqua chlorophyll-a (8-day 4 km L3).

ERDDAP NOAA CoastWatch West Coast node carries this as `erdMH1chla8day_LonPM180`.
We pull annual 8-day blocks, regrid to MASTER_LAT / MASTER_LON, then resample
from 8-day to daily (nearest) and apply a max-gap interpolation.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from .config import (
    CACHE_DIR,
    DOWNLOAD_MAX_WORKERS,
    HTTP_CHUNK_BYTES,
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

log = get_logger("stage3_chl")

ERDDAP_BASE = "https://coastwatch.pfeg.noaa.gov/erddap/griddap"
DATASET_ID = "erdMH1chla8day"  # 4 km Lon 0-360? Check lon scheme
VAR = "chlorophyll"
MAX_GAP_DAYS = 16  # allow bridging ~2 missing 8-day windows


def _year_url(year: int) -> str:
    start = f"{year}-01-01T00:00:00Z"
    end = f"{year}-12-31T23:59:59Z"
    # erdMH1chla8day: lon in -180..180, lat descends 90 -> -90.
    # ERDDAP expects indices in the same order as the axis.
    return (
        f"{ERDDAP_BASE}/{DATASET_ID}.nc?{VAR}"
        f"[({start}):1:({end})]"
        f"[({LAT_MAX}):1:({LAT_MIN})]"
        f"[({LON_MIN}):1:({LON_MAX})]"
    )


def _dataset_time_range() -> tuple[str, str] | None:
    try:
        r = requests.get(
            f"https://coastwatch.pfeg.noaa.gov/erddap/info/{DATASET_ID}/index.json",
            timeout=60,
        )
        rows = r.json()["table"]["rows"]
        for row in rows:
            if row[1] == "time" and row[2] == "actual_range":
                lo, hi = [float(x) for x in row[4].split(",")]
                return (
                    pd.to_datetime(lo, unit="s").isoformat() + "Z",
                    pd.to_datetime(hi, unit="s").isoformat() + "Z",
                )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read MODIS CHL time range: %s", exc)
    return None


def _download_year(year: int, dest: Path, retries: int = 3) -> bool:
    url = _year_url(year)
    for attempt in range(1, retries + 1):
        try:
            log.info("Downloading MODIS CHL %s (attempt %d)", year, attempt)
            r = requests.get(url, timeout=300, stream=True)
            if r.status_code == 404 and attempt == 1:
                # Probably asked past the dataset's end; retry with cap.
                rng = _dataset_time_range()
                if rng is not None:
                    _, hi = rng
                    hi_year = int(hi[:4])
                    if hi_year == year:
                        log.info("CHL %s: capping end at dataset max %s", year, hi)
                        start = f"{year}-01-01T00:00:00Z"
                        capped_url = (
                            f"{ERDDAP_BASE}/{DATASET_ID}.nc?{VAR}"
                            f"[({start}):1:({hi})]"
                            f"[({LAT_MAX}):1:({LAT_MIN})]"
                            f"[({LON_MIN}):1:({LON_MAX})]"
                        )
                        url = capped_url
                        continue  # retry loop with capped url
            if r.status_code != 200:
                log.warning("CHL %s HTTP %s: %s", year, r.status_code, r.text[:200])
                time.sleep(3 * attempt)
                continue
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=HTTP_CHUNK_BYTES):
                    if chunk:
                        f.write(chunk)
            log.info("Saved %s (%.2f MB)", dest.name, dest.stat().st_size / 1e6)
            return True
        except Exception as exc:  # noqa: BLE001
            log.warning("CHL %s attempt %d failed: %s", year, attempt, exc)
            time.sleep(5 * attempt)
    return False


def _nan_chl() -> xr.DataArray:
    nan = np.full(
        (len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)), np.nan, dtype="float32"
    )
    return xr.DataArray(
        nan,
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name="chlorophyll",
    )


def fetch_chl() -> xr.Dataset:
    annual_files = []
    missing = [
        (y, CACHE_DIR / f"modis_chl_{y}.nc")
        for y in YEARS
        if not (CACHE_DIR / f"modis_chl_{y}.nc").exists()
    ]
    if missing:
        n_workers = min(DOWNLOAD_MAX_WORKERS, len(missing))
        log.info(
            "MODIS CHL: downloading %d missing year(s) in parallel (max_workers=%d)",
            len(missing),
            n_workers,
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_download_year, y, path): y for y, path in missing
            }
            for fut in as_completed(futures):
                y = futures[fut]
                try:
                    ok = fut.result()
                    if not ok:
                        log.error("MODIS CHL %s failed", y)
                except Exception as exc:  # noqa: BLE001
                    log.error("MODIS CHL %s failed: %s", y, exc)

    for year in YEARS:
        dest = CACHE_DIR / f"modis_chl_{year}.nc"
        if dest.exists():
            annual_files.append(dest)
        else:
            log.error("Missing MODIS CHL file for year %s — skipping", year)

    if not annual_files:
        log.error("No MODIS CHL files — full NaN chlorophyll")
        chl_daily = _nan_chl()
    else:
        ds_list = []
        for f in annual_files:
            ds = xr.open_dataset(f)[[VAR]]
            ds_list.append(ds)
        ds_all = xr.concat(ds_list, dim="time")
        for d in ds_list:
            d.close()

        # Normalize axes
        rename = {}
        if "latitude" in ds_all.dims:
            rename["latitude"] = "lat"
        if "longitude" in ds_all.dims:
            rename["longitude"] = "lon"
        if rename:
            ds_all = ds_all.rename(rename)

        ds_all = ds_all.sortby("lat")
        chl = ds_all[VAR]
        chl = chl.assign_coords(time=chl["time"].dt.floor("1D"))
        chl = chl.drop_duplicates("time").sortby("time")

        # Spatial regrid from native ~4 km to 0.25 deg via linear interp.
        # The native grid is much denser than the target so this is
        # effectively a block average as far as the coarse cube goes.
        chl = chl.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")

        # Reindex 8-day frames onto daily time axis (nearest) then interpolate
        # time gaps up to MAX_GAP_DAYS.
        chl_daily = chl.reindex(time=MASTER_TIME, method="nearest", tolerance=pd.Timedelta(days=8))
        chl_daily = chl_daily.interpolate_na(
            dim="time", method="linear", max_gap=pd.Timedelta(days=MAX_GAP_DAYS)
        )
        chl_daily.name = "chlorophyll"
        chl_daily.attrs.update(
            units="mg m-3",
            source="NASA MODIS-Aqua L3 8-day chlorophyll via ERDDAP erdMH1chla8day",
        )

    chl_log = np.log10(chl_daily.where(chl_daily > 0))
    chl_log.name = "chlorophyll_log"
    chl_log.attrs.update(units="log10(mg m-3)", long_name="log10 chlorophyll-a")

    qc_variable("chlorophyll", chl_daily)
    qc_variable("chlorophyll_log", chl_log)

    return xr.Dataset({"chlorophyll": chl_daily, "chlorophyll_log": chl_log})


if __name__ == "__main__":
    print(fetch_chl())
