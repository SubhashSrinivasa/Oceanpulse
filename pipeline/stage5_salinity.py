"""Stage 5 - Surface salinity from CalCOFI + GLORYS12 fill.

Pipeline:
  * Server-side ERDDAP query for CalCOFI SIO Hydro Bottle, filtered to:
      depth <= 10 m, valid salinity, bbox, 2018-2022
  * Interpolate to master grid per day using ±7-day windowed IDW/linear interp
  * Gap-fill with GLORYS12 daily mean salinity at surface (z~0.5 m)
  * Final fallback: constant 33.5 psu.
"""
from __future__ import annotations

from io import StringIO
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
    TIME_END,
    TIME_START,
)
from .utils import get_logger, qc_variable

log = get_logger("stage5_sal")

CALCOFI_ERDDAP = (
    "https://coastwatch.pfeg.noaa.gov/erddap/tabledap/siocalcofiHydroBottle.csv"
)
CONST_SAL = 33.5  # practical salinity units


def _fetch_calcofi() -> pd.DataFrame | None:
    cache = CACHE_DIR / "calcofi_bottle_surface.parquet"
    if cache.exists():
        log.info("Using cached CalCOFI surface file")
        return pd.read_parquet(cache)
    qs = (
        "?time,latitude,longitude,depthm,salinity,s_qual"
        f"&time>={TIME_START}T00:00:00Z"
        f"&time<={TIME_END}T23:59:59Z"
        f"&latitude>={LAT_MIN}&latitude<={LAT_MAX}"
        f"&longitude>={LON_MIN}&longitude<={LON_MAX}"
        "&depthm<=10"
        "&salinity!=NaN"
    )
    url = CALCOFI_ERDDAP + qs
    try:
        log.info("Querying CalCOFI ERDDAP...")
        r = requests.get(url, timeout=180)
        if r.status_code != 200:
            log.error("CalCOFI HTTP %s: %s", r.status_code, r.text[:200])
            return None
        # First line is column names, second is units.
        df = pd.read_csv(StringIO(r.text), skiprows=[1])
        log.info("CalCOFI rows: %d", len(df))
        df = df[df["salinity"].notna()].copy()
        # Quality flag: keep s_qual == 0 (good) or NaN (unflagged).
        if "s_qual" in df.columns:
            df = df[(df["s_qual"] == 0) | df["s_qual"].isna()]
        df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert(None).dt.normalize()
        df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
        df = df[["time", "lat", "lon", "salinity"]]
        df.to_parquet(cache, index=False)
        log.info("Cached %d CalCOFI surface rows -> %s", len(df), cache.name)
        return df
    except Exception as exc:  # noqa: BLE001
        log.error("CalCOFI fetch failed: %s", exc)
        return None


def _interp_calcofi_to_master(df: pd.DataFrame) -> xr.DataArray:
    from scipy.interpolate import griddata

    n_t, n_y, n_x = len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)
    out = np.full((n_t, n_y, n_x), np.nan, dtype="float32")

    times = df["time"].to_numpy()
    lats = df["lat"].to_numpy()
    lons = df["lon"].to_numpy()
    vals = df["salinity"].to_numpy()

    Xg, Yg = np.meshgrid(MASTER_LON, MASTER_LAT)
    target = np.column_stack([Yg.ravel(), Xg.ravel()])
    window = np.timedelta64(7, "D")

    master_np = MASTER_TIME.to_numpy()
    for ti, t in enumerate(master_np):
        mask = np.abs(times - t) <= window
        if mask.sum() < 4:
            continue
        pts = np.column_stack([lats[mask], lons[mask]])
        v = vals[mask]
        try:
            g = griddata(pts, v, target, method="linear")
            nan = np.isnan(g)
            if nan.any():
                g[nan] = griddata(pts, v, target[nan], method="nearest")
            out[ti] = g.reshape(n_y, n_x)
        except Exception:  # noqa: BLE001
            continue

    return xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name="sal_calcofi",
    )


def _fetch_glorys12_salinity() -> xr.DataArray | None:
    """Server-side subset of GLORYS12 surface salinity over the master bbox/time."""
    dest = CACHE_DIR / "glorys12_so_surface.nc"
    try:
        import copernicusmarine as cm

        if not dest.exists():
            log.info("Requesting GLORYS12 surface salinity subset...")
            pad = 0.5
            cm.subset(
                dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
                variables=["so"],
                minimum_longitude=LON_MIN - pad,
                maximum_longitude=LON_MAX + pad,
                minimum_latitude=LAT_MIN - pad,
                maximum_latitude=LAT_MAX + pad,
                start_datetime=f"{TIME_START}T00:00:00",
                end_datetime=f"{TIME_END}T23:59:59",
                minimum_depth=0.0,
                maximum_depth=1.0,
                output_filename=dest.name,
                output_directory=str(dest.parent),
                overwrite=True,
            )
        if not dest.exists():
            return None
        ds = xr.open_dataset(dest)
        da = ds["so"]
        rename = {}
        if "latitude" in da.dims:
            rename["latitude"] = "lat"
        if "longitude" in da.dims:
            rename["longitude"] = "lon"
        if rename:
            da = da.rename(rename)
        # Collapse depth if present
        if "depth" in da.dims:
            da = da.isel(depth=0, drop=True)
        da = da.assign_coords(time=da["time"].dt.floor("1D"))
        da = da.drop_duplicates("time").sortby("time")
        da = da.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")
        da = da.reindex(time=MASTER_TIME)
        ds.close()
        return da
    except Exception as exc:  # noqa: BLE001
        log.error("GLORYS12 salinity fetch failed: %s", exc)
        return None


def fetch_salinity() -> xr.Dataset:
    df = _fetch_calcofi()
    cal = None
    if df is not None and len(df) > 0:
        cal = _interp_calcofi_to_master(df)
        log.info("CalCOFI interp coverage: %.1f%%",
                 100 * np.isfinite(cal.values).mean())

    glorys = _fetch_glorys12_salinity()
    if glorys is not None:
        log.info("GLORYS12 coverage: %.1f%%", 100 * np.isfinite(glorys.values).mean())

    shape = (len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON))
    sal = np.full(shape, np.nan, dtype="float32")
    if cal is not None:
        sal = np.where(np.isfinite(cal.values), cal.values, sal)
    if glorys is not None:
        need = ~np.isfinite(sal)
        sal = np.where(need & np.isfinite(glorys.values), glorys.values, sal)
    # Final constant fallback
    sal = np.where(np.isfinite(sal), sal, CONST_SAL)

    sal_da = xr.DataArray(
        sal.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name="salinity",
        attrs={
            "units": "PSU",
            "source": "CalCOFI bottle (surface, <=10 m) + GLORYS12 fill, "
                      f"constant {CONST_SAL} fallback",
        },
    )
    qc_variable("salinity", sal_da)
    return xr.Dataset({"salinity": sal_da})


if __name__ == "__main__":
    print(fetch_salinity())
