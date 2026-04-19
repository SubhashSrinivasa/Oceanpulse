"""Download Sea Surface Height from CMEMS DUACS L4 0.125 deg product.

Also downloads WOA18 monthly dissolved-oxygen climatology from NOAA NCEI
for use as a fallback when in-situ DO is unavailable.

CMEMS credentials: read from CMEMS_USERNAME / CMEMS_PASSWORD environment
variables. If absent the copernicusmarine package uses its stored login
(from a prior ``copernicusmarine login`` interactive session).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import xarray as xr

from .config import (
    CACHE_DIR,
    CMEMS_SSH_DATASET,
    CMEMS_SSH_VARIABLE,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MASTER_LAT,
    MASTER_LON,
    MASTER_TIME,
    TIME_END,
    TIME_START,
    WOA18_DO_URL_TMPL,
)
from .utils import download_to_file, get_logger, qc_variable

log = get_logger("download_cmems")

SSH_CACHE = CACHE_DIR / "cmems_ssh.nc"
WOA_DO_CACHE_TMPL = "woa18_do_m{mm:02d}.nc"


# ---------------------------------------------------------------------------
# CMEMS credentials helper
# ---------------------------------------------------------------------------

def _cmems_credentials() -> dict:
    username = os.getenv("CMEMS_USERNAME")
    password = os.getenv("CMEMS_PASSWORD")
    if not username or not password:
        log.info(
            "CMEMS_USERNAME/CMEMS_PASSWORD not set — "
            "relying on stored copernicusmarine credentials."
        )
        return {}
    return {"username": username, "password": password}


# ---------------------------------------------------------------------------
# SSH
# ---------------------------------------------------------------------------

def fetch_ssh() -> xr.Dataset:
    """Download CMEMS DUACS SSH, regrid to 0.25 deg master grid."""
    if not SSH_CACHE.exists():
        _download_ssh()

    if not SSH_CACHE.exists():
        log.warning("SSH download failed — returning NaN-filled SSH")
        return _nan_ssh()

    try:
        ds = xr.open_dataset(SSH_CACHE)
        var = CMEMS_SSH_VARIABLE if CMEMS_SSH_VARIABLE in ds else list(ds.data_vars)[0]
        ssh = ds[var]
        rename = {}
        if "latitude" in ssh.dims:
            rename["latitude"] = "lat"
        if "longitude" in ssh.dims:
            rename["longitude"] = "lon"
        if rename:
            ssh = ssh.rename(rename)
        ssh = ssh.assign_coords(time=ssh["time"].dt.floor("1D"))
        ssh = ssh.drop_duplicates("time").sortby("time")
        # Native 0.125 deg → master 0.25 deg
        ssh = ssh.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")
        ssh = ssh.reindex(time=MASTER_TIME)
        ssh.name = "ssh"
        ssh.attrs.update(units="m", source=f"CMEMS DUACS L4 ({CMEMS_SSH_VARIABLE})")
        ds.close()
        qc_variable("ssh", ssh.values)
        return xr.Dataset({"ssh": ssh})
    except Exception as exc:  # noqa: BLE001
        log.error("Could not open SSH cache: %s", exc)
        return _nan_ssh()


def _download_ssh() -> None:
    try:
        import copernicusmarine as cm

        creds = _cmems_credentials()
        log.info("Requesting CMEMS SSH subset (%s – %s) ...", TIME_START, TIME_END)
        cm.subset(
            dataset_id=CMEMS_SSH_DATASET,
            variables=[CMEMS_SSH_VARIABLE],
            minimum_longitude=LON_MIN - 0.5,
            maximum_longitude=LON_MAX + 0.5,
            minimum_latitude=LAT_MIN - 0.5,
            maximum_latitude=LAT_MAX + 0.5,
            start_datetime=f"{TIME_START}T00:00:00",
            end_datetime=f"{TIME_END}T23:59:59",
            output_filename=SSH_CACHE.name,
            output_directory=str(SSH_CACHE.parent),
            overwrite=True,
            **creds,
        )
        if SSH_CACHE.exists():
            log.info("CMEMS SSH saved: %.1f MB", SSH_CACHE.stat().st_size / 1e6)
        else:
            log.error("CMEMS subset completed but output file not found")
    except Exception as exc:  # noqa: BLE001
        log.error("CMEMS SSH download failed: %s", exc)


def _nan_ssh() -> xr.Dataset:
    nan = np.full(
        (len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)), np.nan, dtype="float32"
    )
    ssh = xr.DataArray(
        nan,
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name="ssh",
        attrs={"source": "MISSING — NaN filled"},
    )
    return xr.Dataset({"ssh": ssh})


# ---------------------------------------------------------------------------
# WOA18 dissolved oxygen monthly climatology
# ---------------------------------------------------------------------------

def fetch_woa18_dissolved_oxygen() -> xr.DataArray:
    """Return a (12, n_lat, n_lon) monthly DO climatology array.

    Variable is surface (~0 m) dissolved oxygen in ml/L.
    Downloads WOA18 1-degree monthly files and interpolates to master grid.
    """
    def _fetch_do_month(mm: int) -> tuple[int, bool]:
        cache = CACHE_DIR / WOA_DO_CACHE_TMPL.format(mm=mm)
        if cache.exists():
            return mm, True
        url = WOA18_DO_URL_TMPL.format(mm=mm)
        log.info("Downloading WOA18 DO month %02d ...", mm)
        # NCEI servers can return 503 under load — use more retries with longer backoff
        ok = download_to_file(url, cache, logger=log, retries=6, timeout=180, backoff=15.0)
        if not ok:
            log.warning("WOA18 DO month %02d unavailable — using NaN", mm)
        return mm, ok

    from concurrent.futures import ThreadPoolExecutor, as_completed as asc
    month_results = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_fetch_do_month, mm): mm for mm in range(1, 13)}
        for future in asc(futures):
            mm, ok = future.result()
            month_results[mm] = ok

    monthly = []
    for mm in range(1, 13):
        cache = CACHE_DIR / WOA_DO_CACHE_TMPL.format(mm=mm)
        if not month_results.get(mm) or not cache.exists():
            monthly.append(None)
            continue
        try:
            ds = xr.open_dataset(cache, decode_times=False)
            # WOA18 oxygen variable: o_an (objectively analysed mean)
            var = "o_an" if "o_an" in ds else list(ds.data_vars)[0]
            da = ds[var]
            rename = {}
            if "lat" not in da.dims and "latitude" in da.dims:
                rename["latitude"] = "lat"
            if "lon" not in da.dims and "longitude" in da.dims:
                rename["longitude"] = "lon"
            if rename:
                da = da.rename(rename)
            # Take surface depth (depth=0)
            if "depth" in da.dims:
                da = da.isel(depth=0, drop=True)
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            # Interp from 1 deg to master 0.25 deg
            da = da.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")
            monthly.append(da.values.astype("float32"))
            ds.close()
        except Exception as exc:  # noqa: BLE001
            log.warning("WOA18 DO month %02d read error: %s", mm, exc)
            monthly.append(None)

    # Build (12, n_lat, n_lon) array; fill missing months with NaN
    n_y, n_x = len(MASTER_LAT), len(MASTER_LON)
    arr = np.full((12, n_y, n_x), np.nan, dtype="float32")
    for i, m in enumerate(monthly):
        if m is not None:
            arr[i] = m

    clim = xr.DataArray(
        arr,
        dims=("month", "lat", "lon"),
        coords={
            "month": np.arange(1, 13),
            "lat": MASTER_LAT,
            "lon": MASTER_LON,
        },
        name="dissolved_oxygen_climatology",
        attrs={"units": "ml/L", "source": "WOA18 annual objectively analysed mean, surface"},
    )
    log.info(
        "WOA18 DO climatology ready — NaN fraction: %.1f%%",
        100 * np.isnan(arr).mean(),
    )
    return clim
