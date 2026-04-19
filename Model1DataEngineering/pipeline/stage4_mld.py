"""Stage 4 - Argo-derived mixed layer depth.

Strategy:
  * Pull Argo profiles for the bbox/time window via argopy.
  * Compute MLD using de Boyer Montegut density threshold (0.03 kg/m^3 from 10 m ref).
  * Spatial/temporal interpolation to master grid.
  * Gap-fill with WOA23 monthly mixed layer depth climatology.
  * Final fallback: constant 50 m.

mld_source flag: 0 = Argo interp, 1 = WOA23, 2 = constant fallback.
"""
from __future__ import annotations

import warnings
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

log = get_logger("stage4_mld")

DENS_THRESHOLD = 0.03  # kg m^-3
REF_DEPTH = 10.0  # meters


def _mld_from_profile(pres: np.ndarray, sa: np.ndarray, ct: np.ndarray) -> float:
    """Compute MLD (m) from one Argo profile using density threshold.

    pres = pressure (dbar), sa = absolute salinity, ct = conservative temperature.
    Returns NaN if we cannot find a valid crossing.
    """
    import gsw

    mask = np.isfinite(pres) & np.isfinite(sa) & np.isfinite(ct)
    pres, sa, ct = pres[mask], sa[mask], ct[mask]
    if pres.size < 4 or pres.min() > REF_DEPTH + 5 or pres.max() < 40:
        return np.nan
    order = np.argsort(pres)
    pres, sa, ct = pres[order], sa[order], ct[order]
    sigma0 = gsw.sigma0(sa, ct)
    # Interp reference sigma at REF_DEPTH
    ref_sigma = np.interp(REF_DEPTH, pres, sigma0)
    diff = sigma0 - ref_sigma
    over = np.where(diff >= DENS_THRESHOLD)[0]
    if over.size == 0:
        return np.nan
    i = over[0]
    if i == 0:
        return float(pres[0])
    # linear interpolation of crossing depth
    p0, p1 = pres[i - 1], pres[i]
    d0, d1 = diff[i - 1], diff[i]
    if d1 == d0:
        return float(p1)
    frac = (DENS_THRESHOLD - d0) / (d1 - d0)
    return float(p0 + frac * (p1 - p0))


def _fetch_argo_profiles() -> pd.DataFrame | None:
    try:
        from argopy import DataFetcher
    except Exception as exc:  # noqa: BLE001
        log.error("argopy import failed: %s", exc)
        return None
    try:
        f = DataFetcher(src="erddap", parallel=False).region(
            [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, 0, 1500, TIME_START, TIME_END]
        )
        log.info("Requesting Argo profiles for bbox/time...")
        ds = f.to_xarray()
        log.info("Argo fetched: %d measurements", ds.sizes.get("N_POINTS", 0))
        # Convert to profile-level DataFrame
        return ds.to_dataframe().reset_index()
    except Exception as exc:  # noqa: BLE001
        log.error("Argo fetch failed: %s", exc)
        return None


def _mld_from_argo() -> xr.DataArray | None:
    cache = CACHE_DIR / "argo_profiles.parquet"
    if cache.exists():
        log.info("Using cached Argo profiles")
        df = pd.read_parquet(cache)
    else:
        df = _fetch_argo_profiles()
        if df is None or df.empty:
            return None
        df.to_parquet(cache, index=False)
        log.info("Cached Argo to %s (%.2f MB)", cache.name, cache.stat().st_size / 1e6)

    # argopy exposes: PRES, PSAL, TEMP, LATITUDE, LONGITUDE, TIME, PLATFORM_NUMBER, CYCLE_NUMBER
    need = {"PRES", "PSAL", "TEMP", "LATITUDE", "LONGITUDE", "TIME"}
    missing = need - set(df.columns)
    if missing:
        log.error("Argo DataFrame missing cols: %s", missing)
        return None

    import gsw

    records = []
    profile_keys = ["PLATFORM_NUMBER", "CYCLE_NUMBER"] if {"PLATFORM_NUMBER", "CYCLE_NUMBER"}.issubset(df.columns) else ["LATITUDE", "LONGITUDE", "TIME"]
    for _, grp in df.groupby(profile_keys):
        lat = float(grp["LATITUDE"].iloc[0])
        lon = float(grp["LONGITUDE"].iloc[0])
        tval = pd.to_datetime(grp["TIME"].iloc[0])
        pres = grp["PRES"].to_numpy(dtype=float)
        psal = grp["PSAL"].to_numpy(dtype=float)
        temp = grp["TEMP"].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sa = gsw.SA_from_SP(psal, pres, lon, lat)
            ct = gsw.CT_from_t(sa, temp, pres)
            mld = _mld_from_profile(pres, sa, ct)
        if np.isfinite(mld):
            records.append({"lat": lat, "lon": lon, "time": tval.normalize(), "mld": mld})

    if not records:
        log.warning("No valid Argo MLDs computed")
        return None

    mld_df = pd.DataFrame(records)
    log.info("Argo MLD records: %d", len(mld_df))

    # Grid to master by averaging all profiles in each (day, lat, lon) bin.
    from scipy.interpolate import griddata

    # Per-day spatial nearest-neighbor with a temporal smoothing window (30 days).
    # We don't have many profiles per day — take profiles within ±15 days
    # and IDW-average to each grid point.
    out = np.full((len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)), np.nan, dtype="float32")
    mld_df = mld_df.sort_values("time").reset_index(drop=True)

    lats_pts = mld_df["lat"].to_numpy()
    lons_pts = mld_df["lon"].to_numpy()
    times_pts = mld_df["time"].to_numpy()
    mld_pts = mld_df["mld"].to_numpy()

    times_master = MASTER_TIME.to_numpy()
    window = np.timedelta64(15, "D")

    # Simple grid for interp
    Xg, Yg = np.meshgrid(MASTER_LON, MASTER_LAT)
    target_xy = np.column_stack([Yg.ravel(), Xg.ravel()])

    for ti, t in enumerate(times_master):
        mask = np.abs(times_pts - t) <= window
        if mask.sum() < 3:
            continue
        pts = np.column_stack([lats_pts[mask], lons_pts[mask]])
        vals = mld_pts[mask]
        try:
            grid_vals = griddata(pts, vals, target_xy, method="linear")
            # fill remaining with nearest
            nan_mask = np.isnan(grid_vals)
            if nan_mask.any():
                grid_near = griddata(pts, vals, target_xy[nan_mask], method="nearest")
                grid_vals[nan_mask] = grid_near
            out[ti] = grid_vals.reshape(len(MASTER_LAT), len(MASTER_LON))
        except Exception:  # noqa: BLE001
            continue

    da = xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name="mld_argo",
    )
    return da


def _mld_from_woa23() -> xr.DataArray | None:
    """WOA23 mixed layer depth monthly climatology (density-based).

    NOTE: As of WOA23 (release 2024), NCEI does not publish a mixed-layer
    depth product (only T/S/O2/nutrients). We try the WOA18 MLD climatology
    at the same path shape as a best-effort fallback; if that also fails,
    stage 4 falls through to the constant 50 m fill defined in the plan.
    """
    base = "https://www.ncei.noaa.gov/data/oceans/woa/WOA18/DATA/mld/netcdf/decav81B0/1.00"
    monthly = []
    try:
        for mm in range(1, 13):
            out = CACHE_DIR / f"woa18_mld_{mm:02d}.nc"
            if not out.exists():
                url = f"{base}/woa18_decav81B0_M02{mm:02d}_01.nc"
                log.info("Downloading WOA18 MLD month %02d", mm)
                r = requests.get(url, timeout=120, stream=True)
                if r.status_code != 200:
                    log.error("WOA18 month %d HTTP %s", mm, r.status_code)
                    return None
                with open(out, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            monthly.append(out)
        log.info("WOA18 monthly downloads complete")

        # Read each month, subset, stack into 12-month array
        arrs = []
        for mm, p in enumerate(monthly, start=1):
            ds = xr.open_dataset(p, decode_times=False)
            # Find MLD variable (M_an = objectively analyzed mean)
            var_name = None
            for name in ds.data_vars:
                if name.startswith("M_an") or name == "mld" or "mixed" in name.lower():
                    var_name = name
                    break
            if var_name is None:
                # Just take first non-coord var
                var_name = list(ds.data_vars)[0]
            da = ds[var_name]
            # squeeze depth/time if singleton
            for dim in ("time", "depth"):
                if dim in da.dims and da.sizes[dim] == 1:
                    da = da.isel({dim: 0}, drop=True)
            rename = {}
            if "latitude" in da.dims:
                rename["latitude"] = "lat"
            if "longitude" in da.dims:
                rename["longitude"] = "lon"
            if rename:
                da = da.rename(rename)
            da = da.sel(
                lat=slice(LAT_MIN - 1, LAT_MAX + 1),
                lon=slice(LON_MIN - 1, LON_MAX + 1),
            )
            da = da.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")
            arrs.append(da.expand_dims(month=[mm]))
            ds.close()
        woa = xr.concat(arrs, dim="month")  # (12, lat, lon)

        # Expand to daily along MASTER_TIME by month lookup
        month_idx = MASTER_TIME.month
        daily = woa.sel(month=xr.DataArray(month_idx, dims="time", coords={"time": MASTER_TIME}))
        daily = daily.drop_vars("month", errors="ignore")
        daily.name = "mld_woa"
        return daily
    except Exception as exc:  # noqa: BLE001
        log.error("WOA23 MLD fetch/parse failed: %s", exc)
        return None


def fetch_mld() -> xr.Dataset:
    argo = _mld_from_argo()
    if argo is not None:
        n_valid = int(np.isfinite(argo.values).sum())
        log.info("Argo MLD valid cells: %d / %d", n_valid, argo.size)
    woa = _mld_from_woa23()

    shape = (len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON))
    mld = np.full(shape, np.nan, dtype="float32")
    src = np.full(shape, 2, dtype="int8")  # default: constant fallback

    if argo is not None:
        mld = np.where(np.isfinite(argo.values), argo.values, mld)
        src = np.where(np.isfinite(argo.values), 0, src)

    if woa is not None:
        need = ~np.isfinite(mld)
        mld = np.where(need & np.isfinite(woa.values), woa.values, mld)
        src = np.where(need & np.isfinite(woa.values), 1, src)

    remaining = ~np.isfinite(mld)
    mld = np.where(remaining, 50.0, mld)  # constant fallback

    mld_da = xr.DataArray(
        mld.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name="mld",
        attrs={"units": "m", "long_name": "Mixed layer depth (Argo->WOA23->50m)"},
    )
    src_da = xr.DataArray(
        src.astype("int8"),
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name="mld_source",
        attrs={"flag_meanings": "0=Argo, 1=WOA23, 2=constant_50m"},
    )
    qc_variable("mld", mld_da)
    qc_variable(
        "mld_source",
        src_da.astype("float32"),
        {"argo_pct": round(100 * (src == 0).mean(), 2),
         "woa_pct": round(100 * (src == 1).mean(), 2),
         "const_pct": round(100 * (src == 2).mean(), 2)},
    )
    return xr.Dataset({"mld": mld_da, "mld_source": src_da})


if __name__ == "__main__":
    print(fetch_mld())
