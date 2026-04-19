"""Grid alignment utilities.

Two regridding strategies:

1. align_gridded_to_master(ds)   — for xarray Datasets / DataArrays already on
   a regular lat/lon grid; uses xarray.interp() for smooth remapping.

2. align_sparse_to_master(df)    — for sparse point observations (Argo, CalCOFI
   in-situ, species sightings); uses scipy.griddata or direct nearest-neighbour
   snap to the master grid.

Both return data on the exact master lat/lon grid so all datasets share
identical coordinate arrays and can be aligned by label.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

from .config import MASTER_LAT, MASTER_LON, MASTER_TIME
from .utils import get_logger, snap_to_grid

log = get_logger("grid_align")


# ---------------------------------------------------------------------------
# Gridded → master (xarray)
# ---------------------------------------------------------------------------

def align_gridded_to_master(
    ds: xr.Dataset | xr.DataArray,
    *,
    method: str = "linear",
    tolerance: float = 0.13,
) -> xr.Dataset | xr.DataArray:
    """Interpolate a gridded xarray object to the master lat/lon grid.

    Renames latitude/longitude → lat/lon if needed.
    Falls back to nearest-neighbour reindex if interpolation fails.
    """
    # Rename coordinate axes to canonical names
    rename = {}
    dims = ds.dims if isinstance(ds, xr.DataArray) else {d for v in ds.data_vars for d in ds[v].dims}
    coords = ds.coords
    if "latitude" in coords:
        rename["latitude"] = "lat"
    if "longitude" in coords:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    # Normalise time to midnight
    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"].dt.floor("1D"))
        ds = ds.drop_duplicates("time") if hasattr(ds, "drop_duplicates") else ds
        ds = ds.sortby("time")

    # Interpolate spatial dimensions
    try:
        ds_aligned = ds.interp(lat=MASTER_LAT, lon=MASTER_LON, method=method)
    except Exception as exc:  # noqa: BLE001
        log.warning("interp failed (%s) — falling back to reindex nearest", exc)
        ds_aligned = ds.reindex(lat=MASTER_LAT, lon=MASTER_LON, method="nearest", tolerance=tolerance)

    # Reindex time if present
    if "time" in ds_aligned.coords:
        ds_aligned = ds_aligned.reindex(time=MASTER_TIME)

    return ds_aligned


# ---------------------------------------------------------------------------
# Sparse point data → gridded (per timestep)
# ---------------------------------------------------------------------------

def sparse_to_daily_grid(
    df: pd.DataFrame,
    value_col: str,
    *,
    time_window_days: int = 7,
    fill_method: str = "linear",
) -> xr.DataArray:
    """Interpolate sparse (time, lat, lon, value) point data onto the master grid.

    For each master timestep, uses points within ±time_window_days and
    scipy.griddata to produce a gridded field.  Useful for Argo salinity
    and CalCOFI in-situ measurements.

    Parameters
    ----------
    df            : DataFrame with columns time, lat, lon, <value_col>
    value_col     : name of the column to grid
    time_window_days : half-width of the time window for neighbour search
    fill_method   : 'linear' or 'nearest'
    """
    n_t = len(MASTER_TIME)
    n_y = len(MASTER_LAT)
    n_x = len(MASTER_LON)
    out = np.full((n_t, n_y, n_x), np.nan, dtype="float32")

    times = df["time"].to_numpy(dtype="datetime64[ns]")
    lats = df["lat"].to_numpy(dtype=float)
    lons = df["lon"].to_numpy(dtype=float)
    vals = df[value_col].to_numpy(dtype=float)

    Xg, Yg = np.meshgrid(MASTER_LON, MASTER_LAT)
    target = np.column_stack([Yg.ravel(), Xg.ravel()])
    window = np.timedelta64(time_window_days, "D")

    master_np = MASTER_TIME.to_numpy(dtype="datetime64[ns]")
    for ti, t in enumerate(master_np):
        mask = np.abs(times - t) <= window
        if mask.sum() < 4:
            continue
        pts = np.column_stack([lats[mask], lons[mask]])
        v = vals[mask]
        finite = np.isfinite(v)
        if finite.sum() < 4:
            continue
        pts, v = pts[finite], v[finite]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g = griddata(pts, v, target, method=fill_method)
            if fill_method == "linear":
                nan_idx = np.isnan(g)
                if nan_idx.any():
                    g[nan_idx] = griddata(pts, v, target[nan_idx], method="nearest")
            out[ti] = g.reshape(n_y, n_x)
        except Exception:  # noqa: BLE001
            continue

    return xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name=value_col,
    )


# ---------------------------------------------------------------------------
# Monthly climatology → daily broadcast
# ---------------------------------------------------------------------------

def monthly_clim_to_daily(clim: xr.DataArray) -> xr.DataArray:
    """Broadcast a (month, lat, lon) climatology to (time, lat, lon) daily.

    Assigns each calendar day the value of its month.
    """
    months = MASTER_TIME.month.to_numpy()  # shape (n_t,)
    n_t, n_y, n_x = len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)
    out = np.full((n_t, n_y, n_x), np.nan, dtype="float32")
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        if len(idx) == 0:
            continue
        month_data = clim.sel(month=m).values
        out[idx] = month_data[np.newaxis, :, :]
    return xr.DataArray(
        out,
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name=clim.name or "climatology",
    )
