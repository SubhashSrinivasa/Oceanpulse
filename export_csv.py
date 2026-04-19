"""Export ocean_cube.zarr to two CSV layouts for model training.

Output 1 (long / tabular):
  outputs/ocean_cube_long.csv
    One row per (time, lat, lon). Land cells (sst=NaN for all time) dropped.
    Columns: time, lat, lon, month, day_of_year, <8 variables>.

Output 2 (wide per-location sequences):
  outputs/ocean_cube_sequences.csv
    One row per (lat, lon). Each variable column is a JSON list of values
    ordered chronologically on the shared time axis described in the first
    comment line. Requires a uniformly spaced time axis; we verify before
    writing and abort if the check fails.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import griddata

from pipeline.config import OUTPUT_DIR, ZARR_PATH
from pipeline.utils import get_logger

log = get_logger("export_csv")

LONG_PATH = OUTPUT_DIR / "ocean_cube_long.csv"
SEQ_PATH = OUTPUT_DIR / "ocean_cube_sequences.csv"

FLOAT_FMT = 4  # decimals retained in sequence arrays
TEMPORAL_MAX_GAP = pd.Timedelta(days=30)  # max gap bridged by linear time interp


def impute_chlorophyll(ds: xr.Dataset) -> xr.Dataset:
    """Three-pass spatio-temporal imputation for chlorophyll (in log space).

    Pass 1 — temporal linear interp per cell (gap <= 30 days).
    Pass 2 — spatial IDW via griddata per timestep (uses valid ocean neighbors).
    Pass 3 — day-of-year climatology fill (global fallback; covers 2022-H2 block).
    """
    log.info("Imputing chlorophyll (3-pass spatio-temporal)...")
    chl_log = ds["chlorophyll_log"].values.copy().astype("float64")  # (t, lat, lon)
    n_t, n_lat, n_lon = chl_log.shape

    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values

    # Ocean mask: cells where sst is not all-NaN
    sst = ds["sst"].values
    ocean = ~np.isnan(sst).all(axis=0)  # (lat, lon)

    # --- Pass 1: temporal linear interpolation per ocean cell ----------------
    log.info("  Pass 1: temporal interp (max_gap=%s)...", TEMPORAL_MAX_GAP)
    times = ds["time"].values
    chl_da = xr.DataArray(chl_log, dims=("time", "lat", "lon"),
                          coords={"time": times, "lat": lat_vals, "lon": lon_vals})
    chl_da = chl_da.interpolate_na(dim="time", method="linear",
                                   max_gap=TEMPORAL_MAX_GAP)
    chl_log = chl_da.values

    # --- Pass 2: spatial griddata per timestep --------------------------------
    log.info("  Pass 2: spatial IDW per timestep...")
    La, Lo = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    ocean_pts = np.column_stack([La[ocean], Lo[ocean]])   # known-ocean locations
    all_pts = np.column_stack([La.ravel(), Lo.ravel()])   # all grid locations

    for ti in range(n_t):
        slice2d = chl_log[ti]                             # (lat, lon)
        null_ocean = ocean & np.isnan(slice2d)
        if not null_ocean.any():
            continue
        valid = ocean & ~np.isnan(slice2d)
        if valid.sum() < 3:
            continue
        src_pts = np.column_stack([La[valid], Lo[valid]])
        src_vals = slice2d[valid]
        tgt_pts = np.column_stack([La[null_ocean], Lo[null_ocean]])
        filled = griddata(src_pts, src_vals, tgt_pts, method="linear")
        # nearest-neighbor fallback for any remaining NaN (edge cells)
        still_nan = np.isnan(filled)
        if still_nan.any():
            filled[still_nan] = griddata(src_pts, src_vals,
                                         tgt_pts[still_nan], method="nearest")
        chl_log[ti][null_ocean] = filled

    # --- Pass 3: DOY climatology fill (global fallback) ----------------------
    log.info("  Pass 3: DOY climatology fill...")
    doy = pd.to_datetime(times).dayofyear
    doy_clim = {}
    for d in np.unique(doy):
        vals = chl_log[doy == d]          # all years' slice for this DOY
        mean = np.nanmean(vals, axis=0)   # (lat, lon)
        doy_clim[d] = mean
    for ti in range(n_t):
        still_null = np.isnan(chl_log[ti])
        if still_null.any():
            chl_log[ti] = np.where(still_null, doy_clim[doy[ti]], chl_log[ti])

    # Back-transform log -> linear chlorophyll
    chl_linear = np.power(10.0, chl_log).astype("float32")
    chl_log_f32 = chl_log.astype("float32")

    null_remaining = np.isnan(chl_log).sum()
    log.info("  Imputation complete. Remaining nulls: %d", null_remaining)

    ds = ds.assign(
        chlorophyll=xr.DataArray(chl_linear, dims=("time", "lat", "lon"),
                                 coords=ds["chlorophyll"].coords,
                                 attrs=ds["chlorophyll"].attrs),
        chlorophyll_log=xr.DataArray(chl_log_f32, dims=("time", "lat", "lon"),
                                     coords=ds["chlorophyll_log"].coords,
                                     attrs=ds["chlorophyll_log"].attrs),
    )
    return ds


def _check_uniform_time(times: np.ndarray) -> tuple[bool, str]:
    if len(times) < 2:
        return False, "only one timestamp"
    secs = times.astype("datetime64[s]").astype("int64")
    diffs = np.diff(secs)
    if not np.all(diffs == diffs[0]):
        return False, f"non-uniform spacing (min={int(diffs.min())}s, max={int(diffs.max())}s)"
    return True, f"uniform spacing = {int(diffs[0])} s ({int(diffs[0]) // 86400} day)"


def write_long(ds: xr.Dataset) -> None:
    log.info("Building long-format DataFrame...")
    df = ds.to_dataframe().reset_index()
    df = df.drop(columns=[c for c in ("dayofyear",) if c in df.columns])

    before = len(df)
    df = df[df["sst"].notna()].copy()
    log.info("Dropped %d land/gap rows, kept %d", before - len(df), len(df))

    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month.astype("int8")
    df["day_of_year"] = df["time"].dt.dayofyear.astype("int16")

    front = ["time", "lat", "lon", "month", "day_of_year"]
    var_cols = [c for c in df.columns if c not in front]
    df = df[front + var_cols]

    log.info("Writing %s", LONG_PATH)
    df.to_csv(LONG_PATH, index=False)
    log.info("Long CSV: %d rows, %.1f MB", len(df), LONG_PATH.stat().st_size / 1e6)


def _format_sequence(arr: np.ndarray) -> str:
    """Turn a 1-D float array into a compact JSON list. NaN -> null."""
    out = []
    for x in arr:
        if np.isnan(x):
            out.append("null")
        else:
            out.append(f"{x:.{FLOAT_FMT}f}")
    return "[" + ",".join(out) + "]"


def write_sequences(ds: xr.Dataset) -> None:
    times = ds["time"].values
    ok, note = _check_uniform_time(times)
    log.info("Time axis uniform? %s — %s", ok, note)
    if not ok:
        log.error(
            "Sequence export aborted: packing values as arrays requires a uniformly "
            "spaced shared time axis (so position i maps to the same date at every "
            "location). Current axis: %s.",
            note,
        )
        return

    var_names = list(ds.data_vars)
    lat_vals = ds["lat"].values
    lon_vals = ds["lon"].values
    n_t = len(times)

    n_cells = len(lat_vals) * len(lon_vals)
    log.info("Reshaping %d cells x %d timesteps ...", n_cells, n_t)

    arrays = {
        v: np.asarray(ds[v].values, dtype="float32")
            .reshape(n_t, n_cells)
            .T  # -> (n_cells, n_t)
        for v in var_names
    }

    La, Lo = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    latlon = np.column_stack([La.ravel(), Lo.ravel()])

    land_mask = np.isnan(arrays["sst"]).all(axis=1)
    keep = ~land_mask
    log.info("Dropping %d land cells, keeping %d", int(land_mask.sum()), int(keep.sum()))
    latlon = latlon[keep]
    for v in var_names:
        arrays[v] = arrays[v][keep]

    out = pd.DataFrame({"lat": latlon[:, 0], "lon": latlon[:, 1]})
    for v in var_names:
        log.info("  serializing %s", v)
        out[v] = [_format_sequence(row) for row in arrays[v]]

    t_start = str(pd.Timestamp(times[0]).date())
    t_end = str(pd.Timestamp(times[-1]).date())
    dt_days = int(np.diff(times.astype("datetime64[s]").astype("int64"))[0] // 86400)
    header = (
        f"# time_start={t_start} time_end={t_end} n_timesteps={n_t} "
        f"dt_days={dt_days} sequence_length={n_t}\n"
    )

    log.info("Writing %s", SEQ_PATH)
    with open(SEQ_PATH, "w") as f:
        f.write(header)
        out.to_csv(f, index=False)
    log.info(
        "Sequence CSV: %d rows, %.1f MB", len(out), SEQ_PATH.stat().st_size / 1e6
    )


def main() -> None:
    log.info("Opening %s", ZARR_PATH)
    ds = xr.open_zarr(ZARR_PATH, consolidated=True).load()
    ds = impute_chlorophyll(ds)
    write_long(ds)
    write_sequences(ds)
    log.info("Done.")


if __name__ == "__main__":
    main()
