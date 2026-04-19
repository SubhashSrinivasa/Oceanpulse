"""Export ocean_cube.zarr to two CSV layouts for model training.

Output 1 (long / tabular):
  outputs/ocean_cube_long.csv
    One row per (time, lat, lon) over SEA cells only. Land cells (sst=NaN for
    all time) are dropped. All values are imputed; the CSV has no NaNs.
    Columns: time, lat, lon, month, day_of_year, <8 variables>.

Output 2 (wide per-location sequences):
  outputs/ocean_cube_sequences.csv
    One row per SEA (lat, lon) that passes the 50%-drop rule. Each variable
    column is a JSON list of values ordered chronologically on the shared
    time axis described in the first comment line. All arrays are fully
    dense; the CSV has no NaNs.

Imputation / drop policy:
  * All float variables (sst, sst_anomaly, ssh, chlorophyll, chlorophyll_log,
    mld, salinity) go through a 4-pass fill:
      1) temporal linear interp (<= 30 day gap)
      2) spatial griddata per timestep
      3) day-of-year climatology
      4) global ocean mean (final fallback so no NaN survives over ocean)
    Chlorophyll is filled in log10 space; land cells are never touched and
    are removed from both CSV outputs.
  * For `ocean_cube_sequences.csv`: if ANY checked variable at a given
    (lat, lon) has > 50% missing values in the pre-imputation cube, the
    whole location is dropped. Otherwise every variable is emitted as a
    fully dense imputed array.
  * For `ocean_cube_long.csv`: every sea cell is kept; imputation + the
    global-mean fallback guarantee no NaN escapes to the CSV.
"""
from __future__ import annotations

from ensure_deps import ensure_scientific_stack

ensure_scientific_stack()

import time as time_mod

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

# Float variables to run through the generalized 3-pass imputation.
# Chlorophyll is handled separately in log10 space; mld_source is an integer flag.
FLOAT_IMPUTE_VARS = ["sst", "sst_anomaly", "ssh", "mld", "salinity"]

# Variables considered when evaluating the drop rule for sequences.
# chlorophyll_log is derived from chlorophyll; mld_source is a categorical flag.
CHECK_VARS = ["sst", "sst_anomaly", "ssh", "chlorophyll", "mld", "salinity"]

# Drop a sequence row if ANY checked variable has more than this fraction
# of missing values in the pre-imputation cube. Raised from 0.30 to 0.50
# to keep more locations at the cost of heavier reliance on imputation.
DROP_THRESHOLD = 0.50


# --------------------------------------------------------------------------- #
# Shared 3-pass core
# --------------------------------------------------------------------------- #
def _three_pass(
    arr: np.ndarray,
    ocean: np.ndarray,
    times: np.ndarray,
    lat_vals: np.ndarray,
    lon_vals: np.ndarray,
    label: str,
) -> np.ndarray:
    """Four-pass spatio-temporal fill on a (time, lat, lon) float array.

    Pass 1 - temporal linear interp per cell (gap <= TEMPORAL_MAX_GAP).
    Pass 2 - spatial griddata per timestep (uses valid ocean neighbors;
             nearest-neighbor fallback for edge cells).
    Pass 3 - day-of-year climatology fill (global fallback across years).
    Pass 4 - global ocean mean fallback (guarantees zero NaN over ocean
             even if some DOY has no valid ocean data anywhere).

    Cells outside the `ocean` mask are left untouched (stay NaN).
    """
    arr = arr.copy().astype("float64")
    n_t, _, _ = arr.shape

    if not ocean.any():
        log.info("  [%s] empty ocean mask -- skipping 3-pass", label)
        return arr

    ocean_cells_per_t = int(ocean.sum())
    total_ocean = n_t * ocean_cells_per_t
    ocean_nan_count = int(np.isnan(arr[:, ocean]).sum())
    if ocean_nan_count == 0:
        log.info("  [%s] no ocean NaN -- skipping 3-pass", label)
        return arr
    log.info(
        "  [%s] starting: %d / %d (%.2f%%) ocean NaN",
        label, ocean_nan_count, total_ocean,
        100.0 * ocean_nan_count / total_ocean,
    )

    # --- Pass 1: temporal linear interpolation ------------------------------
    t0 = time_mod.time()
    da = xr.DataArray(
        arr,
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": lat_vals, "lon": lon_vals},
    )
    da = da.interpolate_na(dim="time", method="linear", max_gap=TEMPORAL_MAX_GAP)
    arr = da.values
    log.info("  [%s] Pass 1 temporal: %.1fs", label, time_mod.time() - t0)

    # --- Pass 2: spatial griddata per timestep ------------------------------
    t0 = time_mod.time()
    La, Lo = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    skipped = 0
    for ti in range(n_t):
        slice2d = arr[ti]
        null_ocean = ocean & np.isnan(slice2d)
        if not null_ocean.any():
            skipped += 1
            continue
        valid = ocean & ~np.isnan(slice2d)
        if valid.sum() < 3:
            continue
        src_pts = np.column_stack([La[valid], Lo[valid]])
        src_vals = slice2d[valid]
        tgt_pts = np.column_stack([La[null_ocean], Lo[null_ocean]])
        filled = griddata(src_pts, src_vals, tgt_pts, method="linear")
        still_nan = np.isnan(filled)
        if still_nan.any():
            filled[still_nan] = griddata(
                src_pts, src_vals, tgt_pts[still_nan], method="nearest"
            )
        arr[ti][null_ocean] = filled
    log.info(
        "  [%s] Pass 2 spatial: %.1fs  (skipped %d/%d NaN-free timesteps)",
        label, time_mod.time() - t0, skipped, n_t,
    )

    # --- Pass 3: DOY climatology fill --------------------------------------
    t0 = time_mod.time()
    doy = pd.to_datetime(times).dayofyear.to_numpy()
    doy_clim: dict[int, np.ndarray] = {}
    with np.errstate(invalid="ignore"):
        for d in np.unique(doy):
            doy_clim[int(d)] = np.nanmean(arr[doy == d], axis=0)
    for ti in range(n_t):
        still_null = np.isnan(arr[ti]) & ocean
        if still_null.any():
            arr[ti] = np.where(still_null, doy_clim[int(doy[ti])], arr[ti])
    pass3_remaining = int(np.isnan(arr[:, ocean]).sum())
    log.info(
        "  [%s] Pass 3 DOY clim: %.1fs  remaining ocean NaN: %d",
        label, time_mod.time() - t0, pass3_remaining,
    )

    # --- Pass 4: global ocean-mean fallback --------------------------------
    # Covers edge cases where Pass 3 still left NaN (e.g. a DOY with no
    # valid ocean data anywhere across all years).
    if pass3_remaining:
        t0 = time_mod.time()
        with np.errstate(invalid="ignore"):
            global_mean = float(np.nanmean(arr[:, ocean]))
        if not np.isfinite(global_mean):
            # Whole variable is NaN over ocean -- degrade to 0 so the CSV
            # is still writable; the pipeline's upstream QC should have
            # caught this already.
            log.warning(
                "  [%s] global mean is NaN -- falling back to 0.0 for Pass 4",
                label,
            )
            global_mean = 0.0
        for ti in range(n_t):
            still_null = np.isnan(arr[ti]) & ocean
            if still_null.any():
                arr[ti] = np.where(still_null, global_mean, arr[ti])
        remaining = int(np.isnan(arr[:, ocean]).sum())
        log.info(
            "  [%s] Pass 4 global-mean (%.4f): %.1fs  remaining ocean NaN: %d",
            label, global_mean, time_mod.time() - t0, remaining,
        )
    return arr


def _ocean_mask(ds: xr.Dataset) -> np.ndarray:
    """Ocean = cells where SST is not NaN at every single timestep."""
    return ~np.isnan(ds["sst"].values).all(axis=0)


# --------------------------------------------------------------------------- #
# Per-variable imputation wrappers
# --------------------------------------------------------------------------- #
def impute_chlorophyll(ds: xr.Dataset) -> xr.Dataset:
    """Chlorophyll imputation in log10 space; rebuilds both chlorophyll vars."""
    log.info("Imputing chlorophyll (3-pass, log10 space)...")
    # Start from the pre-computed log10 field stage3 already produced.
    chl_log = ds["chlorophyll_log"].values.astype("float64")

    chl_log = _three_pass(
        chl_log,
        _ocean_mask(ds),
        ds["time"].values,
        ds["lat"].values,
        ds["lon"].values,
        label="chlorophyll_log",
    )
    chl_linear = np.power(10.0, chl_log).astype("float32")
    chl_log_f32 = chl_log.astype("float32")

    ds = ds.assign(
        chlorophyll=xr.DataArray(
            chl_linear,
            dims=("time", "lat", "lon"),
            coords=ds["chlorophyll"].coords,
            attrs=ds["chlorophyll"].attrs,
        ),
        chlorophyll_log=xr.DataArray(
            chl_log_f32,
            dims=("time", "lat", "lon"),
            coords=ds["chlorophyll_log"].coords,
            attrs=ds["chlorophyll_log"].attrs,
        ),
    )
    return ds


def impute_variable_3pass(ds: xr.Dataset, var: str) -> xr.Dataset:
    """Linear-space 3-pass imputation for a single float variable."""
    log.info("Imputing %s (3-pass, linear space)...", var)
    arr = ds[var].values.astype("float64")
    filled = _three_pass(
        arr,
        _ocean_mask(ds),
        ds["time"].values,
        ds["lat"].values,
        ds["lon"].values,
        label=var,
    )
    ds = ds.assign(
        {
            var: xr.DataArray(
                filled.astype("float32"),
                dims=("time", "lat", "lon"),
                coords=ds[var].coords,
                attrs=ds[var].attrs,
            )
        }
    )
    return ds


# --------------------------------------------------------------------------- #
# CSV writers
# --------------------------------------------------------------------------- #
def _check_uniform_time(times: np.ndarray) -> tuple[bool, str]:
    if len(times) < 2:
        return False, "only one timestamp"
    secs = times.astype("datetime64[s]").astype("int64")
    diffs = np.diff(secs)
    if not np.all(diffs == diffs[0]):
        return False, f"non-uniform spacing (min={int(diffs.min())}s, max={int(diffs.max())}s)"
    return True, f"uniform spacing = {int(diffs[0])} s ({int(diffs[0]) // 86400} day)"


def write_long(ds: xr.Dataset, raw_ds: xr.Dataset) -> None:
    """Tabular CSV, one row per (time, lat, lon) SEA cell, no NaN values.

    Land identification: in the RAW cube every land cell has sst NaN for
    every time step. The 4-pass fill only touches the ocean mask, so any
    cell that still has sst == NaN after imputation is land. We therefore
    drop rows where sst is NaN; everything that survives is sea.
    """
    log.info("Building long-format DataFrame...")
    df = ds.to_dataframe().reset_index()
    df = df.drop(columns=[c for c in ("dayofyear",) if c in df.columns])

    # Sea filter: sst is NaN everywhere on land and finite everywhere on sea
    # after imputation. Land -> dropped.
    before = len(df)
    df = df[df["sst"].notna()].copy()
    n_sea_cells = df[["lat", "lon"]].drop_duplicates().shape[0]
    log.info(
        "Long: dropped %d land rows, kept %d sea rows (%d sea cells x %d days)",
        before - len(df), len(df), n_sea_cells, ds.sizes["time"],
    )

    # Safety net: imputation + Pass-4 global-mean should have filled every
    # sea-cell NaN. If anything slipped through, fill with column mean to
    # guarantee a NaN-free CSV.
    var_cols_check = [c for c in df.columns if c not in ("time", "lat", "lon")]
    remaining_nan_total = int(df[var_cols_check].isna().sum().sum())
    if remaining_nan_total:
        log.warning(
            "Long: %d residual NaN values on sea cells -- filling with column mean",
            remaining_nan_total,
        )
        for c in var_cols_check:
            col_na = df[c].isna().sum()
            if col_na:
                col_mean = df[c].mean()
                if not np.isfinite(col_mean):
                    col_mean = 0.0
                df[c] = df[c].fillna(col_mean)

    df["time"] = pd.to_datetime(df["time"])
    df["month"] = df["time"].dt.month.astype("int8")
    df["day_of_year"] = df["time"].dt.dayofyear.astype("int16")

    front = ["time", "lat", "lon", "month", "day_of_year"]
    var_cols = [c for c in df.columns if c not in front]
    df = df[front + var_cols]

    total_remaining = int(df[var_cols].isna().sum().sum())
    log.info("Writing %s  (remaining NaN: %d)", LONG_PATH, total_remaining)
    df.to_csv(LONG_PATH, index=False)
    log.info(
        "Long CSV: %d rows, %.1f MB",
        len(df), LONG_PATH.stat().st_size / 1e6,
    )


def _format_sequence(arr: np.ndarray) -> str:
    """Turn a 1-D float array into a compact JSON list. NaN -> null."""
    out = []
    for x in arr:
        if np.isnan(x):
            out.append("null")
        else:
            out.append(f"{x:.{FLOAT_FMT}f}")
    return "[" + ",".join(out) + "]"


def write_sequences(ds: xr.Dataset, raw_ds: xr.Dataset) -> None:
    """Write sequences CSV applying the DROP_THRESHOLD rule on the raw cube."""
    times = ds["time"].values
    ok, note = _check_uniform_time(times)
    log.info("Time axis uniform? %s -- %s", ok, note)
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

    # Post-imputation arrays (what we write out).
    arrays = {
        v: np.asarray(ds[v].values, dtype="float32")
            .reshape(n_t, n_cells)
            .T  # -> (n_cells, n_t)
        for v in var_names
    }
    # Pre-imputation arrays (for the 30% drop check only).
    raw_arrays = {
        v: np.asarray(raw_ds[v].values, dtype="float32")
            .reshape(n_t, n_cells)
            .T
        for v in CHECK_VARS
        if v in raw_ds.data_vars
    }

    La, Lo = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    latlon = np.column_stack([La.ravel(), Lo.ravel()])

    # Land mask: sst all-NaN across time in the RAW cube.
    land_mask = np.isnan(raw_arrays["sst"]).all(axis=1)
    ocean_cells = ~land_mask
    n_ocean = int(ocean_cells.sum())
    log.info("Land cells: %d, ocean cells: %d", int(land_mask.sum()), n_ocean)

    # Per-(cell, var) NaN ratio on pre-imputation data.
    nan_ratios = {v: np.isnan(arr).mean(axis=1) for v, arr in raw_arrays.items()}

    # drop rule: any checked variable exceeding threshold disqualifies the cell.
    thresh_pct = int(round(DROP_THRESHOLD * 100))
    over_threshold = np.zeros(n_cells, dtype=bool)
    per_var_fail: dict[str, int] = {}
    for v, ratio in nan_ratios.items():
        fail = (ratio > DROP_THRESHOLD) & ocean_cells
        per_var_fail[v] = int(fail.sum())
        over_threshold |= fail

    # Per-variable NaN-ratio histogram (ocean cells only).
    log.info("Pre-imputation NaN-ratio distribution (ocean cells, n=%d):", n_ocean)
    bins = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.0001]
    header = "  " + "".ljust(18) + " ".join(f"<{bins[i+1]:>5.2f}" for i in range(len(bins) - 1))
    log.info(header)
    for v, ratio in nan_ratios.items():
        r = ratio[ocean_cells]
        hist, _ = np.histogram(r, bins=bins)
        cells = " ".join(f"{int(c):>6d}" for c in hist)
        log.info(
            "  %-16s %s  fail>%d%%: %d",
            v, cells, thresh_pct, per_var_fail[v],
        )

    keep = ocean_cells & ~over_threshold
    n_drop_land = int(land_mask.sum())
    n_drop_thresh = int(over_threshold.sum())
    n_keep = int(keep.sum())
    log.info(
        "Sequences cell selection: kept=%d  dropped_land=%d  dropped_%d%%=%d  "
        "(of %d grid cells)",
        n_keep, n_drop_land, thresh_pct, n_drop_thresh, n_cells,
    )
    if n_keep == 0:
        log.error("Sequence export aborted: no cells passed the %d%% threshold",
                  thresh_pct)
        return

    latlon = latlon[keep]
    for v in var_names:
        arrays[v] = arrays[v][keep]

    # Guarantee no NaN survives into the sequences CSV.
    for v in var_names:
        nan_count = int(np.isnan(arrays[v]).sum())
        if nan_count:
            log.warning(
                "Sequences: %s still has %d NaN after imputation -- filling with var mean",
                v, nan_count,
            )
            with np.errstate(invalid="ignore"):
                mean_val = float(np.nanmean(arrays[v]))
            if not np.isfinite(mean_val):
                mean_val = 0.0
            arrays[v] = np.where(np.isnan(arrays[v]), mean_val, arrays[v])

    out = pd.DataFrame({"lat": latlon[:, 0], "lon": latlon[:, 1]})
    for v in var_names:
        t0 = time_mod.time()
        log.info("  serializing %s ...", v)
        out[v] = [_format_sequence(row) for row in arrays[v]]
        log.info("    %s done in %.1fs", v, time_mod.time() - t0)

    t_start = str(pd.Timestamp(times[0]).date())
    t_end = str(pd.Timestamp(times[-1]).date())
    dt_days = int(np.diff(times.astype("datetime64[s]").astype("int64"))[0] // 86400)
    header_line = (
        f"# time_start={t_start} time_end={t_end} n_timesteps={n_t} "
        f"dt_days={dt_days} sequence_length={n_t}\n"
    )

    log.info("Writing %s", SEQ_PATH)
    with open(SEQ_PATH, "w") as f:
        f.write(header_line)
        out.to_csv(f, index=False)
    log.info(
        "Sequence CSV: %d rows, %.1f MB", len(out), SEQ_PATH.stat().st_size / 1e6
    )


def main() -> None:
    log.info("Opening %s", ZARR_PATH)
    ds = xr.open_zarr(ZARR_PATH, consolidated=True).load()
    # Keep a raw copy BEFORE any imputation so the drop rule reflects
    # true data support rather than the post-fill dense cube.
    raw_ds = ds.copy(deep=True)

    ds = impute_chlorophyll(ds)
    for v in FLOAT_IMPUTE_VARS:
        ds = impute_variable_3pass(ds, v)

    write_long(ds, raw_ds)
    write_sequences(ds, raw_ds)
    log.info("Done.")


if __name__ == "__main__":
    main()
