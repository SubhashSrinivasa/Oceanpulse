"""Stage 6 - Assemble all variables and write ocean_cube.zarr."""
from __future__ import annotations

import numpy as np
import xarray as xr

from .config import (
    MASTER_LAT,
    MASTER_LON,
    MASTER_TIME,
    SANITY_BOUNDS,
    TIME_END,
    TIME_START,
    ZARR_PATH,
)
from .utils import get_logger

log = get_logger("stage6_assemble")

REQUIRED_VARS = [
    "sst",
    "sst_anomaly",
    "ssh",
    "chlorophyll",
    "chlorophyll_log",
    "salinity",
    "mld",
    "mld_source",
]


def _validate(ds: xr.Dataset) -> list[str]:
    issues: list[str] = []
    for v in REQUIRED_VARS:
        if v not in ds:
            issues.append(f"missing variable {v}")
            continue
        da = ds[v]
        if da.dims != ("time", "lat", "lon"):
            issues.append(f"{v} has dims {da.dims} (expected time/lat/lon)")

    if not np.all(np.diff(ds["lat"].values) > 0):
        issues.append("lat not strictly increasing")
    if not np.all(np.diff(ds["lon"].values) > 0):
        issues.append("lon not strictly increasing")
    if not np.all(np.diff(ds["time"].values.astype("int64")) > 0):
        issues.append("time not strictly increasing")

    for v in REQUIRED_VARS:
        da = ds[v]
        per_t = np.isnan(da.values).all(axis=(1, 2)) if da.dtype.kind == "f" else None
        if per_t is not None and per_t.any():
            n = int(per_t.sum())
            issues.append(f"{v} has {n} all-NaN time slice(s)")

    for v, (lo, hi) in SANITY_BOUNDS.items():
        if v not in ds:
            continue
        vals = ds[v].values
        if vals.size and np.isfinite(vals).any():
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
            if vmin < lo - 1e-6 or vmax > hi + 1e-6:
                issues.append(
                    f"{v} out of sanity bounds [{lo},{hi}] -> [{vmin:.3f},{vmax:.3f}]"
                )
    return issues


def assemble() -> xr.Dataset:
    from .stage1_sst import fetch_sst
    from .stage2_ssh import fetch_ssh
    from .stage3_chl import fetch_chl
    from .stage4_mld import fetch_mld
    from .stage5_salinity import fetch_salinity

    log.info("Running all stages...")
    sst_ds = fetch_sst()
    ssh_ds = fetch_ssh()
    chl_ds = fetch_chl()
    mld_ds = fetch_mld()
    sal_ds = fetch_salinity()

    master = xr.merge(
        [sst_ds, ssh_ds, chl_ds, mld_ds, sal_ds],
        combine_attrs="drop_conflicts",
    )

    master = master.assign_coords(
        lat=MASTER_LAT.astype("float64"),
        lon=MASTER_LON.astype("float64"),
        time=MASTER_TIME,
    )

    master.attrs.update(
        title="OceanPulse harmonized data cube - Central California 2018-2022",
        resolution="0.25 degree daily",
        time_range=f"{TIME_START} .. {TIME_END}",
        created_by="pipeline.stage6_assemble",
    )

    issues = _validate(master)
    if issues:
        log.warning("Validation issues (%d):", len(issues))
        for i in issues:
            log.warning("  - %s", i)
    else:
        log.info("Validation passed.")

    log.info("Writing %s", ZARR_PATH)
    if ZARR_PATH.exists():
        import shutil
        shutil.rmtree(ZARR_PATH)
    # Explicit chunking keeps per-chunk size ~15-40 MB on the expanded bbox
    # and avoids single-chunk writes for the ~41 M point cube.
    chunk_time = min(365, master.sizes["time"])
    chunk_lat = min(50, master.sizes["lat"])
    chunk_lon = min(50, master.sizes["lon"])
    master = master.chunk({"time": chunk_time, "lat": chunk_lat, "lon": chunk_lon})
    master.to_zarr(ZARR_PATH, mode="w", consolidated=True)
    log.info("Zarr written. Shape=%s", dict(master.sizes))
    return master


if __name__ == "__main__":
    ds = assemble()
    print(ds)
