"""Stage 2 - CMEMS DUACS Sea Surface Height.

Uses copernicusmarine client (server-side subset, one file out).
Fallback: synthetic spatially-correlated Gaussian field, flagged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
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

log = get_logger("stage2_ssh")

# DUACS L4 global 0.125 deg daily gridded sea-level anomaly (MY reprocessed).
# The 0.25 deg MY product was retired; we pull native 0.125 deg and regrid.
DATASET_ID = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
VARIABLE = "sla"  # sea level anomaly


def _server_subset(dest: Path) -> bool:
    try:
        import copernicusmarine as cm

        log.info("Requesting CMEMS SSH subset -> %s", dest.name)
        pad = 0.5
        cm.subset(
            dataset_id=DATASET_ID,
            variables=[VARIABLE],
            minimum_longitude=LON_MIN - pad,
            maximum_longitude=LON_MAX + pad,
            minimum_latitude=LAT_MIN - pad,
            maximum_latitude=LAT_MAX + pad,
            start_datetime=f"{TIME_START}T00:00:00",
            end_datetime=f"{TIME_END}T23:59:59",
            output_filename=dest.name,
            output_directory=str(dest.parent),
            overwrite=True,
        )
        return dest.exists()
    except Exception as exc:  # noqa: BLE001
        log.error("CMEMS SSH subset failed: %s", exc)
        return False


def _synthetic_ssh() -> xr.DataArray:
    log.warning("Generating synthetic SSH fallback (Gaussian spatial field)")
    rng = np.random.default_rng(42)
    n_t, n_y, n_x = len(MASTER_TIME), len(MASTER_LAT), len(MASTER_LON)
    # Smooth field: random per-day Gaussian bump + small noise
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, n_y), np.linspace(-1, 1, n_x), indexing="ij"
    )
    base = np.exp(-(yy ** 2 + xx ** 2) / 0.8) * 0.1
    noise = rng.normal(0.0, 0.02, size=(n_t, n_y, n_x)).astype("float32")
    seasonal = 0.05 * np.sin(
        2 * np.pi * (np.arange(n_t) / 365.25)
    ).astype("float32")[:, None, None]
    data = base[None, :, :].astype("float32") + seasonal + noise
    return xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={"time": MASTER_TIME, "lat": MASTER_LAT, "lon": MASTER_LON},
        name="ssh",
        attrs={"source": "SYNTHETIC_FALLBACK", "units": "m"},
    )


def fetch_ssh() -> xr.Dataset:
    dest = CACHE_DIR / "cmems_ssh_2018_2022.nc"
    source_flag = "CMEMS_DUACS"
    if not dest.exists():
        ok = _server_subset(dest)
        if not ok:
            ssh = _synthetic_ssh()
            source_flag = "SYNTHETIC_FALLBACK"
            qc_variable("ssh", ssh, {"source": source_flag})
            return xr.Dataset({"ssh": ssh}, attrs={"ssh_source": source_flag})

    ds = xr.open_dataset(dest)
    var = VARIABLE if VARIABLE in ds else list(ds.data_vars)[0]
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
    # Native CMEMS L4 is 0.125 deg; average down to 0.25 deg master grid.
    ssh = ssh.interp(lat=MASTER_LAT, lon=MASTER_LON, method="linear")
    ssh = ssh.reindex(time=MASTER_TIME)
    ssh.name = "ssh"
    ssh.attrs.update(units="m", source=f"CMEMS DUACS L4 ({VARIABLE})")
    ds.close()

    qc_variable("ssh", ssh, {"source": source_flag})
    return xr.Dataset({"ssh": ssh}, attrs={"ssh_source": source_flag})


if __name__ == "__main__":
    print(fetch_ssh())
