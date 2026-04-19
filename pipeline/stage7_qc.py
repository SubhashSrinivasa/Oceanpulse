"""Stage 7 - QC report and missing-data map."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .config import MISSING_MAP_PATH, QC_REPORT_PATH, SANITY_BOUNDS, ZARR_PATH
from .utils import QC_RECORDS, get_logger

log = get_logger("stage7_qc")


def write_qc_report(ds: xr.Dataset) -> None:
    lines = []
    lines.append("OceanPulse Data Quality Report")
    lines.append("=" * 60)
    lines.append(f"Zarr: {ZARR_PATH}")
    lines.append(f"Dimensions: {dict(ds.sizes)}")
    lines.append(
        f"Lat: {float(ds.lat.min())} .. {float(ds.lat.max())} (n={ds.sizes['lat']})"
    )
    lines.append(
        f"Lon: {float(ds.lon.min())} .. {float(ds.lon.max())} (n={ds.sizes['lon']})"
    )
    lines.append(
        f"Time: {str(ds.time.values[0])[:10]} .. {str(ds.time.values[-1])[:10]} "
        f"(n={ds.sizes['time']})"
    )
    lines.append("")
    lines.append("Per-variable statistics")
    lines.append("-" * 60)
    header = f"{'variable':<18} {'nan%':>7} {'min':>10} {'max':>10} {'bounds_ok':>10}"
    lines.append(header)
    for v in ds.data_vars:
        vals = ds[v].values
        total = vals.size
        if total == 0:
            continue
        nan_pct = 100.0 * float(np.isnan(vals).sum()) / total
        if np.isfinite(vals).any():
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
        else:
            vmin = vmax = float("nan")
        lo, hi = SANITY_BOUNDS.get(v, (None, None))
        if lo is not None and np.isfinite(vmin) and np.isfinite(vmax):
            ok = (vmin >= lo - 1e-6) and (vmax <= hi + 1e-6)
        else:
            ok = True
        lines.append(
            f"{v:<18} {nan_pct:7.2f} {vmin:10.3f} {vmax:10.3f} {str(bool(ok)):>10}"
        )

    lines.append("")
    lines.append("Fallback flags / sources")
    lines.append("-" * 60)
    ssh_source = ds.attrs.get("ssh_source", "CMEMS_DUACS")
    lines.append(f"ssh_source: {ssh_source}")
    src = ds["mld_source"].values
    tot = src.size
    lines.append(
        f"mld_source pct: Argo={100*(src==0).sum()/tot:.1f}% "
        f"WOA18={100*(src==1).sum()/tot:.1f}% const={100*(src==2).sum()/tot:.1f}%"
    )

    QC_REPORT_PATH.write_text("\n".join(lines) + "\n")
    log.info("QC report -> %s", QC_REPORT_PATH)


def write_missing_map(ds: xr.Dataset) -> None:
    core = [
        "sst",
        "ssh",
        "chlorophyll",
        "salinity",
        "mld",
        "sst_anomaly",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    axes = axes.ravel()
    for ax, v in zip(axes, core):
        if v not in ds:
            ax.set_visible(False)
            continue
        da = ds[v]
        pct = (np.isnan(da.values).mean(axis=0)) * 100.0
        im = ax.imshow(
            pct,
            origin="lower",
            extent=[
                float(ds.lon.min()),
                float(ds.lon.max()),
                float(ds.lat.min()),
                float(ds.lat.max()),
            ],
            aspect="auto",
            cmap="viridis",
            vmin=0,
            vmax=100,
        )
        ax.set_title(f"{v} — % NaN over time")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Missing-data map (per grid cell, across full time axis)")
    fig.savefig(MISSING_MAP_PATH, dpi=120)
    plt.close(fig)
    log.info("Missing-data map -> %s", MISSING_MAP_PATH)


def run(ds: xr.Dataset | None = None) -> None:
    if ds is None:
        import zarr  # noqa: F401 - ensure zarr engine available

        ds = xr.open_zarr(ZARR_PATH, consolidated=True)
    write_qc_report(ds)
    write_missing_map(ds)


if __name__ == "__main__":
    run()
