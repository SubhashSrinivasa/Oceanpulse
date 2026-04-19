"""Render species distribution probability maps as PNGs.

Reads the parquet produced by predict_map.py and writes one heatmap per
species into outputs/maps/.

Usage:
    python visualize.py                        # all species, logreg, yearly
    python visualize.py --species sebastes_mystinus
    python visualize.py --top 20 --model gbm --month 7
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ARTIFACTS_DIR,
    LAT_MAX,
    LAT_MIN,
    LAT_STEP,
    LON_MAX,
    LON_MIN,
    LON_STEP,
    MAPS_DIR,
    OUTPUTS_DIR,
)


def load_maps(model_kind, month):
    suffix = f"_m{month:02d}" if month else "_yearly"
    pq = OUTPUTS_DIR / f"probability_maps_{model_kind}{suffix}.parquet"
    csv = OUTPUTS_DIR / f"probability_maps_{model_kind}{suffix}.csv"
    if pq.exists():
        return pd.read_parquet(pq), suffix
    if csv.exists():
        return pd.read_csv(csv), suffix
    raise FileNotFoundError(
        f"No predictions found for model={model_kind} month={month}. "
        f"Run predict_map.py first."
    )


def pick_species(df, species_arg, top_n, model_kind):
    if species_arg:
        return [species_arg]
    if top_n:
        metrics_path = ARTIFACTS_DIR / f"metrics_{model_kind}.csv"
        if metrics_path.exists():
            m = pd.read_csv(metrics_path).dropna(subset=["roc_auc"])
            return m.sort_values("roc_auc", ascending=False).head(top_n)["species"].tolist()
        return df["species"].drop_duplicates().head(top_n).tolist()
    return df["species"].drop_duplicates().tolist()


def render_one(df_species, species, out_path, title_suffix=""):
    lat_bins = np.arange(LAT_MIN, LAT_MAX + LAT_STEP / 2, LAT_STEP)
    lon_bins = np.arange(LON_MIN, LON_MAX + LON_STEP / 2, LON_STEP)

    grid = df_species.pivot_table(
        index="lat", columns="lon", values="probability", aggfunc="mean",
    )
    grid = grid.reindex(index=lat_bins, columns=lon_bins)

    fig, ax = plt.subplots(figsize=(6, 8))
    im = ax.imshow(
        grid.values,
        origin="lower",
        extent=[LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"P({species} present){title_suffix}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("probability")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render(model_kind="logreg", month=None, species=None, top_n=None):
    df, suffix = load_maps(model_kind, month)
    species_list = pick_species(df, species, top_n, model_kind)

    out_dir = MAPS_DIR / f"{model_kind}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    month_tag = f" (month {month})" if month else " (yearly avg)"
    print(f"[viz] rendering {len(species_list)} species -> {out_dir}")
    for i, sp in enumerate(species_list, 1):
        sub = df[df["species"] == sp]
        if sub.empty:
            print(f"[viz] skip {sp}: no rows")
            continue
        out_path = out_dir / f"{sp}.png"
        render_one(sub, sp, out_path, title_suffix=month_tag)
        if i % 10 == 0 or i == len(species_list):
            print(f"[viz] {i}/{len(species_list)}")
    print(f"[viz] done -> {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["logreg", "gbm"], default="logreg")
    p.add_argument("--month", type=int, default=None)
    p.add_argument("--species", type=str, default=None)
    p.add_argument("--top", type=int, default=None,
                   help="render only top-N species by ROC AUC")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render(model_kind=args.model, month=args.month, species=args.species, top_n=args.top)
