"""Per-variable mean/std computed on the training time range.

Stats are computed over ocean cells only (mask applied). Land cells stay at
whatever value they had in the cached grid (NaN) and are zeroed in the
dataset; the ocean_mask channel tells the model where real data lives.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from config import SEQUENCE_VARS, STATS_JSON, TRAIN_END
from data.grid import load_grid


def compute_stats(out_path: Path = STATS_JSON) -> dict:
    g = load_grid()
    data = g["data"]                     # (T, C, H, W)
    ocean = g["ocean_mask"]               # (H, W)
    times = g["time"]

    ts = np.asarray(times, dtype="datetime64[ns]")
    train_mask = ts <= np.datetime64(TRAIN_END)
    print(f"[stats] training timesteps: {int(train_mask.sum())} / {len(ts)}")

    stats: dict[str, dict[str, float]] = {}
    # Flatten (T_train, H, W) -> pull ocean cells only.
    train_data = data[train_mask]        # (T_train, C, H, W)
    for ci, var in enumerate(SEQUENCE_VARS):
        vals = train_data[:, ci][:, ocean]        # (T_train, n_ocean)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            raise RuntimeError(f"no finite ocean values for {var} in training split")
        mu = float(vals.mean())
        sd = float(vals.std())
        if sd < 1e-8:
            print(f"[stats] WARNING {var} std<1e-8, using 1.0")
            sd = 1.0
        stats[var] = {"mean": mu, "std": sd}
        print(f"[stats]   {var:<18} mean={mu: .4f} std={sd:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2))
    print(f"[stats] wrote {out_path}")
    return stats


def load_stats(path: Path = STATS_JSON) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m data.normalize"
        )
    return json.loads(path.read_text())


if __name__ == "__main__":
    compute_stats()
