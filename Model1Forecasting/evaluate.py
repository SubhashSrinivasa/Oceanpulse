"""Evaluate a trained checkpoint on the test split.

Reports per-variable per-lead-day (1..OUTPUT_LEN):
  - RMSE and MAE in physical units vs persistence and DOY climatology
  - Skill score: 1 - RMSE_model / RMSE_persistence
  - Normalised accuracy: 1 - |y - y_hat| / std_var  (clipped to [0,1])

Also saves spatial maps (H, W) for each metric to outputs/metrics/spatial_maps.npz
and renders PNG figures (one per metric) to outputs/metrics/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as C
from data.dataset import OceanWindowDataset, load_and_normalize
from data.grid import load_grid
from data.normalize import load_stats
from data.splits import split_indices
from models.convlstm_seq2seq import ConvLSTMSeq2Seq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_doy_climatology(grid: dict, stats: dict) -> np.ndarray:
    """Return (367, C_tgt, H, W) DOY climatology in NORMALIZED space."""
    data = grid["data"]
    times = grid["time"]
    tgt_idx = np.asarray([C.SEQUENCE_VARS.index(v) for v in C.TARGET_VARS])

    train_mask = np.asarray(times, dtype="datetime64[ns]") <= np.datetime64(C.TRAIN_END)
    doy = pd.to_datetime(times[train_mask]).dayofyear.to_numpy()
    train_data = data[train_mask][:, tgt_idx]

    mean = np.asarray([stats[v]["mean"] for v in C.TARGET_VARS], dtype=np.float32)
    std  = np.asarray([stats[v]["std"]  for v in C.TARGET_VARS], dtype=np.float32)
    train_norm = (train_data - mean[None, :, None, None]) / std[None, :, None, None]
    train_norm = np.where(np.isnan(train_norm), 0.0, train_norm).astype(np.float32)

    clim = np.zeros((367, len(C.TARGET_VARS), train_norm.shape[-2], train_norm.shape[-1]), dtype=np.float32)
    for d in range(1, 367):
        sel = train_norm[doy == d]
        if sel.size > 0:
            clim[d] = sel.mean(axis=0)
    return clim


def denormalize_targets(arr: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray([stats[v]["mean"] for v in C.TARGET_VARS], dtype=np.float32)
    std  = np.asarray([stats[v]["std"]  for v in C.TARGET_VARS], dtype=np.float32)
    return arr * std[None, None, :, None, None] + mean[None, None, :, None, None]


def aggregate_scores(pred: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """Global RMSE/MAE over all ocean cells. Returns (T, C) arrays."""
    diff = pred - y
    m = mask[:, None, None]
    denom = m.sum(axis=(0, 3, 4)).clip(min=1.0)
    rmse = np.sqrt((diff**2 * m).sum(axis=(0, 3, 4)) / denom)
    mae  = (np.abs(diff) * m).sum(axis=(0, 3, 4)) / denom
    return rmse, mae


def spatial_rmse(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-cell RMSE averaged over N windows. Returns (T, C, H, W)."""
    return np.sqrt(((pred - y)**2).mean(axis=0))


def skill_score_map(rmse_model: np.ndarray, rmse_baseline: np.ndarray) -> np.ndarray:
    """1 - RMSE_model / RMSE_baseline, NaN where baseline is 0. Shape (T, C, H, W)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ss = 1.0 - rmse_model / np.where(rmse_baseline == 0, np.nan, rmse_baseline)
    return ss


def norm_accuracy_map(pred: np.ndarray, y: np.ndarray, stats: dict) -> np.ndarray:
    """Mean over N of (1 - |err| / std_var), clipped [0,1]. Shape (T, C, H, W)."""
    std = np.asarray([stats[v]["std"] for v in C.TARGET_VARS], dtype=np.float32)
    abs_err = np.abs(pred - y)                                    # (N, T, C, H, W)
    acc = 1.0 - abs_err / std[None, None, :, None, None]
    return np.clip(acc, 0.0, 1.0).mean(axis=0)                    # (T, C, H, W)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_map_figure(
    maps: np.ndarray,          # (T, C, H, W)
    title: str,
    out_path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: np.ndarray,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "RdYlGn",
) -> None:
    n_leads, n_vars = maps.shape[:2]
    fig, axes = plt.subplots(n_vars, n_leads, figsize=(n_leads * 2.5, n_vars * 2.2), squeeze=False)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    display = maps.copy()
    display[:, :, ~ocean_mask] = np.nan        # mask land to white

    for ci, var in enumerate(C.TARGET_VARS):
        for ti in range(n_leads):
            ax = axes[ci][ti]
            img = ax.imshow(
                display[ti, ci],
                origin="lower",
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                vmin=vmin, vmax=vmax,
                cmap=cmap,
                aspect="auto",
            )
            if ci == 0:
                ax.set_title(f"Day {ti+1}", fontsize=8)
            if ti == 0:
                ax.set_ylabel(var, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(img, ax=axes[ci, -1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(C.CHECKPOINT_DIR / "best.pt"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default=str(C.METRICS_DIR / "test_metrics.json"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"[eval] device={device}")

    preloaded = load_and_normalize()
    ds = OceanWindowDataset("test", tile=False, preloaded=preloaded)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"[eval] test windows: {len(ds)}")

    model = ConvLSTMSeq2Seq(
        n_input_channels=C.N_INPUT_CHANNELS,
        n_target_channels=C.N_TARGET_CHANNELS,
        n_aux_channels=len(C.AUX_CHANNELS),
        hidden_channels=C.HIDDEN_CHANNELS,
        kernel_size=C.KERNEL_SIZE,
    ).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(ck["model"])
    model.eval()

    stats = load_stats()
    grid = load_grid()
    clim = build_doy_climatology(grid, stats)

    test_starts = split_indices(grid["time"])["test"]
    times = pd.to_datetime(grid["time"])

    preds, ys, persists, masks = [], [], [], []
    with torch.no_grad():
        for x, y, aux_fut, init_tgt, mask in dl:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred = model(x.to(device), aux_fut.to(device), init_target=init_tgt.to(device))
            preds.append(pred.cpu().numpy())
            ys.append(y.numpy())
            masks.append(mask.numpy())
            persist = np.broadcast_to(
                init_tgt.numpy()[:, None], (init_tgt.shape[0], C.OUTPUT_LEN, *init_tgt.shape[1:])
            ).copy()
            persists.append(persist)

    pred_all    = np.concatenate(preds,    axis=0)   # (N, T, C, H, W)
    y_all       = np.concatenate(ys,       axis=0)
    persist_all = np.concatenate(persists, axis=0)
    mask_all    = np.concatenate(masks,    axis=0)   # (N, H, W)
    ocean_mask  = mask_all[0] > 0.5                  # (H, W) static

    # DOY climatology per window
    clim_all = np.zeros_like(pred_all)
    for wi, s in enumerate(test_starts[: len(pred_all)]):
        for k in range(C.OUTPUT_LEN):
            d = int(times[s + C.INPUT_LEN + k].dayofyear)
            clim_all[wi, k] = clim[d]

    # Physical units
    pred_phys    = denormalize_targets(pred_all,    stats)
    y_phys       = denormalize_targets(y_all,       stats)
    persist_phys = denormalize_targets(persist_all, stats)
    clim_phys    = denormalize_targets(clim_all,    stats)

    # --- Global aggregate scores (T, C) ---
    rmse_m, mae_m = aggregate_scores(pred_phys,    y_phys, mask_all)
    rmse_p, mae_p = aggregate_scores(persist_phys, y_phys, mask_all)
    rmse_c, mae_c = aggregate_scores(clim_phys,    y_phys, mask_all)

    skill_vs_persist = 1.0 - rmse_m / np.where(rmse_p == 0, np.nan, rmse_p)  # (T, C)
    skill_vs_clim    = 1.0 - rmse_m / np.where(rmse_c == 0, np.nan, rmse_c)

    # --- Spatial maps (T, C, H, W) ---
    rmse_map_m = spatial_rmse(pred_phys,    y_phys)
    rmse_map_p = spatial_rmse(persist_phys, y_phys)
    ss_map     = skill_score_map(rmse_map_m, rmse_map_p)
    na_map     = norm_accuracy_map(pred_phys, y_phys, stats)

    # Print summary
    print("\n[eval] Skill score vs persistence (rows=vars, cols=lead days 1-7)\n")
    for ci, v in enumerate(C.TARGET_VARS):
        row = " ".join(f"{s:+.3f}" for s in skill_vs_persist[:, ci])
        print(f"  {v:<18} {row}")

    print("\n[eval] RMSE (physical units)\n")
    for ci, v in enumerate(C.TARGET_VARS):
        print(f"  {v:<18} model   {rmse_m[:, ci]}")
        print(f"  {v:<18} persist {rmse_p[:, ci]}")

    # Save spatial maps .npz
    maps_path = C.METRICS_DIR / "spatial_maps.npz"
    np.savez(
        maps_path,
        skill_score=ss_map.astype(np.float32),
        norm_accuracy=na_map.astype(np.float32),
        rmse_model=rmse_map_m.astype(np.float32),
        rmse_persist=rmse_map_p.astype(np.float32),
        lat=grid["lat"], lon=grid["lon"],
        vars=np.asarray(C.TARGET_VARS),
        lead_days=np.arange(1, C.OUTPUT_LEN + 1),
    )
    print(f"[eval] saved {maps_path}")

    # Render PNG maps
    lat, lon = grid["lat"], grid["lon"]
    save_map_figure(ss_map, "Skill Score vs Persistence (1 − RMSE_model/RMSE_persist)",
                    C.METRICS_DIR / "map_skill_score.png", lat, lon, ocean_mask,
                    vmin=-0.5, vmax=1.0, cmap="RdYlGn")
    save_map_figure(na_map, "Normalised Accuracy (1 − |err|/std)",
                    C.METRICS_DIR / "map_norm_accuracy.png", lat, lon, ocean_mask,
                    vmin=0.0, vmax=1.0, cmap="Blues")
    save_map_figure(rmse_map_m, "Model RMSE (physical units)",
                    C.METRICS_DIR / "map_rmse_model.png", lat, lon, ocean_mask,
                    cmap="YlOrRd")

    # JSON report
    # Identify variables not supervised during training (weight=0).
    unsupervised = [v for v, w in zip(C.TARGET_VARS, C.LOSS_WEIGHTS) if w == 0.0]
    if unsupervised:
        print(f"\n[eval] NOTE: {unsupervised} had loss weight=0 during training; "
              "their model outputs are persistence-equivalent — use persistence baseline as reference.")

    report = {
        "variables": list(C.TARGET_VARS),
        "lead_days": list(range(1, C.OUTPUT_LEN + 1)),
        "loss_weights": list(C.LOSS_WEIGHTS),
        "unsupervised_variables": unsupervised,
        "model":          {"rmse": rmse_m.tolist(), "mae": mae_m.tolist()},
        "persistence":    {"rmse": rmse_p.tolist(), "mae": mae_p.tolist()},
        "doy_climatology":{"rmse": rmse_c.tolist(), "mae": mae_c.tolist()},
        "skill_vs_persistence": skill_vs_persist.tolist(),
        "skill_vs_climatology": skill_vs_clim.tolist(),
        "caveat": (
            "chlorophyll_log past 2022-06-14 is DOY-climatology-filled; metrics on 2022-H2 are optimistic. "
            "salinity had loss_weight=0 (WOA18 climatology-dominated data); model output is near-persistence."
        ),
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
