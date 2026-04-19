"""Produce a 7-day forecast starting the day after --start-date.

Example:
    python forecast.py --checkpoint outputs/checkpoints/best.pt \\
                       --start-date 2022-06-01 \\
                       --out outputs/forecasts/f_20220601.npz

The model reads the INPUT_LEN days up to and including --start-date from the
cached grid, runs the ConvLSTM on the full 141x161 grid, and writes an .npz
with keys: pred (T_out, C_tgt, H, W) in physical units, lat, lon, time, vars.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

import config as C
from data.dataset import _doy_sin_cos
from data.grid import load_grid
from data.normalize import load_stats
from models.convlstm_seq2seq import ConvLSTMSeq2Seq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(C.CHECKPOINT_DIR / "best.pt"))
    ap.add_argument("--start-date", required=True, help="YYYY-MM-DD; last observed day")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[forecast] device={device}")

    grid = load_grid()
    times = pd.to_datetime(grid["time"])
    start = pd.Timestamp(args.start_date)
    if start not in times:
        raise ValueError(f"{start.date()} not in grid time axis")
    end_idx = int(np.where(times == start)[0][0])
    begin_idx = end_idx - C.INPUT_LEN + 1
    if begin_idx < 0:
        raise ValueError(f"need {C.INPUT_LEN} days of history; --start-date is too early")

    stats = load_stats()
    mean = np.asarray([stats[v]["mean"] for v in C.SEQUENCE_VARS], dtype=np.float32)
    std = np.asarray([stats[v]["std"] for v in C.SEQUENCE_VARS], dtype=np.float32)
    tgt_mean = np.asarray([stats[v]["mean"] for v in C.TARGET_VARS], dtype=np.float32)
    tgt_std = np.asarray([stats[v]["std"] for v in C.TARGET_VARS], dtype=np.float32)

    data = grid["data"][begin_idx:end_idx + 1]                        # (T_in, C_raw, H, W)
    ocean_mask = grid["ocean_mask"].astype(np.float32)

    norm = (data - mean[None, :, None, None]) / std[None, :, None, None]
    norm = np.where(np.isnan(norm), 0.0, norm).astype(np.float32)

    sin_all, cos_all = _doy_sin_cos(grid["time"])
    H, W = ocean_mask.shape

    def aux_block(slice_times: np.ndarray) -> np.ndarray:
        n = len(slice_times)
        sin_s = np.broadcast_to(sin_all[slice_times][:, None, None], (n, H, W))
        cos_s = np.broadcast_to(cos_all[slice_times][:, None, None], (n, H, W))
        mask_s = np.broadcast_to(ocean_mask[None], (n, H, W))
        return np.stack([sin_s, cos_s, mask_s], axis=1).astype(np.float32)

    in_idx = np.arange(begin_idx, end_idx + 1)
    aux_in = aux_block(in_idx)                                         # (T_in, 3, H, W)
    x = np.concatenate([norm, aux_in], axis=1)                         # (T_in, C_in, H, W)

    # Future dates for the 7-day forecast: t_in1 .. t_tg1 - 1
    future_doy = pd.date_range(start + pd.Timedelta(days=1), periods=C.OUTPUT_LEN, freq="D")
    angle = 2.0 * np.pi * (future_doy.dayofyear.to_numpy().astype(np.float32) - 1) / 365.0
    sin_f = np.broadcast_to(np.sin(angle)[:, None, None], (C.OUTPUT_LEN, H, W))
    cos_f = np.broadcast_to(np.cos(angle)[:, None, None], (C.OUTPUT_LEN, H, W))
    mask_f = np.broadcast_to(ocean_mask[None], (C.OUTPUT_LEN, H, W))
    aux_fut = np.stack([sin_f, cos_f, mask_f], axis=1).astype(np.float32)

    tgt_idx = [C.SEQUENCE_VARS.index(v) for v in C.TARGET_VARS]
    init_tgt = norm[-1, tgt_idx]                                       # (C_tgt, H, W)

    model = ConvLSTMSeq2Seq(
        n_input_channels=C.N_INPUT_CHANNELS,
        n_target_channels=C.N_TARGET_CHANNELS,
        n_aux_channels=len(C.AUX_CHANNELS),
        hidden_channels=C.HIDDEN_CHANNELS,
        kernel_size=C.KERNEL_SIZE,
    ).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()

    with torch.no_grad():
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)               # (1, T_in, C_in, H, W)
        aux_t = torch.from_numpy(aux_fut).unsqueeze(0).to(device)       # (1, T_out, C_aux, H, W)
        init_t = torch.from_numpy(init_tgt).unsqueeze(0).to(device)     # (1, C_tgt, H, W)
        pred = model(x_t, aux_t, init_target=init_t)                    # (1, T_out, C_tgt, H, W)
        pred = pred.squeeze(0).cpu().numpy()

    pred_phys = pred * tgt_std[None, :, None, None] + tgt_mean[None, :, None, None]

    out_path = Path(args.out) if args.out else C.FORECAST_DIR / f"f_{start.strftime('%Y%m%d')}.npz"
    np.savez(
        out_path,
        pred=pred_phys.astype(np.float32),
        lat=grid["lat"],
        lon=grid["lon"],
        time=np.asarray([np.datetime64(d) for d in future_doy]),
        vars=np.asarray(C.TARGET_VARS),
        ocean_mask=grid["ocean_mask"],
    )
    print(f"[forecast] wrote {out_path} shape={pred_phys.shape}")


if __name__ == "__main__":
    main()
