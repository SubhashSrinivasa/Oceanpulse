"""Train the encoder-decoder ConvLSTM on ocean_cube_sequences.csv windows.

Usage:
    python train.py                       # full run, defaults from config.py
    python train.py --epochs 1 --tile-size 32 --batch-size 2   # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

import config as C
from data.dataset import OceanWindowDataset, load_and_normalize
from models.convlstm_seq2seq import ConvLSTMSeq2Seq


def masked_mse(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    # pred, tgt: (B, T, C, H, W); mask: (B, H, W); weights: (C,)
    m = mask[:, None, None]                          # (B, 1, 1, H, W)
    w = weights[None, None, :, None, None]            # (1, 1, C, 1, 1)
    err = (pred - tgt) ** 2 * m * w
    # normalise by total weighted ocean·step count so magnitude is stable
    denom = (m * w).sum() * pred.shape[1]
    return err.sum() / denom.clamp_min(1.0)


def build_scheduler(optim: torch.optim.Optimizer, steps_per_epoch: int):
    # Cosine annealing with warm restarts every 10 epochs.
    # Each restart the model can escape shallow local minima found in the
    # previous cycle; T_mult=1 keeps all cycles the same length.
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optim, T_0=steps_per_epoch * 10, T_mult=1, eta_min=1e-5
    )


def teacher_forcing_prob(epoch: int, total_epochs: int) -> float:
    frac = min(max(epoch / max(1, total_epochs - 1), 0.0), 1.0)
    return frac * C.SCHEDULED_SAMPLING_MAX


def run_epoch(model, loader, optim, scheduler, scaler, device, tf_prob: float, train: bool,
              loss_weights: torch.Tensor):
    model.train(train)
    total_loss = 0.0
    total_samples = 0
    t0 = time.time()
    for batch_idx, (x, y, aux_fut, init_tgt, mask) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        aux_fut = aux_fut.to(device, non_blocking=True)
        init_tgt = init_tgt.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                pred = model(
                    x,
                    aux_fut,
                    init_target=init_tgt,
                    teacher_forcing_target=y if train else None,
                    teacher_forcing_prob=tf_prob if train else 0.0,
                )
                loss = masked_mse(pred, y, mask, loss_weights)

            if train:
                optim.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
                    optim.step()
                scheduler.step()

        total_loss += float(loss.item()) * x.size(0)
        total_samples += x.size(0)
    return total_loss / max(1, total_samples), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=C.BATCH_SIZE)
    ap.add_argument("--tile-size", type=int, default=C.TILE_SIZE)
    ap.add_argument("--lr", type=float, default=C.LR)
    ap.add_argument("--num-workers", type=int, default=C.NUM_WORKERS)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # cache best kernel for fixed tile size
    print(f"[train] device={device}")

    # Load and normalize the grid once — avoids reading the 1.2 GB file twice.
    preloaded = load_and_normalize()
    ds_train = OceanWindowDataset("train", tile=True, tile_size=args.tile_size, rng_seed=args.seed, preloaded=preloaded)
    ds_val = OceanWindowDataset("val", tile=True, tile_size=args.tile_size, rng_seed=args.seed + 1, preloaded=preloaded)
    print(f"[train] train windows: {len(ds_train)}  val windows: {len(ds_val)}")

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = ConvLSTMSeq2Seq(
        n_input_channels=C.N_INPUT_CHANNELS,
        n_target_channels=C.N_TARGET_CHANNELS,
        n_aux_channels=len(C.AUX_CHANNELS),
        hidden_channels=C.HIDDEN_CHANNELS,
        kernel_size=C.KERNEL_SIZE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params/1e6:.2f} M")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    scheduler = build_scheduler(optim, len(dl_train))
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and not args.no_amp) else None

    loss_weights = torch.tensor(C.LOSS_WEIGHTS, dtype=torch.float32, device=device)
    print(f"[train] loss weights: { {v: w for v, w in zip(C.TARGET_VARS, C.LOSS_WEIGHTS)} }")

    metrics_path = C.METRICS_DIR / "train.jsonl"
    ckpt_path = C.CHECKPOINT_DIR / "best.pt"
    last_path = C.CHECKPOINT_DIR / "last.pt"

    best_val = float("inf")
    bad_epochs = 0
    with open(metrics_path, "w") as fh:
        for epoch in range(args.epochs):
            tf_prob = teacher_forcing_prob(epoch, args.epochs)
            tr_loss, tr_time = run_epoch(model, dl_train, optim, scheduler, scaler, device, tf_prob, train=True, loss_weights=loss_weights)
            val_loss, val_time = run_epoch(model, dl_val, optim, scheduler, scaler, device, 0.0, train=False, loss_weights=loss_weights)
            rec = {
                "epoch": epoch,
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "tf_prob": tf_prob,
                "lr": scheduler.get_last_lr()[0],
                "train_time_s": tr_time,
                "val_time_s": val_time,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(
                f"[train] epoch {epoch:3d}  tr={tr_loss:.5f}  val={val_loss:.5f}  "
                f"tf_p={tf_prob:.2f}  lr={rec['lr']:.2e}  ({tr_time:.1f}s/{val_time:.1f}s)"
            )

            torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, last_path)
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                bad_epochs = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val_loss": val_loss}, ckpt_path)
                print(f"[train]   -> new best, saved {ckpt_path}")
            else:
                bad_epochs += 1
                if bad_epochs >= C.EARLY_STOP_PATIENCE:
                    print(f"[train] early stop at epoch {epoch} (patience {C.EARLY_STOP_PATIENCE})")
                    break


if __name__ == "__main__":
    main()
