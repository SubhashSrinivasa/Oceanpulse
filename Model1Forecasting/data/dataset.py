"""PyTorch Dataset: sliding windows over the cached grid, optional random tile."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import (
    AUX_CHANNELS,
    GRID_H,
    GRID_W,
    INPUT_LEN,
    OUTPUT_LEN,
    SEQUENCE_VARS,
    TARGET_INDICES_IN_INPUT,
    TILE_OCEAN_MIN,
    TILE_SIZE,
)
from data.grid import load_grid
from data.normalize import load_stats
from data.splits import split_indices


def _doy_sin_cos(times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    doy = pd.to_datetime(times).dayofyear.to_numpy().astype(np.float32)
    angle = 2.0 * np.pi * (doy - 1) / 365.0
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def _normalize_grid(g: dict):
    """Normalize raw grid data. Returns (norm, ocean, sin_t, cos_t, times)."""
    data = g["data"]
    ocean = g["ocean_mask"].astype(bool)
    stats = load_stats()
    mean = np.asarray([stats[v]["mean"] for v in SEQUENCE_VARS], dtype=np.float32)
    std = np.asarray([stats[v]["std"] for v in SEQUENCE_VARS], dtype=np.float32)
    norm = (data - mean[None, :, None, None]) / std[None, :, None, None]
    norm = np.where(np.isnan(norm), 0.0, norm).astype(np.float32)
    sin_t, cos_t = _doy_sin_cos(g["time"])
    return norm, ocean, sin_t, cos_t, g["time"]


def load_and_normalize() -> dict:
    """Load grid.npz once and normalize. Pass the result to OceanWindowDataset
    as `preloaded=` to avoid re-reading the 1.2 GB file for each split."""
    print("[dataset] loading and normalizing grid (once)...")
    g = load_grid()
    norm, ocean, sin_t, cos_t, times = _normalize_grid(g)
    print(f"[dataset] grid ready: {norm.shape}, ocean cells: {ocean.sum()}")
    return {"norm": norm, "ocean_mask": ocean, "sin_t": sin_t, "cos_t": cos_t, "time": times}


class OceanWindowDataset(Dataset):
    """Sliding-window dataset.

    __getitem__ returns:
        x:          (INPUT_LEN, C_in, h, w)       inputs + aux channels (normalized)
        y:          (OUTPUT_LEN, C_tgt, h, w)     targets (normalized)
        aux_future: (OUTPUT_LEN, C_aux, h, w)     sin/cos/mask for the future dates
        init_tgt:   (C_tgt, h, w)                 last observed target frame (decoder seed)
        mask:       (h, w)                        ocean mask (1.0 ocean, 0.0 land)

    When tile is True, a random h=w=TILE_SIZE crop is taken per sample with
    rejection resampling to ensure enough ocean cells (TILE_OCEAN_MIN). When
    tile is False, returns the full grid.
    """

    def __init__(
        self,
        split: str,
        *,
        tile: bool = True,
        tile_size: int = TILE_SIZE,
        tile_ocean_min: int = TILE_OCEAN_MIN,
        max_tile_tries: int = 20,
        rng_seed: Optional[int] = None,
        preloaded: Optional[dict] = None,
    ) -> None:
        """
        preloaded: dict with keys norm, ocean_mask, sin_t, cos_t, time — pass
        the result of load_and_normalize() to avoid re-reading grid.npz for
        every dataset split (train + val each used to load it independently).
        """
        if split not in ("train", "val", "test"):
            raise ValueError(split)

        if preloaded is not None:
            norm = preloaded["norm"]
            ocean = preloaded["ocean_mask"]
            sin_t = preloaded["sin_t"]
            cos_t = preloaded["cos_t"]
            times = preloaded["time"]
        else:
            g = load_grid()
            norm, ocean, sin_t, cos_t, times = _normalize_grid(g)

        self.norm = norm
        self.sin_t = sin_t
        self.cos_t = cos_t
        self.mask = ocean.astype(np.float32)
        self.starts = split_indices(times)[split]
        self.tile = tile
        self.tile_size = tile_size
        self.tile_ocean_min = tile_ocean_min
        self.max_tile_tries = max_tile_tries
        self._rng = np.random.default_rng(rng_seed)
        self.target_idx = np.asarray(TARGET_INDICES_IN_INPUT, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.starts)

    def _sample_tile(self) -> tuple[int, int]:
        s = self.tile_size
        if s >= GRID_H and s >= GRID_W:
            return 0, 0
        for _ in range(self.max_tile_tries):
            i0 = int(self._rng.integers(0, GRID_H - s + 1))
            j0 = int(self._rng.integers(0, GRID_W - s + 1))
            if self.mask[i0:i0 + s, j0:j0 + s].sum() >= self.tile_ocean_min:
                return i0, j0
        # Fallback: return tile centered on the ocean centroid.
        ys, xs = np.where(self.mask > 0.5)
        ci = int(np.clip(np.mean(ys) - s // 2, 0, GRID_H - s))
        cj = int(np.clip(np.mean(xs) - s // 2, 0, GRID_W - s))
        return ci, cj

    def _aux_slice(self, t0: int, t1: int, mask_tile: np.ndarray) -> np.ndarray:
        n = t1 - t0
        h, w = mask_tile.shape
        sin_s = np.broadcast_to(self.sin_t[t0:t1, None, None], (n, h, w))
        cos_s = np.broadcast_to(self.cos_t[t0:t1, None, None], (n, h, w))
        mask_s = np.broadcast_to(mask_tile[None], (n, h, w))
        return np.stack([sin_s, cos_s, mask_s], axis=1).astype(np.float32)

    def __getitem__(self, idx: int):
        t0 = int(self.starts[idx])
        t_in0, t_in1 = t0, t0 + INPUT_LEN
        t_tg1 = t_in1 + OUTPUT_LEN

        if self.tile:
            i0, j0 = self._sample_tile()
            i1, j1 = i0 + self.tile_size, j0 + self.tile_size
        else:
            i0, j0, i1, j1 = 0, 0, GRID_H, GRID_W

        x_raw = self.norm[t_in0:t_in1, :, i0:i1, j0:j1]                  # (T_in, C_raw, h, w)
        y_raw = self.norm[t_in1:t_tg1, :, i0:i1, j0:j1]                  # (T_out, C_raw, h, w)

        mask_tile = self.mask[i0:i1, j0:j1]                              # (h, w)

        aux_in = self._aux_slice(t_in0, t_in1, mask_tile)                # (T_in, 3, h, w)
        aux_fut = self._aux_slice(t_in1, t_tg1, mask_tile)               # (T_out, 3, h, w)

        x = np.concatenate([x_raw, aux_in], axis=1)                      # (T_in, C_in, h, w)
        y = y_raw[:, self.target_idx]                                    # (T_out, C_tgt, h, w)

        # Decoder seed: last observed target frame (normalized).
        init_tgt = x_raw[-1, self.target_idx]                             # (C_tgt, h, w)

        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
            torch.from_numpy(np.ascontiguousarray(aux_fut)),
            torch.from_numpy(np.ascontiguousarray(init_tgt)),
            torch.from_numpy(np.ascontiguousarray(mask_tile)),
        )
