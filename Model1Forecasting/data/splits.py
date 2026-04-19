"""Temporal train/val/test boundaries for sliding-window datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import INPUT_LEN, OUTPUT_LEN, TRAIN_END, VAL_END


def split_indices(times: np.ndarray) -> dict[str, np.ndarray]:
    """Return, per split, the allowed *window start* indices.

    A window starting at t covers [t, t + INPUT_LEN + OUTPUT_LEN). A window
    belongs to a split iff its LAST target timestep (t + INPUT_LEN + OUTPUT_LEN - 1)
    falls within that split's date range. This prevents any window from
    straddling a boundary.
    """
    ts = pd.to_datetime(times)
    train_end = pd.Timestamp(TRAIN_END)
    val_end = pd.Timestamp(VAL_END)

    win = INPUT_LEN + OUTPUT_LEN
    last_start = len(ts) - win
    starts = np.arange(last_start + 1)
    last_target = ts[starts + win - 1]

    return {
        "train": starts[last_target <= train_end],
        "val":   starts[(last_target > train_end) & (last_target <= val_end)],
        "test":  starts[last_target > val_end],
    }
