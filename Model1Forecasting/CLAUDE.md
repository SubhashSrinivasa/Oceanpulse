# CLAUDE.md

Guidance for Claude Code working on `Model1Forecasting/`.

## Project overview

Encoder–decoder ConvLSTM that consumes `INPUT_LEN=30` days of 7 ocean variables
on the 141×161 NE Pacific grid and forecasts `OUTPUT_LEN=7` days of 5 target
variables (sst, ssh, chlorophyll_log, mld, salinity). Inputs are read from
`Model1DataEngineering/outputs/ocean_cube_sequences.csv` (one row per surviving
ocean cell, 1826 daily values per variable).

## Commands

Run everything from the `Model1Forecasting` directory. Training targets a
CUDA GPU; everything else runs on CPU if no GPU is present.

```bash
cd Model1Forecasting
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Install the CUDA-matching torch wheel if the default CPU build isn't what you want:
#   pip install --index-url https://download.pytorch.org/whl/cu121 torch

# 1) One-time: parse the sequences CSV into data/cache/grid.npz
python scripts/prepare_tensors.py

# 2) One-time: compute per-variable mean/std from training years
python -m data.normalize

# 3) Smoke test (fast, verifies shapes + loss decreases)
python train.py --epochs 1 --tile-size 32 --batch-size 2 --num-workers 0

# 4) Full training
python train.py

# 5) Evaluate the best checkpoint on the 2022 test split
python evaluate.py --checkpoint outputs/checkpoints/best.pt

# 6) Produce a 7-day forecast starting the day after --start-date
python forecast.py --checkpoint outputs/checkpoints/best.pt --start-date 2022-06-01
```

## Architecture

```
ocean_cube_sequences.csv (from Model1DataEngineering)
    │
    ▼
scripts/prepare_tensors.py ─► data/cache/grid.npz  (T=1826, C=7, H=141, W=161) + ocean_mask
    │
    ▼
data/normalize.py ─► data/cache/stats.json  (mean/std per variable, training years only)
    │
    ▼
data/dataset.py  OceanWindowDataset
    - sliding windows of INPUT_LEN=30 + OUTPUT_LEN=7 over the daily axis
    - random 64x64 tiles during training (rejection resample: tile needs ≥512 ocean cells)
    - returns (x, y, aux_future, init_tgt, mask)
        x:          (30, 10, h, w)  normalized inputs + [sin_doy, cos_doy, ocean_mask]
        y:          (7,  5,  h, w)  normalized targets
        aux_future: (7,  3,  h, w)  sin/cos/mask for the 7 future dates
        init_tgt:   (5,  h,  w)     last observed target frame, decoder seed
        mask:       (h, w)          ocean mask (1.0 ocean, 0.0 land)
    │
    ▼
models/convlstm_seq2seq.py  ConvLSTMSeq2Seq
    encoder: 2x ConvLSTMCell(10 -> 64 -> 64), kernel 3x3
    decoder: 2x ConvLSTMCell(5+3 -> 64 -> 64); head Conv2d(64, 5, 1)
    autoregressive decode with scheduled sampling (p ramped 0 -> 0.5)
```

## Key design choices

- **Encoder vs decoder input**: the encoder sees the full 10-channel input
  (7 vars + 3 aux). The decoder's input at each step is the previous
  target-channel prediction (5) concatenated with the aux channels (3) for
  the *future* date being predicted. Encoder states seed the decoder.
- **Targets** are the 5 forecastable variables. `sst_anomaly` and `mld_source`
  are input-only channels. Raw `chlorophyll` is dropped in favor of `chlorophyll_log`.
- **Loss**: masked MSE over ocean cells only, in normalized space, equally
  weighted across the 5 target channels.
- **Tiles, not full grid, during training**: reduces memory and acts as
  spatial augmentation. ConvLSTM is fully convolutional so inference runs
  on the full 141×161 grid.
- **Splits**: strict temporal — train 2018–2020, val 2021, test 2022. Window
  assignment uses the window's last target timestep so no window straddles
  a split boundary.
- **Scheduled sampling**: teacher-forcing probability ramps linearly from 0
  at epoch 0 to `SCHEDULED_SAMPLING_MAX=0.5` at the last epoch to avoid
  exposure bias.

## Key non-obvious details

- **`Model1DataEngineering/pipeline/config.py` is authoritative** for grid
  constants (`MASTER_LAT/LON/TIME`). `config.py` imports from it via
  `sys.path` injection rather than copying values.
- **Sequences CSV has 8055 rows today** (≈35.5% of 22,701 grid cells) after
  the 50%-drop rule and 4-pass imputation. Surviving cells are NaN-free.
- **Chlorophyll past 2022-06-14 is DOY-climatology-filled** at export time
  (MODIS source ends there). Test-split chl metrics on 2022-H2 are
  optimistic — `evaluate.py` emits a caveat in the JSON report.
- **Grid .npz size**: `(1826, 7, 141, 161) float32` ≈ 1.17 GB on disk
  (uncompressed). First-time `prepare_tensors.py` takes ~60–90 s parsing
  JSON from the 891 MB CSV.
- **AMP is on by default** on CUDA (`torch.cuda.amp.autocast`). Disable with
  `--no-amp` if you suspect numerical issues.
- **`init_target` at t=0** of the decoder is the *last observed* normalized
  target frame (i.e. encoder sees it, decoder seeds from it). Not a
  learned projection — a simple, defensible choice.
- **ConvLSTM gradients are finicky**: `GRAD_CLIP=1.0` is mandatory; lower
  it further if training diverges.

## Files

| File | Role |
|---|---|
| [config.py](config.py) | Paths, variable lists, window/training hyperparams; imports grid consts from Model1DataEngineering |
| [data/grid.py](data/grid.py) | CSV → dense (T, C, H, W) tensor + ocean mask |
| [data/normalize.py](data/normalize.py) | Per-variable mean/std on training years |
| [data/splits.py](data/splits.py) | Temporal train/val/test window-start indices |
| [data/dataset.py](data/dataset.py) | PyTorch Dataset, sliding windows + random tile |
| [models/convlstm_cell.py](models/convlstm_cell.py) | Single ConvLSTM cell (Shi et al., peephole-free) |
| [models/convlstm_seq2seq.py](models/convlstm_seq2seq.py) | Encoder-decoder with scheduled sampling |
| [train.py](train.py) | Training loop, AMP, cosine LR, early stopping |
| [evaluate.py](evaluate.py) | Test-split RMSE/MAE vs persistence + DOY climatology baselines |
| [forecast.py](forecast.py) | CLI: produce a 7-day forecast .npz from an arbitrary `--start-date` |
| [scripts/prepare_tensors.py](scripts/prepare_tensors.py) | One-time CSV → grid.npz parse |
