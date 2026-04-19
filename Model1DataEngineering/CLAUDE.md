# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OceanPulse Model1 Data Engineering: a 7-stage pipeline that ingests oceanographic data from multiple public sources, harmonizes it onto a shared 0.25° daily grid over a Northeast Pacific bbox (20–55°N, −150 to −110°W, 2018–2022), and produces a Zarr cube + two ML-ready CSV exports.

## Commands

Run everything from the `Model1DataEngineering` directory. Dependencies live in `.venv`; **do not use bare `python` from the OS** unless that interpreter has `pip install -r requirements.txt` applied.

```bash
cd Model1DataEngineering

# Option A — activate, then use `python` as usual
source .venv/bin/activate
pip install -r requirements.txt   # first time (or after recreating the venv)

# Option B — no activation (always uses project deps)
# .venv/bin/python run_pipeline.py

# Run the full pipeline (download → zarr → QC → CSVs)
python run_pipeline.py

# Re-export CSVs only (reads existing zarr, re-runs imputation)
python export_csv.py

# Analyse both CSV outputs (null rates, stats)
python analyse_outputs.py

# Authenticate with Copernicus Marine (one-time setup)
copernicusmarine login
```

All pipeline output is tailed to both stdout and `logs/pipeline.log`. Downloaded source files are cached in `data/cache/` — re-running the pipeline skips already-downloaded files.

**Interrupted downloads / resume:** Caching is **per file**, not HTTP byte-range resume. OISST and MODIS use **one NetCDF per year**; WOA18 MLD uses **one file per month**; Argo and CalCOFI use **one parquet each**; CMEMS SSH and GLORYS salinity each use **one NetCDF**. If you stop the run after some of those finished saving, the next run **only downloads what is still missing**. If you stop **while** a file is being written, that path may exist but be **truncated** — the code only checks `exists()`, so delete that broken `.nc` / `.parquet` (or the obvious outlier) under `data/cache/` and rerun so it redownloads cleanly.

**Faster downloads (optional):** Independent HTTP fetches run in parallel — OISST and MODIS (one job per missing year) and WOA18 MLD (one job per missing month). Tune with `OCEANPULSE_DOWNLOAD_WORKERS` (default `4`; set to `1` for sequential). Larger read chunks: `OCEANPULSE_HTTP_CHUNK_BYTES` (default 4 MiB). CMEMS (`copernicusmarine subset` for SSH and GLORYS salinity) is still a single server-side job per product; Argo is one argopy request — those are not parallelized here.

## Architecture

### Data Flow

```
NOAA ERDDAP ──────────────────────────────► stage1_sst.py      → sst, sst_anomaly
Argo / argopy ────────────────────────────► stage4_mld.py      → mld, mld_source
CMEMS copernicusmarine ───────────────────► stage2_ssh.py      → ssh
                       ───────────────────► stage5_salinity.py → salinity
WOA18 (NCEI HTTPS) ──────────────────────► stage4_mld.py
CalCOFI ERDDAP ──────────────────────────► stage5_salinity.py
NASA MODIS ERDDAP ────────────────────────► stage3_chl.py      → chlorophyll, chlorophyll_log

All stages ──► stage6_assemble.py → outputs/ocean_cube.zarr  (1826×141×161, 8 vars)
                    └──► stage7_qc.py  → data_quality_report.txt, missing_data_map.png
                    └──► export_csv.py → ocean_cube_long.csv, ocean_cube_sequences.csv
```

### Master Grid (defined in `pipeline/config.py`)
- **Lat**: 20.0–55.0°N, 0.25° spacing, 141 points
- **Lon**: −150.0–−110.0°W, 0.25° spacing, 161 points
- **Time**: 2018-01-01–2022-12-31, daily, 1826 steps
- **Cube**: 1826 × 141 × 161 ≈ 41 M points per variable, 8 variables

All source grids are regridded/reindexed to this master before merge.

### Pipeline Modules

| File | Role |
|------|------|
| `pipeline/config.py` | Master grid constants, all file paths, physical sanity bounds |
| `pipeline/utils.py` | Shared logger (`get_logger`), QC record list (`qc_variable`) |
| `pipeline/stage1_sst.py` | OISST via ERDDAP, one year per cache file (parallel when several years missing); bridges ≤3-day gaps; computes DOY climatology + anomaly |
| `pipeline/stage2_ssh.py` | CMEMS DUACS 0.125° → regridded to 0.25°; Gaussian synthetic fallback if unavailable |
| `pipeline/stage3_chl.py` | MODIS-Aqua 8-day L3, one year per cache file (parallel when several years missing); probes dataset end-date to cap requests; regrid + daily resample |
| `pipeline/stage4_mld.py` | Argo profiles (density-threshold MLD via `gsw`) → WOA18 monthly climatology → constant 50 m |
| `pipeline/stage5_salinity.py` | CalCOFI bottle (≤10 m) → GLORYS12 → constant 33.5 PSU |
| `pipeline/stage6_assemble.py` | Merges all datasets; validates dims/axes/bounds; writes Zarr |
| `pipeline/stage7_qc.py` | Text report + 2×3 NaN-rate map |
| `export_csv.py` | Loads Zarr, runs 4-pass imputation on all float vars, applies 50%-drop rule for sequences, writes both CSV formats (NaN-free) |
| `analyse_outputs.py` | Stats on both CSVs (null counts, mean, std, min, max) |

### CSV Outputs

**`ocean_cube_long.csv`** — tabular, one row per (time, lat, lon) **sea** cell, **zero NaN values**.  
Columns: `time, lat, lon, month, day_of_year, sst, sst_anomaly, ssh, chlorophyll, chlorophyll_log, mld, mld_source, salinity`  
Land cells (SST always NaN in the raw cube, stays NaN through imputation) are dropped. At the current bbox this is ~15–19 k ocean locations × 1826 days ≈ 28–35 M rows ≈ **3.0–3.5 GB**.

**`ocean_cube_sequences.csv`** — wide, one row per **sea** location that survives the 50%-drop rule (below), **zero `null` values in any array**.  
Each variable column is a JSON list of 1826 daily values. First line is a `#` comment with `time_start`, `time_end`, `n_timesteps`, `dt_days`. Load with `pd.read_csv(..., comment='#')` then `df['sst'].apply(json.loads)`. Far more locations pass at 50% than at 30%; expect the file to grow ~10× vs. the old 30% run.

### Imputation + 50%-drop policy (in `export_csv.py`)

All imputation happens at export time; the Zarr stays raw. A shared `_three_pass` core fills NaN over ocean cells in four passes:
1. **Temporal**: linear interp per cell, gap ≤ 30 days — handles cloud gaps.
2. **Spatial**: `scipy.interpolate.griddata` per timestep using valid ocean neighbors (nearest-neighbor fallback for edge cells).
3. **DOY climatology**: day-of-year mean across years — fills long blackouts (e.g. MODIS Jun–Dec 2022).
4. **Global ocean-mean fallback**: any remaining NaN inside the ocean mask is filled with the variable's global ocean mean (degrades to `0.0` only if the variable is NaN everywhere). This guarantees no NaN survives inside the ocean mask.

Variables imputed: `chlorophyll` / `chlorophyll_log` (in log₁₀ space) and `sst, sst_anomaly, ssh, mld, salinity` (in linear space). Land cells are **never** touched — they stay NaN and get filtered out by `sst.notna()` in `write_long` and by the land mask in `write_sequences`. `mld_source` is a categorical flag and is not imputed.

**Sequences 50%-drop rule** (`DROP_THRESHOLD = 0.50` in `export_csv.py`): before writing `ocean_cube_sequences.csv`, for every (lat, lon) cell the per-variable NaN ratio is computed on the **pre-imputation** cube across `sst, sst_anomaly, ssh, chlorophyll, mld, salinity`. If ANY of those variables exceeds 50% missing at that cell, the whole location is dropped. Raised from the previous 30% to keep more locations at the cost of heavier reliance on imputation. Surviving locations emit fully dense imputed arrays (0% nulls). This rule does NOT apply to the long CSV, which keeps every sea cell and relies on the 4-pass fill alone.

Both writers also run a final **per-variable safety net** (fill residual NaN with column/variable mean) so the written CSVs are guaranteed NaN-free even if the 4-pass fill misses anything.

## Key Non-Obvious Details

- **CMEMS product ID changed**: the 0.25° MY DUACS product was retired; pipeline uses `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` and regridds it.
- **MODIS dataset ends 2022-06-14**: stage 3 auto-detects the dataset's actual time range and caps the 2022 request rather than failing.
- **WOA23 has no MLD product**: the pipeline falls back to WOA18 (`decav81B0` series, `woa18_decav81B0_M02{mm}_01.nc`).
- **Sanity bounds** are defined in `config.py` and checked in stage 6 as warnings, not errors — the pipeline completes even with out-of-bounds values so downstream QC can decide.
- **`mld_source` flag**: 0 = Argo-derived, 1 = WOA18 climatology, 2 = constant 50 m fallback. Use this in ML to weight or mask low-confidence MLD values.
- **CMEMS requires authentication**: run `copernicusmarine login` once before the pipeline; credentials are stored by the package in `~/.copernicusmarine/`.
- **Validation tolerances**: lat/lon reindexing uses `tolerance=0.13–0.15` (roughly half a 0.25° cell) to snap nearby grid points.
- **Cache filenames don't encode the bbox**: files in `data/cache/` (e.g. `oisst_2018.nc`, `modis_chl_2018.nc`, `cmems_ssh_2018_2022.nc`, `argo_profiles.parquet`) are server-side subsets pinned to whatever `LAT_MIN/MAX/LON_MIN/MAX` was in effect when they were downloaded. Wipe `data/cache/*` (keeping `woa18_mld_*.nc`, which is global raw data) and delete `outputs/ocean_cube.zarr` whenever you change the master grid in `pipeline/config.py`.
- **Zarr chunking**: `stage6_assemble.py` chunks the cube `(time=365, lat=50, lon=50)` before `to_zarr` so no variable is written as a single giant chunk on the expanded bbox.
- **Peak memory**: `export_csv.py` loads the full Zarr into memory, keeps a pre-imputation copy for the 50% drop rule, and runs 4-pass fill on 6 variables. Expect ~4–6 GB RSS during export at the current bbox.
- **Dependencies**: `requirements.txt` lists packages; `ensure_deps.py` fails fast with a venv hint if `numpy` is missing (wrong interpreter). Entrypoints: `run_pipeline.py`, `export_csv.py`, `analyse_outputs.py`.

## External Service Dependencies

| Service | Auth | Used by |
|---------|------|---------|
| NOAA ERDDAP (`coastwatch.pfeg.noaa.gov`) | None | SST, chlorophyll, CalCOFI |
| Copernicus Marine (`copernicusmarine`) | Login required | SSH, GLORYS12 salinity |
| Argo / `argopy` ERDDAP backend | None | MLD |
| NOAA NCEI HTTPS | None | WOA18 MLD climatology |
