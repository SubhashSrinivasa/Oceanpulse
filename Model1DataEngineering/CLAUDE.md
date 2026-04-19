# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OceanPulse Model1 Data Engineering: a 7-stage pipeline that ingests oceanographic data from multiple public sources, harmonizes it onto a shared 0.25° daily grid over the Central California Upwelling Zone (34–42°N, 124–118°W, 2018–2022), and produces a Zarr cube + two ML-ready CSV exports.

## Commands

```bash
# Activate virtual environment (always required first)
source .venv/bin/activate

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

All stages ──► stage6_assemble.py → outputs/ocean_cube.zarr  (1826×33×25, 8 vars)
                    └──► stage7_qc.py  → data_quality_report.txt, missing_data_map.png
                    └──► export_csv.py → ocean_cube_long.csv, ocean_cube_sequences.csv
```

### Master Grid (defined in `pipeline/config.py`)
- **Lat**: 34.0–42.0°N, 0.25° spacing, 33 points
- **Lon**: −124.0–−118.0°W, 0.25° spacing, 25 points
- **Time**: 2018-01-01–2022-12-31, daily, 1826 steps

All source grids are regridded/reindexed to this master before merge.

### Pipeline Modules

| File | Role |
|------|------|
| `pipeline/config.py` | Master grid constants, all file paths, physical sanity bounds |
| `pipeline/utils.py` | Shared logger (`get_logger`), QC record list (`qc_variable`) |
| `pipeline/stage1_sst.py` | OISST via ERDDAP, one year at a time; bridges ≤3-day gaps; computes DOY climatology + anomaly |
| `pipeline/stage2_ssh.py` | CMEMS DUACS 0.125° → regridded to 0.25°; Gaussian synthetic fallback if unavailable |
| `pipeline/stage3_chl.py` | MODIS-Aqua 8-day L3; probes dataset end-date to cap requests; regrid + daily resample |
| `pipeline/stage4_mld.py` | Argo profiles (density-threshold MLD via `gsw`) → WOA18 monthly climatology → constant 50 m |
| `pipeline/stage5_salinity.py` | CalCOFI bottle (≤10 m) → GLORYS12 → constant 33.5 PSU |
| `pipeline/stage6_assemble.py` | Merges all datasets; validates dims/axes/bounds; writes Zarr |
| `pipeline/stage7_qc.py` | Text report + 2×3 NaN-rate map |
| `export_csv.py` | Loads Zarr, runs 3-pass chlorophyll imputation, writes both CSV formats |
| `analyse_outputs.py` | Stats on both CSVs (null counts, mean, std, min, max) |

### CSV Outputs

**`ocean_cube_long.csv`** — tabular, one row per (time, lat, lon) ocean cell  
Columns: `time, lat, lon, month, day_of_year, sst, sst_anomaly, ssh, chlorophyll, chlorophyll_log, mld, mld_source, salinity`  
Land cells (SST always NaN) are dropped → ~197 ocean locations × 1826 days = ~360K rows.

**`ocean_cube_sequences.csv`** — wide, one row per ocean location  
Each variable column is a JSON list of 1826 daily values. First line is a `#` comment with `time_start`, `time_end`, `n_timesteps`, `dt_days`. Load with `pd.read_csv(..., comment='#')` then `df['sst'].apply(json.loads)`.

### Chlorophyll Imputation (in `export_csv.py`)

Imputation happens at export time (Zarr stays raw). Three passes, all in log₁₀ space:
1. **Temporal**: linear interp per cell, gap ≤ 30 days — handles cloud gaps
2. **Spatial**: `scipy.interpolate.griddata` per timestep using valid ocean neighbors — handles longer gaps
3. **DOY climatology**: day-of-year mean from 2018–2021 — fills the Jun–Dec 2022 MODIS blackout

Result is back-transformed to linear chlorophyll. Both CSVs emerge with 0% nulls.

## Key Non-Obvious Details

- **CMEMS product ID changed**: the 0.25° MY DUACS product was retired; pipeline uses `cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D` and regridds it.
- **MODIS dataset ends 2022-06-14**: stage 3 auto-detects the dataset's actual time range and caps the 2022 request rather than failing.
- **WOA23 has no MLD product**: the pipeline falls back to WOA18 (`decav81B0` series, `woa18_decav81B0_M02{mm}_01.nc`).
- **Sanity bounds** are defined in `config.py` and checked in stage 6 as warnings, not errors — the pipeline completes even with out-of-bounds values so downstream QC can decide.
- **`mld_source` flag**: 0 = Argo-derived, 1 = WOA18 climatology, 2 = constant 50 m fallback. Use this in ML to weight or mask low-confidence MLD values.
- **CMEMS requires authentication**: run `copernicusmarine login` once before the pipeline; credentials are stored by the package in `~/.copernicusmarine/`.
- **Validation tolerances**: lat/lon reindexing uses `tolerance=0.13–0.15` (roughly half a 0.25° cell) to snap nearby grid points.

## External Service Dependencies

| Service | Auth | Used by |
|---------|------|---------|
| NOAA ERDDAP (`coastwatch.pfeg.noaa.gov`) | None | SST, chlorophyll, CalCOFI |
| Copernicus Marine (`copernicusmarine`) | Login required | SSH, GLORYS12 salinity |
| Argo / `argopy` ERDDAP backend | None | MLD |
| NOAA NCEI HTTPS | None | WOA18 MLD climatology |
