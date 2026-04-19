# OceanPulse — Model 2 Species Classification Dataset Pipeline

Builds a production-quality, flat ML training dataset for predicting fish species likelihood from oceanographic conditions.

**Output**: `outputs/training_data.csv` — each row is one independent observation:
```
(time, lat, lon, sst, sst_anomaly, chlorophyll, salinity, dissolved_oxygen, ssh, day_of_year, month, species, presence)
```

---

## Architecture

```
NOAA ERDDAP (OISST) ──────────────► download_erddap.py   → sst, sst_anomaly
NOAA ERDDAP (MODIS) ──────────────► download_erddap.py   → chlorophyll
CMEMS DUACS ──────────────────────► download_cmems.py    → ssh
Argo / argopy ────────────────────► download_argo.py     → salinity
WOA18 NCEI HTTPS ─────────────────► download_cmems.py    → dissolved_oxygen

CalCOFI ERDDAP ────────────────────► process_calcofi.py  → species (presence=1)
                                                           + in-situ features
iNaturalist / GBIF API ────────────► process_inaturalist.py → species (presence=1)
NOAA WCGBT (NWFSC API) ───────────► process_noaa_trawl.py  → species (presence 0/1)

All three species sources ─────────► load_species_data.py  → combined species table
Ocean data + species obs ──────────► feature_store.py       → LEFT JOIN features
Feature patching (CalCOFI in-situ)► join_species.py         → fills NaN features
Size estimation + downscaling ─────► size_control.py        → ≤5 GB enforcement
Final export ──────────────────────► export_dataset.py      → training_data.csv
```

### Master Grid
- **Lat**: 30.0–48.0°N, 0.25° spacing (73 points)
- **Lon**: −124.0–−116.0°W, 0.25° spacing (33 points)
- **Time**: 2010-01-01–2024-12-31, daily (15 years)

---

## Setup

### 1. Create virtual environment

```bash
cd Model2DataEngineering
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# or
.venv\Scripts\activate          # Windows
```

### 2. Install dependencies

```bash
pip install numpy pandas xarray scipy requests argopy copernicusmarine pyarrow
```

### 3. CMEMS Authentication

The SSH download requires Copernicus Marine Service credentials.

**Option A — Environment variables (recommended):**
```bash
export CMEMS_USERNAME="your_cmems_username"
export CMEMS_PASSWORD="your_cmems_password"
```

**Option B — Persistent login (one-time interactive):**
```bash
copernicusmarine login
```
Credentials are stored in `~/.copernicusmarine/` and reused automatically.

Register for a free account at: https://marine.copernicus.eu

---

## Running the Pipeline

### Full pipeline (all downloads + build)
```bash
python run_pipeline.py
```

### Skip download (use cached files)
```bash
python run_pipeline.py --skip-download
```

### Force rebuild (clear caches, re-download everything)
```bash
python run_pipeline.py --force-rebuild
```

All output is logged to both `stdout` and `logs/pipeline.log`.

---

## Output Schema

`outputs/training_data.csv`:

| Column | Type | Description |
|--------|------|-------------|
| `time` | str (YYYY-MM-DD) | Observation date |
| `lat` | float32 | Latitude (snapped to 0.25° grid) |
| `lon` | float32 | Longitude (snapped to 0.25° grid) |
| `sst` | float32 | Sea surface temperature (°C) |
| `sst_anomaly` | float32 | SST deviation from DOY climatology (°C) |
| `chlorophyll` | float32 | Surface chlorophyll-a (mg m⁻³) |
| `salinity` | float32 | Surface salinity (PSU) |
| `dissolved_oxygen` | float32 | Surface dissolved oxygen (ml/L) |
| `ssh` | float32 | Sea surface height anomaly (m) |
| `day_of_year` | int16 | 1–366 |
| `month` | int8 | 1–12 |
| `species` | str | Standardised scientific name (snake_case) |
| `presence` | int8 | 1 = observed, 0 = confirmed absent (NOAA trawl only) |

---

## Species Data Sources

### 1. CalCOFI (California Cooperative Oceanic Fisheries Investigations)
- Long-term larvae counts from ERDDAP (`siocalcofiLarvalCounts`)
- Bottle data used for in-situ feature patching (`siocalcofiHydroBottle`)
- Coverage: California Current since 1949; this pipeline uses 2010–2024
- Presence = 1 only (no confirmed absences from larvae surveys)

### 2. iNaturalist (via GBIF)
- Research-grade adult fish sightings
- Dataset key: `50c9509d-22c7-4a22-a47d-8c48425ef4a7` on GBIF
- Classes: Actinopterygii + Elasmobranchii
- Paginated year-by-year via `https://api.gbif.org/v1/occurrence/search`
- Presence = 1 only

### 3. NOAA West Coast Groundfish Bottom Trawl Survey (WCGBT)
- NWFSC Data Warehouse API
- **Provides both presences AND confirmed absences** (CPUE = 0 → absence)
- Strongest supervised learning signal
- Presence = 1 (CPUE > 0) or 0 (CPUE = 0)

---

## Dataset Size Handling

The pipeline targets a final CSV of **1–5 GB**. If the dataset exceeds 5 GB,
an automatic downscaling cascade is applied in order:

1. **Temporal downsampling**: sample every 2 → 3 → 5 → 7 days
2. **Species filtering**: keep top 50 → 40 → 30 → 20 species by observation count
3. **Spatial coarsening**: round to 0.5° resolution (last resort)

Size is estimated before writing (15 bytes per CSV cell, conservative). Actual
file size is logged after writing along with row count and species count.

To adjust thresholds, edit `pipeline/config.py`:
```python
MAX_SIZE_GB = 5.0
MAX_SPECIES_TOP_N = 50
TEMPORAL_STRIDE_DAYS = 2
YEARS = list(range(2010, 2025))   # reduce range to shrink dataset
```

---

## Cache Structure

Downloaded files are cached and re-used on subsequent runs:

```
data/cache/
  oisst_{year}.nc                — OISST SST annual files
  modis_chl_{year}.nc            — MODIS chlorophyll annual files
  cmems_ssh.nc                   — CMEMS SSH full period
  glorys12_salinity.nc           — GLORYS12 salinity (if used)
  argo_surface_salinity.parquet  — Argo profiles
  woa18_do_m{01-12}.nc           — WOA18 monthly DO climatology
  calcofi_species.parquet        — CalCOFI larvae species obs
  calcofi_insitu_features.parquet — CalCOFI bottle in-situ measurements
  inaturalist_species.parquet    — iNaturalist species obs
  noaa_trawl_species.parquet     — NOAA WCGBT species obs
  sst_doy_climatology.nc         — Pre-computed SST DOY climatology
  features_joined.parquet        — Intermediate joined table
```

Delete individual cache files to force re-download of specific sources.

---

## Key Design Decisions

- **Every row is an independent observation** — no sequences, no sliding windows,
  no time-series tensors. This is a flat tabular dataset for classification.
- **LEFT JOIN from species**: ocean features are joined onto species observations,
  not the other way around. Empty ocean grid cells without species observations
  are never included.
- **CalCOFI feature patching**: in-situ bottle measurements fill gridded gaps at
  survey station locations (within ±3 days, same grid cell).
- **SST anomaly**: computed from the full 2010–2024 DOY climatology (fixed, not
  rolling) to prevent data leakage into future training splits.
- **Intermediate caches are parquet** (columnar, fast); only the final output is CSV.
- **Salinity fallback hierarchy**: Argo profiles → GLORYS12 → 33.5 PSU constant.

---

## Important Notes

- The `mld_source` flag from Model 1 is **not** in this dataset; MLD is also
  omitted as it was not listed in the spec's feature schema.
- SSH may be NaN for periods outside CMEMS DUACS coverage (currently limited to
  the MY reprocessed product ending around 2023).
- iNaturalist GBIF queries return at most 100 000 records per year-taxon
  combination. For denser coverage, consider using the GBIF Download API with
  a registered account.
