from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

TRAINING_CSV = REPO_ROOT / "Model2DataEngineering" / "outputs" / "training_data.csv"

ARTIFACTS_DIR = ROOT / "artifacts"
OUTPUTS_DIR = ROOT / "outputs"
MAPS_DIR = OUTPUTS_DIR / "maps"

FEATURE_COLUMNS = [
    "sst",
    "sst_anomaly",
    "chlorophyll",
    "salinity",
    "dissolved_oxygen",
    "ssh",
    "day_of_year",
    "month",
    "lat",
    "lon",
]

META_COLUMNS = ["time", "lat", "lon"]
NON_SPECIES_COLUMNS = [
    "time", "lat", "lon", "sst", "sst_anomaly", "chlorophyll",
    "salinity", "dissolved_oxygen", "ssh", "day_of_year", "month",
]

MIN_POSITIVES = 20
RANDOM_STATE = 42
TEST_FRACTION = 0.2

LAT_MIN, LAT_MAX, LAT_STEP = 30.0, 48.0, 0.25
LON_MIN, LON_MAX, LON_STEP = -124.0, -116.0, 0.25
