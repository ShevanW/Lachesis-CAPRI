# config.py
from pathlib import Path

# --------- Paths ---------
DATA_DIR = Path("data")
AQI_XLSX = DATA_DIR / "2024_All_sites_air_quality_hourly_avg_AIR-I-F-V-VH-O-S1-DB-M2-4-0.xlsx"
AIHW_XLSX = DATA_DIR / "AIHW-ACM-42-Chronic-respiratory-conditions-data-tables.xlsx"  # optional later

# Artifacts/output
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# --------- Modeling scope ---------
# Start with one site for clarity; we can generalize later.
SITE_SHEET = "Alphington_10001"     # e.g., "Melbourne CBD_10239", "Dandenong_10022"

# Predict how far ahead (in hours)
HORIZON_HOURS = 6

# Lag features (hours)
LAGS = [1, 2, 3, 6]

# Which pollutant columns to consider (after renaming SBPM25 -> PM25)
POLLUTANTS = ["PM25", "PM10", "NO2", "O3", "SO2", "CO"]

# --------- Label strategy ---------
# A) Percentile-based thresholds per pollutant (safe default until units confirmed)
USE_PERCENTILES = True
PERCENTILE_CUT = 0.90  # top 10% labeled as "high-risk" per pollutant

# B) Fixed thresholds (swap in later with authoritative AU NEPM/WHO values and correct units)
FIXED_THRESHOLDS = {
    "PM25": 35.0,   # µg/m³ (24h) – placeholder
    "PM10": 50.0,   # µg/m³ (24h) – placeholder
    "NO2": 200.0,   # µg/m³ (1h)  – placeholder
    "O3": 120.0,    # µg/m³ (8h)  – placeholder
    "SO2": 125.0,   # µg/m³ (24h) – placeholder
    "CO": 10.0      # mg/m³ (8h)  – placeholder
}

# --------- Plotting / split ---------
N_IMPORTANCES = 20
TEST_FRACTION = 0.2   # time-based 80/20 split
RANDOM_STATE = 42

# Streamlit app defaults
APP_SITE_TITLE = f"AQI → Asthma Risk Prototype ({SITE_SHEET})"
