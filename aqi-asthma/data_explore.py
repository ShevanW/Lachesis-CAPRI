# data_explore.py
import pandas as pd
from pathlib import Path
from config import AQI_XLSX, SITE_SHEET, ARTIFACT_DIR

def main():
    print("Loading workbook:", AQI_XLSX.resolve())
    if not AQI_XLSX.exists():
        raise FileNotFoundError(
            f"Can't find {AQI_XLSX}. "
            "Make sure your AQI Excel file is copied into the 'data' folder."
        )

    xls = pd.ExcelFile(AQI_XLSX)

    print("\nAvailable sheets (first 20):")
    for s in xls.sheet_names[:20]:
        print(" -", s)

    # Peek Metadata
    if "Metadata" in xls.sheet_names:
        meta = pd.read_excel(AQI_XLSX, sheet_name="Metadata")
        print("\n[Metadata] head:")
        with pd.option_context("display.max_columns", 12, "display.width", 120):
            print(meta.head(10))

    # Sample AllData (tall table of all sites/parameters)
    if "AllData" in xls.sheet_names:
        all_sample = pd.read_excel(AQI_XLSX, sheet_name="AllData", nrows=500)
        print("\n[AllData] columns:")
        print(list(all_sample.columns))
        print("\n[AllData] sample head():")
        with pd.option_context("display.max_columns", 18, "display.width", 160):
            print(all_sample.head(5))

    # Load the chosen site (wide format)
    print(f"\nLoading site sheet: {SITE_SHEET}")
    site_df = pd.read_excel(AQI_XLSX, sheet_name=SITE_SHEET)

    # Rename SBPM25 -> PM25 if present for consistency
    if "SBPM25" in site_df.columns:
        site_df = site_df.rename(columns={"SBPM25": "PM25"})

    print("\n[Site] columns:")
    print(list(site_df.columns))

    print("\n[Site] head:")
    with pd.option_context("display.max_columns", 18, "display.width", 160):
        print(site_df.head(5))

    # Identify pollutant & met columns
    pollutant_maybe = ["PM25","PM10","NO2","O3","SO2","CO","DBT","BSP","SWD","SWS","VWD","VWS","Sigma60"]
    pollutant_cols = [c for c in pollutant_maybe if c in site_df.columns]

    # Basic missingness
    base_cols = [c for c in ["datetime_local","location_id","location_name"] if c in site_df.columns]
    keep_cols = base_cols + pollutant_cols
    sub = site_df[keep_cols].copy()
    if "datetime_local" in sub.columns:
        sub["datetime_local"] = pd.to_datetime(sub["datetime_local"], errors="coerce")
        sub = sub.sort_values("datetime_local")

    print("\n[Site] pollutant/meteorology detected:")
    print(pollutant_cols)

    miss = sub.isna().mean().sort_values(ascending=False)
    print("\nMissingness (fraction NaN) by column:")
    print(miss)

    # Save a small CSV sample for manual check
    out_csv = ARTIFACT_DIR / "site_sample_first500.csv"
    site_df.head(500).to_csv(out_csv, index=False)
    print(f"\nWrote sample rows to: {out_csv.resolve()}")

if __name__ == "__main__":
    main()
