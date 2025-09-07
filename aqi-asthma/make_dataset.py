# make_dataset.py
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AQI_XLSX, ARTIFACT_DIR, SITE_SHEET,
    HORIZON_HOURS, LAGS, POLLUTANTS,
    USE_PERCENTILES, PERCENTILE_CUT, FIXED_THRESHOLDS,
    TEST_FRACTION
)

def load_site_df() -> pd.DataFrame:
    if not AQI_XLSX.exists():
        raise FileNotFoundError(
            f"Can't find {AQI_XLSX}. "
            "Make sure your AQI Excel file is in the 'data' folder."
        )
    df = pd.read_excel(AQI_XLSX, sheet_name=SITE_SHEET)
    # Standardize PM2.5 name
    if "SBPM25" in df.columns:
        df = df.rename(columns={"SBPM25": "PM25"})
    # Parse and sort time
    if "datetime_local" not in df.columns:
        raise ValueError(f"'datetime_local' column not found in sheet {SITE_SHEET}")
    df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors="coerce")
    df = df.sort_values("datetime_local").drop_duplicates(subset=["datetime_local"])
    return df

def compute_thresholds(df: pd.DataFrame) -> dict:
    """Return effective thresholds per pollutant based on config."""
    thr = {}
    if USE_PERCENTILES:
        for p in POLLUTANTS:
            if p in df.columns:
                val = df[p].quantile(PERCENTILE_CUT)
                if pd.notna(val):
                    thr[p] = float(val)
    else:
        # Use provided fixed thresholds, but only for columns found in df
        for p, v in FIXED_THRESHOLDS.items():
            if p in df.columns:
                thr[p] = float(v)
    if not thr:
        raise RuntimeError("No thresholds computed. Check POLLUTANTS and dataset columns.")
    return thr

def build_label(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    """Label = 1 if any pollutant exceeds its threshold, else 0."""
    flags = []
    for p, t in thresholds.items():
        if p in df.columns:
            flags.append(df[p] > t)
    if not flags:
        raise RuntimeError("No pollutant flags created. Check thresholds & columns.")
    high = np.logical_or.reduce(flags)
    return pd.Series(np.where(high, 1, 0), index=df.index, name="risk_label")

def make_lag_features(df: pd.DataFrame, cols: list, lags: list) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        for L in lags:
            out[f"{c}_lag{L}"] = out[c].shift(L)
    return out

def chronological_split(df: pd.DataFrame, test_fraction: float):
    n = len(df)
    split_idx = int(round(n * (1 - test_fraction)))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    return train, test

def main():
    print(f"Loading site: {SITE_SHEET}")
    df = load_site_df()

    # Keep base columns + potential features
    base_cols = ["datetime_local", "location_id", "location_name"]
    feature_candidates = [
        c for c in df.columns
        if c not in ["datetime_AEST", "datetime_local", "location_id", "location_name"]
    ]

    # Compute thresholds and label current hour
    thresholds = compute_thresholds(df)
    df["risk_label"] = build_label(df, thresholds)

    # Build *future* target (predict H hours ahead)
    df = df.set_index("datetime_local")
    df["risk_label_future"] = df["risk_label"].shift(-HORIZON_HOURS)

    # Create lagged features for all candidate columns
    df = make_lag_features(df, feature_candidates, LAGS)

    # Drop rows that don't have full lags or the future label
    lag_cols = [f"{c}_lag{L}" for c in feature_candidates for L in LAGS]
    model_df = df.dropna(subset=lag_cols + ["risk_label_future"]).copy()

    # Prepare final modeling frame
    model_df["risk_label_future"] = model_df["risk_label_future"].astype(int)

    # Chronological train/test split
    train_df, test_df = chronological_split(model_df, TEST_FRACTION)

    # Save outputs
    processed_csv = ARTIFACT_DIR / "processed_site_full.csv"
    train_csv     = ARTIFACT_DIR / "train.csv"
    test_csv      = ARTIFACT_DIR / "test.csv"
    info_json     = ARTIFACT_DIR / "labeling_info.json"

    # (Optional) also save the raw site df with time index reset for reference
    raw_out_csv = ARTIFACT_DIR / "site_raw_ordered.csv"
    df.reset_index().to_csv(raw_out_csv, index=False)

    train_df.to_csv(train_csv, index=True)  # index is datetime_local
    test_df.to_csv(test_csv, index=True)
    model_df.to_csv(processed_csv, index=True)

    info = {
        "site_sheet": SITE_SHEET,
        "horizon_hours": HORIZON_HOURS,
        "lags": LAGS,
        "use_percentiles": USE_PERCENTILES,
        "percentile_cut": PERCENTILE_CUT,
        "thresholds_effective": thresholds,
        "n_total_after_dropna": int(len(model_df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_candidates": feature_candidates,
        "lag_feature_count": len(lag_cols),
        "test_fraction": TEST_FRACTION,
    }
    with open(info_json, "w") as f:
        json.dump(info, f, indent=2)

    print("\n=== Summary ===")
    print(f"Rows after lag/target dropna: {len(model_df)}")
    print(f"Train/Test: {len(train_df)} / {len(test_df)}")
    print("Effective thresholds:")
    for k, v in thresholds.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved:\n  {raw_out_csv}\n  {processed_csv}\n  {train_csv}\n  {test_csv}\n  {info_json}")

if __name__ == "__main__":
    main()
