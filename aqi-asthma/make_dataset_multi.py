# make_dataset_multi.py
"""
Build a combined train/test set from multiple monitoring sites.
- Loads every site in config.SITE_SHEETS
- Standardises columns (SBPM25 -> PM25)
- Computes thresholds ON THE COMBINED DATA (percentiles or fixed)
- Builds future label per site (shift within site)
- Creates lag features per site (groupby to avoid leakage)
- Chronologically splits the combined frame
Outputs: artifacts/train.csv, artifacts/test.csv (overwrites previous)
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    AQI_XLSX, ARTIFACT_DIR,
    SITE_SHEETS, SITE_SHEET,
    HORIZON_HOURS, LAGS, POLLUTANTS,
    USE_PERCENTILES, PERCENTILE_CUT, FIXED_THRESHOLDS,
    TEST_FRACTION
)

def load_site_sheet(sheet_name: str) -> pd.DataFrame:
    if not AQI_XLSX.exists():
        raise FileNotFoundError(f"Can't find {AQI_XLSX}. Put the Excel in the 'data' folder.")
    df = pd.read_excel(AQI_XLSX, sheet_name=sheet_name)
    # Normalise PM2.5 column name
    if "SBPM25" in df.columns:
        df = df.rename(columns={"SBPM25": "PM25"})
    if "datetime_local" not in df.columns:
        raise ValueError(f"'datetime_local' missing in sheet {sheet_name}")
    df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors="coerce")
    df = df.sort_values("datetime_local").drop_duplicates(subset=["datetime_local"])
    df["site_sheet"] = sheet_name  # keep site identity
    return df

def compute_thresholds_combined(df_all: pd.DataFrame) -> dict:
    thr = {}
    if USE_PERCENTILES:
        for p in POLLUTANTS:
            if p in df_all.columns:
                val = df_all[p].quantile(PERCENTILE_CUT)
                if pd.notna(val):
                    thr[p] = float(val)
    else:
        for p, v in FIXED_THRESHOLDS.items():
            if p in df_all.columns:
                thr[p] = float(v)
    if not thr:
        raise RuntimeError("No thresholds computed. Check POLLUTANTS vs columns.")
    return thr

def build_label_any_exceeds(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    flags = []
    for p, t in thresholds.items():
        if p in df.columns:
            flags.append(df[p] > t)
    if not flags:
        raise RuntimeError("No pollutant flags created. Check thresholds & columns.")
    return pd.Series(np.where(np.logical_or.reduce(flags), 1, 0), index=df.index, name="risk_label")

def add_lag_features_grouped(df: pd.DataFrame, feature_cols: List[str], lags: List[int], group_key: str) -> pd.DataFrame:
    out = df.copy()
    # Add lags within each site to avoid cross-site leakage
    for c in feature_cols:
        if c not in out.columns:
            continue
        for L in lags:
            out[f"{c}_lag{L}"] = out.groupby(group_key)[c].shift(L)
    return out

def chronological_split(df: pd.DataFrame, test_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split_idx = int(round(n * (1 - test_fraction)))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    return train, test

def main():
    # 1) Load all requested sites
    sheets = SITE_SHEETS if SITE_SHEETS else [SITE_SHEET]
    print("Multi-site build from sheets:", ", ".join(sheets))

    site_frames = []
    for s in sheets:
        df_s = load_site_sheet(s)
        site_frames.append(df_s)

    # 2) Combine
    df_all = pd.concat(site_frames, ignore_index=True)
    df_all = df_all.sort_values(["datetime_local", "site_sheet"])

    # 3) Identify candidate feature columns (exclude time/id/meta)
    exclude = {"datetime_AEST", "datetime_local", "location_id", "location_name", "site_sheet"}
    feature_candidates = [c for c in df_all.columns if c not in exclude]

    # 4) Compute thresholds on COMBINED data and label per row
    thresholds = compute_thresholds_combined(df_all)
    df_all["risk_label"] = build_label_any_exceeds(df_all, thresholds)

    # 5) Future label (per site)
    df_all = df_all.set_index("datetime_local")
    df_all["risk_label_future"] = df_all.groupby("site_sheet")["risk_label"].shift(-HORIZON_HOURS)

    # 6) Lag features (per site)
    df_all = add_lag_features_grouped(df_all, feature_candidates, LAGS, group_key="site_sheet")

    # 7) Drop rows without full lags / target
    lag_cols = [f"{c}_lag{L}" for c in feature_candidates for L in LAGS]
    model_df = df_all.dropna(subset=lag_cols + ["risk_label_future"]).copy()
    model_df["risk_label_future"] = model_df["risk_label_future"].astype(int)

    # 8) Chronological split over the combined index
    model_df = model_df.sort_index()
    train_df, test_df = chronological_split(model_df, TEST_FRACTION)

    # 9) Save
    processed_csv = ARTIFACT_DIR / "processed_multisite_full.csv"
    train_csv     = ARTIFACT_DIR / "train.csv"   # overwrite on purpose (train_model.py reads these)
    test_csv      = ARTIFACT_DIR / "test.csv"
    info_json     = ARTIFACT_DIR / "labeling_info_multi.json"
    raw_out_csv   = ARTIFACT_DIR / "multisite_raw_ordered.csv"

    # Save raw (with index back)
    df_all_reset = df_all.reset_index()
    df_all_reset.to_csv(raw_out_csv, index=False)
    train_df.to_csv(train_csv, index=True)  # index is datetime_local
    test_df.to_csv(test_csv, index=True)
    model_df.to_csv(processed_csv, index=True)

    info = {
        "sheets_used": sheets,
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

    print("\n=== Multi-site Summary ===")
    print(f"Sites: {', '.join(sheets)}")
    print(f"Rows after lag/target dropna: {len(model_df)}")
    print(f"Train/Test: {len(train_df)} / {len(test_df)}")
    print("Effective thresholds (combined):")
    for k, v in thresholds.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved:\n  {raw_out_csv}\n  {processed_csv}\n  {train_csv}\n  {test_csv}\n  {info_json}")

if __name__ == "__main__":
    main()
