# make_dataset_multi_feats.py
"""
Multi-site dataset with temporal + rolling features.
- Uses SITE_SHEETS from config.py
- Temporal: hour-of-day, day-of-week, month (plus cyclical sin/cos)
- Rolling features (per site): 3h, 6h, 24h means for pollutants & met columns
- Lags (1,2,3,6h) and 6h-ahead target as before
Outputs: overwrites artifacts/train.csv and artifacts/test.csv
"""

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from config import (
    AQI_XLSX, ARTIFACT_DIR,
    SITE_SHEETS, SITE_SHEET,  # fallback if SITE_SHEETS empty
    HORIZON_HOURS, LAGS, POLLUTANTS,
    USE_PERCENTILES, PERCENTILE_CUT, FIXED_THRESHOLDS,
    TEST_FRACTION
)

# ---- helpers ----

def load_site_sheet(sheet_name: str) -> pd.DataFrame:
    if not AQI_XLSX.exists():
        raise FileNotFoundError(f"Can't find {AQI_XLSX}. Put the Excel in 'data/'.")
    df = pd.read_excel(AQI_XLSX, sheet_name=sheet_name)
    if "SBPM25" in df.columns:
        df = df.rename(columns={"SBPM25": "PM25"})
    if "datetime_local" not in df.columns:
        raise ValueError(f"'datetime_local' missing in sheet {sheet_name}")
    df["datetime_local"] = pd.to_datetime(df["datetime_local"], errors="coerce")
    df = df.sort_values("datetime_local").drop_duplicates(subset=["datetime_local"])
    df["site_sheet"] = sheet_name
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

def label_any_exceeds(df: pd.DataFrame, thresholds: dict) -> pd.Series:
    flags = []
    for p, t in thresholds.items():
        if p in df.columns:
            flags.append(df[p] > t)
    if not flags:
        raise RuntimeError("No pollutant flags created. Check thresholds & columns.")
    return pd.Series(np.where(np.logical_or.reduce(flags), 1, 0), index=df.index, name="risk_label")

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = out.index  # datetime index
    out["hour"]  = idx.hour
    out["dow"]   = idx.dayofweek  # Monday=0
    out["month"] = idx.month

    # Cyclical encodings
    out["hour_sin"]  = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"]  = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["dow_sin"]   = np.sin(2 * np.pi * out["dow"] / 7.0)
    out["dow_cos"]   = np.cos(2 * np.pi * out["dow"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    return out

def add_rolling_means(df: pd.DataFrame, cols: List[str], windows: List[int], group_key: str) -> pd.DataFrame:
    """
    Add rolling means (past window) per site to avoid leakage.
    windows are in 'hours' assuming regular hourly sampling.
    """
    out = df.copy()
    grouped = out.groupby(group_key)
    for c in cols:
        if c not in out.columns:
            continue
        for w in windows:
            out[f"{c}_rollmean_{w}h"] = grouped[c].transform(lambda s: s.rolling(window=w, min_periods=max(1, w//2)).mean())
    return out

def add_lag_features_grouped(df: pd.DataFrame, feature_cols: List[str], lags: List[int], group_key: str) -> pd.DataFrame:
    out = df.copy()
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

# ---- main ----

def main():
    sheets = SITE_SHEETS if SITE_SHEETS else [SITE_SHEET]
    print("Multi-site (with features) from sheets:", ", ".join(sheets))

    # 1) Load and combine
    site_frames = [load_site_sheet(s) for s in sheets]
    df_all = pd.concat(site_frames, ignore_index=True)
    df_all = df_all.sort_values(["datetime_local", "site_sheet"])

    # 2) Identify base numeric columns for features
    exclude = {"datetime_AEST", "datetime_local", "location_id", "location_name", "site_sheet"}
    base_numeric = [c for c in df_all.columns if c not in exclude]

    # 3) Build label (on combined thresholds)
    thresholds = compute_thresholds_combined(df_all)
    df_all = df_all.set_index("datetime_local")
    df_all["risk_label"] = label_any_exceeds(df_all, thresholds)
    df_all["risk_label_future"] = df_all.groupby("site_sheet")["risk_label"].shift(-HORIZON_HOURS)

    # 4) Temporal features
    df_all = add_time_features(df_all)

    # 5) Rolling features (per site) — 3h, 6h, 24h means
    # Choose columns to roll: pollutants + some met vars if present
    roll_candidates = [c for c in base_numeric if c in df_all.columns]
    windows = [3, 6, 24]
    df_all = add_rolling_means(df_all, roll_candidates, windows, group_key="site_sheet")

    # 6) Lags (per site) — on base numeric + rolling features
    lag_sources = roll_candidates + [f"{c}_rollmean_{w}h" for c in roll_candidates for w in windows]
    df_all = add_lag_features_grouped(df_all, lag_sources, LAGS, group_key="site_sheet")

    # 7) Assemble modeling columns
    lag_cols = [f"{c}_lag{L}" for c in lag_sources for L in LAGS]
    # Keep temporal features as **direct** features (no lag)
    time_cols = ["hour","dow","month","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]
    needed = lag_cols + time_cols + ["risk_label_future", "site_sheet"]

    model_df = df_all[needed].dropna(subset=lag_cols + ["risk_label_future"]).copy()
    model_df["risk_label_future"] = model_df["risk_label_future"].astype(int)
    model_df = model_df.sort_index()

    # 8) Split
    train_df, test_df = chronological_split(model_df, TEST_FRACTION)

    # 9) Save
    processed_csv = ARTIFACT_DIR / "processed_multisite_feats_full.csv"
    train_csv     = ARTIFACT_DIR / "train.csv"  # overwrite for train_model.py
    test_csv      = ARTIFACT_DIR / "test.csv"
    info_json     = ARTIFACT_DIR / "labeling_info_multi_feats.json"
    raw_out_csv   = ARTIFACT_DIR / "multisite_raw_ordered.csv"  # reuse

    # Save combined raw with index reset (for reference)
    df_all_reset = df_all.reset_index()
    df_all_reset.to_csv(raw_out_csv, index=False)
    train_df.to_csv(train_csv, index=True)
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
        "time_features": time_cols,
        "rolling_windows_h": windows,
        "lag_feature_count": len(lag_cols),
        "test_fraction": TEST_FRACTION,
    }
    with open(info_json, "w") as f:
        json.dump(info, f, indent=2)

    print("\n=== Multi-site + Features Summary ===")
    print(f"Sites: {', '.join(sheets)}")
    print(f"Rows after dropna: {len(model_df)}")
    print(f"Train/Test: {len(train_df)} / {len(test_df)}")
    print("Effective thresholds (combined):")
    for k, v in thresholds.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved:\n  {raw_out_csv}\n  {processed_csv}\n  {train_csv}\n  {test_csv}\n  {info_json}")

if __name__ == "__main__":
    main()
