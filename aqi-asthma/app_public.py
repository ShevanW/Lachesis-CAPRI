# app_public.py — Public-facing AQI → Asthma Risk (6h ahead)
# Simple UI:
# - Pick location (monitoring site)
# - Pick model (RF/XGB)
# - Sensitivity (Cautious/Balanced/Strict) instead of raw threshold
# - Show: current risk status, mini forecast, and pollutants vs thresholds

from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt

ARTIFACTS = Path("artifacts")
MODELS = {
    "XGBoost":       ARTIFACTS / "aqi_asthma_xgb_model.joblib",
    "Random Forest": ARTIFACTS / "aqi_asthma_rf_model.joblib",
}

# Full processed datasets (prefer features version)
FULL_DATA_CANDIDATES = [
    ARTIFACTS / "processed_multisite_feats_full.csv",  # from make_dataset_multi_feats.py
    ARTIFACTS / "processed_multisite_full.csv",        # from make_dataset_multi.py
    ARTIFACTS / "processed_site_full.csv",             # single-site fallback
]

RAW_CSV   = ARTIFACTS / "multisite_raw_ordered.csv"    # for “what’s driving risk”
THRESH_JSON = ARTIFACTS / "labeling_info_multi_feats.json"  # thresholds
TEMPORAL_COLS = [
    "hour","dow","month","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"
]

SENS_TO_THR = {
    "Cautious": 0.25,   # more alerts (higher recall)
    "Balanced": 0.40,
    "Strict":   0.60,   # fewer alerts (higher precision)
}

# ---------- helpers ----------
@st.cache_data
def _load_csv(path: Path, dt_index: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    if dt_index:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "datetime_local"
    else:
        df = pd.read_csv(path)
        if "datetime_local" in df.columns:
            df["datetime_local"] = pd.to_datetime(df["datetime_local"])
    return df

@st.cache_data
def _load_full_processed() -> pd.DataFrame:
    # Try the candidates in order; first one that exists wins
    for p in FULL_DATA_CANDIDATES:
        if p.exists():
            df = _load_csv(p, dt_index=True)
            return df
    raise FileNotFoundError(
        "No processed dataset found. Expected one of: "
        + ", ".join(str(p) for p in FULL_DATA_CANDIDATES)
        + ". Run a dataset builder script first."
    )

@st.cache_data
def _load_thresholds(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            info = json.load(f)
        return info.get("thresholds_effective", {})
    # fallback: try older multi file
    alt_path = ARTIFACTS / "labeling_info_multi.json"
    if alt_path.exists():
        with open(alt_path, "r") as f:
            info = json.load(f)
        return info.get("thresholds_effective", {})
    return {}

def _feature_cols(df: pd.DataFrame) -> list:
    lag_cols = [c for c in df.columns if "_lag" in c]
    time_cols = [c for c in TEMPORAL_COLS if c in df.columns]
    return lag_cols + time_cols

def _load_model(name: str):
    mp = MODELS[name]
    if not mp.exists():
        st.error(f"Model not found: {mp}")
        st.stop()
    return joblib.load(mp)

def _risk_label(prob: float, thr: float) -> tuple[str, str]:
    if prob >= thr:
        severity = "High" if prob >= max(0.9, thr + 0.3) else "Elevated"
        return "⚠️ " + severity + " risk (next 6h)", "alert"
    return "✅ Low risk (next 6h)", "safe"

def _pollutant_panel(raw_df_site_latest: pd.Series, thresholds: dict) -> pd.DataFrame:
    ordered = ["PM25","PM10","NO2","O3","SO2","CO"]
    rows = []
    for p in ordered:
        if p in raw_df_site_latest.index and pd.notna(raw_df_site_latest[p]):
            val = float(raw_df_site_latest[p])
            thr = thresholds.get(p, None)
            status = "—"
            if thr is not None:
                status = "Above" if val > thr else "Below"
            rows.append({"Pollutant": p, "Current": val, "Threshold": thr, "Status": status})
    return pd.DataFrame(rows)

def _forecast_chart(times: pd.Index, probs: np.ndarray, thr: float) -> alt.Chart:
    dfp = pd.DataFrame({"datetime_local": times, "risk_prob": probs})
    base = alt.Chart(dfp).encode(x="datetime_local:T")
    line = base.mark_line().encode(y=alt.Y("risk_prob:Q", title="Risk probability"))
    rule = alt.Chart(pd.DataFrame({"thr":[thr]})).mark_rule(strokeDash=[4,4]).encode(y="thr:Q")
    return alt.layer(line, rule).properties(width="container", height=220, title="Risk outlook")

# ---------- UI ----------
st.set_page_config(page_title="Asthma Risk (next 6 hours)", layout="wide")
st.title("Asthma Risk (next 6 hours)")

# Load data once
full_df = _load_full_processed()

# --- ensure 'site_sheet' exists (compat shim) ---
if "site_sheet" not in full_df.columns:
    if {"location_name", "location_id"}.issubset(full_df.columns):
        # e.g., "Melbourne CBD_10239"
        full_df["site_sheet"] = (
            full_df["location_name"].astype(str)
            + "_" +
            full_df["location_id"].astype(str)
        )
    elif "location_name" in full_df.columns:
        full_df["site_sheet"] = full_df["location_name"].astype(str)
    else:
        st.error("No site identifier found in processed dataset (need site_sheet or location_name[/location_id]).")
        st.stop()


# Sidebar: location, model, sensitivity
with st.sidebar:
    st.header("Choose your location")

    # Build site list strictly from the processed data we actually predict on
    locations = sorted(full_df["site_sheet"].dropna().unique().tolist())
    if not locations:
        st.error("No sites found in the processed dataset.")
        st.stop()

    # Pick CBD if present, else first
    default_idx = locations.index("Melbourne CBD_10239") if "Melbourne CBD_10239" in locations else 0
    site_choice = st.selectbox("Monitoring site", options=locations, index=default_idx)

    st.header("Model & Sensitivity")
    available_models = [m for m, p in MODELS.items() if p.exists()]
    if not available_models:
        st.error("No saved models in artifacts/. Train a model first.")
        st.stop()
    model_name = st.selectbox(
        "Model",
        options=available_models,
        index=available_models.index("XGBoost") if "XGBoost" in available_models else 0
    )
    sensitivity = st.radio("Alert sensitivity", options=list(SENS_TO_THR.keys()),
                           horizontal=True, index=0)
    thr = float(SENS_TO_THR[sensitivity])


# Prepare feature set for the chosen site from the full dataset
if "site_sheet" not in full_df.columns:
    st.error("Processed dataset is missing 'site_sheet'. Rebuild with multi-site builder.")
    st.stop()

feature_cols = _feature_cols(full_df)
site_mask = (full_df["site_sheet"] == site_choice)
site_df = full_df.loc[site_mask].sort_index()
if site_df.empty:
    st.warning("No rows available for this site in the processed dataset.")
    st.stop()

X_site = site_df[feature_cols]
times_site = site_df.index

# Load model & predict probabilities for the site rows
model = _load_model(model_name)
if hasattr(model, "predict_proba"):
    y_prob_site = model.predict_proba(X_site)[:, 1]
else:
    st.error("Loaded model has no predict_proba.")
    st.stop()

# Current (latest) prediction for this site
latest_x = X_site.iloc[[-1]]
latest_ts = times_site[-1]
latest_prob = model.predict_proba(latest_x)[:, 1][0].item()
headline, state = _risk_label(latest_prob, thr)

# Thresholds + current pollutant readings (from raw if present)
thresholds = _load_thresholds(THRESH_JSON)
if RAW_CSV.exists():
    raw_df_all = _load_csv(RAW_CSV, dt_index=False)
    raw_site = raw_df_all[raw_df_all["site_sheet"] == site_choice].sort_values("datetime_local")
    latest_raw = raw_site.iloc[-1] if len(raw_site) else pd.Series(dtype=float)
else:
    latest_raw = pd.Series(dtype=float)
poll_df = _pollutant_panel(latest_raw, thresholds) if not latest_raw.empty else pd.DataFrame()

# ---------- Layout ----------
col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader(f"{site_choice.replace('_', ' ')}")
    # big status chip
    color = "#e67e22" if state == "alert" else "#2ecc71"
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:12px;background:{color}22;border:1px solid {color};">
            <div style="font-size:22px;font-weight:600;color:{color};">{headline}</div>
            <div style="font-size:14px;margin-top:6px;">Model: <b>{model_name}</b> · Sensitivity: <b>{sensitivity}</b> (threshold {thr:.2f})</div>
            <div style="font-size:14px;margin-top:2px;">Latest reading time: {latest_ts}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Mini “forecast”: show last ~24 timestamps' predicted probabilities for this site
    N = min(24, len(y_prob_site))
    st.altair_chart(_forecast_chart(times_site[-N:], y_prob_site[-N:], thr), use_container_width=True)

with col2:
    st.subheader("What’s driving risk right now?")
    if poll_df.empty:
        st.info("No pollutant readings available for this site’s latest timestamp.")
    else:
        def _style_row(row):
            if pd.notna(row["Threshold"]) and row["Current"] > row["Threshold"]:
                return ["background-color: #fdecea"]*len(row)
            return [""]*len(row)
        st.dataframe(
            poll_df.style.apply(_style_row, axis=1).format({"Current":"{:.2f}", "Threshold":"{:.2f}"}),
            hide_index=True,
            use_container_width=True,
        )

st.caption(
    "This tool estimates the probability that air conditions could trigger asthma symptoms in the next ~6 hours, "
    "based on pollutant history at the selected monitoring site. "
    "“Cautious” raises alerts earlier; “Strict” raises alerts only at higher risk."
)
