# app_streamlit.py
# Run with:
#   streamlit run app_streamlit.py
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from config import ARTIFACT_DIR, APP_SITE_TITLE, HORIZON_HOURS, LAGS

st.set_page_config(page_title="AQI → Asthma Risk", layout="wide")
st.title(APP_SITE_TITLE)

# --- Paths ---
model_path = ARTIFACT_DIR / "aqi_asthma_rf_model.joblib"
train_csv  = ARTIFACT_DIR / "train.csv"                  # to discover feature columns
raw_csv    = ARTIFACT_DIR / "site_raw_ordered.csv"       # raw ordered site data we saved in make_dataset.py
summary_js = ARTIFACT_DIR / "labeling_info.json"         # thresholds & info

# --- Load artifacts ---
if not model_path.exists() or not train_csv.exists() or not raw_csv.exists():
    st.error("Missing artifacts. Please run make_dataset.py and train_model.py first.")
    st.stop()

pipe = joblib.load(model_path)
train_df = pd.read_csv(train_csv, index_col=0)
raw_df   = pd.read_csv(raw_csv)

# Ensure datetime is parsed and sorted
if "datetime_local" in raw_df.columns:
    raw_df["datetime_local"] = pd.to_datetime(raw_df["datetime_local"], errors="coerce")
    raw_df = raw_df.sort_values("datetime_local")

# Figure out which lagged features were used in training
feature_cols = [c for c in train_df.columns if "_lag" in c]
base_feats = sorted({c.split("_lag")[0] for c in feature_cols})

st.sidebar.header("Settings")
st.sidebar.write(f"Prediction horizon: **{HORIZON_HOURS} hours**")
st.sidebar.write(f"Lags used: **{LAGS} hours**")
thresh = st.sidebar.slider("Alert threshold (probability)", 0.05, 0.95, 0.5, 0.05)

# Rename SBPM25 -> PM25 if present (consistency with training)
if "SBPM25" in raw_df.columns:
    raw_df = raw_df.rename(columns={"SBPM25": "PM25"})

# Build the required lag features on the raw_df
df = raw_df.copy()
for base in base_feats:
    if base in df.columns:
        for L in LAGS:
            df[f"{base}_lag{L}"] = df[base].shift(L)

# Drop rows that don't have all lag features
df_model = df.dropna(subset=feature_cols).copy()

if len(df_model) == 0:
    st.warning("Not enough rows with lag features to make predictions yet.")
    st.stop()

# Predict probabilities
probs = pipe.predict_proba(df_model[feature_cols])[:, 1]
df_model["risk_prob"] = probs
df_model = df_model.set_index("datetime_local")

# --- Charts ---
st.subheader("Predicted risk probability over time")
st.line_chart(df_model[["risk_prob"]])

# Latest snapshot
st.subheader("Latest prediction")
latest = df_model.iloc[-1]
st.metric(f"Predicted risk (next {HORIZON_HOURS}h)", f"{latest['risk_prob']:.2%}")

# Show last 48 hours table
st.caption("Last 48 rows with predictions")
st.dataframe(df_model[["risk_prob"]].tail(48))

# Simple alert band
n_alerts = int((df_model["risk_prob"] >= thresh).sum())
st.sidebar.success(f"Alerts (prob ≥ {thresh:.2f}): {n_alerts}")

# Optional: show thresholds info used to create labels
if summary_js.exists():
    with open(summary_js) as f:
        info = json.load(f)
    with st.expander("Labeling info (from make_dataset.py)"):
        st.json(info)
