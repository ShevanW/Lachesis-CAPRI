# app_city.py — Public dashboard for Melbourne asthma risk (with lat/lon fallback)
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt
import pydeck as pdk

ARTIFACTS = Path("artifacts")
DATA_PATH = Path("data")

FULL_CANDIDATES = [
    ARTIFACTS / "processed_multisite_feats_full.csv",
    ARTIFACTS / "processed_multisite_full.csv",
    ARTIFACTS / "processed_site_full.csv",
]

MODELS = {
    "XGBoost": ARTIFACTS / "aqi_asthma_xgb_model.joblib",
    "Random Forest": ARTIFACTS / "aqi_asthma_rf_model.joblib",
}

RAW_CSV = ARTIFACTS / "multisite_raw_ordered.csv"   # fallback for coords & pollutant panel
THRESH_JSON = ARTIFACTS / "labeling_info_multi_feats.json"
TEMPORAL_COLS = ["hour","dow","month","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"]
SENS_TO_THR = {"Cautious":0.25, "Balanced":0.40, "Strict":0.60}

# Advisory thresholds & display units (tune later as needed)
POLLUTANT_THRESH = {"PM25": 25, "PM10": 50, "NO2": 80, "O3": 80, "SO2": 100, "CO": 9}
POLLUTANT_UNITS  = {"PM25": "µg/m³", "PM10": "µg/m³", "NO2": "ppb", "O3": "ppb", "SO2": "ppb", "CO": "ppm"}
SNAPSHOT_COLS    = ["PM25", "PM10", "NO2", "O3", "SO2", "CO"]

# ---------- utils ----------
@st.cache_data
def load_df(path: Path, dt_index=True) -> pd.DataFrame:
    if dt_index:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "datetime_local"
    else:
        df = pd.read_csv(path)
        if "datetime_local" in df.columns:
            df["datetime_local"] = pd.to_datetime(df["datetime_local"])
    return df

@st.cache_data
def load_full_processed() -> pd.DataFrame:
    for p in FULL_CANDIDATES:
        if p.exists():
            return load_df(p, dt_index=True)
    st.error("No processed dataset found in artifacts/. Run make_dataset_multi_feats.py.")
    st.stop()

@st.cache_data
def load_thresholds() -> dict:
    for p in [THRESH_JSON, ARTIFACTS / "labeling_info_multi.json"]:
        if p.exists():
            with open(p, "r") as f:
                info = json.load(f)
            return info.get("thresholds_effective", {})
    return {}

def feature_cols(df: pd.DataFrame) -> list:
    lags = [c for c in df.columns if "_lag" in c]
    times = [c for c in TEMPORAL_COLS if c in df.columns]
    return lags + times

def ensure_site_sheet(df: pd.DataFrame) -> pd.DataFrame:
    if "site_sheet" not in df.columns:
        if {"location_name","location_id"}.issubset(df.columns):
            df = df.copy()
            df["site_sheet"] = df["location_name"].astype(str) + "_" + df["location_id"].astype(str)
        elif "location_name" in df.columns:
            df = df.copy()
            df["site_sheet"] = df["location_name"].astype(str)
        else:
            st.error("Dataset has no site identifier (site_sheet/location_name).")
            st.stop()
    return df

def load_model() -> tuple[str, object]:
    for name, p in MODELS.items():
        if p.exists():
            return name, joblib.load(p)
    st.error("No trained model found in artifacts/.")
    st.stop()

@st.cache_data
def load_suburbs_csv() -> pd.DataFrame | None:
    p = DATA_PATH / "suburbs_melbourne.csv"
    if not p.exists(): return None
    df = pd.read_csv(p)
    needed = {"suburb","latitude","longitude"}
    if not needed.issubset(df.columns):
        st.warning("suburbs_melbourne.csv needs columns: suburb, latitude, longitude")
        return None
    return df.dropna(subset=["latitude","longitude"]).copy()

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(p1)*np.cos(p2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def nearest_station(sub_lat, sub_lon, sites_df):
    d = haversine(sub_lat, sub_lon, sites_df["latitude"].values, sites_df["longitude"].values)
    idx = int(np.argmin(d))
    return sites_df.iloc[idx], float(d[idx])

def risk_label(prob: float, thr: float) -> tuple[str,str]:
    if prob >= thr:
        sev = "High" if prob >= max(0.9, thr + 0.3) else "Elevated"
        return f"⚠️ {sev} asthma risk (next ~6h)", "alert"
    return "✅ Low asthma risk (next ~6h)", "safe"

def forecast_chart(times: pd.Index, probs: np.ndarray, thr: float) -> alt.Chart:
    dfp = pd.DataFrame({"datetime_local": times, "risk_prob": probs})
    base = alt.Chart(dfp).encode(x="datetime_local:T")
    line = base.mark_line().encode(y=alt.Y("risk_prob:Q", title="Risk probability"))
    rule = alt.Chart(pd.DataFrame({"thr":[thr]})).mark_rule(strokeDash=[4,4]).encode(y="thr:Q")
    return alt.layer(line, rule).properties(height=220, title="Risk outlook (per timestamp → next ~6h)")

def pollutant_panel(latest_row: pd.Series, thresholds: dict) -> pd.DataFrame:
    order = ["PM25","PM10","NO2","O3","SO2","CO"]
    rows = []
    for p in order:
        if p in latest_row.index and pd.notna(latest_row[p]):
            val = float(latest_row[p])
            thr = thresholds.get(p, None)
            status = "Above" if (thr is not None and val > thr) else "Below"
            rows.append({"Pollutant": p, "Current": val, "Threshold": thr, "Status": status})
    return pd.DataFrame(rows)

# ---------- UI ----------
st.set_page_config(page_title="Melbourne Asthma Risk", layout="wide")
st.title("Melbourne Asthma Risk")

full_df = ensure_site_sheet(load_full_processed())
fcols = feature_cols(full_df)

# ---- site directory: processed -> raw -> stations_vic.csv ----
raw_df_for_meta = None
site_dir = None

if {"site_sheet","latitude","longitude"}.issubset(full_df.columns):
    site_dir = full_df[["site_sheet","latitude","longitude"]].drop_duplicates().dropna()
else:
    if RAW_CSV.exists():
        raw_df_for_meta = load_df(RAW_CSV, dt_index=False)
        raw_df_for_meta = ensure_site_sheet(raw_df_for_meta)
        if {"site_sheet","latitude","longitude"}.issubset(raw_df_for_meta.columns):
            site_dir = raw_df_for_meta[["site_sheet","latitude","longitude"]].drop_duplicates().dropna()

    if site_dir is None:
        stations_csv = DATA_PATH / "stations_vic.csv"
        if stations_csv.exists():
            stations = pd.read_csv(stations_csv)
            need = {"site_sheet","latitude","longitude"}
            if need.issubset(stations.columns):
                site_dir = stations[list(need)].drop_duplicates().dropna()

if site_dir is None or site_dir.empty:
    st.error("No latitude/longitude found in processed/raw/stations datasets. Rebuild dataset or create data/stations_vic.csv.")
    st.stop()


thresh = load_thresholds()
# also keep RAW (if present) for pollutant panel
if raw_df_for_meta is None and RAW_CSV.exists():
    raw_df_for_meta = load_df(RAW_CSV, dt_index=False)
    raw_df_for_meta = ensure_site_sheet(raw_df_for_meta)

with st.sidebar:
    st.header("Find your location")
    sub_df = load_suburbs_csv()
    mode = st.radio("Select by:", options=["Suburb","Monitoring site"], horizontal=True)
    if mode == "Suburb" and sub_df is not None:
        suburb = st.selectbox("Suburb", options=sorted(sub_df["suburb"].unique().tolist()))
        row = sub_df.loc[sub_df["suburb"] == suburb].iloc[0]
        user_lat, user_lon = float(row["latitude"]), float(row["longitude"])
        nearest, dist_km = nearest_station(user_lat, user_lon, site_dir)
        chosen_site = nearest["site_sheet"]
        st.caption(f"Nearest station to **{suburb}** → **{chosen_site}** ({dist_km:.1f} km)")
    else:
        chosen_site = st.selectbox("Monitoring site", options=sorted(site_dir["site_sheet"].tolist()))

    st.header("Settings")
    model_name, model = load_model()
    sens = st.radio("Alert sensitivity", options=list(SENS_TO_THR.keys()), horizontal=True, index=0)
    thr = float(SENS_TO_THR[sens])

# filter rows for site & predict
site_rows = full_df.loc[full_df["site_sheet"] == chosen_site].sort_index()
if site_rows.empty:
    st.warning("No rows available for this site in the processed dataset.")
    st.stop()

X_site = site_rows[fcols]
times_site = site_rows.index

if hasattr(model, "predict_proba"):
    y_prob = model.predict_proba(X_site)[:, 1]
else:
    st.error("Loaded model lacks predict_proba.")
    st.stop()

latest_prob = float(y_prob[-1])
latest_ts = times_site[-1]
headline, state = risk_label(latest_prob, thr)

# pollutant table (from raw fallback)
if raw_df_for_meta is not None and "site_sheet" in raw_df_for_meta.columns:
    raw_site = raw_df_for_meta[raw_df_for_meta["site_sheet"] == chosen_site].sort_values("datetime_local")
    latest_raw = raw_site.iloc[-1] if len(raw_site) else pd.Series(dtype=float)
    poll_df = pollutant_panel(latest_raw, thresh) if not latest_raw.empty else pd.DataFrame()
else:
    poll_df = pd.DataFrame()

# layout
left, right = st.columns([1.2, 1])
with left:
    st.subheader(chosen_site.replace("_", " "))
    color = "#e67e22" if state == "alert" else "#2ecc71"
    st.markdown(
        f"""
        <div style="padding:18px;border-radius:12px;background:{color}22;border:1px solid {color};">
            <div style="font-size:22px;font-weight:600;color:{color};">{headline}</div>
            <div style="font-size:14px;margin-top:6px;">Sensitivity: <b>{sens}</b> (threshold {thr:.2f}) · Model: <b>{model_name}</b></div>
            <div style="font-size:14px;margin-top:2px;">Latest reading time: {latest_ts}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    N = min(24, len(y_prob))
    st.altair_chart(forecast_chart(times_site[-N:], y_prob[-N:], thr), use_container_width=True)

with right:
    st.subheader("What’s high right now?")
    if poll_df.empty:
        st.info("No pollutant table available for the latest timestamp at this site.")
    else:
        def sty(r):
            if pd.notna(r["Threshold"]) and r["Current"] > r["Threshold"]:
                return ["background-color: #fdecea"]*len(r)
            return [""]*len(r)
        st.dataframe(poll_df.style.apply(sty, axis=1).format({"Current":"{:.2f}","Threshold":"{:.2f}"}),
                     hide_index=True, use_container_width=True)

# map
st.subheader("Map")
latest_by_site = site_dir.copy()
# set neutral risk probs; highlight current site with latest prob
latest_by_site["risk_prob"] = 0.3
if chosen_site in latest_by_site["site_sheet"].values:
    latest_by_site.loc[latest_by_site["site_sheet"] == chosen_site, "risk_prob"] = latest_prob

def risk_color(p):
    if p >= max(0.9, thr + 0.3): return [200,60,60]
    if p >= thr:                 return [230,160,60]
    return [60,180,120]
latest_by_site["color"] = latest_by_site["risk_prob"].apply(risk_color)

circles = pdk.Layer(
    "ScatterplotLayer",
    data=latest_by_site,
    get_position='[longitude, latitude]',
    get_radius=800,
    get_fill_color='color',
    pickable=True,
    opacity=0.6,
)
heat = pdk.Layer(
    "HeatmapLayer",
    data=latest_by_site,
    get_position='[longitude, latitude]',
    get_weight='risk_prob',
    radius_pixels=60,
)
view_state = pdk.ViewState(latitude=-37.8136, longitude=144.9631, zoom=9)
st.pydeck_chart(pdk.Deck(layers=[heat, circles], initial_view_state=view_state,
                         tooltip={"text":"{site_sheet}\nRisk prob: {risk_prob}"}))
st.caption("Estimates the probability that air conditions could trigger asthma symptoms in ~6 hours at the nearest monitoring site.")
