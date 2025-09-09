# app_streamlit.py
# Streamlit UI for AQI -> Asthma Risk (6h ahead)
# - Model picker: Random Forest vs XGBoost
# - Threshold slider: tune recall vs precision
# - Timeline plot: predicted probability vs actual future risk
# - Latest prediction card

from pathlib import Path
import json

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import altair as alt

# -----------------
# Paths & constants
# -----------------
ARTIFACTS = Path("artifacts")
MODELS = {
    "Random Forest": ARTIFACTS / "aqi_asthma_rf_model.joblib",
    "XGBoost":       ARTIFACTS / "aqi_asthma_xgb_model.joblib",
}
TRAIN_CSV = ARTIFACTS / "train.csv"
TEST_CSV  = ARTIFACTS / "test.csv"

TEMPORAL_COLS = [
    "hour","dow","month","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"
]

# -------------
# Helper funcs
# -------------
@st.cache_data
def load_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}. Build it with a dataset script first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "datetime_local"
    return df

def get_feature_cols(df: pd.DataFrame) -> list:
    # Same selection used in trainers: all lagged + any temporal present
    lag_cols = [c for c in df.columns if "_lag" in c]
    time_cols = [c for c in TEMPORAL_COLS if c in df.columns]
    return lag_cols + time_cols

def load_model(model_name: str):
    path = MODELS[model_name]
    if not path.exists():
        st.error(f"Model file not found: {path}")
        st.stop()
    return joblib.load(path)

def compute_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    acc = (tp + tn) / max(1, len(y_true))
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": acc
    }

def timeline_chart(test_index: pd.Index, y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> alt.Chart:
    dfp = pd.DataFrame({
        "datetime_local": test_index,
        "risk_true": y_true.astype(int),
        "risk_prob": y_prob.astype(float)
    })
    base = alt.Chart(dfp).encode(x="datetime_local:T")

    prob_line = base.mark_line().encode(
        y=alt.Y("risk_prob:Q", title="Predicted probability")
    )

    # Actual future risk as a stepped line on 2nd axis (0/1)
    true_line = base.mark_line(color="#888", interpolate="step-after").encode(
        y=alt.Y("risk_true:Q", axis=alt.Axis(title="Actual risk (0/1)"), scale=alt.Scale(domain=[-0.05, 1.05]))
    )

    thr_rule = alt.Chart(pd.DataFrame({"thr": [thr]})).mark_rule(strokeDash=[4,4]).encode(
        y=alt.Y("thr:Q")
    )

    return alt.layer(prob_line, thr_rule, true_line).resolve_scale(y="independent").properties(
        width="container", height=260, title="Risk probability vs. actual future risk"
    )

# ----------
# UI Layout
# ----------
st.set_page_config(page_title="AQI → Asthma Risk (6h ahead)", layout="wide")
st.title("AQI → Asthma Risk (6h ahead)")

# Sidebar controls
st.sidebar.header("Model & Threshold")

# Pick model
available_models = [name for name, p in MODELS.items() if p.exists()]
if not available_models:
    st.error("No saved models found in 'artifacts/'. Train a model first.")
    st.stop()

default_model = "XGBoost" if "XGBoost" in available_models else available_models[0]
model_name = st.sidebar.selectbox("Choose model", options=available_models,
                                  index=available_models.index(default_model))

# Threshold slider
default_thr = 0.25 if model_name == "XGBoost" else 0.50
threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.05, max_value=0.95, value=float(default_thr), step=0.01,
    help="Lower = higher recall (more alerts). Higher = higher precision (fewer alerts)."
)

# Load data + model
test_df = load_frame(TEST_CSV)
train_df = load_frame(TRAIN_CSV)  # kept for future use if needed
feature_cols = get_feature_cols(test_df)
target_col = "risk_label_future"
if target_col not in test_df.columns:
    st.error(f"Column '{target_col}' missing in {TEST_CSV}. Rebuild dataset.")
    st.stop()

X_test = test_df[feature_cols]
y_test = test_df[target_col].astype(int).values
test_index = test_df.index

model = load_model(model_name)

# Inference on test set (for timeline + metrics)
with st.spinner("Scoring test set…"):
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        # Safety fallback: some models only output decision_function
        if hasattr(model, "decision_function"):
            raw = model.decision_function(X_test)
            # min-max to [0,1] for display only
            y_prob = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        else:
            st.error("Loaded model does not support predict_proba or decision_function.")
            st.stop()

metrics = compute_metrics_at_threshold(y_test, y_prob, threshold)

# -------------
# Top summary
# -------------
left, mid, right = st.columns([1,1,2])

with left:
    st.subheader("Selected model")
    st.write(f"**{model_name}**")
    st.metric("Threshold", f"{threshold:.2f}")

with mid:
    st.subheader("Latest prediction")
    # Use the last available feature row as “latest”
    latest_x = X_test.iloc[[-1]]
    latest_ts = test_index[-1]
    if hasattr(model, "predict_proba"):
        latest_prob = model.predict_proba(latest_x)[:, 1][0].item()
    else:
        latest_prob = float(y_prob[-1])
    latest_label = int(latest_prob >= threshold)
    status_str = "⚠️ Risk in next 6h" if latest_label == 1 else "✅ Safe"
    st.metric("Status", status_str, f"{latest_prob:.2f}")

with right:
    st.subheader("Test-set metrics @ threshold")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision (1)", f"{metrics['precision']:.2f}")
    c2.metric("Recall (1)", f"{metrics['recall']:.2f}")
    c3.metric("F1 (1)", f"{metrics['f1']:.2f}")
    c4.metric("Accuracy", f"{metrics['accuracy']:.2f}")

    st.caption(
        "Precision = how many predicted alerts were truly risky. "
        "Recall = how many truly risky hours we caught. "
        "Tune the threshold to trade precision vs recall."
    )

# -------------
# Timeline plot
# -------------
st.altair_chart(timeline_chart(test_index, y_test, y_prob, threshold), use_container_width=True)

# -------------
# Confusion matrix table
# -------------
st.subheader("Confusion Matrix (test set)")
cm_df = pd.DataFrame(
    [[metrics["tn"], metrics["fp"]],
     [metrics["fn"], metrics["tp"]]],
    columns=["Pred 0", "Pred 1"],
    index=["Actual 0", "Actual 1"],
)
st.table(cm_df)

# -------------
# Feature importances (if available)
# -------------
st.subheader("Top feature importances")
# Try to read the PNG produced by the corresponding trainer (RF or XGB)
fi_png = ARTIFACTS / ("feature_importances_xgb.png" if model_name == "XGBoost" else "feature_importances.png")
if fi_png.exists():
    st.image(str(fi_png))
else:
    st.info("Feature importances plot not found yet. Train the selected model to generate it.")

# -------------
# Evaluation summary (if present)
# -------------
summary_path = ARTIFACTS / ("eval_summary_xgb.json" if model_name == "XGBoost" else "eval_summary.json")
if summary_path.exists():
    with open(summary_path, "r") as f:
        summary = json.load(f)
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC AUC", f"{summary.get('roc_auc', np.nan):.3f}")
    col2.metric("PR AUC", f"{summary.get('pr_auc', np.nan):.3f}")
    cm = summary.get("confusion_matrix", None)
    if cm:
        st.caption(f"Trainer-reported CM: {cm}")
else:
    st.info("Evaluation summary JSON not found. Train the selected model to generate it.")
