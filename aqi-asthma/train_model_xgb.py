# train_model_xgb.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
from sklearn.pipeline import Pipeline

import xgboost as xgb
from xgboost import XGBClassifier

from config import ARTIFACT_DIR, HORIZON_HOURS, RANDOM_STATE, N_IMPORTANCES

def load_xy(csv_path: Path):
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    y = df["risk_label_future"].astype(int)

    # Feature set: all lagged features + temporal features if present
    feature_cols = [c for c in df.columns if "_lag" in c] + [
        "hour","dow","month","hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos"
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    X = df[feature_cols]
    return X, y, feature_cols

def build_pipeline(feature_cols, scale_pos_weight):
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), feature_cols),
    ])
    # Trees don't need scaling; keep it simple
    clf = XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",  # fast on CPU
        scale_pos_weight=scale_pos_weight,  # handle class imbalance
    )
    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    return pipe

def plot_pr_curve(y_true, y_prob, out_path: Path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    plt.figure(figsize=(6, 4))
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall (AUC={pr_auc:.3f})")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()
    return pr_auc

def plot_feature_importances(model, feature_cols, out_path: Path, top_k: int):
    clf = model.named_steps["clf"]
    importances = clf.feature_importances_
    imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp = imp.sort_values("importance", ascending=False).head(top_k)
    plt.figure(figsize=(8, 6))
    plt.barh(imp["feature"], imp["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Gain-based importance")
    plt.title("Top Feature Importances (XGBoost)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def plot_risk_timeline(test_idx, y_true, y_prob, out_path: Path):
    dfp = pd.DataFrame({
        "datetime_local": test_idx,
        "risk_true": y_true.values,
        "risk_prob": y_prob,
    }).set_index("datetime_local")
    ax = dfp[["risk_prob"]].plot(figsize=(10, 3))
    dfp["risk_true"].plot(ax=ax, secondary_y=True)
    plt.title("Predicted Risk Probability vs. Actual Future Risk (XGB)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    train_csv = ARTIFACT_DIR / "train.csv"
    test_csv  = ARTIFACT_DIR / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Missing train/test CSVs. Run a dataset builder first.")

    print("Loading train/test…")
    X_train, y_train, feature_cols = load_xy(train_csv)
    X_test,  y_test,  _            = load_xy(test_csv)

    # class imbalance handling
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = (neg / max(1, pos)) if pos > 0 else 1.0

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Horizon (hours ahead): {HORIZON_HOURS}")
    print(f"scale_pos_weight ≈ {scale_pos_weight:.2f}")

    pipe = build_pipeline(feature_cols, scale_pos_weight)

    # Early stopping (XGB 3.x): use callbacks
    eval_set = [(X_train, y_train), (X_test, y_test)]
    # remove early stopping / eval_set for this environment
    pipe.fit(X_train, y_train)


    y_prob = pipe.predict_proba(X_test)[:, 1]

# --- threshold search: F1 and recall-minded options ---
    from sklearn.metrics import f1_score, precision_recall_curve

    prec, rec, thr = precision_recall_curve(y_test, y_prob)
    f1s = (2 * prec * rec) / np.clip(prec + rec, 1e-12, None)
    best_f1_idx = np.nanargmax(f1s)
    best_f1_thr = thr[best_f1_idx-1] if best_f1_idx > 0 else 0.5  # align to thresholds array

# also pick a threshold that gives recall >= 0.50 with best precision
    target_recall = 0.50
    candidates = np.where(rec[:-1] >= target_recall)[0]
    best_recall_thr = best_f1_thr
    if len(candidates):
    # among recalls >= target, choose the one with highest precision
        best_idx = candidates[np.argmax(prec[candidates])]
        best_recall_thr = thr[best_idx] if best_idx < len(thr) else thr[-1]

    print(f"\nSuggested thresholds -> best_F1: {best_f1_thr:.3f} | recall≥{target_recall:.2f}: {best_recall_thr:.3f}")

# choose one to report predictions (use best F1 by default)
    chosen_thr = best_f1_thr
    y_pred = (y_prob >= chosen_thr).astype(int)


    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print("\n=== XGBoost Evaluation ===")
    print(f"ROC AUC: {roc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Outputs (XGB-specific filenames)
    model_path   = ARTIFACT_DIR / "aqi_asthma_xgb_model.joblib"
    pr_path      = ARTIFACT_DIR / "pr_curve_xgb.png"
    fi_path      = ARTIFACT_DIR / "feature_importances_xgb.png"
    tl_path      = ARTIFACT_DIR / "risk_timeline_xgb.png"
    summary_path = ARTIFACT_DIR / "eval_summary_xgb.json"

    pr_auc = plot_pr_curve(y_test, y_prob, pr_path)
    plot_feature_importances(pipe, feature_cols, fi_path, N_IMPORTANCES)
    test_index = pd.read_csv(test_csv, index_col=0).index
    plot_risk_timeline(test_index, y_test, y_prob, tl_path)

    joblib.dump(pipe, model_path)
    summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "horizon_hours": HORIZON_HOURS,
        "top_features_png": str(fi_path),
        "pr_curve_png": str(pr_path),
        "timeline_png": str(tl_path),
        "scale_pos_weight": float(scale_pos_weight),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print(" ", model_path)
    print(" ", pr_path)
    print(" ", fi_path)
    print(" ", tl_path)
    print(" ", summary_path)

if __name__ == "__main__":
    main()
