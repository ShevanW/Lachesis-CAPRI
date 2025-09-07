# train_model.py
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    auc,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

from config import ARTIFACT_DIR, N_IMPORTANCES, RANDOM_STATE, HORIZON_HOURS

def load_xy(csv_path: Path):
    """Load features/target from a processed csv with datetime index."""
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
    y = df["risk_label_future"].astype(int)
    # All columns that are lag features become X
    feature_cols = [c for c in df.columns if "_lag" in c]
    X = df[feature_cols]
    return X, y, feature_cols

def build_pipeline(feature_cols):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, feature_cols),
    ])
    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        n_jobs=-1,
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
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
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
    plt.title("Predicted Risk Probability vs. Actual Future Risk")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def main():
    train_csv = ARTIFACT_DIR / "train.csv"
    test_csv  = ARTIFACT_DIR / "test.csv"

    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError("Missing train/test CSVs. Run make_dataset.py first.")

    print("Loading train/test…")
    X_train, y_train, feature_cols = load_xy(train_csv)
    X_test,  y_test,  _            = load_xy(test_csv)

    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print(f"Horizon (hours ahead): {HORIZON_HOURS}")

    pipe = build_pipeline(feature_cols)
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print("\n=== Evaluation ===")
    print(f"ROC AUC: {roc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Outputs
    model_path = ARTIFACT_DIR / "aqi_asthma_rf_model.joblib"
    pr_path = ARTIFACT_DIR / "pr_curve.png"
    fi_path = ARTIFACT_DIR / "feature_importances.png"
    tl_path = ARTIFACT_DIR / "risk_timeline.png"
    summary_path = ARTIFACT_DIR / "eval_summary.json"

    pr_auc = plot_pr_curve(y_test, y_prob, pr_path)
    plot_feature_importances(pipe, feature_cols, fi_path, N_IMPORTANCES)
    # keep datetime index from test for plotting
    test_index = pd.read_csv(test_csv, index_col=0).index
    plot_risk_timeline(test_index, y_test, y_prob, tl_path)

    # Save model + summary
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
