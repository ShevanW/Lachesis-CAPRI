# AQI → Asthma Risk Prototype  
**Model Summary (current Streamlit run)**

## Task
Predict the likelihood of high asthma-risk air conditions **6 hours ahead** at a monitoring site.

- **Target (label):**  
  `risk_label_future` = 1 if **any pollutant exceeds its 90th percentile threshold** in the *future horizon* (6 hours ahead).  
  Otherwise, 0.

- **Features:**  
  Pollutant & meteorology readings (PM2.5, PM10, NO₂, O₃, SO₂, CO, DBT, BSP, SWD, SWS, VWD, VWS, Sigma60).  
  Each is represented with lagged values at **1, 2, 3, 6 hours**.

- **Train/Test split:**  
  Chronological 80% train / 20% test.

## Model
- **Type:** Random Forest Classifier  
- **Library:** scikit-learn (`sklearn.ensemble.RandomForestClassifier`)  
- **Pipeline:**
  - `SimpleImputer(strategy="median")` – fills missing values
  - `StandardScaler(with_mean=False)` – scales numeric features
  - `RandomForestClassifier`  
    - `n_estimators=300`  
    - `random_state=42`  
    - `class_weight="balanced_subsample"`  
    - `n_jobs=-1`

## Evaluation (Alphington site, percentile thresholds)
- **ROC AUC:** ~0.688  
- **Precision–Recall AUC:** ~0.57  
- **Confusion Matrix (test set):**
[[690 48]
[315 103]]

- **TN (True Negatives):** 690 → correctly predicted *safe* (0)  
- **FP (False Positives):** 48 → predicted *risk* when it was actually *safe*  
- **FN (False Negatives):** 315 → predicted *safe* when it was actually *risk*  
- **TP (True Positives):** 103 → correctly predicted *risk* (1)  

- **Interpretation:**  
The model is strong at identifying safe conditions but misses many risk events.  
This is partly due to class imbalance (only ~36% of samples were “risk”).  