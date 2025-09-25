# Modelling

## Targets
- Primary: Overall AQI (max pollutant AQI)
- Optional: Next-day AQI (forecasting); pollutant-level AQIs

## Features
- Recent lags (t-1, t-2, …), rolling means
- Calendar (day-of-week, season, holiday flags)
- Weather/confounders (if available)
- Site/LGA encodings for multi-site data

## Models
- Linear Regression (baseline)
- Random Forest Regressor
- Gradient Boosting Regressor

## Validation & metrics
- Time-series aware split (rolling origin)
- Report MAE, RMSE, R²; include residual plots

## Reporting
- Single comparison table (models × metrics)
- Optional interpretability (permutation importance/SHAP)
