# Data & ML pipeline

## 1) Ingest
- EPA Vic Air Watch yearly hourly averages (XLSX/CSV)
- AIHW respiratory data (CSV/Excel) – scoped to matching time/geography
- Optional: WAQI/IQAir APIs for real-time checks

## 2) Clean
- Parse timestamps and set timezone
- Coerce pollutants to numeric, standardise units
- Handle missing values (median/ffill as appropriate); log missingness
- Drop duplicates and invalid rows; persist a clean snapshot

## 3) Align
- Temporal: resample to desired window (e.g., 8-hour or daily)
- Spatial: map site → LGA/SA2 (record shapefile/version used)
- Lag tests: explore pollutant → health lags (e.g., 0–7 days)

## 4) AQI compute
- Compute pollutant AQIs and **overall AQI = max(pollutant AQIs)** per row

## 5) EDA
- Distributions, seasonal patterns, correlations, lag plots

## 6) Model
- Baselines: Linear Regression
- Ensembles: RandomForestRegressor, GradientBoostingRegressor
- Validation: rolling-origin CV; metrics: MAE/RMSE/R²

## 7) Visualise
- Trends, spatial heat maps, hotspots, model diagnostics

## 8) Deliver
- Export tables/figures; compile report sections; prepare slides
