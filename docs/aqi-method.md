# AQI method (project standard)

We compute per-pollutant AQI using the US EPA method and select **max(AQI_pollutant)** as the overall AQI for each record.

## Pollutants considered
- PM2.5, PM10, O3, NO2, SO2, CO (as available)

## Equation (per pollutant)
Given concentration `C`, find the breakpoint interval `[C_low, C_high]` with AQI breakpoints `[I_low, I_high]` and compute:

```
AQI = (I_high - I_low) / (C_high - C_low) * (C - C_low) + I_low
```

Apply pollutant-specific truncation/rounding rules per EPA guidance. For overall AQI, take the maximum among pollutant AQIs.

## Notes
- Ensure units match the breakpoint tables (e.g., µg/m³, ppm).
- EPA updated PM2.5 breakpoints in 2024; use the latest tables.
- Document any local conventions or deviations.

## References
- US EPA / AirNow: Technical guidance for Daily AQI reporting (latest).
- EPA: “How is the AQI calculated?” with equations and breakpoints.
