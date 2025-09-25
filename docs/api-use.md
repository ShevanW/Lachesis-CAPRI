# API use (WAQI/IQAir)

## WAQI (aqicn)
- Obtain a token and store it in the environment variable `WAQI_TOKEN`.
- Respect rate limits; cache responses and log calls.
- Record endpoint versions and schema assumptions.

## IQAir (AirVisual)
- Similar token-based access; document endpoints and attribution.

## Security & attribution
- Never commit secrets; use `.env` or OS key stores.
- Follow provider attribution requirements in UIs/reports.
