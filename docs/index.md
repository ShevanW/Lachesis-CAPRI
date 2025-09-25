# Lachesis-CAPRI Documentation

Exploring the relationship between air quality and respiratory-related hospital admissions in Australia.

**Data**: EPA Victoria Air Watch (hourly site data), AIHW respiratory statistics, optional WAQI/IQAir APIs  
**Tech stack**: Python (pandas, numpy, scikit-learn), (optional) GeoPandas/Folium for maps, matplotlib  
**Key challenge**: Merging spatial & temporal datasets and controlling for confounders

---

## Quick start

1. Clone the repository and create a Python virtual environment.
2. Install project requirements (`pip install -r requirements.txt`).
3. (Optional) Set a WAQI API token via environment variable `WAQI_TOKEN` if using real‑time ingestion.

See:
- `docs/pipeline.md` for the end-to-end data & modelling pipeline
- `docs/aqi-method.md` for AQI computation details
- `docs/modelling.md` for model choices and evaluation
- `docs/visualisation.md` for plot guidelines
- `docs/api-use.md` for API usage notes
- `docs/data-sources.md` for source links and attributions
- `docs/governance-ethics.md` for governance & ethics

---

## Project structure (suggested)

```
.
├── apps/
│   └── lachesis_app_plus/          # (prototype UI/app if used)
├── aqi-asthma/                     # notebooks and analysis scripts
├── src/                            # reusable Python modules (optional)
├── data/                           # data folder (gitignored)
├── docs/                           # documentation (this folder)
├── AQICode.py                      # AQI helper(s)
├── LICENSE
└── README.md
```

## Reproducibility

- Use feature branches and pull requests (see `docs/CONTRIBUTING.md`).
- Keep heavy logic in `src/` modules; keep notebooks lightweight.
- Record key decisions in a short decision log (`docs/decisions.md`, optional).

## License

MIT (see `LICENSE` in the repository root).
