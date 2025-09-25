# Contributing

## Workflow
1. Create a feature branch from `main` (e.g., `feat/etl-aqi`).
2. Commit with clear messages; link issues.
3. Open a PR with:
   - What changed and why
   - Data/schema impacts
   - Screenshots for visuals (if applicable)
4. One reviewer approval required.

## Code style
- Python: format with black/ruff (or flake8); add type hints where feasible.
- Keep notebooks light; move logic to `src/` or `utils/` modules.

## Data hygiene
- Do not commit raw data; use `/data/` with `.gitignore` and per-dataset README.
- Do not commit secrets (API keys, tokens); use environment variables.
