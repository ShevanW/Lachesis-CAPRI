# Lachesis-CAPRI

> Exploring the relationship between air quality and respiratory-related hospital admissions in Australia.

**Data**: EPA Victoria Air Watch (hourly site data), AIHW respiratory statistics, optional WAQI/IQAir APIs  
**Tech stack**: Python (pandas, numpy, scikit-learn), (optional) GeoPandas/Folium for maps, matplotlib  
**Key challenge**: Merging spatial & temporal datasets and controlling for confounders

---

## Quick start

```bash
# 1) Clone
git clone https://github.com/ShevanW/Lachesis-CAPRI.git
cd Lachesis-CAPRI

# 2) Create env (pick one)
python -m venv .venv && source .venv/bin/activate        # macOS/Linux
# or
py -m venv .venv && .\.venv\Scripts\activate              # Windows

# 3) Install (edit requirements.txt as needed)
pip install -r requirements.txt
