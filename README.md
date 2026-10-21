# SpaceX Falcon 9 First-Stage Landing Prediction
**IBM Applied Data Science Capstone — project bundle**

Repository: https://github.com/premanand404/spacex-data-science-capstone

## Important
This is a reproducible project scaffold and analysis code. Run the notebooks/scripts to generate metrics and plots from the official course datasets before treating the report as a final results report. Do not claim model scores or findings that you have not run and verified.

## Contents
- `notebooks/01_eda_and_sql.ipynb` — data exploration and SQL queries
- `notebooks/02_predictive_analysis.ipynb` — four classification models, metrics and confusion matrix
- `src/analyze_spacex.py` — end-to-end analysis and chart generation
- `src/dashboard.py` — Plotly Dash interactive dashboard
- `src/folium_map.py` — interactive launch-site map
- `requirements.txt`
- `Data Science Capstone Project Report.pdf` — report framework with methods and repository URL; refresh results after running analysis

## Quick start
1. Install Python 3.10+.
2. `python -m pip install -r requirements.txt`
3. Run `python src/analyze_spacex.py` (downloads official IBM course data if missing, creates charts and model metrics in `outputs/`).
4. Run `python src/folium_map.py` to create `outputs/launch_map.html`.
5. Run `python src/dashboard.py` to start the dashboard at the local URL shown in terminal.
6. Open the notebooks and run all cells.
7. Upload all files to your GitHub repository, then regenerate the PDF with actual metrics/charts from `outputs/`.

## Research question
Can launch and mission features help predict whether the Falcon 9 first stage will land successfully?

## Data provenance
The script loads the IBM Skills Network course dataset from the official course URL. The prepared dataset is a teaching dataset and does not necessarily represent all SpaceX launches.
