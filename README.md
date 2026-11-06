# Road Safety ML Greece

A machine learning pipeline and interactive decision-support dashboard for analyzing road traffic collision severity (**KSI — Killed or Seriously Injured**) across the 13 NUTS-2 regions of Greece.

The project is built on official **Hellenic Statistical Authority (ELSTAT, Survey SDT04)** road accident tables (`2015–2024` national series and `2021–2025` regional monthly panel), complemented by field-observed road safety indicators from **NTUA (ΕΜΠ)**, **Copernicus ERA5**, and **EKAB** for policy simulation.

---

## Key Motivation: The Greek Regional Paradox

Official ELSTAT statistics highlight a sharp geographic split in how road trauma occurs in Greece:
- **Attica** accounts for more than half (**53.5%**) of all injury collisions nationally because of dense urban traffic, yet its fatality and severe injury rate per collision (**~6.5%–7.0% KSI**) is relatively low due to lower urban speeds and close proximity to major trauma hospitals.
- **Island and rural regions** (*South Aegean, Crete, Peloponnese, Central Greece, Ionian Islands*) record fewer total crashes, but **18% to 25%** of those crashes result in death or severe injury—driven by high-speed undivided rural roads, seasonal tourism inflows, lower helmet compliance, and longer ambulance response times.

This project models monthly regional KSI severity (`target_high_ksi = 1` when `ksi_severity_rate >= 0.21`) and translates model forecasts into actionable regional interventions.

---

## What's Inside

1. **Leak-Free Time-Series ML Pipeline (`src/road_safety/`)**
   - Chronological split: `2021–2023` (Train, $N=468$), `2024` (Validation / Threshold calibration, $N=156$), and `2025` (Held-out Test, $N=156$).
   - Programmatic anti-leakage checks (`data_validation.py`) ensure contemporaneous outcome variables never enter the feature matrix `X`.
   - Benchmarks 5 classifiers with 95% bootstrap confidence intervals, feature-group ablation studies, permutation importance, and regional calibration audits.

2. **2026 Regional Early-Warning Forecasts (`forecast.py`)**
   - Generates 12-month prospective predictions (`13 NUTS-2 regions × 12 months`) for KSI severity rates and severe-regime probabilities, flagging peak summer island surges and high-severity interurban corridors.

3. **Counterfactual Policy Simulator & 10-Year Foresight (`simulator.py`, `foresight.py`)**
   - Evaluates 5 Safe System intervention axes (average-speed cameras, helmet/sobriety checks, 2+1 median barriers, urban 30 km/h zones, and mobile EMS units) using empirical EU *SafetyCube* / ERSO Crash Modification Factors (CMFs).
   - Projects regional trajectories over a 10-year horizon (`2026–2035`) across 4 scenarios to track when each region reaches the EU `-50% KSI` target.

4. **Human-in-the-Loop Decision Copilot (`decision_support.py` & `app/streamlit_app.py`)**
   - Combines ML predictions with regional field KPIs under user-defined budget and reaction-time constraints to propose three ranked intervention packages (*Max Life-Saving*, *Best ROI*, and *24–72h Tactical Deployment*), generating a step-by-step execution plan and a downloadable decision protocol.

---

## Main Results on the 2025 Held-Out Test Set

### 1. Model Benchmark (`2025` Test Set, $N = 156$)

| Model | ROC-AUC (95% CI) | PR-AUC | F1-Score | Precision | Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `DummyClassifier (Baseline)` | `0.5000 [0.5000, 0.5000]` | `0.4679` | `0.0000` | `0.0000` | `0.0000` | `0.2496` |
| `DecisionTree` | `0.7958 [0.7209, 0.8642]` | `0.7522` | `0.7579` | `0.6154` | `0.9863` | `0.1903` |
| `RandomForest` | **`0.8567 [0.7973, 0.9105]`** | **`0.8374`** | `0.7594` | `0.6228` | `0.9726` | **`0.1552`** |
| `HistGradientBoosting` | `0.8199 [0.7509, 0.8808]` | `0.8036` | `0.7543` | **`0.6471`** | `0.9041` | `0.1919` |
| `XGBoost` | `0.8498 [0.7835, 0.9043]` | `0.8259` | **`0.7609`** | `0.6306` | `0.9589` | `0.1597` |

### 2. Feature Group Ablation Study (`2025` Test Set)

| Feature Subset | Features | ROC-AUC (95% CI) | PR-AUC | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `1. Calendar & Seasonality Only` | 4 | `0.5805 [0.4954, 0.6586]` | `0.5658` | `0.6376` |
| `2. Calendar + NUTS-2 Region` | 5 | **`0.8404 [0.7717, 0.8971]`** | **`0.8298`** | **`0.7735`** |
| `3. Calendar + Structural Exposure (Urban & Moto Share)` | 6 | `0.8320 [0.7538, 0.8907]` | `0.8141` | `0.7660` |
| `4. Full Baseline (+ Autoregressive Lags t-1, t-2, t-3, t-12)` | 14 | `0.8199 [0.7509, 0.8808]` | `0.8036` | `0.7543` |

### 3. Counterfactual Policy Simulation (`2026` National Totals Across All 13 Regions)

| Policy Scenario | Baseline KSI | Simulated KSI | KSI Prevented | Reduction (%) | Estimated Benefit (€M) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `0. Baseline (Business as Usual 2026)` | `1,354` | `1,354` | `0` | `0.00%` | `€0.00M` |
| `1. Urban VRU & 30 km/h Shield` | `1,354` | `999` | `355` | `26.22%` | `€408.25M` |
| `2. Summer Island & Two-Wheeler Blitz` | `1,354` | `987` | `367` | `27.10%` | `€422.05M` |
| `3. Interurban Highway & 2+1 Barrier Upgrade` | `1,354` | `969` | `385` | `28.43%` | `€442.75M` |
| `4. Vision Zero Greece 2030 (All 5 Axes)` | `1,354` | **`788`** | **`566`** | **`41.80%`** | **`€650.90M`** |

---

## Repository Structure

```text
road-safety-ml-greece/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── run.py
├── configs/
│   └── baseline.yaml
├── data/
│   ├── README.md
│   ├── elstat_national_annual_2015_2024.csv
│   ├── elstat_regional_monthly_panel_2021_2025.csv
│   ├── independent_road_safety_kpis_greece.csv
│   ├── regional_climate_telematics_kpis_nuts2.csv
│   └── operational_interventions_catalog.csv
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/
│   └── road_safety/
│       ├── __init__.py
│       ├── data_validation.py
│       ├── preprocessing.py
│       ├── features.py
│       ├── train.py
│       ├── evaluate.py
│       ├── explain.py
│       ├── forecast.py
│       ├── simulator.py
│       ├── foresight.py
│       ├── decision_support.py
│       ├── generate_pdf_guide.py
│       └── elstat_updater.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   ├── test_evaluation.py
│   └── test_simulator.py
├── reports/
│   ├── figures/
│   ├── benchmark_comparison.csv
│   ├── ablation_study.csv
│   ├── feature_importance.csv
│   ├── regional_subgroup_audit.csv
│   ├── regional_forecast_next_12m.csv
│   ├── policy_simulation_scenarios.csv
│   ├── foresight_trajectories_2026_2035.csv
│   ├── foresight_regional_backcasting.csv
│   ├── foresight_megatrends_matrix.csv
│   ├── Presentation_Guide_Road_Safety_Dashboard_GR.pdf
│   ├── benchmark_metrics.json
│   └── model_card.md
└── docs/
    ├── DATA_SOURCES.md
    ├── DATA_DICTIONARY.md
    ├── METHODOLOGY.md
    ├── LIMITATIONS.md
    ├── CODE_AUDIT_REPORT.md
    └── POLICY_RECOMMENDATIONS.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full ML, forecasting, simulation, and foresight pipeline
python run.py

# 3. Run the unit test suite (19 tests)
pytest

# 4. Launch the interactive Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## Scope & Limitations

- **Spatial Resolution**: Models operate on monthly NUTS-2 regional aggregates ($N=780$), not individual street coordinates, as row-level Greek police microdata (ΔΟΤΑ) are restricted. Event-level modeling on 513,801 police crash records is covered in our companion UK STATS19 project ([`KSI-Accident-Severity-Prediction`](https://github.com/dppel/KSI-Accident-Severity-Prediction)).
- **Injury Collisions Only**: ELSTAT Survey `SDT04` records collisions involving at least one injured or killed person; property-damage-only incidents are not part of the census.
- Further details are documented in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## License

Released under the [MIT License](LICENSE). Primary accident data originate from the **Hellenic Statistical Authority (ELSTAT)**.
