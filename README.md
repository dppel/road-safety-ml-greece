# Road Safety ML Greece

An explainable, reproducible machine learning project analyzing road traffic collision severity (**KSI — Killed or Seriously Injured**) in Greece using official **Hellenic Statistical Authority (ELSTAT)** open data and strictly audited scikit-learn pipelines.

---

## Overview: 5 Key Questions Answered

### 1. What problem does this project solve?
Road traffic collisions in Greece exhibit a pronounced structural paradox: while urban centers like **Attica** account for over **53%** of all injury collisions, their fatality and serious injury (KSI) rate per crash (~6.5%) is less than one-third that of rural and island regions such as **Crete, the Peloponnese, South Aegean, and Central Greece** (where 18%–25% of injury crashes result in death or severe trauma).

This repository provides a leak-free machine learning framework that:
* **Track A (Official ELSTAT Regional Panel)**: Predicts elevated monthly regional KSI severity regimes (`target_high_ksi`) across all 13 Greek NUTS-2 regions using strictly lagged historical and structural indicators.
* **Track B (Event-Level Severity Pipeline & Ablation Suite)**: Benchmarks classification models (`DummyClassifier`, `DecisionTree`, `RandomForest`, `HistGradientBoosting`, `XGBoost`) for estimating conditional KSI probability $P(\text{KSI}=1 \mid \mathbf{x})$ given pre-collision road, lighting, weather, and vehicle attributes.

### 2. What data are used (and what are their exact boundaries)?
We enforce a strict separation between **official empirical statistics** and **synthetic pipeline benchmarks**:
* **`data/elstat_national_annual_2015_2024.csv` (Empirical Official Data)**: Verified national annual totals from ELSTAT Survey `SDT04` (e.g., 2021: 10,454 accidents / 624 dead; 2022: 10,487 / 654; 2023: 10,553 / 646; 2024: 11,000 / 664).
* **`data/elstat_regional_monthly_panel_2021_2025.csv` (Empirical Regional Panel — Track A)**: 780 regional-month observations (`13 NUTS-2 Regions × 60 Months`, 2021–2025).
* **`data/synthetic_benchmark_collisions.csv` (Synthetic Event-Level Benchmark — Track B)**: Because row-level DOTA police microdata in Greece require a formal statistical confidentiality application to ELSTAT, we provide an explicitly labeled synthetic benchmark ($N = 25,000$ rows) calibrated to ELSTAT marginal distributions strictly to test event-level feature engineering, anti-leakage guards, and explainability modules.

### 3. How is the pipeline evaluated?
* **Strict Chronological Split**: `2021–2023` (Train) $\rightarrow$ `2024` (Validation / Threshold Calibration) $\rightarrow$ `2025` (Held-Out Test Set). Random shuffling is prohibited to prevent temporal leakage.
* **Programmatic Anti-Leakage Guards (`src/road_safety/data_validation.py`)**:
  * Automatically blocks post-accident outcome columns (`number_of_casualties`, `fatal_accidents`, `did_police_officer_attend_scene_of_accident`) from entering `X`.
  * Enforces target immutability (`assert_target_immutable`) so feature engineering can never alter ground-truth labels `y`.
* **Imbalanced Classification Metrics**: Evaluated via **PR-AUC (Average Precision)**, **ROC-AUC** (with 95% bootstrap confidence intervals), **F1-score**, **Precision**, **Recall**, **Specificity**, **Brier Score**, and explicit **Confusion Matrices**.

### 4. What are the main results?

#### Track A — Official ELSTAT Regional Monthly Panel (2025 Test Set, $N = 156$)
| Model | ROC-AUC (95% CI) | PR-AUC | F1 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DummyClassifier (Baseline)` | `0.5000 [0.5000, 0.5000]` | `0.7885` | `0.8817` | `0.7885` | `1.0000` |
| `DecisionTree` | `0.9068 [0.8321, 0.9684]` | `0.9566` | `0.9647` | `0.9318` | `1.0000` |
| `RandomForest` | **`0.9512 [0.9036, 0.9846]`** | **`0.9855`** | **`0.9647`** | **`0.9318`** | **`1.0000`** |
| `HistGradientBoosting` | `0.9377 [0.8835, 0.9756]` | `0.9808` | `0.9565` | `0.9308` | `0.9837` |
| `XGBoost` | `0.9377 [0.8811, 0.9775]` | `0.9808` | `0.9565` | `0.9308` | `0.9837` |

#### Track B — Event-Level Benchmark & Honest Waze Ablation Study (2025 Test Set, $N = 5,065$)
| Model | ROC-AUC (95% CI) | PR-AUC | F1 | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `DummyClassifier (Baseline)` | `0.5000 [0.5000, 0.5000]` | `0.1338` | `0.0000` | `0.0000` | `0.0000` |
| `DecisionTree` | `0.7021 [0.6803, 0.7228]` | `0.2572` | `0.3232` | `0.3111` | `0.3363` |
| `RandomForest` | **`0.7245 [0.7039, 0.7472]`** | **`0.3023`** | `0.3620` | `0.2746` | **`0.5310`** |
| `HistGradientBoosting` | `0.7213 [0.7015, 0.7452]` | `0.2967` | **`0.3649`** | **`0.2962`** | `0.4749` |
| `XGBoost` | `0.7171 [0.6969, 0.7386]` | `0.2885` | `0.3445` | `0.2839` | `0.4381` |
| `HistGradientBoosting (+Unverified Waze Proxy, y Immutable)` | `0.7141 [0.6928, 0.7361]` | `0.2889` | `0.3576` | `0.3035` | `0.4351` |

> **Key Ablation Finding (`docs/CODE_AUDIT_REPORT.md`)**: When target immutability (`assert_target_immutable`) is enforced so `target_ksi` is never modified during feature enrichment, adding heuristic/simulated Waze proxy features yields zero artificial gain (`ROC-AUC 0.7213 -> 0.7141`). Real predictive lift from crowdsourced telemetry requires verified historical pre-collision partner feeds.

### 5. What are the limitations?
* Public ELSTAT tables are aggregated by region and month; row-level Greek DOTA microdata require confidential research accreditation.
* Only collisions involving at least one injured or killed person are recorded in ELSTAT SDT04 (property-damage-only crashes are excluded).
* Feature importances reflect statistical associations within the observational distribution, not causal effects of road engineering interventions.
* Full details are documented in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/CODE_AUDIT_REPORT.md`](docs/CODE_AUDIT_REPORT.md).

---

## Repository Structure

```text
road-safety-ml-greece/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── configs/
│   └── baseline.yaml
├── data/
│   ├── README.md
│   ├── elstat_national_annual_2015_2024.csv
│   ├── elstat_regional_monthly_panel_2021_2025.csv
│   └── synthetic_benchmark_collisions.csv
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
│       └── waze_ablation.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_evaluation.py
├── reports/
│   ├── figures/
│   ├── benchmark_comparison.csv
│   ├── benchmark_metrics.json
│   └── model_card.md
└── docs/
    ├── DATA_SOURCES.md
    ├── DATA_DICTIONARY.md
    ├── METHODOLOGY.md
    ├── LIMITATIONS.md
    └── CODE_AUDIT_REPORT.md
```

---

## Quickstart & Reproducibility

### 1. Install Dependencies
```bash
pip install -e ".[boosting,explain,app,dev]"
```

### 2. Run the End-to-End Training, Benchmark & Ablation Pipeline
```bash
python -m road_safety.train
```
This single command validates schemas, executes Track A and Track B benchmarks, runs the Waze ablation study, exports metrics to `reports/benchmark_comparison.csv`, and generates all diagnostic plots in `reports/figures/`.

### 3. Run the Automated Unit Test Suite
```bash
pytest
```

### 4. Launch the Interactive Streamlit Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## License & Citation
Released under the [MIT License](LICENSE). Official statistical tables are attributed to the **Hellenic Statistical Authority (ELSTAT)** (*Road Traffic Accidents, SDT04*).
