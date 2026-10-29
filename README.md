# Road Safety ML Greece

An explainable, reproducible machine learning project analyzing road traffic collision severity (**KSI — Killed or Seriously Injured**) in Greece using **100% official Hellenic Statistical Authority (ELSTAT SDT04)** open data and strictly audited `scikit-learn` pipelines.

> **Strict Scientific Integrity Policy**: This repository operates **exclusively on official, verified ELSTAT statistical tables** (`2015–2024` national series and `2021–2025` regional monthly panel across all 13 Greek NUTS-2 regions). **Zero synthetic, Monte Carlo, or unverified crowdsourced Waze datasets are used.**

---

## Overview: 5 Key Questions Answered

### 1. What problem does this project solve?
Official ELSTAT road safety statistics reveal a structural paradox across Greece: while metropolitan **Attica** accounts for over **53.5%** of all injury collisions nationally, its fatality and serious injury (**KSI**) rate per collision (~6.5%) is less than one-third that of rural and island regions such as **Crete, the Peloponnese, South Aegean, and Central Greece** (where **18%–25%** of injury collisions result in death or severe trauma).

This repository implements an end-to-end, leak-free machine learning pipeline that predicts whether a Greek NUTS-2 region in a given month enters a **Severe KSI Regime** (`target_high_ksi = 1` when `ksi_severity_rate >= 0.21`) using strictly pre-period information: calendar seasonality, structural road exposure, and historical autoregressive lags ($t-1, t-2, t-3, t-12$).

### 2. What data are used?
* **[`data/elstat_national_annual_2015_2024.csv`](data/elstat_national_annual_2015_2024.csv)**: Verified official national annual totals from ELSTAT Survey `SDT04` (e.g., `2021`: 10,454 injury accidents / 624 killed / 610 serious; `2022`: 10,487 / 654 / 664; `2023`: 10,553 / 646 / 659; `2024`: 11,000 / 664 / 568).
* **[`data/elstat_regional_monthly_panel_2021_2025.csv`](data/elstat_regional_monthly_panel_2021_2025.csv)**: Official regional monthly panel (`13 Greek NUTS-2 Regions × 60 Months = 780 observations`, `2021–2025`).
* *Note on Event-Level Microdata*: Because row-level Greek police DOTA records require a confidential statistical research application to ELSTAT, we model Greek public data strictly at its authentic `Region × Month` unit of observation (while event-level crash modeling on 513,801 real police records is maintained in our companion UK STATS19 repository, [`KSI-Accident-Severity-Prediction`](https://github.com/dppel/KSI-Accident-Severity-Prediction)).

### 3. How is the pipeline evaluated without data leakage?
* **Strict Chronological Split**: `2021–2023` (**Train**, $N=468$) $\rightarrow$ `2024` (**Validation / Threshold Calibration**, $N=156$) $\rightarrow$ `2025` (**Held-Out Test Set**, $N=156$). Random shuffling is prohibited.
* **Programmatic Anti-Leakage Guards ([`src/road_safety/data_validation.py`](src/road_safety/data_validation.py))**:
  * `validate_no_leakage(X)`: Automatically blocks contemporaneous outcome variables (`ksi_accidents`, `fatal_accidents`, `serious_accidents`, `slight_accidents`, `ksi_severity_rate`, `total_accidents`) from entering `X`.
  * `assert_target_immutable(y_before, y_after)`: Guarantees that feature engineering never modifies the ground-truth target `y`.
* **Uncertainty Quantification**: All test metrics are evaluated on the exact same 2025 partition with **95% non-parametric bootstrap confidence intervals**.

### 4. What are the main results on the 2025 Held-Out Test Set?

#### 5-Model Benchmark on Official ELSTAT Regional Panel (`2025` Test Set, $N = 156$)
| Model | ROC-AUC (95% CI) | PR-AUC | F1-Score | Precision | Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `DummyClassifier (Baseline)` | `0.5000 [0.5000, 0.5000]` | `0.4679` | `0.0000` | `0.0000` | `0.0000` | `0.2496` |
| `DecisionTree` | `0.7958 [0.7209, 0.8642]` | `0.7522` | `0.7579` | `0.6154` | `0.9863` | `0.1903` |
| `RandomForest` | **`0.8567 [0.7973, 0.9105]`** | **`0.8374`** | `0.7594` | `0.6228` | `0.9726` | **`0.1552`** |
| `HistGradientBoosting` | `0.8199 [0.7509, 0.8808]` | `0.8036` | `0.7543` | **`0.6471`** | `0.9041` | `0.1919` |
| `XGBoost` | `0.8498 [0.7835, 0.9043]` | `0.8259` | **`0.7609`** | `0.6306` | `0.9589` | `0.1597` |

#### Empirical Feature Group Ablation Study (`2025` Test Set)
| Feature Subset | Features | ROC-AUC (95% CI) | PR-AUC | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| `1. Calendar & Seasonality Only` | 4 | `0.5805 [0.4954, 0.6586]` | `0.5658` | `0.6376` |
| `2. Calendar + NUTS-2 Region` | 5 | **`0.8404 [0.7717, 0.8971]`** | **`0.8298`** | **`0.7735`** |
| `3. Calendar + Structural Exposure (Urban & Moto Share)` | 6 | `0.8320 [0.7538, 0.8907]` | `0.8141` | `0.7660` |
| `4. Full Baseline (+ Autoregressive Lags t-1, t-2, t-3, t-12)` | 14 | `0.8199 [0.7509, 0.8808]` | `0.8036` | `0.7543` |

#### Top Global Features by Permutation Importance (`RandomForest`, 2025 Test Set)
1. **`lag_12m_ksi_rate`** (`+0.0440` PR-AUC drop) — Regional KSI severity rate in the same calendar month one year earlier.
2. **`urban_ratio`** (`+0.0254`) — Structural proportion of urban vs. rural road network collisions.
3. **`lag_12m_total_accidents`** (`+0.0186`) — Prior-year same-month collision volume.
4. **`lag_1m_total_accidents`** (`+0.0174`) — Immediate prior-month collision volume.
5. **`motorcycle_share`** (`+0.0162`) — Regional two-wheeler involvement rate.

### 5. What are the limitations?
* **Regional Granularity**: Predictions apply to `NUTS-2 Region × Month` aggregates ($N=780$), not individual street intersections.
* **Sample Size**: Because Greece has 13 NUTS-2 regions, the held-out 2025 test partition comprises $13 \times 12 = 156$ observations; bootstrap 95% confidence intervals are explicitly reported.
* **Injury Collisions Only**: ELSTAT `SDT04` records only collisions involving death or bodily injury (property-damage-only incidents are excluded).
* Full documentation is available in [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) and [`docs/CODE_AUDIT_REPORT.md`](docs/CODE_AUDIT_REPORT.md).

---

## Repository Structure

```text
road-safety-ml-greece/
├── README.md
├── LICENSE
├── pyproject.toml
├── run.py
├── .gitignore
├── configs/
│   └── baseline.yaml
├── data/
│   ├── README.md
│   ├── elstat_national_annual_2015_2024.csv
│   └── elstat_regional_monthly_panel_2021_2025.csv
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
│       └── elstat_updater.py
├── app/
│   └── streamlit_app.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_evaluation.py
├── reports/
│   ├── figures/
│   ├── benchmark_comparison.csv
│   ├── ablation_study.csv
│   ├── feature_importance.csv
│   ├── regional_subgroup_audit.csv
│   ├── regional_forecast_next_12m.csv
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

## Quickstart & Reproducibility

```bash
# 1. Run the entire pipeline (Validation -> Train -> Benchmark -> Ablation -> Plots)
python run.py

# 2. Run the automated unit test suite (10/10 tests)
pytest

# 3. Launch the interactive Streamlit dashboard
streamlit run app/streamlit_app.py
```

---

## License & Citation
Released under the [MIT License](LICENSE). Statistical data are attributed to the **Hellenic Statistical Authority (ELSTAT)** (*Road Traffic Accidents, SDT04*).
