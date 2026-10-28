# Model Card — Road Safety ML Greece (v1.1.0 — 100% Official ELSTAT Data)

## 1. Model Details
* **Project Name**: `road-safety-ml-greece`
* **Version**: `1.1.0` (October 2026)
* **Data Provenance Policy**: **100% Official ELSTAT Aggregated Statistics (`SDT04`) — Zero Synthetic Data**
* **Model Types Evaluated**:
  * `DummyClassifier (Prior)` — Naive statistical baseline
  * `DecisionTreeClassifier` — Interpretable shallow tree (`max_depth=4`, `min_samples_leaf=15`)
  * `RandomForestClassifier` — Bagged tree ensemble (`n_estimators=200`, `max_depth=6`, `min_samples_leaf=10`)
  * `HistGradientBoostingClassifier` — Native scikit-learn histogram gradient boosting (`max_iter=200`, `learning_rate=0.04`)
  * `XGBClassifier` — Regularized gradient boosted trees (`n_estimators=200`, `max_depth=4`, `learning_rate=0.04`)
* **License**: MIT

---

## 2. Intended Use & Scope
* **Intended Use**:
  * Portfolio demonstration of leak-free, reproducible tabular machine learning for road safety analysis in Greece using exclusively verified public statistics.
  * Regional severity regime monitoring (`target_high_ksi = 1` when `ksi_severity_rate >= 0.21`) on official Hellenic Statistical Authority (**ELSTAT SDT04**) monthly statistics across Greece's 13 NUTS-2 regions.
* **Out-of-Scope Uses**:
  * **Not for real-time emergency dispatch or clinical triage**: Predictions operate at the `Region × Month` level and must not be used for individual emergency dispatch.
  * **Not causal policy truth**: Feature importances measure predictive associations across historical ELSTAT observations, not causal treatment effects.

---

## 3. Training & Evaluation Data
* **Dataset**: `data/elstat_regional_monthly_panel_2021_2025.csv` ($N = 780$ regional-month observations across all 13 Greek NUTS-2 regions; Severe KSI Regime $\ge 21\%$ positive rate = `45.26%`).
* **Chronological Split Protocol**:
  * **Training Partition**: `2021–2023` ($N = 468$ region-months, 60% of timeline)
  * **Validation Partition**: `2024` ($N = 156$ region-months, 20% of timeline — used for F1 threshold calibration)
  * **Held-Out Test Partition**: `2025` ($N = 156$ region-months, 20% of timeline — strictly unseen during training and threshold tuning)

---

## 4. Quantitative Benchmark Results (2025 Held-Out ELSTAT Test Set, $N = 156$)

### 5-Model Benchmark (`target_high_ksi`: Severe Regional KSI Regime $\ge 21\%$)
| Model | ROC-AUC | 95% Bootstrap CI | PR-AUC | F1-Score | Precision | Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DummyClassifier (Baseline)` | 0.5000 | `[0.5000, 0.5000]` | 0.4679 | 0.0000 | 0.0000 | 0.0000 | 0.2496 |
| `DecisionTree` | 0.7958 | `[0.7209, 0.8642]` | 0.7522 | 0.7579 | 0.6154 | 0.9863 | 0.1903 |
| `RandomForest` | **0.8567** | `[0.7973, 0.9105]` | **0.8374** | 0.7594 | 0.6228 | 0.9726 | **0.1552** |
| `HistGradientBoosting` | 0.8199 | `[0.7509, 0.8808]` | 0.8036 | 0.7543 | **0.6471** | 0.9041 | 0.1919 |
| `XGBoost` | 0.8498 | `[0.7835, 0.9043]` | 0.8259 | **0.7609** | 0.6306 | 0.9589 | 0.1597 |

### Week 4 Empirical Feature Group Ablation Study (`HistGradientBoosting` on 2025 Test Set)
| Feature Subset | Features (`#`) | ROC-AUC | 95% Bootstrap CI | PR-AUC | F1-Score | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1. Calendar & Seasonality Only` | 4 | 0.5805 | `[0.4954, 0.6586]` | 0.5658 | 0.6376 | 1.0000 |
| `2. Calendar + NUTS-2 Region` | 5 | **0.8404** | `[0.7717, 0.8971]` | **0.8298** | **0.7735** | 0.9589 |
| `3. Calendar + Structural Exposure (Urban & Moto Share)` | 6 | 0.8320 | `[0.7538, 0.8907]` | 0.8141 | 0.7660 | 0.9863 |
| `4. Full Baseline (+ Autoregressive Lags t-1, t-2, t-3, t-12)` | 14 | 0.8199 | `[0.7509, 0.8808]` | 0.8036 | 0.7543 | 0.9041 |

---

## 5. Key Scientific & Explainability Findings
1. **Substantial Lift Over Naive Baseline**: On the held-out 2025 ELSTAT test set, `RandomForest` (`ROC-AUC = 0.8567`, `PR-AUC = 0.8374`, `Brier = 0.1552`) and `XGBoost` (`ROC-AUC = 0.8498`, `PR-AUC = 0.8259`, `F1 = 0.7609`) substantially outperform the naive prior baseline (`ROC-AUC = 0.5000`, `PR-AUC = 0.4679`, `F1 = 0.0000`).
2. **Dominant Empirical Predictors**: Permutation importance on the 2025 test set shows that same-month prior-year severity (`lag_12m_ksi_rate`, `+0.0440` PR-AUC drop), structural road network urbanization (`urban_ratio`, `+0.0254`), lagged regional collision volume (`lag_12m_total_accidents`, `+0.0186`; `lag_1m_total_accidents`, `+0.0174`), and two-wheeler share (`motorcycle_share`, `+0.0162`) drive the model's out-of-sample discrimination.
