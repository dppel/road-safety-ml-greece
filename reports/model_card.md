# Model Card — Road Safety ML Greece (v1.0.0)

## 1. Model Details
* **Project Name**: `road-safety-ml-greece`
* **Version**: `1.0.0` (October 2026)
* **Model Types Evaluated**:
  * `DummyClassifier (Prior)` — Naive statistical baseline
  * `DecisionTreeClassifier` — Interpretable shallow tree (`max_depth=5`)
  * `RandomForestClassifier` — Bagged tree ensemble (`n_estimators=200`, `max_depth=8`)
  * `HistGradientBoostingClassifier` — Native scikit-learn histogram gradient boosting (`max_iter=250`, `learning_rate=0.04`)
  * `XGBClassifier` — Regularized gradient boosted trees (`n_estimators=250`, `max_depth=5`, `learning_rate=0.04`)
* **License**: MIT

---

## 2. Intended Use & Scope
* **Intended Use**:
  * Portfolio demonstration of leak-free, reproducible tabular machine learning for road safety analysis in Greece.
  * Regional severity regime monitoring (Track A) on official Hellenic Statistical Authority (ELSTAT SDT04) monthly statistics across Greece's 13 NUTS-2 regions.
  * Methodological benchmarking and ablation auditing (Track B) for event-level collision severity pipelines.
* **Out-of-Scope Uses**:
  * **Not for real-time emergency dispatch or clinical triage**: Predictions are research estimates and must not replace human emergency dispatchers or medical protocols.
  * **Not causal policy truth**: Feature importances measure predictive associations, not causal effects of infrastructure interventions.

---

## 3. Training & Evaluation Data
* **Chronological Split Protocol**:
  * **Training Partition**: `2021–2023` (60% of timeline)
  * **Validation Partition**: `2024` (20% of timeline — used for F1 threshold calibration)
  * **Held-Out Test Partition**: `2025` (20% of timeline — strictly unseen during training and threshold tuning)
* **Track A Dataset**: `data/elstat_regional_monthly_panel_2021_2025.csv` ($N = 780$ regional-month rows across 13 Greek NUTS-2 regions; 2025 test set $N = 156$).
* **Track B Dataset**: `data/synthetic_benchmark_collisions.csv` ($N = 25,000$ synthetic event-level collisions calibrated to ELSTAT regional, seasonal, and road-type marginal distributions; 2025 test set $N = 5,065$, KSI positive rate = `13.38%`).

---

## 4. Quantitative Benchmark Results (2025 Held-Out Test Set)

### Track A: Official ELSTAT Regional Monthly Panel (`target_high_ksi`)
| Model | ROC-AUC | 95% Bootstrap CI | PR-AUC | F1-Score | Precision | Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DummyClassifier (Baseline)` | 0.5000 | `[0.5000, 0.5000]` | 0.7885 | 0.8817 | 0.7885 | 1.0000 | 0.1669 |
| `DecisionTree` | 0.9068 | `[0.8321, 0.9684]` | 0.9566 | 0.9647 | 0.9318 | 1.0000 | 0.0681 |
| `RandomForest` | **0.9512** | `[0.9036, 0.9846]` | **0.9855** | **0.9647** | **0.9318** | **1.0000** | **0.0561** |
| `HistGradientBoosting` | 0.9377 | `[0.8835, 0.9756]` | 0.9808 | 0.9565 | 0.9308 | 0.9837 | 0.0574 |
| `XGBoost` | 0.9377 | `[0.8811, 0.9775]` | 0.9808 | 0.9565 | 0.9308 | 0.9837 | 0.0609 |

### Track B: Event-Level Collision Benchmark & Honest Waze Ablation (`target_ksi`)
| Model | ROC-AUC | 95% Bootstrap CI | PR-AUC | F1-Score | Precision | Recall | Brier Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `DummyClassifier (Baseline)` | 0.5000 | `[0.5000, 0.5000]` | 0.1338 | 0.0000 | 0.0000 | 0.0000 | 0.1159 |
| `DecisionTree` | 0.7021 | `[0.6803, 0.7228]` | 0.2572 | 0.3232 | 0.3111 | 0.3363 | 0.2116 |
| `RandomForest` | **0.7245** | `[0.7039, 0.7472]` | **0.3023** | 0.3620 | 0.2746 | **0.5310** | 0.1941 |
| `HistGradientBoosting` | 0.7213 | `[0.7015, 0.7452]` | 0.2967 | **0.3649** | **0.2962** | 0.4749 | 0.1981 |
| `XGBoost` | 0.7171 | `[0.6969, 0.7386]` | 0.2885 | 0.3445 | 0.2839 | 0.4381 | 0.1726 |
| `HistGradientBoosting (+Unverified Waze Proxy, y Immutable)` | 0.7141 | `[0.6928, 0.7361]` | 0.2889 | 0.3576 | 0.3035 | 0.4351 | 0.1980 |

---

## 5. Key Scientific & Audit Findings
1. **Substantial Lift Over Naive Baseline**: On Track B, `RandomForest` (`PR-AUC = 0.3023`) and `HistGradientBoosting` (`PR-AUC = 0.2967`, `F1 = 0.3649`) more than double the precision-recall performance of the naive baseline (`PR-AUC = 0.1338`).
2. **Honest Waze Ablation Verification**: When `assert_target_immutable(y_before, y_after)` is enforced so that feature engineering cannot overwrite `target_ksi`, adding heuristic simulated Waze features (`waze_flow_speed_kmh`, `waze_jam_level`) slightly reduces test ROC-AUC (`0.7213 -> 0.7141`) due to injected feature noise. This confirms that genuine predictive lift from crowdsourced telemetry requires real historical pre-collision partner feeds.
3. **Dominant Physical Risk Factors**: Permutation importance on the 2025 test set shows that rural vs. urban road environment (`is_rural`, `rural_x_speed`), motorcycle involvement (`has_motorcycle`, `moto_x_speed`), pedestrian involvement (`has_pedestrian`), and unlit nighttime road conditions (`is_dark_unlit`) drive the majority of discriminating signal.
