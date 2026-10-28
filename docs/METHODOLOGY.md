# Methodology & Experimental Design (`METHODOLOGY.md`)

## 1. Problem Formulation (100% Official ELSTAT Data)

This repository models Greek road traffic collision severity strictly at its verified public observation granularity:

* **Unit of Observation**: `Region (NUTS-2) × Year × Month` ($N = 780$ panel observations across Greece's 13 NUTS-2 regions over 60 months, `2021–2025`).
* **Classification Task**: Predict whether a Greek NUTS-2 region in month $t$ enters a **Severe KSI Regime** (`target_high_ksi = 1` when `ksi_severity_rate >= 0.21`, i.e., at least 21% of injury collisions result in death or serious injury; minority class prevalence = `45.26%`) using exclusively pre-period information:
  1. Calendar seasonality (`month`, `month_sin`, `month_cos`, `is_summer`).
  2. Structural regional indicators (`region`, `urban_ratio`, `motorcycle_share`).
  3. Strictly lagged historical indicators from months $\le t - 1$ (`lag_1m_ksi_rate`, `lag_2m_ksi_rate`, `lag_12m_ksi_rate`, `rolling_3m_ksi_rate`, `yoy_ksi_rate_diff`, `lag_1m_total_accidents`, `lag_12m_total_accidents`).

---

## 2. Temporal Train / Validation / Test Splitting Strategy

Random cross-validation (`train_test_split(..., shuffle=True)`) causes temporal data leakage in panel datasets because macroeconomic trends and multi-month rolling averages overlap across time.

We enforce a **strictly chronological split**:
* **Training Set (`train`)**: `2021–2023` ($N = 468$ region-months; 60% of timeline) — used to fit all imputers, encoders, scalers, and model parameters.
* **Validation Set (`val`)**: `2024` ($N = 156$ region-months; 20% of timeline) — used for hyperparameter selection and decision threshold calibration (F1-score maximization).
* **Test Set (`test`)**: `2025` ($N = 156$ region-months; 20% of timeline) — held out strictly for final, out-of-time evaluation.

---

## 3. Leakage Prevention Protocol

Our pipeline enforces three programmatic safeguards in `src/road_safety/data_validation.py` and `src/road_safety/preprocessing.py`:

1. **Forbidden Column Guard**: `validate_no_leakage(X, forbidden_cols)` raises a `ValueError` if any contemporaneous outcome column (`ksi_accidents`, `fatal_accidents`, `serious_accidents`, `slight_accidents`, `ksi_severity_rate`, `total_accidents`) appears in the feature matrix `X`.
2. **Target Immutability Guard**: All feature engineering transformations operate strictly on `X` and never accept or mutate the target vector `y`. `assert_target_immutable(y_before, y_after)` verifies exact bitwise equality of `y`.
3. **Train-Only Transformer Fitting**: Median imputation (`SimpleImputer(strategy='median')`), standardization (`StandardScaler`), and categorical encoding (`OneHotEncoder(handle_unknown='ignore')`) are encapsulated inside a `scikit-learn` `ColumnTransformer` fit exclusively on `X_train`.

---

## 4. Model Benchmark Suite & Feature Group Ablation

1. **5-Model Benchmark Suite**:
   * `DummyClassifier(strategy="prior")` — Naive baseline (`ROC-AUC = 0.5000`, `PR-AUC = 0.4679`, `F1 = 0.0000`).
   * `DecisionTreeClassifier` (`max_depth=4`, `min_samples_leaf=15`, `class_weight="balanced"`).
   * `RandomForestClassifier` (`n_estimators=200`, `max_depth=6`, `min_samples_leaf=10`, `class_weight="balanced"`).
   * `HistGradientBoostingClassifier` (`max_iter=200`, `learning_rate=0.04`, `max_leaf_nodes=15`).
   * `XGBClassifier` (`n_estimators=200`, `max_depth=4`, `learning_rate=0.04`, `subsample=0.8`).

2. **Empirical Feature Group Ablation Study**:
   Evaluates progressive feature groups on the 2025 test set:
   * *Group 1*: Calendar & Seasonality Only (`month`, `month_sin`, `month_cos`, `is_summer`)
   * *Group 2*: Calendar + NUTS-2 Region
   * *Group 3*: Calendar + Structural Exposure (`urban_ratio`, `motorcycle_share`)
   * *Group 4*: Full Baseline (+ Historical Autoregressive Lags $t-1, t-2, t-3, t-12$)
