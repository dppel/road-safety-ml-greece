# Methodology & Experimental Design (`METHODOLOGY.md`)

## 1. Problem Formulation

This repository addresses two complementary, methodologically distinct modeling tasks in Greek road safety:

1. **Track A — Official ELSTAT Regional Panel Modeling (Empirical Open Data)**:
   * **Unit of Observation**: `Region (NUTS-2) × Year × Month` ($N = 780$ panel observations, 13 regions × 60 months, 2021–2025).
   * **Classification Task**: Predict whether a region-month enters a **High-KSI Severity Regime** (`target_high_ksi = 1` when `ksi_severity_rate >= 0.16`, i.e., at least 16% of injury collisions result in death or serious injury) using only structural regional characteristics, calendar seasonality, and strictly lagged historical indicators ($t-1$, $t-3$, $t-12$).
   * **Regression / Anomaly Detection Task**: Forecast the expected monthly `ksi_severity_rate` per region and flag statistical anomalies when newly published ELSTAT bulletins deviate from historical expectations by $\ge 2.5$ percentage points (`250 bps`).

2. **Track B — Event-Level Collision Severity Pipeline (Synthetic Benchmark & Transferable Architecture)**:
   * **Unit of Observation**: Individual injury collision event ($N = 25,000$ benchmark records calibrated to ELSTAT regional, seasonal, and road-type marginal distributions).
   * **Classification Task**: Estimate the probability $P(\text{KSI} = 1 \mid \mathbf{x})$ that a reported injury collision involves at least one fatality or serious injury (`target_ksi = 1`) vs. slight injury only (`target_ksi = 0`).

---

## 2. Temporal Train / Validation / Test Splitting Strategy

Random cross-validation (`train_test_split(..., shuffle=True)`) causes temporal data leakage in road safety datasets because macroeconomic trends, vehicle fleet aging, and seasonal anomalies correlate across contemporaneous observations.

We enforce a **strictly chronological split**:
* **Training Set (`train`)**: `2021–2023` (36 months; 60% of timeline) — used to fit all imputers, encoders, scalers, and model parameters.
* **Validation Set (`val`)**: `2024` (12 months; 20% of timeline) — used for hyperparameter selection and decision threshold calibration (F1-score maximization).
* **Test Set (`test`)**: `2025` (12 months; 20% of timeline) — held out strictly for final, out-of-time evaluation.

---

## 3. Leakage Prevention Protocol

Our pipeline enforces three programmatic safeguards in `src/road_safety/data_validation.py` and `src/road_safety/preprocessing.py`:

1. **Forbidden Column Guard**: `validate_no_leakage(X, forbidden_cols)` raises a `ValueError` if any post-collision or contemporaneous outcome column (`number_of_casualties`, `fatal_accidents`, `serious_accidents`, `ksi_accidents`, `did_police_officer_attend_scene_of_accident`) appears in the feature matrix `X`.
2. **Target Immutability Guard**: All feature engineering transformations operate strictly on `X` (`transform(X) -> X_fe`) and never accept or mutate the target vector `y`. Unit tests in `tests/test_features.py` verify exact bitwise equality of `y` before and after feature engineering.
3. **Train-Only Transformer Fitting**: Sentinel replacement (`-1, 99 -> np.nan`), median imputation (`SimpleImputer(strategy='median')`), standardization (`StandardScaler`), and categorical encoding (`OneHotEncoder(handle_unknown='ignore')`) are encapsulated inside a `scikit-learn` `ColumnTransformer` fit exclusively on `X_train`.

---

## 4. Model Benchmark Suite

Every experiment compares five models on the **exact same temporal test set (`2025`)**:

1. **`DummyClassifier(strategy="prior")`**: Naive baseline predicting the training class prior. Establishes the floor for ROC-AUC (`0.5000`) and PR-AUC (equal to test-set KSI prevalence).
2. **`DecisionTreeClassifier`**: Shallow, interpretable single-tree baseline (`max_depth=5`, `min_samples_leaf=20`, `class_weight="balanced"`).
3. **`RandomForestClassifier`**: Bagged ensemble of decision trees (`n_estimators=200`, `max_depth=8`, `class_weight="balanced"`).
4. **`HistGradientBoostingClassifier`**: Native histogram-based gradient boosted decision trees (`max_iter=250`, `learning_rate=0.04`, `class_weight="balanced"`).
5. **`XGBClassifier`** (when `xgboost` is installed): Regularized gradient boosting (`n_estimators=250`, `max_depth=5`, `learning_rate=0.04`, `subsample=0.8`).

---

## 5. Evaluation Metrics & Uncertainty Quantification

Because KSI collisions represent the minority class (~11.8%–18.7% of injury collisions in Greece), raw accuracy is misleading. A model predicting `0` for every collision achieves >81% accuracy while failing to identify a single fatal or serious crash. We report:

* **PR-AUC (Average Precision)**: Primary ranking metric under class imbalance.
* **ROC-AUC**: Overall discrimination capacity across all thresholds with **95% bootstrap confidence intervals** ($B = 500$ resamples).
* **F1-Score, Precision, Recall (Sensitivity), and Specificity**: Evaluated both at the default `0.50` threshold and at the validation-optimal threshold $\tau^*$ that maximizes F1 on the 2024 validation set.
* **Confusion Matrix (TN, FP, FN, TP)**: Explicit breakdown of false alarms vs. missed severe collisions.
* **Brier Score**: Evaluates probability calibration quality.
