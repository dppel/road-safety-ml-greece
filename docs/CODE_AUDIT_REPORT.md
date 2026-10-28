# Code & Scientific Integrity Audit Report (`CODE_AUDIT_REPORT.md`)

**Audited Prototype Artifacts**:
1. Prototype script: `waze_feature_enrichment.py`
2. Prototype notebook: `02_greece_elstat_road_safety_ksi.ipynb`

In accordance with **Week 1 & Week 4** of the Road Safety ML Roadmap, we conducted a line-by-line audit of the initial Greek prototype scripts and remediated every methodological flaw so that `road-safety-ml-greece` operates **100% on official ELSTAT data with zero synthetic datasets**.

---

## 1. Executive Summary Table

| Category | Audited Component / Finding | Status in Prototype | Remediation in `road-safety-ml-greece` (v1.1) |
| :--- | :--- | :---: | :--- |
| **1. Confirmed (Επιβεβαιωμένο)** | Official ELSTAT national annual statistics (`2015–2024`, including `2021`: 10,454 accidents / 624 dead; `2022`: 10,487 / 654; `2023`: 10,553 / 646; `2024`: 11,000 / 664) | Verified (`SDT04`) | Preserved in `data/elstat_national_annual_2015_2024.csv` and `docs/DATA_SOURCES.md` |
| **1. Confirmed (Επιβεβαιωμένο)** | Structural urban vs. rural/island severity paradox in Greece (Attica ~53.5% of crashes but ~6.5% KSI rate vs. rural/island regions >18%–25% KSI rate) | Verified (`SDT04`) | Modeled at its true `Region × Month` granularity in `data/elstat_regional_monthly_panel_2021_2025.csv` |
| **1. Confirmed (Επιβεβαιωμένο)** | `scikit-learn` `Pipeline` + `ColumnTransformer` architecture with train-only imputation/scaling and cyclical time encoding (`sin`/`cos`) | Verified | Modularized into `src/road_safety/preprocessing.py` and `src/road_safety/features.py` with 10 unit tests |
| **2. Needs Correction (Χρειάζεται Διόρθωση)** | **Target Overwrite Bug (`waze_feature_enrichment.py` lines 101–114)**: `df['target_ksi'] = np.random.binomial(1, prob_ksi)` overwrote the target using newly generated random Waze features | **Critical Flaw** | **Eliminated**: Deleted `waze_feature_enrichment.py` and added programmatic `assert_target_immutable` guard in `src/road_safety/data_validation.py` |
| **2. Needs Correction (Χρειάζεται Διόρθωση)** | **Synthetic Monte Carlo Microdata (`elstat_dota_microdata_2021_2025.csv`)**: Training ML models on synthetic rows merely rediscovers the simulation formula | **Methodological Flaw** | **Eliminated**: Removed all synthetic collision files. The repository now models **100% official ELSTAT tables (Zero Synthetic Data)** |
| **2. Needs Correction (Χρειάζεται Διόρθωση)** | **Single-Model Evaluation without Naive Baseline**: Evaluating only `HistGradientBoostingClassifier` without comparing against `DummyClassifier`, `DecisionTree`, `RandomForest`, and `XGBoost` | **Incomplete** | **Fixed**: Implemented full 5-model benchmark + empirical feature group ablation study in `src/road_safety/train.py` with 95% bootstrap CIs |
| **3. Requires External Data (Απαιτεί Εξωτερικά Δεδομένα)** | **Waze Telemetry Features (`waze_flow_speed_kmh`, `waze_jam_level`, `waze_hazard_density`)**: No public historical archive exists for Greece | **Unavailable Publicly** | **Excluded from Empirical Models**: Documented in `docs/DATA_SOURCES.md` and `docs/LIMITATIONS.md` as requiring authorized *Waze for Cities* municipal partner feeds |
| **3. Requires External Data (Απαιτεί Εξωτερικά Δεδομένα)** | **Row-Level Greek DOTA Microdata**: Individual police crash records require statistical confidentiality clearance | **Restricted Access** | Documented formal ELSTAT research access rules in `docs/LIMITATIONS.md`; kept real event-level modeling in the companion UK STATS19 repository |
