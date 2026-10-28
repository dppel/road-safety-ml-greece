# Code & Scientific Integrity Audit Report (`CODE_AUDIT_REPORT.md`)

**Audited Artifacts**:
1. Prototype script: `waze_feature_enrichment.py`
2. Prototype notebook: `02_greece_elstat_road_safety_ksi.ipynb`

In accordance with **Week 1 & Week 4** of the Road Safety ML Roadmap, we conducted a line-by-line audit of the initial Greek prototype scripts to separate verified empirical components from methodological errors that required remediation.

---

## 1. Executive Summary Table

| Category | Audited Component / Finding | Status in Prototype | Remediation in `road-safety-ml-greece` (v1.0) |
| :--- | :--- | :---: | :--- |
| **1. Confirmed (Επιβεβαιωμένο)** | Official ELSTAT national annual statistics (2021–2024: 10,454 / 10,487 / 10,553 / 11,000 injury accidents and exact dead/serious/slight breakdowns) | Verified | Preserved in `data/elstat_national_annual_2015_2024.csv` and `docs/DATA_SOURCES.md` |
| **1. Confirmed (Επιβεβαιωμένο)** | Structural urban vs. rural/island severity paradox in Greece (Attica ~53.5% of crashes but ~6.5% KSI rate vs. rural/island regions >18%–25% KSI rate) | Verified | Modeled at true regional granularity in Track A (`elstat_regional_monthly_panel_2021_2025.csv`) |
| **1. Confirmed (Επιβεβαιωμένο)** | `scikit-learn` `Pipeline` + `ColumnTransformer` architecture with train-only imputation/scaling and cyclical time encoding (`sin`/`cos`) | Verified | Modularized into `src/road_safety/preprocessing.py` and `src/road_safety/features.py` with unit tests |
| **2. Needs Correction (Χρειάζεται Διόρθωση)** | **Target Overwrite Bug (`waze_feature_enrichment.py` lines 101–114)**: `df['target_ksi'] = np.random.binomial(1, prob_ksi)` overwrote the ground-truth target using the newly synthesized Waze features | **Critical Flaw** | **Fixed**: Removed target overwrite completely. Added strict unit test (`test_target_immutability`) ensuring `y` is never touched by feature engineering |
| **2. Needs Correction (Χρειάζεται Διόρθωση)** | **Conflation of Aggregated ELSTAT Tables with Simulated DOTA Microdata**: Presenting Monte Carlo sampled rows (`elstat_dota_microdata_2021_2025.csv`) as official ELSTAT DOTA records | **Methodological Flaw** | **Fixed**: Separated into Track A (Official Aggregated Regional Panel) and Track B (Explicitly labeled `synthetic_benchmark_collisions.csv`) |
| **2. Needs Correction (Χρειάζεται Διόρθωση)** | **Single-Model Evaluation without Naive Baseline**: Evaluating only `HistGradientBoostingClassifier` without comparing against `DummyClassifier`, `DecisionTree`, and `RandomForest` on identical splits | **Incomplete** | **Fixed**: Implemented full 5-model benchmark suite in `src/road_safety/train.py` and `evaluate.py` with 95% bootstrap CIs |
| **3. Requires External Data (Απαιτεί Εξωτερικά Δεδομένα)** | **Waze Telemetry Features (`waze_flow_speed_kmh`, `waze_jam_level`, `waze_hazard_density`)**: Generated via `np.random` heuristics (`lines 56–88`) rather than historical pre-collision telemetry | **Unverified / Simulated** | **Isolated**: Removed from baseline pipeline; moved to opt-in `src/road_safety/waze_ablation.py` clearly documented as a simulation/schema study |
| **3. Requires External Data (Απαιτεί Εξωτερικά Δεδομένα)** | **Real Event-Level Greek DOTA Microdata**: Individual crash records with exact hour, driver age, and vehicle age | **Restricted Access** | Documented formal ELSTAT research microdata request process in `docs/DATA_SOURCES.md` and `docs/LIMITATIONS.md` |

---

## 2. Detailed Technical Breakdown of the `waze_feature_enrichment.py` Bug

### What Happened in the Prototype
In the prototype `waze_feature_enrichment.py` (lines 101–114), after generating synthetic columns `waze_flow_speed_kmh`, `waze_jam_level`, and `waze_hazard_density`, the script executed:

```python
# PROTOTYPE FLAW (DO NOT USE):
logit_adj = (
    -3.35
    + 0.90 * (df['area_type'] == 2)
    + 0.024 * (flow_speed_kmh - 45.0)
    - 0.65 * (jam_levels >= 3)
    + 0.18 * hazard_density_score
    ...
)
prob_ksi = 1.0 / (1.0 + np.exp(-logit_adj))
df['target_ksi'] = np.random.binomial(1, prob_ksi)
```

### Why This Invalidated the Ablation Comparison
1. **Circular Target Redefinition**: By redefining `df['target_ksi']` as a direct mathematical function of `flow_speed_kmh`, `jam_levels`, and `hazard_density_score`, the target variable $y$ was synthetically coupled to the Waze features.
2. **Unfair Baseline Comparison**: When the baseline model (without Waze features) and the Waze-enriched model were subsequently evaluated on this overwritten `target_ksi`, the jump in ROC-AUC (`0.7209 -> 0.8285`) and PR-AUC (`0.4368 -> 0.6246`) merely reflected that the enriched model had access to the exact random numbers used to overwrite `target_ksi` five lines earlier.

### How It Is Fixed in `road-safety-ml-greece`
1. **Strict Separation of `X` and `y`**: In `src/road_safety/features.py` and `src/road_safety/waze_ablation.py`, feature engineering functions accept only the feature DataFrame `X` and return `X_transformed`. They have no access to `y`.
2. **Authentic Ablation Protocol**: When testing proxy or simulated features against a fixed ground-truth target `y` that was generated independently of those features, adding pure random noise or heuristic proxies **does not artificially inflate test ROC-AUC/PR-AUC**. Our ablation study in `src/road_safety/waze_ablation.py` demonstrates this honestly.
