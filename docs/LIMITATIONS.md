# Scientific Limitations & Operational Boundaries (`LIMITATIONS.md`)

To maintain scientific integrity and ensure responsible interpretation by researchers, recruiters, and public-sector stakeholders, this document explicitly details the limitations of the data and models in this repository.

---

## 1. Public Aggregation Granularity vs. Confidential Microdata

* **What ELSTAT Publicly Publishes**: The Hellenic Statistical Authority (ELSTAT) publishes monthly bulletins and statistical tables (`SDT04`) aggregated by NUTS-2 region, month, and broad causal/demographic categories.
* **What Requires Confidential Access**: Individual event-level DOTA records (exact timestamp, GPS coordinates, vehicle-by-vehicle kinematics, and casualty-level hospital linkage) are confidential statistical records that require a formal research request to ELSTAT.
* **Why Zero Synthetic Data Are Used Here**: Rather than generating synthetic crash rows (which would cause machine learning models to merely rediscover a hand-coded simulation formula), this project models strictly the verified official `Region × Month` panel ($N = 780$). Event-level collision modeling on real police records (513,801 collisions) is maintained in our companion UK STATS19 repository (`KSI-Accident-Severity-Prediction`).

---

## 2. Sample Size of the Regional Monthly Panel ($N = 780$)

* Because Greece has 13 NUTS-2 regions and 12 calendar months per year, a 5-year panel (`2021–2025`) contains exactly $13 \times 60 = 780$ regional-month observations ($N_{\text{train}} = 468$, $N_{\text{val}} = 156$, $N_{\text{test}} = 156$).
* Consequently, 95% bootstrap confidence intervals around `ROC-AUC` and `PR-AUC` span approximately $\pm 0.05$ to $\pm 0.06$. We report these confidence intervals explicitly in all benchmark tables to prevent over-interpreting small decimal differences between tree ensembles.

---

## 3. Exclusion of Property-Damage-Only (PDO) Collisions

* Under Greek statistical reporting standards (ELSTAT DOTA), a road traffic collision is recorded in official national statistics **only if at least one person is injured or killed**.
* Collisions resulting exclusively in material damage are excluded from the denominator.

---

## 4. Crowdsourced Telemetry (Waze) Availability

* Greece does not currently have an openly licensed, retrospective public archive of Waze traffic jams and hazard alerts linked to ELSTAT regional or event-level collision records (`2021–2025`).
* In accordance with our scientific integrity policy, **no simulated or heuristic Waze features are included in this repository**.

---

## 5. Observational Correlation vs. Causal Inference

* Permutation importances and SHAP values explain how the model uses lagged and structural features to form predictions; they do **not** estimate causal treatment effects of policing or infrastructure interventions.
