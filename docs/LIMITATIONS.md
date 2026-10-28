# Scientific Limitations & Operational Boundaries (`LIMITATIONS.md`)

To maintain scientific integrity and ensure responsible interpretation by researchers, recruiters, and public-sector stakeholders, this document explicitly details the limitations of the data and models in this repository.

---

## 1. Public Aggregation vs. Event-Level Microdata in Greece

* **What ELSTAT Publicly Publishes**: The Hellenic Statistical Authority (ELSTAT) publishes monthly press releases and Excel tables (`SDT04`) aggregated by NUTS-2 region, month, and broad causal/demographic categories.
* **What Requires Confidential Access**: Individual event-level DOTA records (exact timestamp, GPS coordinates, vehicle-by-vehicle kinematics, and casualty-level hospital linkage) are confidential statistical records that require a formal research request to ELSTAT.
* **Implication for Modeling**:
  * **Track A (`elstat_regional_monthly_panel_2021_2025.csv`)** operates on real, verified regional aggregates. It predicts *regional monthly severity regimes*, **not** individual crash outcomes at a specific street corner.
  * **Track B (`synthetic_benchmark_collisions.csv`)** demonstrates the event-level ML engineering pipeline using a synthetic benchmark calibrated to ELSTAT marginals. Metrics on Track B validate software correctness and feature engineering logic, not empirical field accuracy on confidential police records.

---

## 2. Exclusion of Property-Damage-Only (PDO) Collisions

* Under Greek statistical reporting standards (ELSTAT DOTA), a road traffic collision is recorded in official national statistics **only if at least one person is injured or killed**.
* Collisions resulting exclusively in material damage (Δελτίο Συμβάντος Τροχαίου με Υλικές Ζημιές) are excluded from the denominator.
* Consequently, any probability $P(\text{KSI} \mid \mathbf{x})$ estimated by these models is conditional on an **injury collision having occurred**, not the unconditional probability of a crash occurring per vehicle-kilometer traveled (VKT).

---

## 3. Crowdsourced Telemetry (Waze) Availability & Bias

* **No Public Historical Archive**: Greece does not currently have an openly licensed, retrospective public archive of Waze traffic jams and hazard alerts linked spatio-temporally to past ELSTAT collisions (2021–2025).
* **Look-Ahead & Reverse-Causality Risk**: If traffic congestion (`jam_level`) or hazard alerts are queried *after* the collision timestamp $t_0$, the telemetry measures the traffic jam **caused by the crash itself** (a severe post-collision leak). Any real-world integration must strictly query telemetry within the pre-incident window $[t_0 - 30\text{ min}, t_0 - 5\text{ min}]$.
* **Reporting Selection Bias**: Crowdsourced app penetration is substantially higher on urban arterial roads (e.g., Attica, Thessaloniki) than on rural provincial or island networks (e.g., Crete, Peloponnese, South Aegean) where KSI severity rates are highest.

---

## 4. Observational Correlation vs. Causal Inference

* **Feature Importance $\neq$ Causality**: SHAP values and permutation importances explain how the model uses input features to form predictions; they do **not** estimate causal treatment effects.
* For example, a negative association between `urban_ratio` and `ksi_severity_rate` reflects lower average impact speeds and denser emergency medical proximity in urban centers (Attica), not that urban intersections are inherently "safe" in absolute collision frequency (Attica accounts for >53% of total Greek injury collisions).

---

## 5. Non-Operational Status

* This repository is a **research and portfolio engineering project**. Its predictions and Streamlit dashboard visualizations must not be used for real-time emergency dispatch or clinical triage without formal validation on audited, real-time operational data streams.
