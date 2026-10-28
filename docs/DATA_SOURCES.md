# Data Sources & Provenance (`DATA_SOURCES.md`)

This document records the exact provenance, licensing, geographic granularity, and temporal coverage of all data sources evaluated or included in the **Road Safety ML Greece** repository.

## 1. Primary Statistical Source: Hellenic Statistical Authority (ELSTAT)

* **Official Publication**: *Road Traffic Accidents resulting in Death or Injury (Οδικά Τροχαία Ατυχήματα)* — Survey Code `SDT04`.
* **Portal URL**: [https://www.statistics.gr/el/statistics/-/publication/SDT04/-](https://www.statistics.gr/el/statistics/-/publication/SDT04/-)
* **Collection Instrument**: Collected across Greece via the **Road Traffic Accident Bulletin (Δελτίο Οδικού Τροχαίου Ατυχήματος — ΔΟΤΑ)** filled out by Traffic Police and Port Authorities whenever a road collision results in at least one injured or killed person.
* **Geographic Coverage**: All 13 NUTS-2 Regions (Περιφέρειες) of Greece.
* **Temporal Coverage in Repository**:
  * `data/elstat_national_annual_2015_2024.csv`: Verified official annual national totals (Accidents, Dead, Seriously Injured, Slightly Injured) from 2015 to 2024.
  * `data/elstat_regional_monthly_panel_2021_2025.csv`: Regional monthly panel (`Region × Year × Month`) across all 13 Greek NUTS-2 regions (2021–2025).
* **Public Granularity vs. Confidential Microdata**:
  * **Publicly Available**: Aggregated statistical tables (monthly bulletins and annual tables broken down by NUTS-2 region, month, severity class, broad road category, and age/vehicle groups).
  * **Restricted / On-Request**: Anonymized event-level microdata (individual DOTA records) require a formal research request to ELSTAT under the Statistical Confidentiality Regulation.
* **License**: ELSTAT Open Statistical Data Terms of Use (CC BY 4.0 equivalent — attribution required to the Hellenic Statistical Authority).

### Verified Official National Totals (ELSTAT 2021–2024)

| Year | Total Injury Accidents | Killed (30-day definition) | Seriously Injured | Slightly Injured | Total Casualties |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **2021** | 10,454 | 624 | 610 | 11,746 | 12,980 |
| **2022** | 10,487 | 654 | 664 | 11,961 | 13,279 |
| **2023** | 10,553 | 646 | 659 | 12,156 | 13,461 |
| **2024** | 11,000 | 664 | 568 | 12,803 | 14,035 |

---

## 2. Secondary Open Government Source: `data.gov.gr`

* **Portal URL**: [https://www.data.gov.gr](https://www.data.gov.gr)
* **Relevant Datasets**:
  * Ministry of Citizen Protection / Hellenic Police (`mcp_traffic_accidents`, `mcp_traffic_violations`): Aggregated summaries of traffic policing actions, violations, and collision counts.
  * Road traffic volume counters (Attica / Regional highways where available via API).
* **Usage in Project**: Used for cross-checking regional exposure indicators and violation trends. Not mixed with non-public operational systems.

---

## 3. European Comparative Context: Eurostat (`tran_sf_road`)

* **Portal URL**: [https://ec.europa.eu/eurostat/web/transport/database](https://ec.europa.eu/eurostat/web/transport/database)
* **Datasets**: `tran_sf_roadnu` (Road accidents victims by NUTS-2 regions) and CARE (Community database on Accidents on the Roads in Europe).
* **Usage in Project**: Benchmarks Greece's regional KSI (Killed or Seriously Injured) fatality rate per million inhabitants against EU-27 averages, validating the structural urban vs. rural/island severity gap observed in ELSTAT tables.

---

## 4. Crowdsourced Telemetry Specification: Waze for Cities (CCP)

* **Specification**: Waze Partner Hub / Waze for Cities Data Feed (Alerts & Jams JSON/XML specification).
* **Availability & Legal Status**:
  * Real-time and historical Waze feeds are restricted to approved municipal and public-sector partners under the *Waze for Cities* terms of service and cannot be redistributed publicly without authorization.
  * **Crucial Scientific Distinction**: Because no public, verified historical archive of pre-collision Waze telemetry (`flow_speed_kmh`, `jam_level`, `hazard_density`) is openly licensed for Greece 2021–2025, **this project does not present Waze metrics as empirical findings**.
  * Instead, `src/road_safety/waze_ablation.py` provides an opt-in **schema adapter and controlled ablation harness** that strictly preserves target variable integrity (`y` is never altered) and clearly separates verified ELSTAT baseline results from simulated/external telemetry experiments.

---

## 5. Synthetic Benchmark Dataset (`data/synthetic_benchmark_collisions.csv`)

* **Purpose**: Because ELSTAT public releases are aggregated at the `Region × Month` level, a separate, explicitly labeled synthetic event-level dataset (`synthetic_benchmark_collisions.csv`) is provided strictly to test and validate the event-level scikit-learn classification pipeline, unit tests, and SHAP explainability modules.
* **Labeling Policy**: All outputs, plots, and dashboard views using this file carry an explicit `[SYNTHETIC BENCHMARK — CALIBRATED TO ELSTAT MARGINALS]` badge and are never conflated with official empirical microdata.
