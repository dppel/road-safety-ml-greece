# Data Sources & Provenance (`DATA_SOURCES.md`)

This document records the exact provenance, licensing, geographic granularity, and temporal coverage of all data sources included in **Road Safety ML Greece**.

> **Strict Empirical Policy**: This repository operates **exclusively on official, verified statistical tables** from the Hellenic Statistical Authority (ELSTAT). **Zero synthetic, Monte Carlo, or simulated Waze datasets are included or modeled.**

---

## 1. Primary Statistical Source: Hellenic Statistical Authority (ELSTAT)

* **Official Publication**: *Road Traffic Accidents resulting in Death or Injury (Οδικά Τροχαία Ατυχήματα)* — Survey Code `SDT04`.
* **Portal URL**: [https://www.statistics.gr/el/statistics/-/publication/SDT04/-](https://www.statistics.gr/el/statistics/-/publication/SDT04/-)
* **Collection Instrument**: Collected across Greece via the **Road Traffic Accident Bulletin (Δελτίο Οδικού Τροχαίου Ατυχήματος — ΔΟΤΑ)** completed by Traffic Police and Port Authorities whenever a road collision results in at least one injured or killed person.
* **Geographic Coverage**: All 13 NUTS-2 Regions (Περιφέρειες) of Greece.
* **Datasets Included in `data/`**:
  * [`data/elstat_national_annual_2015_2024.csv`](../data/elstat_national_annual_2015_2024.csv): Verified official annual national totals (Accidents, Dead, Seriously Injured, Slightly Injured) from 2015 to 2024.
  * [`data/elstat_regional_monthly_panel_2021_2025.csv`](../data/elstat_regional_monthly_panel_2021_2025.csv): Regional monthly panel (`Region × Year × Month`, $N = 780$ rows) across all 13 Greek NUTS-2 regions (2021–2025).
* **Public Granularity vs. Confidential Microdata**:
  * **Publicly Available (Used Here)**: Aggregated statistical tables broken down by NUTS-2 region, month, year, and severity class.
  * **Restricted / Confidential (Not Used)**: Anonymized event-level microdata (individual DOTA records) require a formal research request to ELSTAT under the Statistical Confidentiality Regulation. Rather than fabricating synthetic microdata, we model the data strictly at its true public observation unit (`Region × Month`).
* **License**: ELSTAT Open Statistical Data Terms of Use (CC BY 4.0 equivalent — attribution required to the Hellenic Statistical Authority).

### Verified Official National Totals (ELSTAT 2015–2024)

| Year | Total Injury Accidents | Killed (30-day definition) | Seriously Injured | Slightly Injured | Total Casualties | KSI per 100 Accidents |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2015** | 11,440 | 793 | 999 | 13,097 | 14,889 | 15.66 |
| **2016** | 11,318 | 824 | 879 | 12,946 | 14,649 | 15.05 |
| **2017** | 10,848 | 731 | 706 | 12,565 | 14,002 | 13.25 |
| **2018** | 10,737 | 700 | 727 | 12,422 | 13,849 | 13.29 |
| **2019** | 10,712 | 688 | 652 | 12,350 | 13,690 | 12.51 |
| **2020** | 9,083 | 584 | 518 | 10,202 | 11,304 | 12.13 |
| **2021** | 10,454 | 624 | 610 | 11,746 | 12,980 | 11.80 |
| **2022** | 10,487 | 654 | 664 | 11,961 | 13,279 | 12.57 |
| **2023** | 10,553 | 646 | 659 | 12,156 | 13,461 | 12.37 |
| **2024** | 11,000 | 664 | 568 | 12,803 | 14,035 | 11.20 |

---

## 2. European Comparative Context: Eurostat (`tran_sf_road`)

* **Portal URL**: [https://ec.europa.eu/eurostat/web/transport/database](https://ec.europa.eu/eurostat/web/transport/database)
* **Datasets**: `tran_sf_roadnu` (Road accidents victims by NUTS-2 regions) and CARE (Community database on Accidents on the Roads in Europe).
* **Usage in Project**: Cross-validates Greece's regional KSI fatality and serious injury patterns across NUTS-2 regions.

---

## 3. Why Crowdsourced Telemetry (Waze) Is Excluded from the Empirical Dataset

* **Availability & Legal Status**: Real-time and historical Waze feeds (`flow_speed_kmh`, `jam_level`, `hazard_density`) are restricted to approved municipal and public-sector partners under the *Waze for Cities* terms of service.
* **Scientific Integrity Decision**: Because no public, verified historical archive of pre-collision Waze telemetry is openly licensed for Greece (2021–2025), **we exclude Waze features completely from our empirical dataset and models** rather than simulating them with random number generators.
