# Data Dictionary & Feature Availability Audit (`DATA_DICTIONARY.md`)

A central requirement in road safety machine learning is defining the **Unit of Observation**, the **Target Variable**, and **Feature Availability at Prediction Time** to prevent data leakage.

---

## 1. Official ELSTAT Regional Monthly Panel (`data/elstat_regional_monthly_panel_2021_2025.csv`)

* **Unit of Observation**: One NUTS-2 Greek Region in one calendar month (`Region × Year × Month`, $N = 780$ rows for 13 regions × 60 months, 2021–2025).
* **Primary Classification Target (`target_high_ksi`)**: Binary indicator (`1` if the region-month `ksi_severity_rate >= 0.21`, `0` otherwise), representing a **Severe Regional KSI Regime** (at least 21% of injury collisions resulting in death or serious injury; positive class prevalence = `45.26%`).
* **Continuous Rate (`ksi_severity_rate`)**: Ratio of fatal + serious injury collisions (`ksi_accidents`) to `total_accidents` in that region-month.

### Schema & Pre-Period Leakage Audit

| Column Name | Type | Description | Available Prior to Target Month $t$? | Role in ML Pipeline |
| :--- | :--- | :--- | :---: | :--- |
| `year` | `int` | Calendar year (`2021`–`2025`) | Yes | Temporal split index (`train <= 2023`, `val == 2024`, `test == 2025`) |
| `month` | `int` | Calendar month (`1`–`12`) | Yes | Input Feature (`X`) + Cyclical `month_sin`, `month_cos` |
| `region` | `str` | Canonical Greek NUTS-2 Region (13 categories) | Yes | Input Feature (`X`) |
| `urban_ratio` | `float` | Structural proportion of urban road network collisions in the region | Yes | Input Feature (`X`) |
| `motorcycle_share` | `float` | Structural share of two-wheeler involvement in the region | Yes | Input Feature (`X`) |
| `lag_1m_ksi_rate` | `float` | Regional `ksi_severity_rate` at $t - 1$ month (computed strictly chronologically) | Yes | Engineered Lag Feature (`X`) |
| `lag_2m_ksi_rate` | `float` | Regional `ksi_severity_rate` at $t - 2$ months | Yes | Engineered Lag Feature (`X`) |
| `lag_12m_ksi_rate` | `float` | Regional `ksi_severity_rate` at $t - 12$ months (same month previous year) | Yes | Engineered Lag Feature (`X`) |
| `rolling_3m_ksi_rate` | `float` | Mean `ksi_severity_rate` over $[t-3, t-1]$ months | Yes | Engineered Lag Feature (`X`) |
| `yoy_ksi_rate_diff` | `float` | Year-over-year lagged KSI rate shift ($\text{rate}_{t-1} - \text{rate}_{t-12}$) | Yes | Engineered Lag Feature (`X`) |
| `lag_1m_total_accidents` | `float` | Total collision volume at $t - 1$ month | Yes | Engineered Lag Feature (`X`) |
| `lag_12m_total_accidents` | `float` | Total collision volume at $t - 12$ months | Yes | Engineered Lag Feature (`X`) |
| `total_accidents` | `int` | Total injury collisions recorded during month $t$ | **No (Contemporaneous)** | **Excluded from `X` (Leakage Guard)** |
| `ksi_accidents` | `int` | Sum of `fatal_accidents + serious_accidents` in month $t$ | **No (Target Numerator)** | **Excluded from `X` (Direct Leakage)** |
| `fatal_accidents` | `int` | Collisions resulting in death (within 30 days) in month $t$ | **No (Post-Period Outcome)** | **Excluded from `X` (Direct Leakage)** |
| `serious_accidents` | `int` | Collisions resulting in serious injury in month $t$ | **No (Post-Period Outcome)** | **Excluded from `X` (Direct Leakage)** |
| `slight_accidents` | `int` | Collisions resulting in slight injury in month $t$ | **No (Post-Period Outcome)** | **Excluded from `X` (Direct Leakage)** |
| `ksi_severity_rate` | `float` | `ksi_accidents / total_accidents` in month $t$ | **No (Target Variable)** | **Excluded from `X` (Direct Leakage)** |
| `target_high_ksi` | `int` | Binary indicator `1[ksi_severity_rate >= 0.21]` | **No (Target Variable)** | **Target (`y`) — Immutable across all feature pipelines** |

---

## 2. Official ELSTAT National Annual Series (`data/elstat_national_annual_2015_2024.csv`)

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `year` | `int` | Calendar year (`2015`–`2024`) |
| `total_injury_accidents` | `int` | Official ELSTAT count of road traffic collisions resulting in death or injury |
| `killed_30d` | `int` | Persons killed within 30 days of the collision |
| `seriously_injured` | `int` | Persons sustaining serious injuries (hospitalization / severe trauma) |
| `slightly_injured` | `int` | Persons sustaining slight injuries |
| `total_casualties` | `int` | Sum of `killed_30d + seriously_injured + slightly_injured` |
| `ksi_casualties` | `int` | Sum of `killed_30d + seriously_injured` |
| `ksi_per_100_accidents` | `float` | Ratio `(ksi_casualties / total_injury_accidents) * 100` |
