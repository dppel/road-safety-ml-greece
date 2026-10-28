# Data Dictionary & Feature Availability Audit (`DATA_DICTIONARY.md`)

A central requirement in road safety machine learning is defining the **Unit of Observation**, the **Target Variable**, and **Feature Availability at Prediction Time** to prevent data leakage.

---

## 1. Track A: Official ELSTAT Regional Panel (`data/elstat_regional_monthly_panel_2021_2025.csv`)

* **Unit of Observation**: One NUTS-2 Greek Region in one calendar month (`Region × Year × Month`, $N = 780$ rows for 13 regions × 60 months, 2021–2025).
* **Primary Target (`target_high_ksi`)**: Binary indicator (`1` if the region-month `ksi_severity_rate >= 0.16`, `0` otherwise), representing an elevated regional KSI severity regime.
* **Continuous Target (`ksi_severity_rate`)**: Ratio of fatal + serious injury collisions (`ksi_accidents`) to `total_accidents` in that region-month.

### Schema & Leakage Audit (Track A)

| Column Name | Type | Description | Available Prior to Target Period? | Role in ML Pipeline |
| :--- | :--- | :--- | :---: | :--- |
| `year` | `int` | Calendar year (2021–2025) | Yes | Temporal split index (`train <= 2023`, `val == 2024`, `test == 2025`) |
| `month` | `int` | Calendar month (1–12) | Yes | Input Feature (`X`) + Cyclical `sin`/`cos` |
| `region` | `str` | Canonical Greek NUTS-2 Region (13 categories) | Yes | Input Feature (`X`) |
| `urban_ratio` | `float` | Structural proportion of urban road network collisions in the region | Yes | Input Feature (`X`) |
| `motorcycle_share` | `float` | Structural share of two-wheeler involvement in the region | Yes | Input Feature (`X`) |
| `lag_1m_ksi_rate` | `float` | Regional `ksi_severity_rate` at $t - 1$ month (computed strictly chronologically) | Yes | Engineered Lag Feature (`X`) |
| `lag_12m_ksi_rate` | `float` | Regional `ksi_severity_rate` at $t - 12$ months (same month previous year) | Yes | Engineered Lag Feature (`X`) |
| `rolling_3m_ksi_rate` | `float` | Mean `ksi_severity_rate` over $[t-3, t-1]$ months | Yes | Engineered Lag Feature (`X`) |
| `lag_1m_total_accidents` | `float` | Total collision volume at $t - 1$ month | Yes | Engineered Lag Feature (`X`) |
| `total_accidents` | `int` | Total injury collisions recorded during month $t$ | **No (Contemporaneous)** | **Excluded from `X` (Leakage Guard)** |
| `ksi_accidents` | `int` | Sum of `fatal_accidents + serious_accidents` in month $t$ | **No (Target Numerator)** | **Excluded from `X` (Direct Leakage)** |
| `fatal_accidents` | `int` | Collisions resulting in death (within 30 days) in month $t$ | **No (Post-Period Outcome)** | **Excluded from `X` (Direct Leakage)** |
| `serious_accidents` | `int` | Collisions resulting in serious injury in month $t$ | **No (Post-Period Outcome)** | **Excluded from `X` (Direct Leakage)** |
| `slight_accidents` | `int` | Collisions resulting in slight injury in month $t$ | **No (Post-Period Outcome)** | **Excluded from `X` (Direct Leakage)** |
| `ksi_severity_rate` | `float` | `ksi_accidents / total_accidents` in month $t$ | **No (Target Variable)** | **Target (`y_reg`) — Excluded from `X`** |
| `target_high_ksi` | `int` | Binary indicator `1[ksi_severity_rate >= 0.16]` | **No (Target Variable)** | **Target (`y_clf`) — Excluded from `X`** |

---

## 2. Track B: Event-Level Collision Schema (`data/synthetic_benchmark_collisions.csv`)

* **Unit of Observation**: A single injury road traffic collision event (calibrated to ELSTAT DOTA marginal distributions for pipeline benchmarking; compatible with UK STATS19 event schemas).
* **Target Variable (`target_ksi`)**:
  * `1` = **KSI (Killed or Seriously Injured)**: At least one person died within 30 days or sustained a serious injury (hospitalization / severe trauma).
  * `0` = **Slight Injury**: All casualties sustained only minor/slight injuries.
  * *Note*: Property-damage-only (PDO) collisions are not recorded in ELSTAT SDT04 statistics and are not part of the sample space.

### Schema & Pre-Collision Availability Audit (Track B)

| Variable | Type | Values / Domain | Pre-Collision / Dispatch Available? | Leakage Status & Pipeline Decision |
| :--- | :--- | :--- | :---: | :--- |
| `collision_year` | `int` | `2021`–`2025` | Yes | Used strictly for chronological Train/Val/Test splitting |
| `region` | `str` | 13 Greek NUTS-2 Regions | Yes | Included in `X` (Categorical) |
| `month` | `int` | `1`–`12` | Yes | Included in `X` (Numeric + Cyclical `month_sin`, `month_cos`) |
| `day_of_week` | `int` | `1` (Mon) – `7` (Sun) | Yes | Included in `X` (Numeric + `is_weekend`) |
| `hour` | `int` | `0`–`23` | Yes | Included in `X` (Numeric + Cyclical `hour_sin`, `hour_cos`, `is_night`) |
| `area_type` | `int` | `1` = Urban (Κατοικημένη περιοχή), `2` = Rural (Εκτός κατοικημένης) | Yes | Included in `X` (Categorical + `is_rural`) |
| `road_type` | `int` | `1`=National Highway, `2`=Motorway, `3`=Provincial, `6`=Municipal/Urban | Yes | Included in `X` (Categorical) |
| `speed_limit` | `float` | `30`, `50`, `70`, `90`, `110`, `130` km/h (`-1`/`99` = Missing) | Yes | Included in `X` (Sentinel values `-1, 99` -> `NaN` -> Median Imputed on Train) |
| `is_junction` | `int` | `0` = Mid-block segment, `1` = Intersection/Junction | Yes | Included in `X` (Binary) |
| `light_conditions` | `int` | `1`=Daylight, `4`=Dark (lit), `6`=Dark (unlit), `-1`/`99`=Missing | Yes | Included in `X` (Sentinel cleaned; `is_dark_unlit` indicator) |
| `weather_conditions` | `int` | `1`=Fine, `2`=Raining, `3`=Snow/Fog, `4`=High winds, `-1`/`99`=Missing | Yes | Included in `X` (Sentinel cleaned) |
| `road_surface_conditions` | `int` | `1`=Dry, `2`=Wet/Damp, `3`=Slippery/Icy, `-1`/`99`=Missing | Yes | Included in `X` (Sentinel cleaned) |
| `number_of_vehicles` | `int` | `1`–`6` vehicles involved | Yes (Dispatch) | Included in `X` (Numeric) |
| `has_motorcycle` | `int` | `0` / `1` (Two-wheeler involved) | Yes (Dispatch) | Included in `X` (Binary + interaction `moto_x_speed`) |
| `has_pedestrian` | `int` | `0` / `1` (Pedestrian involved) | Yes (Dispatch) | Included in `X` (Binary) |
| `has_heavy_vehicle` | `int` | `0` / `1` (HGV / Bus involved) | Yes (Dispatch) | Included in `X` (Binary) |
| `min_driver_age` | `float` | Minimum age among involved drivers (`16`–`88`) | Conditionally (Post-ID) | Included in `X` for post-scene severity analysis; excluded in pure pre-dispatch mode |
| `max_vehicle_age` | `float` | Age of oldest involved vehicle in years (`0`–`28`) | Conditionally (Post-ID) | Included in `X` for vehicle fleet analysis |
| `number_of_casualties` | `int` | Total injured/killed persons | **No (Post-Accident)** | **FORBIDDEN — Blocked by `data_validation.py` leakage guard** |
| `did_police_officer_attend` | `int` | Police attendance flag | **No (Endogenous Response)** | **FORBIDDEN — Blocked by `data_validation.py` leakage guard** |
| `waze_flow_speed_kmh` | `float` | Segment speed 5–15 min prior to crash | **External Only** | **Excluded from Baseline**; isolated in `waze_ablation.py` |
| `waze_jam_level` | `int` | Congestion level `0`–`4` prior to crash | **External Only** | **Excluded from Baseline**; isolated in `waze_ablation.py` |
| `target_ksi` | `int` | `0` = Slight, `1` = Killed/Seriously Injured | **Target (`y`)** | **Target Variable — Immutable across all feature pipelines** |
