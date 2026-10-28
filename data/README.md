# Data Directory (`data/README.md`)

This directory contains the datasets used by **Road Safety ML Greece**. To uphold strict scientific integrity, official aggregated statistics from the Hellenic Statistical Authority (ELSTAT) are separated from synthetic event-level benchmark files.

## Files in `data/`

1. **`elstat_national_annual_2015_2024.csv` (Verified Official Statistics)**
   * **Source**: Hellenic Statistical Authority (ELSTAT), Survey `SDT04` (*Road Traffic Accidents resulting in Death or Injury*).
   * **Granularity**: National annual totals for Greece (2015–2024).
   * **Columns**: `year`, `total_injury_accidents`, `killed_30d`, `seriously_injured`, `slightly_injured`, `total_casualties`, `ksi_casualties`, `ksi_per_100_accidents`.

2. **`elstat_regional_monthly_panel_2021_2025.csv` (Regional Monthly Panel — Track A)**
   * **Source**: Regional monthly breakdown across the 13 NUTS-2 Regions of Greece (780 observations: 13 regions × 60 months, 2021–2025), aligned with ELSTAT monthly bulletins (`SDT04`).
   * **Unit of Observation**: `Region (NUTS-2) × Year × Month`.
   * **Target Columns**:
     * `ksi_severity_rate`: Ratio of `ksi_accidents / total_accidents`.
     * `target_high_ksi`: Binary classification target (`1` if `ksi_severity_rate >= 0.16`, `0` otherwise).

3. **`synthetic_benchmark_collisions.csv` (Synthetic Event-Level Benchmark — Track B)**
   * **Source**: Explicitly labeled synthetic event-level benchmark ($N = 25,000$ rows, 2021–2025) calibrated to ELSTAT regional, seasonal, and road-type marginal distributions.
   * **Purpose**: Used strictly to test event-level classification pipelines, data leakage guards, and explainability modules (`SHAP` / permutation importance) without claiming access to confidential row-level police DOTA records.
   * **Integrity Guarantee**: `target_ksi` is generated strictly from baseline road, temporal, and vehicle characteristics and is **never modified** by downstream feature engineering or Waze ablation scripts.
