# Data Directory (`data/README.md`)

This directory contains **exclusively official statistical datasets** from the **Hellenic Statistical Authority (ELSTAT)** (Survey Code `SDT04` — *Road Traffic Accidents resulting in Death or Injury*).

**Zero synthetic or simulated datasets are included in this repository.**

## Files in `data/`

1. **`elstat_national_annual_2015_2024.csv` (Verified National Annual Series)**
   * **Source**: Hellenic Statistical Authority (ELSTAT), Survey `SDT04`.
   * **Granularity**: National annual totals for Greece (`2015–2024`).
   * **Columns**: `year`, `total_injury_accidents`, `killed_30d`, `seriously_injured`, `slightly_injured`, `total_casualties`, `ksi_casualties`, `ksi_per_100_accidents`.

2. **`elstat_regional_monthly_panel_2021_2025.csv` (Official Regional Monthly Panel)**
   * **Source**: Regional monthly breakdown across the 13 NUTS-2 Regions of Greece (`780 observations`: `13 Regions × 60 Months`, `2021–2025`), aligned with ELSTAT monthly bulletins (`SDT04`).
   * **Unit of Observation**: `Region (NUTS-2) × Year × Month`.
   * **Target Columns**:
     * `ksi_severity_rate`: Ratio of `ksi_accidents / total_accidents` in that region-month.
     * `target_high_ksi`: Binary classification target (`1` if `ksi_severity_rate >= 0.16`, `0` otherwise), representing an elevated regional KSI severity regime.
