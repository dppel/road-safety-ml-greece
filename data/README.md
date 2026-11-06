# Data Overview (`data/`)

This folder holds the empirical datasets used across the training pipeline, regional forecasting, counterfactual simulations, and the Streamlit dashboard.

## Included Datasets

1. **`elstat_national_annual_2015_2024.csv`**
   - **Source**: Hellenic Statistical Authority (ELSTAT), Survey `SDT04` (*Road Traffic Accidents resulting in Death or Injury*).
   - **Scope**: National annual totals for Greece (`2015–2024`).
   - **Columns**: `year`, `total_injury_accidents`, `killed_30d`, `seriously_injured`, `slightly_injured`, `total_ksi_casualties`, `ksi_per_100_accidents`.

2. **`elstat_regional_monthly_panel_2021_2025.csv`**
   - **Source**: ELSTAT monthly regional bulletins (`SDT04`).
   - **Scope**: `780` monthly observations (`13 NUTS-2 Regions × 60 Months`, `2021–2025`).
   - **Target Definitions**:
     - `ksi_severity_rate`: Ratio of `ksi_accidents / total_accidents` in a given region and month.
     - `target_high_ksi`: Binary indicator (`1` when `ksi_severity_rate >= 0.21`, `0` otherwise).

3. **`independent_road_safety_kpis_greece.csv`**
   - **Source**: European Commission Road Safety KPIs (*Baseline* & *Trendline* field studies coordinated in Greece by NTUA / ΕΜΠ) and European Transport Safety Council (ETSC) benchmarks.
   - **Scope**: Field-observed compliance rates (helmet wearing, seatbelt use, speed compliance) comparing Greece against the EU-27 average across road types and user groups.

4. **`regional_climate_telematics_kpis_nuts2.csv`**
   - **Source**: Multi-source regional indicators across the 13 NUTS-2 regions (NTUA field compliance, OSeven smartphone telematics harsh-braking rates, Copernicus ERA5 winter rainfall, and EKAB median EMS response times).
   - **Scope**: Used in the dashboard and the Human-in-the-Loop decision module to contextualize regional risk triggers.

5. **`operational_interventions_catalog.csv`**
   - **Source**: Empirical Crash Modification Factors (CMFs) and unit costs derived from the EU *SafetyCube* Decision Support System, ERSO, and NTUA studies.
   - **Scope**: 10 operational and infrastructure road-safety interventions (`INT_01`–`INT_10`) with deployment horizons, personnel needs, unit costs, and expected KSI reduction ranges.
