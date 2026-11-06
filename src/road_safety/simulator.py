"""
Counterfactual Policy Intervention Simulator & Geospatial NUTS-2 Engine (`simulator.py`)
========================================================================================
Provides:
  1. Official NUTS-2 geospatial coordinates (`GREEK_REGIONS_GEODATA`) for all 13 Greek
     administrative regions to power static and interactive geospatial risk mapping.
  2. Evidence-based Counterfactual Policy Intervention Simulator (`simulate_policy_interventions`)
     grounded in EU Road Safety Observatory (ERSO), SafetyCube, and NTUA (ΕΜΠ) Crash
     Modification Factors (CMFs).
  3. Multi-scenario Policy Package evaluation (`evaluate_policy_packages`) and publication
     visualizations (`save_policy_simulation_figure`, `save_geospatial_risk_map`).
"""

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Official NUTS-2 codes and geographic centroids (lat, lon) for all 13 Greek Regions
GREEK_REGIONS_GEODATA: Dict[str, Dict[str, object]] = {
    "Attica (Αττική)": {
        "nuts2_code": "EL30",
        "lat": 37.9838,
        "lon": 23.7275,
        "zone_type": "Metropolitan Urban",
    },
    "Central Macedonia (Κεντρική Μακεδονία)": {
        "nuts2_code": "EL52",
        "lat": 40.6401,
        "lon": 22.9444,
        "zone_type": "Metropolitan Urban",
    },
    "Western Greece (Δυτική Ελλάδα)": {
        "nuts2_code": "EL63",
        "lat": 38.2466,
        "lon": 21.7346,
        "zone_type": "Mainland Corridor",
    },
    "Crete (Κρήτη)": {
        "nuts2_code": "EL43",
        "lat": 35.3387,
        "lon": 25.1442,
        "zone_type": "Island / Tourism",
    },
    "Thessaly (Θεσσαλία)": {
        "nuts2_code": "EL61",
        "lat": 39.6390,
        "lon": 22.4191,
        "zone_type": "Mainland Corridor",
    },
    "Peloponnese (Πελοπόννησος)": {
        "nuts2_code": "EL65",
        "lat": 37.5089,
        "lon": 22.3794,
        "zone_type": "Mainland Corridor",
    },
    "Central Greece (Στερεά Ελλάδα)": {
        "nuts2_code": "EL64",
        "lat": 38.8956,
        "lon": 22.4346,
        "zone_type": "Mainland Corridor",
    },
    "South Aegean (Νότιο Αιγαίο)": {
        "nuts2_code": "EL42",
        "lat": 37.4467,
        "lon": 24.9429,
        "zone_type": "Island / Tourism",
    },
    "Eastern Macedonia & Thrace (ΑΜΘ)": {
        "nuts2_code": "EL51",
        "lat": 41.1192,
        "lon": 25.4054,
        "zone_type": "Mainland Corridor",
    },
    "Ionian Islands (Ιόνια Νησιά)": {
        "nuts2_code": "EL62",
        "lat": 39.6243,
        "lon": 19.9217,
        "zone_type": "Island / Tourism",
    },
    "Epirus (Ήπειρος)": {
        "nuts2_code": "EL54",
        "lat": 39.6650,
        "lon": 20.8537,
        "zone_type": "Mountainous / Mainland",
    },
    "North Aegean (Βόρειο Αιγαίο)": {
        "nuts2_code": "EL41",
        "lat": 39.1077,
        "lon": 26.5553,
        "zone_type": "Island / Tourism",
    },
    "Western Macedonia (Δυτική Μακεδονία)": {
        "nuts2_code": "EL53",
        "lat": 40.3007,
        "lon": 21.7889,
        "zone_type": "Mountainous / Mainland",
    },
}

# Standard EU SafetyCube / Hellenic Transport Socio-Economic Valuation per KSI Collision Prevented (€ Millions)
# Blended average accounting for 30-day fatalities and severe life-altering injuries in Greece
SOCIOECONOMIC_VALUE_PER_KSI_EUR_M: float = 1.15


def attach_geospatial_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a regional dataframe containing a `region` column with `nuts2_code`,
    `lat`, `lon`, and `zone_type`.
    """
    out = df.copy()
    out["nuts2_code"] = out["region"].map(lambda r: GREEK_REGIONS_GEODATA.get(str(r), {}).get("nuts2_code", "ELXX"))
    out["lat"] = out["region"].map(lambda r: float(GREEK_REGIONS_GEODATA.get(str(r), {}).get("lat", 38.0)))
    out["lon"] = out["region"].map(lambda r: float(GREEK_REGIONS_GEODATA.get(str(r), {}).get("lon", 23.7)))
    out["zone_type"] = out["region"].map(lambda r: GREEK_REGIONS_GEODATA.get(str(r), {}).get("zone_type", "Mixed"))
    return out


def simulate_policy_interventions(
    forecast_df: pd.DataFrame,
    speed_cameras_intensity: float = 0.0,
    helmet_sobriety_intensity: float = 0.0,
    rural_barriers_intensity: float = 0.0,
    urban_30kmh_intensity: float = 0.0,
    ems_golden_hour_intensity: float = 0.0,
) -> pd.DataFrame:
    """
    Applies a counterfactual Safe System Crash Modification Factor (CMF) model
    to the 2026 prospective regional forecast (`forecast_df`).

    All 5 intensity parameters must lie in [0.0, 1.0], where:
      - 0.0 = Business as Usual (0% additional deployment)
      - 1.0 = Full benchmark deployment (100% target coverage under Vision Zero 2030)

    Empirical CMF parameterization (ERSO / Elvik / SafetyCube):
      1. `speed_cameras_intensity`: Automated speed & section control reduces KSI severity
         rate by up to 16% on rural/interurban roads and 10% on urban arterials, plus a
         6% reduction in total injury collisions.
      2. `helmet_sobriety_intensity`: Mandatory helmet enforcement & nighttime sobriety
         checkpoints reduce KSI severity in proportion to regional `motorcycle_share`
         (up to 22% effect on two-wheeler KSI, amplified by 1.25x during summer tourism months).
      3. `rural_barriers_intensity`: Median separation (2+1 barriers) & forgiving roadsides
         reduce KSI severity in proportion to rural road share (`1 - urban_ratio`) by up to 18%.
      4. `urban_30kmh_intensity`: 30 km/h urban zones & protected pedestrian crossings
         reduce urban collision volume by up to 12% and KSI severity by up to 14%
         in proportion to `urban_ratio`.
      5. `ems_golden_hour_intensity`: Dynamic EMS (ΕΚΑΒ) mobile staging reduces fatal/serious
         outcome conversion by up to 9% (higher impact in rural & summer island settings).
    """
    for name, val in [
        ("speed_cameras_intensity", speed_cameras_intensity),
        ("helmet_sobriety_intensity", helmet_sobriety_intensity),
        ("rural_barriers_intensity", rural_barriers_intensity),
        ("urban_30kmh_intensity", urban_30kmh_intensity),
        ("ems_golden_hour_intensity", ems_golden_hour_intensity),
    ]:
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"Policy lever `{name}` must be in [0.0, 1.0]; got {val}")

    sim = attach_geospatial_metadata(forecast_df)

    u_ratio = sim["urban_ratio"].astype(float).to_numpy()
    r_ratio = 1.0 - u_ratio
    m_share = sim["motorcycle_share"].astype(float).to_numpy()
    summer_boost = np.where(sim["is_summer"].astype(int).to_numpy() == 1, 1.25, 1.0)

    # Multiplicative Crash Modification Factors (CMFs) on KSI Severity Rate
    cmf_speed_sev = 1.0 - speed_cameras_intensity * (0.16 * r_ratio + 0.10 * u_ratio)
    cmf_helmet_sev = 1.0 - helmet_sobriety_intensity * (0.22 * m_share * summer_boost)
    cmf_rural_sev = 1.0 - rural_barriers_intensity * (0.18 * r_ratio)
    cmf_urban_sev = 1.0 - urban_30kmh_intensity * (0.14 * u_ratio)
    cmf_ems_sev = 1.0 - ems_golden_hour_intensity * (0.09 * (0.5 + 0.5 * r_ratio) * summer_boost)

    combined_severity_cmf = np.clip(
        cmf_speed_sev * cmf_helmet_sev * cmf_rural_sev * cmf_urban_sev * cmf_ems_sev,
        0.35,
        1.0,
    )

    # Multiplicative Crash Modification Factors (CMFs) on Total Injury Collision Volume
    cmf_speed_vol = 1.0 - speed_cameras_intensity * 0.06
    cmf_helmet_vol = 1.0 - helmet_sobriety_intensity * (0.05 * summer_boost)
    cmf_urban_vol = 1.0 - urban_30kmh_intensity * (0.12 * u_ratio)
    combined_volume_cmf = np.clip(cmf_speed_vol * cmf_helmet_vol * cmf_urban_vol, 0.50, 1.0)

    base_rate = sim["predicted_ksi_rate"].astype(float).to_numpy()
    base_total_acc = sim["expected_total_accidents"].astype(float).to_numpy()
    base_ksi_acc = sim["expected_ksi_accidents"].astype(int).to_numpy()

    sim_rate = np.clip(base_rate * combined_severity_cmf, 0.01, 1.0)
    sim_total_acc = np.maximum(1, np.round(base_total_acc * combined_volume_cmf).astype(int))

    # When all policy intensities are 0, preserve exact baseline integer KSI accidents
    is_zero_policy = (
        speed_cameras_intensity == 0.0
        and helmet_sobriety_intensity == 0.0
        and rural_barriers_intensity == 0.0
        and urban_30kmh_intensity == 0.0
        and ems_golden_hour_intensity == 0.0
    )
    if is_zero_policy:
        sim_ksi_acc = base_ksi_acc.copy()
    else:
        raw_sim_ksi = np.round(base_ksi_acc * combined_severity_cmf * combined_volume_cmf).astype(int)
        sim_ksi_acc = np.minimum(base_ksi_acc, np.maximum(0, raw_sim_ksi))

    sim["baseline_ksi_rate_pct"] = np.round(base_rate * 100.0, 2)
    sim["simulated_ksi_rate"] = np.round(sim_rate, 4)
    sim["simulated_ksi_rate_pct"] = np.round(sim_rate * 100.0, 2)
    sim["simulated_total_accidents"] = sim_total_acc
    sim["baseline_ksi_accidents"] = base_ksi_acc
    sim["simulated_ksi_accidents"] = sim_ksi_acc
    sim["ksi_accidents_prevented"] = np.maximum(0, base_ksi_acc - sim_ksi_acc)
    sim["relative_ksi_rate_reduction_pct"] = np.round((1.0 - combined_severity_cmf) * 100.0, 2)
    sim["socioeconomic_benefit_eur_m"] = np.round(
        sim["ksi_accidents_prevented"] * SOCIOECONOMIC_VALUE_PER_KSI_EUR_M, 2
    )

    return sim


def evaluate_policy_packages(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates 5 structured national/regional policy packages across all 13 Greek NUTS-2 regions
    for the full 2026 forecast horizon.
    """
    packages = [
        {
            "package_name": "0. Baseline (Business as Usual 2026)",
            "focus_area": "No new interventions",
            "speed_cameras": 0.0,
            "helmet_sobriety": 0.0,
            "rural_barriers": 0.0,
            "urban_30kmh": 0.0,
            "ems_golden_hour": 0.0,
        },
        {
            "package_name": "1. Urban VRU & 30 km/h Shield (Αττική & Θεσσαλονίκη)",
            "focus_area": "Urban 30 km/h zones, red-light/speed cameras & VRU crossings",
            "speed_cameras": 0.50,
            "helmet_sobriety": 0.40,
            "rural_barriers": 0.0,
            "urban_30kmh": 0.85,
            "ems_golden_hour": 0.20,
        },
        {
            "package_name": "2. Summer Island & Two-Wheeler Enforcement Blitz",
            "focus_area": "Helmet/alcohol checkpoints, rental fleet audits & island EMS",
            "speed_cameras": 0.35,
            "helmet_sobriety": 0.90,
            "rural_barriers": 0.25,
            "urban_30kmh": 0.20,
            "ems_golden_hour": 0.75,
        },
        {
            "package_name": "3. Interurban Highway & 2+1 Barrier Upgrade",
            "focus_area": "Median separation, forgiving roadsides & section control",
            "speed_cameras": 0.80,
            "helmet_sobriety": 0.30,
            "rural_barriers": 0.85,
            "urban_30kmh": 0.15,
            "ems_golden_hour": 0.60,
        },
        {
            "package_name": "4. Comprehensive Vision Zero Greece 2030 (All 5 Axes)",
            "focus_area": "Full national deployment across all 5 Safe System pillars",
            "speed_cameras": 0.85,
            "helmet_sobriety": 0.85,
            "rural_barriers": 0.80,
            "urban_30kmh": 0.80,
            "ems_golden_hour": 0.80,
        },
    ]

    summary_rows: List[Dict[str, object]] = []
    for pkg in packages:
        sim_df = simulate_policy_interventions(
            forecast_df,
            speed_cameras_intensity=float(pkg["speed_cameras"]),
            helmet_sobriety_intensity=float(pkg["helmet_sobriety"]),
            rural_barriers_intensity=float(pkg["rural_barriers"]),
            urban_30kmh_intensity=float(pkg["urban_30kmh"]),
            ems_golden_hour_intensity=float(pkg["ems_golden_hour"]),
        )
        base_ksi = int(sim_df["baseline_ksi_accidents"].sum())
        sim_ksi = int(sim_df["simulated_ksi_accidents"].sum())
        prevented = int(sim_df["ksi_accidents_prevented"].sum())
        rel_red = round((prevented / max(1, base_ksi)) * 100.0, 2)
        mean_rate_pct = round(float(sim_df["simulated_ksi_rate_pct"].mean()), 2)
        benefit_m = round(float(sim_df["socioeconomic_benefit_eur_m"].sum()), 2)

        summary_rows.append(
            {
                "package_name": pkg["package_name"],
                "focus_area": pkg["focus_area"],
                "baseline_annual_ksi_accidents": base_ksi,
                "simulated_annual_ksi_accidents": sim_ksi,
                "annual_ksi_accidents_prevented": prevented,
                "annual_ksi_reduction_pct": rel_red,
                "mean_regional_ksi_rate_pct": mean_rate_pct,
                "estimated_socioeconomic_benefit_eur_m": benefit_m,
            }
        )

    return pd.DataFrame(summary_rows)


def save_policy_simulation_figure(
    forecast_df: pd.DataFrame,
    packages_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Saves a two-panel publication figure showing:
      (Left) Annual KSI Accidents Prevented across the 4 active policy packages.
      (Right) Regional Baseline vs. Vision Zero 2030 KSI Severity Rate (%) across all 13 Greek Regions.
    """
    full_sim = simulate_policy_interventions(
        forecast_df,
        speed_cameras_intensity=0.85,
        helmet_sobriety_intensity=0.85,
        rural_barriers_intensity=0.80,
        urban_30kmh_intensity=0.80,
        ems_golden_hour_intensity=0.80,
    )
    reg_comp = (
        full_sim.groupby("region", as_index=False)
        .agg(
            base_rate=("baseline_ksi_rate_pct", "mean"),
            sim_rate=("simulated_ksi_rate_pct", "mean"),
            prevented=("ksi_accidents_prevented", "sum"),
        )
        .sort_values("base_rate", ascending=True)
    )
    reg_comp["short_region"] = reg_comp["region"].apply(lambda s: s.split(" (")[0])

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6))

    # Left panel: Policy Packages comparison
    active_pkgs = packages_df.iloc[1:].copy()
    short_names = [p.split(" (")[0] for p in active_pkgs["package_name"]]
    bars = axes[0].barh(
        short_names,
        active_pkgs["annual_ksi_accidents_prevented"],
        color=["#2563eb", "#d97706", "#059669", "#dc2626"],
    )
    for bar, ben in zip(bars, active_pkgs["estimated_socioeconomic_benefit_eur_m"]):
        w = bar.get_width()
        axes[0].text(
            w + 4,
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(w)} KSI saved (€{ben:.0f}M)",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[0].set_xlim(0, max(active_pkgs["annual_ksi_accidents_prevented"]) * 1.32)
    axes[0].set_title("Annual KSI Collisions Prevented by Policy Package (2026)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Prevented Fatal & Serious Injury (KSI) Collisions per Year")

    # Right panel: Baseline vs Vision Zero 2030 by Greek Region
    y_pos = np.arange(len(reg_comp))
    axes[1].barh(
        y_pos + 0.18,
        reg_comp["base_rate"],
        height=0.35,
        label="2026 Baseline Forecast KSI Rate (%)",
        color="#ef4444",
    )
    axes[1].barh(
        y_pos - 0.18,
        reg_comp["sim_rate"],
        height=0.35,
        label="Post-Intervention KSI Rate (Vision Zero Package)",
        color="#10b981",
    )
    axes[1].axvline(21.0, color="#7f1d1d", linestyle="--", lw=1.3, label="Severe Regime Threshold (21%)")
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(reg_comp["short_region"])
    axes[1].set_xlabel("Mean Regional KSI Severity Rate (%)")
    axes[1].set_title("Regional KSI Rate Reduction: Baseline vs. 5-Axis Package", fontsize=12, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=9)

    plt.suptitle(
        "Counterfactual Safe System Policy Simulation Across Greece (2026 Horizon)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def save_geospatial_risk_map(forecast_df: pd.DataFrame, output_path: Path) -> None:
    """
    Saves a static geospatial bubble map of the 13 Greek NUTS-2 regions plotted at their
    geographic coordinates, colored by August 2026 Peak KSI Severity Rate (%) and sized
    by Expected KSI Collisions.
    """
    aug = attach_geospatial_metadata(forecast_df[forecast_df["month"] == 8].copy())

    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        aug["lon"],
        aug["lat"],
        s=aug["expected_ksi_accidents"] * 22 + 90,
        c=aug["predicted_ksi_rate_pct"],
        cmap="YlOrRd",
        edgecolors="#1e293b",
        linewidth=1.5,
        alpha=0.9,
        zorder=4,
    )
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("August 2026 Forecasted KSI Severity Rate (%)", fontweight="bold")

    for _, r in aug.iterrows():
        short_name = str(r["region"]).split(" (")[0]
        ax.annotate(
            f"{short_name} ({r['nuts2_code']})\n{r['predicted_ksi_rate_pct']:.1f}% | {int(r['expected_ksi_accidents'])} KSI",
            (float(r["lon"]), float(r["lat"])),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.85),
            zorder=5,
        )

    ax.set_xlim(19.0, 27.5)
    ax.set_ylim(34.6, 41.8)
    ax.set_xlabel("Longitude (°E)", fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontweight="bold")
    ax.set_title(
        "Geospatial NUTS-2 Early-Warning Map of Greece (August 2026 Peak KSI Risk)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)
