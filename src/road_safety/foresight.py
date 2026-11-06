"""
Strategic Foresight & 10-Year Horizon Scanning Engine (`foresight.py`)
======================================================================
Extends the 12-month empirical forecasting pipeline into a 10-year Strategic
Foresight & Backcasting system (2026–2035) across all 13 Greek NUTS-2 regions:
  1. Models 4 plausible macro-futures ("Cone of Plausibility" 2026–2035):
     - S1: Stress / Drift (Aging vehicle fleet, tourism/climate expansion, micromobility growth)
     - S2: Digital Enforcement & Urban 30 km/h Transition (AI cameras, section control, VRU zones)
     - S3: Fleet Renewal (EU GSR2 ADAS) & Major Highway Corridors (BOAK Crete, Patras-Pyrgos, E65)
     - S4: Full Safe System / Vision Zero Transformation (EU convergence across all 5 pillars)
  2. Quantifies the 6 key Road Safety Megatrends & Wildcards for Greece (2026–2035).
  3. Performs Regional Backcasting to determine when each Greek NUTS-2 region achieves
     the EU -50% KSI reduction target and the 10-year cumulative lives/serious injuries saved.
"""

from pathlib import Path
from typing import Dict, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from road_safety.simulator import SOCIOECONOMIC_VALUE_PER_KSI_EUR_M, attach_geospatial_metadata


# Major infrastructure completion multipliers (BOAK in Crete, Patras-Pyrgos in Western Greece/Peloponnese, E65 in Central Greece/Thessaly)
MAJOR_CORRIDOR_REGIONS = {
    "Crete (Κρήτη)": 1.28,
    "Western Greece (Δυτική Ελλάδα)": 1.25,
    "Peloponnese (Πελοπόννησος)": 1.18,
    "Central Greece (Στερεά Ελλάδα)": 1.18,
    "Thessaly (Θεσσαλία)": 1.15,
}


def get_megatrends_matrix() -> pd.DataFrame:
    """
    Returns the structured Horizon Scanning matrix of 6 key Megatrends & Wildcards
    shaping Greek road safety over the 2026–2035 decade.
    """
    rows = [
        {
            "megatrend_id": "MT_01_MICROMOBILITY_DELIVERY",
            "category": "Αστική Κινητικότητα & Στόλος",
            "megatrend_name_el": "Έκρηξη Μικροκινητικότητας (e-Scooters) & Στόλων Διανομής (Last-Mile Delivery)",
            "direction": "Αυξητικός Κίνδυνος (+)",
            "annual_ksi_drift_pct": +1.4,
            "primary_affected_regions": "Αττική, Κεντρική Μακεδονία, Αστικά Κέντρα",
            "mitigation_lever": "Υποχρεωτικό κράνος/φωτισμός e-scooters, ζώνες 30 km/h & έλεγχος εργοδοτών delivery.",
        },
        {
            "megatrend_id": "MT_02_CLIMATE_TOURISM_EXTENSION",
            "category": "Κλίμα & Τουριστική Έκθεση",
            "megatrend_name_el": "Κλιματική Αλλαγή & Επιμήκυνση Θερινής Τουριστικής Περιόδου (Απρ–Νοέ)",
            "direction": "Αυξητικός Κίνδυνος (+)",
            "annual_ksi_drift_pct": +1.9,
            "primary_affected_regions": "Νότιο Αιγαίο, Κρήτη, Ιόνια Νησιά, Βόρειο Αιγαίο",
            "mitigation_lever": "Επιμήκυνση θερινής ανάπτυξης ΕΚΑΒ/Τροχαίας από Απρίλιο έως Οκτώβριο & έλεγχοι ενοικιαζόμενων δικύκλων.",
        },
        {
            "megatrend_id": "MT_03_DEMOGRAPHIC_AGING",
            "category": "Δημογραφία & Ευαλωτότητα",
            "megatrend_name_el": "Δημογραφική Γήρανση Οδηγών & Πεζών στην Ελληνική Περιφέρεια (>65 ετών)",
            "direction": "Αυξητικός Κίνδυνος (+)",
            "annual_ksi_drift_pct": +0.9,
            "primary_affected_regions": "Ήπειρος, Δυτική Μακεδονία, Πελοπόννησος, Βόρειο Αιγαίο",
            "mitigation_lever": "Φωτεινές διαβάσεις πεζών, ιατρικοί έλεγχοι ικανότητας οδήγησης & «συγχωρητικές» επαρχιακές οδοί.",
        },
        {
            "megatrend_id": "MT_04_ADAS_GSR2_FLEET",
            "category": "Τεχνολογία Οχημάτων (ΕΕ GSR2)",
            "megatrend_name_el": "Διείσδυση Συστημάτων ADAS (Αυτόματο Φρενάρισμα AEB, ISA, Διατήρηση Λωρίδας)",
            "direction": "Μείωση Κινδύνου (-)",
            "annual_ksi_drift_pct": -3.1,
            "primary_affected_regions": "Όλες οι 13 Περιφέρειες (ταχύτερα σε Αττική/Κεντρική Μακεδονία)",
            "mitigation_lever": "Κίνητρα απόσυρσης παλαιών ΙΧ (>17 ετών) και υποχρεωτικό ABS/CBS σε όλα τα δίκυκλα.",
        },
        {
            "megatrend_id": "MT_05_MEGA_INFRASTRUCTURE",
            "category": "Οδικές Υποδομές & Δίκτυα",
            "megatrend_name_el": "Ολοκλήρωση Μεγάλων Οδικών Αξόνων (ΒΟΑΚ Κρήτης, Πατρών–Πύργου, Ε65 & Στηθαία 2+1)",
            "direction": "Μείωση Κινδύνου (-)",
            "annual_ksi_drift_pct": -4.2,
            "primary_affected_regions": "Κρήτη (ΒΟΑΚ), Δυτική Ελλάδα, Πελοπόννησος, Στερεά Ελλάδα, Θεσσαλία",
            "mitigation_lever": "Πλήρης διαχωρισμός κατευθύνσεων κυκλοφορίας και αναβάθμιση επικίνδυνων επαρχιακών κόμβων.",
        },
        {
            "megatrend_id": "MT_06_AI_ENFORCEMENT_ECALL",
            "category": "Ψηφιακή Διακυβέρνηση & ΕΚΑΒ",
            "megatrend_name_el": "Ψηφιακές Κάμερες AI, Section Control & Αυτόματη Τηλεειδοποίηση eCall / ΕΚΑΒ",
            "direction": "Μείωση Κινδύνου (-)",
            "annual_ksi_drift_pct": -4.8,
            "primary_affected_regions": "Όλες οι 13 Περιφέρειες (Αστικές Αρτηρίες & Εθνικοί Άξονες)",
            "mitigation_lever": "Αυτοματοποιημένη βεβαίωση παραβάσεων (ταχύτητα, κράνος, ερυθρός) και μείωση χρόνου ΕΚΑΒ <15 min.",
        },
    ]
    return pd.DataFrame(rows)


def generate_foresight_trajectories(
    forecast_2026_df: pd.DataFrame,
    start_year: int = 2026,
    end_year: int = 2035,
    tourism_climate_shock: float = 1.0,
    adas_fleet_renewal_speed: float = 1.0,
    infra_enforcement_speed: float = 1.0,
) -> pd.DataFrame:
    """
    Generates 10-year annual KSI trajectories (2026–2035) across all 13 Greek NUTS-2 regions
    under 4 strategic foresight scenarios, incorporating customizable megatrend multipliers.
    """
    enriched_2026 = attach_geospatial_metadata(forecast_2026_df)
    reg_base = (
        enriched_2026.groupby(["region", "nuts2_code", "zone_type"], as_index=False)
        .agg(
            base_2026_total_accidents=("expected_total_accidents", "sum"),
            base_2026_ksi_accidents=("expected_ksi_accidents", "sum"),
            base_2026_ksi_rate_pct=("predicted_ksi_rate_pct", "mean"),
            urban_ratio=("urban_ratio", "mean"),
            motorcycle_share=("motorcycle_share", "mean"),
        )
    )

    scenarios = [
        {
            "scenario_id": "S1_STRESS",
            "scenario_name": "1. Αδράνεια & Τουριστική/Κλιματική Πίεση (Stress Scenario)",
            "short_name": "S1: Αδράνεια & Πίεση",
            "color": "#ef4444",
        },
        {
            "scenario_id": "S2_DIGITAL",
            "scenario_name": "2. Ψηφιακή Επιτήρηση AI & Αστικές Ζώνες 30 km/h",
            "short_name": "S2: Ψηφιακή Επιτήρηση",
            "color": "#f59e0b",
        },
        {
            "scenario_id": "S3_TECH_INFRA",
            "scenario_name": "3. Ανανέωση Στόλου (ADAS) & Νέοι Αυτοκινητόδρομοι (ΒΟΑΚ/Ε65)",
            "short_name": "S3: ADAS & Υποδομές",
            "color": "#3b82f6",
        },
        {
            "scenario_id": "S4_VISION_ZERO",
            "scenario_name": "4. Πλήρες Safe System / Vision Zero 2030–2035 (Στόχος ΕΕ -50%)",
            "short_name": "S4: Vision Zero (-50%)",
            "color": "#10b981",
        },
    ]

    years = list(range(start_year, end_year + 1))
    records: List[Dict[str, object]] = []

    for _, reg_row in reg_base.iterrows():
        reg = str(reg_row["region"])
        nuts2 = str(reg_row["nuts2_code"])
        ztype = str(reg_row["zone_type"])
        base_ksi = float(reg_row["base_2026_ksi_accidents"])
        base_rate = float(reg_row["base_2026_ksi_rate_pct"])
        u_ratio = float(reg_row["urban_ratio"])
        m_share = float(reg_row["motorcycle_share"])
        corridor_boost = float(MAJOR_CORRIDOR_REGIONS.get(reg, 1.0))
        is_island = "Island" in ztype

        # Target -50% KSI relative to 2026 baseline
        eu_target_ksi = max(1, int(np.round(base_ksi * 0.50)))

        # First compute S1_STRESS trajectory for this region so other scenarios can compute savings vs Stress
        stress_by_year: Dict[int, int] = {}
        for yr in years:
            t = yr - start_year
            # Islands face higher upward drift from tourism & climate extension if unmitigated
            drift_rate = (0.018 if is_island else 0.009) * float(tourism_climate_shock)
            mult_stress = (1.0 + drift_rate) ** t
            stress_by_year[yr] = max(1, int(np.round(base_ksi * mult_stress)))

        for sc in scenarios:
            sc_id = sc["scenario_id"]
            for yr in years:
                t = yr - start_year
                if t == 0:
                    proj_ksi = int(np.round(base_ksi))
                    proj_rate = round(base_rate, 2)
                else:
                    if sc_id == "S1_STRESS":
                        drift_rate = (0.018 if is_island else 0.009) * float(tourism_climate_shock)
                        factor = (1.0 + drift_rate) ** t
                    elif sc_id == "S2_DIGITAL":
                        # Digital cameras & 30 km/h have strongest impact in urban + high-moto regions
                        ann_red = (0.036 * u_ratio + 0.030 * m_share + 0.018) * float(infra_enforcement_speed)
                        ann_red = min(0.08, max(0.015, ann_red))
                        factor = (1.0 - ann_red) ** t
                    elif sc_id == "S3_TECH_INFRA":
                        # ADAS + major highway upgrades (BOAK Crete, Patras-Pyrgos, E65)
                        ann_red = (0.028 * float(adas_fleet_renewal_speed)) + (
                            0.024 * corridor_boost * float(infra_enforcement_speed)
                        )
                        ann_red = min(0.09, max(0.02, ann_red))
                        factor = (1.0 - ann_red) ** t
                    else:
                        # S4_VISION_ZERO: Full 5-pillar Safe System + ADAS + Infrastructure
                        ann_red = (
                            0.038 * float(infra_enforcement_speed)
                            + 0.026 * float(adas_fleet_renewal_speed)
                            + 0.022 * corridor_boost
                        )
                        ann_red = min(0.12, max(0.045, ann_red))
                        factor = (1.0 - ann_red) ** t

                    proj_ksi = max(1, int(np.round(base_ksi * factor)))
                    proj_rate = round(max(2.0, min(35.0, base_rate * (factor ** 0.78))), 2)

                saved_vs_baseline = max(0, int(np.round(base_ksi)) - proj_ksi)
                saved_vs_stress = max(0, stress_by_year[yr] - proj_ksi)

                records.append(
                    {
                        "year": yr,
                        "region": reg,
                        "nuts2_code": nuts2,
                        "zone_type": ztype,
                        "scenario_id": sc_id,
                        "scenario_name": sc["scenario_name"],
                        "short_scenario": sc["short_name"],
                        "baseline_2026_ksi": int(np.round(base_ksi)),
                        "eu_target_50pct_ksi": eu_target_ksi,
                        "projected_ksi_accidents": proj_ksi,
                        "projected_ksi_rate_pct": proj_rate,
                        "annual_ksi_saved_vs_2026": saved_vs_baseline,
                        "annual_ksi_saved_vs_stress": saved_vs_stress,
                        "annual_socioeconomic_benefit_eur_m": round(
                            saved_vs_baseline * SOCIOECONOMIC_VALUE_PER_KSI_EUR_M, 2
                        ),
                        "achieves_eu_50pct_target": bool(proj_ksi <= eu_target_ksi),
                    }
                )

    return pd.DataFrame(records)


def compute_regional_backcasting_table(trajectories_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the 10-year Regional Backcasting Roadmap (2026–2035) showing for each of the
    13 Greek NUTS-2 regions:
      - 2026 Baseline KSI
      - EU -50% Target KSI
      - 2035 Projected KSI under Vision Zero (S4) vs. Stress (S1)
      - First Year EU -50% Target is Achieved under Vision Zero
      - Cumulative 10-Year KSI Prevented (2026–2035) & Socio-Economic Benefit (€M)
    """
    s4 = trajectories_df[trajectories_df["scenario_id"] == "S4_VISION_ZERO"].copy()
    s1 = trajectories_df[trajectories_df["scenario_id"] == "S1_STRESS"].copy()

    rows: List[Dict[str, object]] = []
    for reg, grp in s4.groupby("region"):
        grp_sorted = grp.sort_values("year")
        base_2026 = int(grp_sorted.iloc[0]["baseline_2026_ksi"])
        target_50 = int(grp_sorted.iloc[0]["eu_target_50pct_ksi"])
        proj_2030 = int(grp_sorted[grp_sorted["year"] == 2030]["projected_ksi_accidents"].iloc[0])
        proj_2035 = int(grp_sorted[grp_sorted["year"] == 2035]["projected_ksi_accidents"].iloc[0])

        s1_reg = s1[s1["region"] == reg].sort_values("year")
        stress_2035 = int(s1_reg[s1_reg["year"] == 2035]["projected_ksi_accidents"].iloc[0])

        hit_years = grp_sorted[grp_sorted["achieves_eu_50pct_target"]]["year"].tolist()
        target_year_str = str(hit_years[0]) if hit_years else ">2035"

        cum_saved_vs_2026 = int(grp_sorted["annual_ksi_saved_vs_2026"].sum())
        cum_saved_vs_stress = int(grp_sorted["annual_ksi_saved_vs_stress"].sum())
        cum_benefit_m = round(float(grp_sorted["annual_socioeconomic_benefit_eur_m"].sum()), 2)
        reduction_2035_pct = round(((base_2026 - proj_2035) / max(1, base_2026)) * 100.0, 1)

        rows.append(
            {
                "region": str(reg),
                "nuts2_code": str(grp_sorted.iloc[0]["nuts2_code"]),
                "zone_type": str(grp_sorted.iloc[0]["zone_type"]),
                "baseline_2026_ksi": base_2026,
                "eu_target_50pct_ksi": target_50,
                "stress_2035_ksi": stress_2035,
                "vision_zero_2030_ksi": proj_2030,
                "vision_zero_2035_ksi": proj_2035,
                "reduction_by_2035_pct": reduction_2035_pct,
                "year_eu_target_achieved": target_year_str,
                "cumulative_10yr_ksi_saved": cum_saved_vs_2026,
                "cumulative_10yr_saved_vs_stress": cum_saved_vs_stress,
                "cumulative_10yr_benefit_eur_m": cum_benefit_m,
            }
        )

    return pd.DataFrame(rows).sort_values("baseline_2026_ksi", ascending=False).reset_index(drop=True)


def save_foresight_cone_figure(
    trajectories_df: pd.DataFrame,
    backcast_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Saves a 2-panel publication figure (`reports/figures/foresight_2026_2035_cone.png`)
    illustrating the 10-Year National Foresight Cone of Plausibility (2026–2035) and
    the 10-Year Cumulative KSI Prevented across the 13 Greek NUTS-2 regions.
    """
    nat_traj = (
        trajectories_df.groupby(["year", "scenario_id", "short_scenario"], as_index=False)
        .agg(
            projected_ksi=("projected_ksi_accidents", "sum"),
            baseline_2026=("baseline_2026_ksi", "sum"),
            eu_target=("eu_target_50pct_ksi", "sum"),
        )
    )

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6))

    palette = {
        "S1: Αδράνεια & Πίεση": "#ef4444",
        "S2: Ψηφιακή Επιτήρηση": "#f59e0b",
        "S3: ADAS & Υποδομές": "#2563eb",
        "S4: Vision Zero (-50%)": "#10b981",
    }

    for sc_label, grp in nat_traj.groupby("short_scenario"):
        grp_s = grp.sort_values("year")
        axes[0].plot(
            grp_s["year"],
            grp_s["projected_ksi"],
            marker="o",
            linewidth=2.6,
            label=sc_label,
            color=palette.get(str(sc_label), "#334155"),
        )

    eu_target_val = int(nat_traj["eu_target"].iloc[0])
    axes[0].axhline(
        eu_target_val,
        color="#047857",
        linestyle="--",
        linewidth=1.8,
        label=f"EU Vision Zero -50% Target ({eu_target_val} KSI)",
    )
    axes[0].set_title(
        "10-Year Strategic Foresight Cone of Plausibility (Greece 2026–2035)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Horizon Year")
    axes[0].set_ylabel("Annual Fatal & Serious Injury (KSI) Collisions")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # Right panel: 10-year cumulative KSI prevented by region under Vision Zero
    bc_sorted = backcast_df.sort_values("cumulative_10yr_ksi_saved", ascending=True).copy()
    bc_sorted["short_reg"] = bc_sorted["region"].apply(lambda s: s.split(" (")[0])
    axes[1].barh(
        bc_sorted["short_reg"],
        bc_sorted["cumulative_10yr_ksi_saved"],
        color="#059669",
    )
    axes[1].set_title(
        "10-Year Cumulative KSI Collisions Prevented by Region (2026–2035)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Cumulative KSI Collisions Prevented (10-Year Vision Zero)")

    plt.suptitle(
        "Road Safety Strategic Foresight & EU Vision Zero Backcasting (Greece 2026–2035)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)
