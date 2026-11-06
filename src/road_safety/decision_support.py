"""
Human-in-the-Loop (HITL) Prescriptive Decision Support Engine (`decision_support.py`)
=====================================================================================
Implements an EU AI Act (Article 14 — Human Oversight) compliant Decision Recommender
for critical road-safety moments across the 13 Greek NUTS-2 regions:
  1. fuses ML KSI forecasts (`regional_forecast_next_12m.csv`) with independent field
     KPIs (`regional_climate_telematics_kpis_nuts2.csv`: NTUA/Trendline helmet & speed
     compliance, OSeven harsh braking, ERA5 rainfall, and EKAB EMS response times).
  2. evaluates the Operational Interventions Catalog (`operational_interventions_catalog.csv`)
     under user-defined constraints (available budget, deployment time window, crisis trigger).
  3. produces 3 transparent, ranked Decision Options (Option A: Max Life-Saving,
     Option B: Optimal ROI / Cost-Effectiveness, Option C: Immediate 24-72h Tactical Blitz)
     with plain-language explanations so the human commander makes the final decision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

SOCIOECONOMIC_COST_PER_KSI_EUR_K = 1150.0  # €1.15M = €1,150K per KSI (EU SafetyCube)


def diagnose_critical_moment(
    fc_row: pd.Series,
    ext_row: pd.Series,
    crisis_override: str = "AUTO",
) -> Dict[str, object]:
    """
    Synthesizes ML forecast metrics and non-ELSTAT field indicators to diagnose
    why a specific region-month represents a critical decision moment.
    """
    month = int(fc_row["month"])
    is_summer = month in (6, 7, 8, 9)
    is_winter_rain = month in (10, 11, 12, 1, 2, 3)

    ksi_rate_pct = float(fc_row["predicted_ksi_rate_pct"])
    severe_prob_pct = float(fc_row["severe_regime_prob_pct"])
    urban_ratio = float(fc_row["urban_ratio"])
    moto_share = float(fc_row["motorcycle_share"])

    helmet_pct = float(ext_row.get("field_helmet_compliance_pct", 70.0))
    speed_comp_pct = float(ext_row.get("rural_speed_compliance_pct", 62.0))
    harsh_braking = float(ext_row.get("telematics_harsh_braking_per_100km", 14.0))
    tourism_mult = float(ext_row.get("summer_tourism_pressure_index", 1.5))
    rain_mm = float(ext_row.get("winter_rainfall_mm", 350.0))
    ems_min = float(ext_row.get("ems_median_response_min", 16.0))

    # Compute normalized mechanism weights (0.2 to 2.0)
    w_tourism_moto = (
        (1.35 if is_summer else 0.65)
        * (moto_share / 0.38)
        * ((100.0 - helmet_pct) / 28.0)
        * np.clip(tourism_mult / 2.2, 0.7, 1.8)
    )
    w_ems = np.clip((ems_min / 15.0) * (ksi_rate_pct / 18.0), 0.5, 2.0)
    w_rural_speed = np.clip(((1.0 - urban_ratio) / 0.55) * ((100.0 - speed_comp_pct) / 36.0), 0.4, 2.0)
    w_weather = np.clip((1.4 if is_winter_rain else 0.65) * (rain_mm / 380.0), 0.4, 1.9)
    w_urban_vru = np.clip((urban_ratio / 0.65) * (harsh_braking / 14.0), 0.4, 2.0)

    if crisis_override == "TOURISM_MOTO":
        w_tourism_moto *= 1.65
        w_ems *= 1.35
    elif crisis_override == "WEATHER_RAIN":
        w_weather *= 1.85
        w_rural_speed *= 1.25
    elif crisis_override == "RURAL_SPEED":
        w_rural_speed *= 1.70
        w_ems *= 1.25
    elif crisis_override == "URBAN_VRU":
        w_urban_vru *= 1.75

    weights = {
        "TOURISM_MOTO": round(float(w_tourism_moto), 3),
        "EMS_GOLDEN_HOUR": round(float(w_ems), 3),
        "RURAL_SPEED": round(float(w_rural_speed), 3),
        "WEATHER_RAIN": round(float(w_weather), 3),
        "URBAN_VRU": round(float(w_urban_vru), 3),
    }

    triggers = []
    if helmet_pct < 68.0:
        triggers.append(f"Χαμηλή επιτόπια χρήση κράνους ({helmet_pct:.1f}% vs 94.8% ΕΕ-27)")
    if ems_min >= 18.0:
        triggers.append(f"Καθυστέρηση απόκρισης ΕΚΑΒ ({ems_min:.1f} min > Golden Hour)")
    if is_summer and tourism_mult >= 2.2:
        triggers.append(f"Θερινή τουριστική υπερφόρτωση δικτύου ({tourism_mult:.1f}×)")
    if speed_comp_pct < 60.0 and urban_ratio < 0.65:
        triggers.append(f"Χαμηλή τήρηση ορίων ταχύτητας στις επαρχιακές οδούς ({speed_comp_pct:.1f}%)")
    if is_winter_rain and rain_mm >= 420:
        triggers.append(f"Υψηλή χειμερινή/φθινοπωρινή βροχόπτωση ({int(rain_mm)} mm) & ολισθηρότητα")
    if urban_ratio >= 0.72:
        triggers.append(f"Υψηλός μητροπολιτικός φόρτος ({urban_ratio*100:.0f}% αστικό) & {harsh_braking:.1f} απότομα φρεναρίσματα/100km")
    if not triggers:
        triggers.append(f"Πρόβλεψη KSI {ksi_rate_pct:.1f}% (Πιθανότητα έξαρσης {severe_prob_pct:.1f}%)")

    criticality_index = float(
        np.clip(
            0.45 * severe_prob_pct
            + 1.6 * ksi_rate_pct
            + 0.8 * max(0.0, ems_min - 11.0)
            + 0.4 * max(0.0, 80.0 - helmet_pct),
            15.0,
            99.0,
        )
    )

    if criticality_index >= 75.0 or severe_prob_pct >= 75.0:
        urgency_level = "🔴 ΚΡΙΣΙΜΗ ΕΠΙΧΕΙΡΗΣΙΑΚΗ ΚΑΤΑΣΤΑΣΗ (RED ALERT)"
    elif criticality_index >= 55.0 or severe_prob_pct >= 50.0:
        urgency_level = "🟠 ΥΨΗΛΗ ΕΠΙΚΙΝΔΥΝΟΤΗΤΑ (HIGH PRIORITY)"
    else:
        urgency_level = "🟡 ΠΡΟΛΗΠΤΙΚΗ ΕΤΟΙΜΟΤΗΤΑ (TACTICAL WATCH)"

    return {
        "criticality_index": round(criticality_index, 1),
        "urgency_level": urgency_level,
        "mechanism_weights": weights,
        "triggers": triggers,
        "helmet_pct": helmet_pct,
        "speed_comp_pct": speed_comp_pct,
        "ems_min": ems_min,
        "tourism_mult": tourism_mult,
        "rain_mm": rain_mm,
        "harsh_braking": harsh_braking,
    }


def score_interventions_for_context(
    catalog_df: pd.DataFrame,
    fc_row: pd.Series,
    diagnosis: Dict[str, object],
    horizon_filter: str = "ALL",
) -> pd.DataFrame:
    """
    Scores each intervention in `catalog_df` for the given region-month crisis context,
    computing contextual effectiveness, expected KSI reduction, and benefit-cost ROI.
    """
    df = catalog_df.copy()
    if horizon_filter == "24H_ONLY":
        df = df[df["deployment_horizon"] == "24H_CRITICAL"].copy()
    elif horizon_filter == "UP_TO_SEASONAL":
        df = df[df["deployment_horizon"].isin(["24H_CRITICAL", "SEASONAL_30D"])].copy()

    weights: Dict[str, float] = diagnosis["mechanism_weights"]  # type: ignore[assignment]
    # Annualized baseline KSI context for ROI comparison (monthly * 12 seasonal equivalent or annual regional baseline)
    monthly_ksi = float(fc_row["expected_ksi_accidents"])
    planning_ksi_pool = max(12.0, monthly_ksi * 12.0)

    suitability_list = []
    eff_reduction_list = []
    ksi_saved_list = []
    benefit_eur_k_list = []
    roi_ratio_list = []
    why_recommended_list = []

    reason_templates = {
        "TOURISM_MOTO": f"Στοχεύει άμεσα στη χαμηλή χρήση κράνους ({diagnosis['helmet_pct']:.1f}%) και τη θερινή πίεση ({diagnosis['tourism_mult']:.1f}×).",
        "EMS_GOLDEN_HOUR": f"Μειώνει τον διάμεσο χρόνο απόκρισης ΕΚΑΒ ({diagnosis['ems_min']:.1f} λεπτά) εντός της Χρυσής Ώρας.",
        "RURAL_SPEED": f"Ανακόπτει τις υπερβάσεις ταχύτητας στο επαρχιακό δίκτυο (συμμόρφωση μόλις {diagnosis['speed_comp_pct']:.1f}%).",
        "WEATHER_RAIN": f"Αντισταθμίζει τον κίνδυνο ολισθηρότητας (χειμερινή βροχόπτωση {int(diagnosis['rain_mm'])} mm).",
        "URBAN_VRU": f"Προστατεύει πεζούς και δίκυκλα σε ζώνες με {diagnosis['harsh_braking']:.1f} απότομα φρεναρίσματα/100km.",
    }

    for _, row in df.iterrows():
        mech = str(row["target_mechanism"])
        w = float(weights.get(mech, 1.0))
        suitability = float(np.clip(55.0 + 25.0 * w, 35.0, 99.0))
        eff_red_pct = float(np.clip(float(row["max_ksi_reduction_pct"]) * np.clip(w, 0.55, 1.35), 3.0, 32.0))
        ksi_saved = round(planning_ksi_pool * (eff_red_pct / 100.0), 1)
        benefit_k = round(ksi_saved * SOCIOECONOMIC_COST_PER_KSI_EUR_K, 1)
        cost_k = max(1.0, float(row["unit_cost_eur_k"]))
        roi = round(benefit_k / cost_k, 1)

        suitability_list.append(round(suitability, 1))
        eff_reduction_list.append(round(eff_red_pct, 2))
        ksi_saved_list.append(ksi_saved)
        benefit_eur_k_list.append(benefit_k)
        roi_ratio_list.append(roi)
        why_recommended_list.append(reason_templates.get(mech, str(row["operational_description_el"])))

    df["suitability_score"] = suitability_list
    df["contextual_ksi_reduction_pct"] = eff_reduction_list
    df["estimated_annual_ksi_saved"] = ksi_saved_list
    df["socioeconomic_benefit_eur_k"] = benefit_eur_k_list
    df["roi_benefit_cost_ratio"] = roi_ratio_list
    df["ai_rationale_el"] = why_recommended_list

    return df.sort_values(["suitability_score", "roi_benefit_cost_ratio"], ascending=[False, False]).reset_index(drop=True)


def evaluate_intervention_portfolio(
    scored_df: pd.DataFrame,
    selected_ids: List[str],
    fc_row: pd.Series,
) -> Dict[str, object]:
    """
    Computes multiplicative counterfactual impact (ERSO / SafetyCube standard)
    for any human-selected or AI-recommended subset of interventions.
    """
    chosen = scored_df[scored_df["intervention_id"].isin(selected_ids)].copy()
    base_ksi_rate = float(fc_row["predicted_ksi_rate_pct"])
    monthly_ksi = float(fc_row["expected_ksi_accidents"])
    annual_equiv_ksi = max(12.0, monthly_ksi * 12.0)

    if chosen.empty:
        return {
            "selected_count": 0,
            "total_cost_eur_k": 0.0,
            "total_personnel": 0,
            "combined_ksi_reduction_pct": 0.0,
            "new_ksi_rate_pct": round(base_ksi_rate, 2),
            "monthly_ksi_saved": 0.0,
            "annual_equiv_ksi_saved": 0.0,
            "socioeconomic_benefit_eur_m": 0.0,
            "portfolio_roi": 0.0,
            "selected_names": [],
            "agencies_involved": [],
        }

    # Multiplicative residual risk with diminishing returns floor (0.42)
    residual = 1.0
    for red_pct in chosen["contextual_ksi_reduction_pct"]:
        residual *= 1.0 - (float(red_pct) / 100.0)
    residual = float(np.clip(residual, 0.42, 1.0))
    combined_red_pct = (1.0 - residual) * 100.0

    new_rate = base_ksi_rate * residual
    monthly_saved = monthly_ksi * (1.0 - residual)
    annual_saved = annual_equiv_ksi * (1.0 - residual)
    total_cost_k = float(chosen["unit_cost_eur_k"].sum())
    benefit_k = annual_saved * SOCIOECONOMIC_COST_PER_KSI_EUR_K
    benefit_m = benefit_k / 1000.0
    roi = benefit_k / max(1.0, total_cost_k)

    return {
        "selected_count": int(len(chosen)),
        "total_cost_eur_k": round(total_cost_k, 1),
        "total_personnel": int(chosen["personnel_units"].sum()),
        "combined_ksi_reduction_pct": round(combined_red_pct, 1),
        "new_ksi_rate_pct": round(new_rate, 2),
        "monthly_ksi_saved": round(monthly_saved, 1),
        "annual_equiv_ksi_saved": round(annual_saved, 1),
        "socioeconomic_benefit_eur_m": round(benefit_m, 2),
        "portfolio_roi": round(roi, 1),
        "selected_names": chosen["name_el"].tolist(),
        "agencies_involved": sorted(chosen["lead_agency_el"].unique().tolist()),
    }


def generate_three_hitl_options(
    scored_df: pd.DataFrame,
    fc_row: pd.Series,
    max_budget_eur_k: float,
) -> Dict[str, Dict[str, object]]:
    """
    Builds 3 distinct Prescriptive AI Options under the human's budget constraint:
      - Option A: Maximum Life-Saving Impact within Budget
      - Option B: Highest Cost-Benefit ROI (Lean & High-Efficiency Package)
      - Option C: Immediate 24-72h Field Mobilization (Zero-Delay Tactical Blitz)
    """
    # Option A: Greedy selection by contextual_ksi_reduction_pct within max_budget_eur_k
    opt_a_ids: List[str] = []
    rem_budget = max_budget_eur_k
    for _, r in scored_df.sort_values("contextual_ksi_reduction_pct", ascending=False).iterrows():
        c = float(r["unit_cost_eur_k"])
        if c <= rem_budget:
            opt_a_ids.append(str(r["intervention_id"]))
            rem_budget -= c
    if not opt_a_ids and not scored_df.empty:
        cheapest = scored_df.sort_values("unit_cost_eur_k").iloc[0]
        opt_a_ids = [str(cheapest["intervention_id"])]

    # Option B: Top 3 highest ROI interventions within 65% of max_budget_eur_k
    opt_b_ids: List[str] = []
    rem_b = max(60.0, max_budget_eur_k * 0.65)
    for _, r in scored_df.sort_values("roi_benefit_cost_ratio", ascending=False).iterrows():
        c = float(r["unit_cost_eur_k"])
        if c <= rem_b and len(opt_b_ids) < 3:
            opt_b_ids.append(str(r["intervention_id"]))
            rem_b -= c
    if not opt_b_ids:
        opt_b_ids = opt_a_ids[:2]

    # Option C: Strictly 24H_CRITICAL tactical actions within budget
    opt_c_ids: List[str] = []
    rem_c = max_budget_eur_k
    fast_df = scored_df[scored_df["deployment_horizon"] == "24H_CRITICAL"].sort_values(
        "suitability_score", ascending=False
    )
    for _, r in fast_df.iterrows():
        c = float(r["unit_cost_eur_k"])
        if c <= rem_c:
            opt_c_ids.append(str(r["intervention_id"]))
            rem_c -= c
    if not opt_c_ids and not fast_df.empty:
        opt_c_ids = [str(fast_df.iloc[0]["intervention_id"])]

    return {
        "OPTION_A": {
            "code": "Επιλογή Α (Πρόταση Μέγιστης Ασφάλειας)",
            "title": "🛡️ Μέγιστη Αποτροπή Θυμάτων KSI ( εντός Προϋπολογισμού )",
            "subtitle": "Εξαντλεί έξυπνα τον διαθέσιμο προϋπολογισμό για τη μέγιστη δυνατή μείωση θανατηφόρων/σοβαρών ατυχημάτων.",
            "ids": opt_a_ids,
            "metrics": evaluate_intervention_portfolio(scored_df, opt_a_ids, fc_row),
        },
        "OPTION_B": {
            "code": "Επιλογή Β (Πρόταση Βέλτιστου ROI)",
            "title": "⚖️ Βέλτιστη Σχέση Κόστους–Οφέλους (High-Efficiency ROI)",
            "subtitle": "Επιλέγει τις 3 πιο αποδοτικές παρεμβάσεις με το χαμηλότερο δημοσιονομικό κόστος ανά σωζόμενη ζωή.",
            "ids": opt_b_ids,
            "metrics": evaluate_intervention_portfolio(scored_df, opt_b_ids, fc_row),
        },
        "OPTION_C": {
            "code": "Επιλογή Γ (Άμεση Κρίση 24–72 Ωρών)",
            "title": "⚡ Ταχεία Επιχειρησιακή Κινητοποίηση Πεδίου (24–72h Blitz)",
            "subtitle": "Ενεργοποιεί αποκλειστικά δράσεις άμεσης αντίδρασης (Τροχαία, ΕΚΑΒ, Δυναμικά Όρια VMS) χωρίς αναμονή έργων.",
            "ids": opt_c_ids,
            "metrics": evaluate_intervention_portfolio(scored_df, opt_c_ids, fc_row),
        },
    }
