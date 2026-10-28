"""
Streamlit Interactive Road Safety ML Dashboard (`streamlit_app.py`)
===================================================================
Week 7 Deliverable:
  - Displays verified historical ELSTAT statistics (2015-2024 national & 2021-2025 regional panel).
  - Presents out-of-time 2025 benchmark metrics across Dummy, Decision Tree, Random Forest,
    HistGradientBoosting, and XGBoost models.
  - Visualizes global permutation importance, regional calibration audits, and the Week 4 Waze
    Ablation Study.
  - Clearly distinguishes Empirical Official ELSTAT Data from Synthetic Event-Level Benchmark scenarios.

Run with:
  streamlit run app/streamlit_app.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

st.set_page_config(
    page_title="Road Safety ML Greece — Explainable AI Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Road Safety ML Greece: Explainable Accident Severity Analysis")
st.caption(
    "Reproducible Machine Learning & Statistical Analysis based on Hellenic Statistical Authority (ELSTAT SDT04) Open Data"
)

# Data Provenance & Scientific Integrity Banner
st.info(
    "**Data Provenance & Scientific Integrity Notice:** "
    "**Track A** uses official aggregated ELSTAT regional monthly statistics (13 NUTS-2 regions, 2021–2025). "
    "**Track B** evaluates event-level classification pipelines on an explicitly labeled synthetic benchmark calibrated "
    "to ELSTAT marginal distributions (because row-level DOTA police microdata are restricted under statistical confidentiality). "
    "No simulated or crowdsourced Waze telemetry is presented as empirical field measurements."
)

# Load datasets and benchmark reports
nat_path = DATA_DIR / "elstat_national_annual_2015_2024.csv"
reg_path = DATA_DIR / "elstat_regional_monthly_panel_2021_2025.csv"
bench_path = REPORTS_DIR / "benchmark_comparison.csv"
imp_path = REPORTS_DIR / "feature_importance.csv"
sub_path = REPORTS_DIR / "regional_subgroup_audit.csv"

nat_df = pd.read_csv(nat_path) if nat_path.exists() else pd.DataFrame()
reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
bench_df = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()
imp_df = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
sub_df = pd.read_csv(sub_path) if sub_path.exists() else pd.DataFrame()

# Sidebar Filters
st.sidebar.header("Navigation & Filters")
section = st.sidebar.radio(
    "Select View",
    [
        "1. Official ELSTAT Overview (Greece)",
        "2. Model Benchmark & Waze Ablation",
        "3. Explainable AI & Regional Audit",
        "4. Interactive Scenario Explorer (Demo)",
    ],
)

if section == "1. Official ELSTAT Overview (Greece)":
    st.subheader("1. Verified National & Regional Road Safety Trends (ELSTAT SDT04)")
    st.markdown("`[DATA BADGE: OFFICIAL AGGREGATED ELSTAT STATISTICS]`")

    if not nat_df.empty:
        latest = nat_df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("2024 Injury Accidents (Greece)", f"{int(latest['total_injury_accidents']):,}")
        c2.metric("2024 Killed (30-Day Def.)", f"{int(latest['killed_30d']):,}")
        c3.metric("2024 Seriously Injured", f"{int(latest['seriously_injured']):,}")
        c4.metric("2024 KSI per 100 Accidents", f"{latest['ksi_per_100_accidents']:.2f}%")

        st.markdown("#### National Annual Series (ELSTAT 2015–2024)")
        st.dataframe(nat_df, use_container_width=True)

    if not reg_df.empty:
        st.markdown("#### Regional Monthly Panel Explorer (13 NUTS-2 Regions)")
        selected_regions = st.multiselect(
            "Filter Greek NUTS-2 Regions:",
            options=sorted(reg_df["region"].unique()),
            default=["Attica (Αττική)", "Central Macedonia (Κεντρική Μακεδονία)", "Crete (Κρήτη)", "Peloponnese (Πελοπόννησος)"],
        )
        filtered_reg = reg_df[reg_df["region"].isin(selected_regions)] if selected_regions else reg_df
        reg_agg = (
            filtered_reg.groupby("region", as_index=False)
            .agg(
                total_accidents=("total_accidents", "sum"),
                ksi_accidents=("ksi_accidents", "sum"),
                fatal_accidents=("fatal_accidents", "sum"),
                serious_accidents=("serious_accidents", "sum"),
                mean_ksi_rate=("ksi_severity_rate", "mean"),
            )
            .sort_values("total_accidents", ascending=False)
        )
        reg_agg["mean_ksi_rate_pct"] = (reg_agg["mean_ksi_rate"] * 100).round(2)
        st.dataframe(reg_agg, use_container_width=True)

    fig_paradox = FIGURES_DIR / "elstat_regional_paradox_greece.png"
    if fig_paradox.exists():
        st.image(str(fig_paradox), caption="Urban Volume vs. Rural/Island Severity in Greece (ELSTAT 2021–2025)")

elif section == "2. Model Benchmark & Waze Ablation":
    st.subheader("2. Out-of-Time Model Evaluation (2021–2023 Train | 2024 Val | 2025 Test)")
    if not bench_df.empty:
        st.markdown("#### Track A: Official ELSTAT Regional Panel (`target_high_ksi` Regime Classification)")
        st.markdown("`[DATA BADGE: OFFICIAL AGGREGATED ELSTAT PANEL — STRICTLY LAGGED FEATURES]`")
        ta = bench_df[bench_df["track"].str.contains("Track A")]
        st.dataframe(
            ta[["model", "roc_auc", "roc_auc_ci_95", "pr_auc", "pr_auc_ci_95", "f1", "precision", "recall", "brier_score", "tn", "fp", "fn", "tp"]],
            use_container_width=True,
        )

        st.markdown("#### Track B: Event-Level Collision Benchmark & Week 4 Waze Ablation Study")
        st.markdown("`[DATA BADGE: SYNTHETIC BENCHMARK CALIBRATED TO ELSTAT MARGINALS — TARGET y IMMUTABLE]`")
        tb = bench_df[bench_df["track"].str.contains("Track B")]
        st.dataframe(
            tb[["model", "roc_auc", "roc_auc_ci_95", "pr_auc", "pr_auc_ci_95", "f1", "precision", "recall", "brier_score", "tn", "fp", "fn", "tp"]],
            use_container_width=True,
        )

        st.warning(
            "**Week 4 Waze Ablation Finding:** When target immutability (`assert_target_immutable`) is enforced so that "
            "`target_ksi` is never overwritten during feature engineering, adding heuristic/simulated Waze proxy columns "
            "(`waze_flow_speed_kmh`, `waze_jam_level`) provides zero artificial lift over the clean baseline model. "
            "Real predictive gain from crowdsourced telemetry requires verified historical pre-collision partner feeds."
        )

    fig_roc = FIGURES_DIR / "benchmark_roc_pr_curves.png"
    if fig_roc.exists():
        st.image(str(fig_roc), caption="Out-of-Time 2025 ROC and Precision-Recall Curves Across Benchmark Models")

elif section == "3. Explainable AI & Regional Audit":
    st.subheader("3. Explainability, Feature Attribution & Subgroup Calibration")
    col1, col2 = st.columns(2)
    with col1:
        fig_imp = FIGURES_DIR / "feature_importance_and_ablation.png"
        if fig_imp.exists():
            st.image(str(fig_imp), caption="Permutation Feature Importance on Held-Out 2025 Test Set")
    with col2:
        fig_cal = FIGURES_DIR / "regional_subgroup_calibration.png"
        if fig_cal.exists():
            st.image(str(fig_cal), caption="Actual vs. Predicted KSI Rate by Greek Region (2025 Test Set)")

    if not sub_df.empty:
        st.markdown("#### Regional Subgroup Performance Table (2025 Test Set)")
        st.dataframe(sub_df, use_container_width=True)

elif section == "4. Interactive Scenario Explorer (Demo)":
    st.subheader("4. Interactive Risk Factor Sensitivity Explorer")
    st.markdown("`[DEMO MODE — NON-OPERATIONAL EDUCATIONAL SIMULATION]`")
    st.caption(
        "Explore how physical road environment and vulnerable road user factors shift relative KSI severity risk in the baseline model."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        area = st.selectbox("Area Type", ["Urban (Κατοικημένη Περιοχή)", "Rural (Εκτός Κατοικημένης)"])
        speed = st.select_slider("Posted Speed Limit (km/h)", options=[30, 50, 70, 90, 110, 130], value=50)
    with c2:
        light = st.selectbox("Lighting Conditions", ["Daylight", "Dark — Street Lights Lit", "Dark — Unlit Rural Road"])
        moto = st.checkbox("Motorcycle / Two-Wheeler Involved", value=False)
    with c3:
        ped = st.checkbox("Pedestrian Involved", value=False)
        heavy = st.checkbox("Heavy Goods Vehicle / Bus Involved", value=False)

    is_rural = 1 if "Rural" in area else 0
    is_unlit = 1 if "Unlit" in light else 0
    logit_est = (
        -3.05
        + 0.88 * is_rural
        + 0.82 * int(moto)
        + 0.68 * int(ped)
        + 0.016 * (speed - 50.0)
        + 0.58 * is_unlit
        + 0.38 * int(heavy)
    )
    prob_est = float(1.0 / (1.0 + np.exp(-logit_est)))

    st.metric("Estimated Conditional KSI Probability P(KSI | Injury Crash)", f"{prob_est:.1%}")
    st.progress(min(max(prob_est, 0.01), 0.99))
    st.caption(
        "Note: This probability is conditional on an injury collision having occurred and represents relative physical risk factors (kinetic energy & rider vulnerability), not causal treatment effects."
    )
