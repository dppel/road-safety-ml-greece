"""
Streamlit Interactive Road Safety ML Dashboard (`streamlit_app.py`)
===================================================================
Week 7 Deliverable (100% Official ELSTAT Data — Zero Synthetic Data):
  - Displays verified historical ELSTAT statistics (2015-2024 national & 2021-2025 regional panel).
  - Presents out-of-time 2025 benchmark metrics across Dummy, Decision Tree, Random Forest,
    HistGradientBoosting, and XGBoost models.
  - Visualizes the Empirical Feature Group Ablation Study, global permutation importance,
    and the 13-Region calibration audit.

Run with:
  streamlit run app/streamlit_app.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

st.set_page_config(
    page_title="Road Safety ML Greece — Official ELSTAT Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Road Safety ML Greece: Explainable Regional KSI Analysis")
st.caption(
    "100% Empirical Machine Learning & Statistical Analysis based on Hellenic Statistical Authority (ELSTAT SDT04) Open Data"
)

st.info(
    "**Data Provenance & Scientific Integrity Guarantee:** "
    "This project operates **exclusively on official ELSTAT statistical tables** "
    "(`2015–2024` national series and `2021–2025` regional monthly panel across all 13 Greek NUTS-2 regions). "
    "**Zero synthetic crash rows or simulated Waze features are used.**"
)

nat_path = DATA_DIR / "elstat_national_annual_2015_2024.csv"
reg_path = DATA_DIR / "elstat_regional_monthly_panel_2021_2025.csv"
bench_path = REPORTS_DIR / "benchmark_comparison.csv"
abl_path = REPORTS_DIR / "ablation_study.csv"
imp_path = REPORTS_DIR / "feature_importance.csv"
sub_path = REPORTS_DIR / "regional_subgroup_audit.csv"

nat_df = pd.read_csv(nat_path) if nat_path.exists() else pd.DataFrame()
reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
bench_df = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()
abl_df = pd.read_csv(abl_path) if abl_path.exists() else pd.DataFrame()
imp_df = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
sub_df = pd.read_csv(sub_path) if sub_path.exists() else pd.DataFrame()

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Select View",
    [
        "1. Official ELSTAT Overview (Greece)",
        "2. 2025 Model Benchmark & Feature Ablation",
        "3. Explainable AI & 13-Region Audit",
    ],
)

if section == "1. Official ELSTAT Overview (Greece)":
    st.subheader("1. Verified National & Regional Road Safety Trends (ELSTAT SDT04)")
    st.markdown("`[DATA PROVENANCE: 100% OFFICIAL ELSTAT STATISTICS — SURVEY SDT04]`")

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
        st.markdown("#### Regional Monthly Panel Explorer (13 NUTS-2 Regions, 2021–2025)")
        selected_regions = st.multiselect(
            "Filter Greek NUTS-2 Regions:",
            options=sorted(reg_df["region"].unique()),
            default=[
                "Attica (Αττική)",
                "Central Macedonia (Κεντρική Μακεδονία)",
                "Crete (Κρήτη)",
                "Peloponnese (Πελοπόννησος)",
                "South Aegean (Νότιο Αιγαίο)",
            ],
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

elif section == "2. 2025 Model Benchmark & Feature Ablation":
    st.subheader("2. Out-of-Time Model Evaluation (2021–2023 Train | 2024 Val | 2025 Test)")
    st.markdown("`[UNIT OF OBSERVATION: GREEK NUTS-2 REGION x MONTH (N_test = 156)]`")

    if not bench_df.empty:
        st.markdown("#### 5-Model Benchmark (`target_high_ksi`: Severe Regional KSI Regime >= 21%)")
        st.dataframe(
            bench_df[
                [
                    "model",
                    "roc_auc",
                    "roc_auc_ci_95",
                    "pr_auc",
                    "pr_auc_ci_95",
                    "f1",
                    "precision",
                    "recall",
                    "brier_score",
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                ]
            ],
            use_container_width=True,
        )

    if not abl_df.empty:
        st.markdown("#### Empirical Feature Group Ablation Study (2025 Test Set)")
        st.dataframe(
            abl_df[["model", "num_features", "roc_auc", "roc_auc_ci_95", "pr_auc", "f1", "precision", "recall"]],
            use_container_width=True,
        )

    fig_roc = FIGURES_DIR / "benchmark_roc_pr_curves.png"
    if fig_roc.exists():
        st.image(str(fig_roc), caption="Out-of-Time 2025 ROC and Precision-Recall Curves on Official ELSTAT Data")

elif section == "3. Explainable AI & 13-Region Audit":
    st.subheader("3. Explainability, Feature Attribution & 13-Region Calibration Audit")
    col1, col2 = st.columns(2)
    with col1:
        fig_imp = FIGURES_DIR / "feature_importance_and_ablation.png"
        if fig_imp.exists():
            st.image(str(fig_imp), caption="Permutation Importance & Feature Group Ablation on 2025 Test Set")
    with col2:
        fig_cal = FIGURES_DIR / "regional_subgroup_calibration.png"
        if fig_cal.exists():
            st.image(str(fig_cal), caption="Actual vs. Predicted Severe KSI Regime Rate by Greek Region (2025 Test Set)")

    if not imp_df.empty:
        st.markdown("#### Global Permutation Feature Importance Table (2025 Test Set)")
        st.dataframe(imp_df, use_container_width=True)

    if not sub_df.empty:
        st.markdown("#### 13-Region Subgroup Calibration & Performance Table (2025 Test Set)")
        st.dataframe(sub_df, use_container_width=True)
