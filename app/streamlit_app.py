"""
Streamlit Interactive Road Safety ML Dashboard (`streamlit_app.py`)
===================================================================
Week 7 Deliverable (100% Official ELSTAT Data — Zero Synthetic Data):
  - Displays verified historical ELSTAT statistics (2015-2024 national & 2021-2025 regional panel).
  - Presents out-of-time 2025 benchmark metrics across Dummy, Decision Tree, Random Forest,
    HistGradientBoosting, and XGBoost models.
  - Visualizes the Empirical Feature Group Ablation Study, global permutation importance,
    and the 13-Region calibration audit.
  - Provides a 2026 Prospective Regional KSI Forecast & Evidence-Based Policy Action Plan
    across all 13 Greek NUTS-2 Regions.

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
DOCS_DIR = PROJECT_ROOT / "docs"

st.set_page_config(
    page_title="Road Safety ML Greece — Official ELSTAT Portfolio",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Road Safety ML Greece: Explainable Regional KSI Analysis & Forecasting")
st.caption(
    "100% Empirical Machine Learning, Forecasting & Policy Decision-Support based on Hellenic Statistical Authority (ELSTAT SDT04) Open Data"
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
fc_path = REPORTS_DIR / "regional_forecast_next_12m.csv"
policy_doc_path = DOCS_DIR / "POLICY_RECOMMENDATIONS.md"

nat_df = pd.read_csv(nat_path) if nat_path.exists() else pd.DataFrame()
reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
bench_df = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()
abl_df = pd.read_csv(abl_path) if abl_path.exists() else pd.DataFrame()
imp_df = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
sub_df = pd.read_csv(sub_path) if sub_path.exists() else pd.DataFrame()
fc_df = pd.read_csv(fc_path) if fc_path.exists() else pd.DataFrame()

st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Select View",
    [
        "1. Official ELSTAT Overview (Greece)",
        "2. 2025 Model Benchmark & Feature Ablation",
        "3. Explainable AI & 13-Region Audit",
        "4. 2026 Πρόγνωση & Εισήγηση Μέτρων (Forecast & Policy)",
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

elif section == "4. 2026 Πρόγνωση & Εισήγηση Μέτρων (Forecast & Policy)":
    st.subheader("4. 2026 Μηνιαία Πρόγνωση Επικινδυνότητας KSI & Εισήγηση Μέτρων Αντιμετώπισης")
    st.markdown("`[EARLY-WARNING FORECASTING & SAFE SYSTEM POLICY DECISION-SUPPORT]`")

    fig_hm = FIGURES_DIR / "regional_forecast_heatmap.png"
    if fig_hm.exists():
        st.image(
            str(fig_hm),
            caption="12-Month Prospective Regional KSI Severity Rate Forecast (%) Across All 13 Greek Regions (2026)",
        )

    if not fc_df.empty:
        col_m, col_r = st.columns(2)
        month_names = {
            1: "1 - Ιανουάριος",
            2: "2 - Φεβρουάριος",
            3: "3 - Μάρτιος",
            4: "4 - Απρίλιος",
            5: "5 - Μάιος",
            6: "6 - Ιούνιος (Θερινή Περίοδος)",
            7: "7 - Ιούλιος (Θερινή Αιχμή)",
            8: "8 - Αύγουστος (Θερινή Αιχμή)",
            9: "9 - Σεπτέμβριος (Θερινή Περίοδος)",
            10: "10 - Οκτώβριος",
            11: "11 - Νοέμβριος",
            12: "12 - Δεκέμβριος",
        }
        with col_m:
            sel_month = st.selectbox(
                "Επιλογή Μήνα Πρόγνωσης (2026):",
                options=list(month_names.keys()),
                format_func=lambda k: month_names[k],
                index=7,  # Default August
            )
        with col_r:
            sel_reg = st.selectbox(
                "Εστίαση σε Περιφέρεια (προαιρετικό για αναλυτική εισήγηση):",
                options=["Όλες οι Περιφέρειες (Συγκριτικός Πίνακας)"] + sorted(fc_df["region"].unique()),
            )

        month_fc = fc_df[fc_df["month"] == sel_month].sort_values(
            "predicted_ksi_rate_pct", ascending=False
        )

        if sel_reg != "Όλες οι Περιφέρειες (Συγκριτικός Πίνακας)":
            row_sel = month_fc[month_fc["region"] == sel_reg].iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Προβλεπόμενο Ποσοστό KSI (%)", f"{row_sel['predicted_ksi_rate_pct']:.2f}%")
            k2.metric("Πιθανότητα Καθεστώτος Υψηλού KSI (>=21%)", f"{row_sel['severe_regime_prob_pct']:.1f}%")
            k3.metric("Αναμενόμενα Σοβαρά/Θανατηφόρα Ατυχήματα", f"{int(row_sel['expected_ksi_accidents'])}")
            st.warning(f"**Επίπεδο Κινδύνου:** {row_sel['risk_tier']}")
            st.markdown(f"**Διάγνωση Μηχανισμού Κινδύνου:** {row_sel['primary_risk_mechanism']}")
            st.success(f"**Προτεινόμενη Εισήγηση Παρεμβάσεων:** {row_sel['recommended_interventions']}")

        st.markdown(f"#### Πίνακας Πρόγνωσης & Προτεινόμενων Παρεμβάσεων — {month_names[sel_month]} 2026")
        st.dataframe(
            month_fc[
                [
                    "region",
                    "predicted_ksi_rate_pct",
                    "severe_regime_prob_pct",
                    "expected_total_accidents",
                    "expected_ksi_accidents",
                    "risk_tier",
                    "recommended_interventions",
                ]
            ],
            use_container_width=True,
        )

    if policy_doc_path.exists():
        with st.expander("Πλήρης Στρατηγική Εισήγηση Αντιμετώπισης (5 Άξονες Οδικής Ασφάλειας — Vision Zero)", expanded=True):
            st.markdown(policy_doc_path.read_text(encoding="utf-8"))
