"""
Streamlit Interactive Road Safety ML Dashboard (`streamlit_app.py`)
===================================================================
Modern SaaS / Executive Observatory Interface (100% Official ELSTAT Data — Zero Synthetic Data):
  - Dark Glassmorphic Sidebar + Bento-Grid Central Canvas.
  - Built-in Plain-Language Legends & Contextual Glossaries (Υπομνήματα & Επεξηγήσεις Όρων)
    across all views, KPI cards, charts, and tables so users never need external literature.
  - Displays verified historical ELSTAT statistics (2015-2024 national & 2021-2025 regional panel).
  - Presents out-of-time 2025 benchmark metrics across Dummy, Decision Tree, Random Forest,
    HistGradientBoosting, and XGBoost models.
  - Visualizes the Empirical Feature Group Ablation Study, global permutation importance,
    and the 13-Region calibration audit.
  - Provides a 2026 Prospective Regional KSI Forecast & Evidence-Based Policy Action Plan
    across all 13 Greek NUTS-2 Regions.
  - Features an interactive NUTS-2 Geospatial Map & 5-Axis What-If Policy Simulator
    with one-click preset scenarios.

Run with:
  streamlit run app/streamlit_app.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from road_safety.decision_support import (
    diagnose_critical_moment,
    evaluate_intervention_portfolio,
    generate_three_hitl_options,
    score_interventions_for_context,
)
from road_safety.foresight import (
    compute_regional_backcasting_table,
    generate_foresight_trajectories,
    get_megatrends_matrix,
)
from road_safety.simulator import (
    attach_geospatial_metadata,
    simulate_policy_interventions,
)

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"

st.set_page_config(
    page_title="Παρατηρητήριο Οδικής Ασφάλειας Ελλάδας — ML & ΕΛΣΤΑΤ",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Modern SaaS Design System (Sidebar + Central Bento Canvas + Legend Cards)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Soft slate background so white Bento cards pop */
    .stApp {
        background: linear-gradient(180deg, #f1f5f9 0%, #f8fafc 100%) !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 1.1rem !important;
        padding-bottom: 3rem !important;
        max-width: 1480px !important;
    }

    /* ===================================================================
       1. ULTRA-MODERN HERO COMMAND HEADER
       =================================================================== */
    .hero-banner {
        background: radial-gradient(circle at 85% 20%, rgba(56, 189, 248, 0.22) 0%, transparent 45%),
                    linear-gradient(135deg, #090d16 0%, #0f172a 45%, #1e3a8a 100%);
        color: #ffffff;
        padding: 1.75rem 2.2rem;
        border-radius: 20px;
        margin-bottom: 1.35rem;
        box-shadow: 0 16px 36px -8px rgba(15, 23, 42, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.12);
        position: relative;
        overflow: hidden;
    }
    .hero-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-bottom: 0.55rem;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(56, 189, 248, 0.14);
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #7dd3fc;
        font-size: 0.74rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
    }
    .hero-title {
        font-size: 1.85rem;
        font-weight: 800;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.025em;
        color: #ffffff;
    }
    .hero-subtitle {
        font-size: 0.95rem;
        color: #cbd5e1;
        margin: 0 0 1.05rem 0;
        max-width: 960px;
        line-height: 1.55;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
    }
    .hero-badge {
        background: rgba(255, 255, 255, 0.09);
        backdrop-filter: blur(8px);
        color: #f1f5f9;
        padding: 0.32rem 0.85rem;
        border-radius: 999px;
        font-size: 0.77rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.16);
    }

    /* ===================================================================
       2. BENTO KPI CARDS (Central Canvas)
       =================================================================== */
    .bento-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.15rem 1.3rem;
        box-shadow: 0 6px 20px -6px rgba(15, 23, 42, 0.06);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    .bento-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 28px -8px rgba(15, 23, 42, 0.12);
    }
    .bento-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.65rem;
    }
    .bento-icon {
        width: 40px;
        height: 40px;
        border-radius: 11px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }
    .bento-pill {
        font-size: 0.73rem;
        font-weight: 700;
        padding: 0.22rem 0.65rem;
        border-radius: 999px;
    }
    .pill-emerald { background: #ecfdf5; color: #047857; border: 1px solid #a7f3d0; }
    .pill-blue    { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }
    .pill-amber   { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
    .pill-rose    { background: #fef2f2; color: #be123c; border: 1px solid #fecdd3; }
    .bento-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.03em;
        line-height: 1.15;
        margin-bottom: 0.25rem;
    }
    .bento-label {
        font-size: 0.84rem;
        font-weight: 600;
        color: #475569;
    }
    .bento-hint {
        font-size: 0.74rem;
        color: #64748b;
        margin-top: 0.35rem;
        padding-top: 0.35rem;
        border-top: 1px dashed #e2e8f0;
        line-height: 1.35;
    }

    /* ===================================================================
       3. ELEVATED WHITE CONTAINERS, SECTION BANNERS & LEGEND CARDS
       =================================================================== */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 16px !important;
        padding: 1.25rem 1.35rem !important;
        box-shadow: 0 8px 24px -8px rgba(15, 23, 42, 0.05) !important;
        color: #0f172a !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] ol,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] strong,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stMarkdownContainer"] b,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stWidgetLabel"] p,
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stCaptionContainer"] p {
        color: #0f172a !important;
    }
    div.stButton > button[kind="primary"] p {
        color: #ffffff !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] code {
        background: #eff6ff !important;
        color: #1e3a8a !important;
        border: 1px solid #bfdbfe !important;
        padding: 0.12rem 0.42rem !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] blockquote {
        background: #f8fafc !important;
        border-left: 4px solid #2563eb !important;
        padding: 0.65rem 0.9rem !important;
        border-radius: 8px !important;
        color: #1e293b !important;
        margin: 0.4rem 0 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] blockquote p {
        color: #1e293b !important;
        font-weight: 500 !important;
        margin: 0 !important;
    }
    .section-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #2563eb;
        border-radius: 14px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.03);
    }
    .section-header-bar h3 {
        margin: 0 0 0.2rem 0;
        color: #0f172a;
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }
    .section-header-bar p {
        margin: 0;
        color: #64748b;
        font-size: 0.88rem;
    }
    .card-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.65rem;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .card-title-text {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0f172a;
    }
    .card-title-tag {
        font-size: 0.72rem;
        font-weight: 600;
        color: #475569;
        background: #f1f5f9;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
    }

    /* Legend / Plain-Language Explainer Grid */
    .legend-banner {
        background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 1.1rem 1.35rem;
        margin-bottom: 1.2rem;
    }
    .legend-header {
        font-size: 0.92rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }
    .legend-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 0.75rem;
    }
    .legend-item {
        background: #ffffff;
        border: 1px solid #dbeafe;
        border-radius: 11px;
        padding: 0.75rem 0.9rem;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
    }
    .legend-term {
        font-size: 0.83rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.22rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .legend-tag {
        font-size: 0.68rem;
        font-weight: 700;
        padding: 0.1rem 0.45rem;
        border-radius: 6px;
        background: #eff6ff;
        color: #1d4ed8;
    }
    .legend-desc {
        font-size: 0.78rem;
        color: #475569;
        line-height: 1.42;
        margin: 0;
    }

    /* Modern Preset Buttons */
    div.stButton > button {
        border-radius: 11px !important;
        font-weight: 600 !important;
        font-size: 0.84rem !important;
        padding: 0.55rem 0.9rem !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        color: #1e293b !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04) !important;
        transition: all 0.16s ease !important;
    }
    div.stButton > button:hover {
        border-color: #2563eb !important;
        color: #1d4ed8 !important;
        background: #eff6ff !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.12) !important;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.28) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        color: #ffffff !important;
    }

    /* Risk Tier Badges & Policy Action Card */
    .risk-pill {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 0.6rem;
    }
    .risk-critical { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
    .risk-high     { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; }
    .risk-moderate { background: #fefce8; color: #a16207; border: 1px solid #fef08a; }
    .risk-urban    { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }

    .policy-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #10b981;
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-top: 0.8rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 8px 24px -8px rgba(15, 23, 42, 0.06);
    }

    /* ===================================================================
       4. MODERN DARK SAAS SIDEBAR (Left Vertical Navigation Rail)
       =================================================================== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #0f172a 55%, #172554 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1.1rem !important;
    }
    .sb-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.85rem 0.95rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 14px;
        margin-bottom: 1.1rem;
        backdrop-filter: blur(10px);
    }
    .sb-icon {
        width: 42px;
        height: 42px;
        border-radius: 11px;
        background: linear-gradient(135deg, #2563eb 0%, #38bdf8 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.45);
        flex-shrink: 0;
    }
    .sb-brand-title {
        color: #f8fafc;
        font-weight: 800;
        font-size: 0.95rem;
        line-height: 1.2;
        letter-spacing: -0.01em;
    }
    .sb-brand-sub {
        color: #94a3b8;
        font-size: 0.73rem;
        margin-top: 0.18rem;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }
    .sb-dot {
        width: 7px;
        height: 7px;
        background: #22c55e;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 8px #22c55e;
    }
    .sb-section-label {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0.4rem 0 0.45rem 0.25rem;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.42rem !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        border-radius: 11px !important;
        padding: 0.68rem 0.85rem !important;
        margin: 0 !important;
        width: 100% !important;
        transition: all 0.18s ease-in-out !important;
        cursor: pointer !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
        color: #cbd5e1 !important;
        font-size: 0.86rem !important;
        font-weight: 500 !important;
        margin: 0 !important;
        line-height: 1.3 !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        background: rgba(59, 130, 246, 0.16) !important;
        border-color: rgba(96, 165, 250, 0.4) !important;
        transform: translateX(3px);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover p {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%) !important;
        border: 1px solid #60a5fa !important;
        box-shadow: 0 6px 16px -3px rgba(37, 99, 235, 0.55) !important;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    .sb-kpi-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 0.9rem;
    }
    .sb-kpi-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 0.55rem 0.65rem;
    }
    .sb-kpi-val {
        color: #f8fafc;
        font-weight: 800;
        font-size: 0.98rem;
    }
    .sb-kpi-lbl {
        color: #94a3b8;
        font-size: 0.68rem;
        margin-top: 0.1rem;
    }
    .sb-glass-card {
        background: rgba(30, 58, 138, 0.35);
        border: 1px solid rgba(96, 165, 250, 0.25);
        border-radius: 12px;
        padding: 0.85rem 0.9rem;
        color: #e2e8f0;
        font-size: 0.79rem;
        line-height: 1.45;
    }
    .sb-footer {
        color: #64748b;
        font-size: 0.72rem;
        margin-top: 0.9rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_bento_kpi(
    icon: str,
    icon_bg: str,
    label: str,
    value: str,
    pill_text: str,
    pill_class: str = "pill-blue",
    hint: str = "",
) -> str:
    """Returns modern Bento-style KPI card HTML with optional plain-language micro-legend (`hint`)."""
    hint_html = f'<div class="bento-hint">💡 {hint}</div>' if hint else ""
    return f"""
    <div class="bento-card" title="{hint}">
        <div class="bento-top">
            <div class="bento-icon" style="background:{icon_bg};">{icon}</div>
            <span class="bento-pill {pill_class}">{pill_text}</span>
        </div>
        <div class="bento-value">{value}</div>
        <div class="bento-label">{label}</div>
        {hint_html}
    </div>
    """


def style_plotly_chart(
    fig: go.Figure,
    title_text: str = "",
    height: int = 450,
    show_legend: bool = True,
) -> go.Figure:
    """
    Enforces crisp dark-slate (#0f172a) typography for chart titles, axis titles,
    tick labels, legends, and colorbars on white (#ffffff) card backgrounds so
    Streamlit's dark/light theme overrides never make titles invisible.
    """
    layout_kwargs = dict(
        height=height,
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
        showlegend=show_legend,
        margin=dict(r=20, t=68 if title_text else 35, l=20, b=25),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
            bgcolor="rgba(255,255,255,0.9)",
        ),
        coloraxis_colorbar=dict(
            title_font=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
            tickfont=dict(color="#0f172a", size=11, family="Inter, sans-serif"),
        ),
    )
    if title_text:
        layout_kwargs["title"] = dict(
            text=f"<b>{title_text}</b>",
            x=0.01,
            y=0.96,
            xanchor="left",
            yanchor="top",
            font=dict(size=14, color="#0f172a", family="Inter, sans-serif"),
        )
    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        title_font=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
        tickfont=dict(color="#1e293b", size=11, family="Inter, sans-serif"),
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1",
        automargin=True,
    )
    fig.update_yaxes(
        title_font=dict(color="#0f172a", size=12, family="Inter, sans-serif"),
        tickfont=dict(color="#1e293b", size=11, family="Inter, sans-serif"),
        gridcolor="#f1f5f9",
        linecolor="#cbd5e1",
        automargin=True,
    )
    return fig


# ---------------------------------------------------------------------------
# Load Verified Datasets & Reports
# ---------------------------------------------------------------------------
nat_path = DATA_DIR / "elstat_national_annual_2015_2024.csv"
reg_path = DATA_DIR / "elstat_regional_monthly_panel_2021_2025.csv"
bench_path = REPORTS_DIR / "benchmark_comparison.csv"
abl_path = REPORTS_DIR / "ablation_study.csv"
imp_path = REPORTS_DIR / "feature_importance.csv"
sub_path = REPORTS_DIR / "regional_subgroup_audit.csv"
fc_path = REPORTS_DIR / "regional_forecast_next_12m.csv"
pkg_path = REPORTS_DIR / "policy_simulation_scenarios.csv"
ext_kpi_path = DATA_DIR / "independent_road_safety_kpis_greece.csv"
ext_reg_path = DATA_DIR / "regional_climate_telematics_kpis_nuts2.csv"
cat_path = DATA_DIR / "operational_interventions_catalog.csv"
policy_doc_path = DOCS_DIR / "POLICY_RECOMMENDATIONS.md"

nat_df = pd.read_csv(nat_path) if nat_path.exists() else pd.DataFrame()
reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()
bench_df = pd.read_csv(bench_path) if bench_path.exists() else pd.DataFrame()
abl_df = pd.read_csv(abl_path) if abl_path.exists() else pd.DataFrame()
imp_df = pd.read_csv(imp_path) if imp_path.exists() else pd.DataFrame()
sub_df = pd.read_csv(sub_path) if sub_path.exists() else pd.DataFrame()
fc_df = pd.read_csv(fc_path) if fc_path.exists() else pd.DataFrame()
pkg_df = pd.read_csv(pkg_path) if pkg_path.exists() else pd.DataFrame()
ext_kpi_df = pd.read_csv(ext_kpi_path) if ext_kpi_path.exists() else pd.DataFrame()
ext_reg_df = pd.read_csv(ext_reg_path) if ext_reg_path.exists() else pd.DataFrame()
cat_df = pd.read_csv(cat_path) if cat_path.exists() else pd.DataFrame()

# ---------------------------------------------------------------------------
# Hero Command Header Banner
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-top-row">
            <span class="hero-eyebrow">⚡ DECISION-SUPPORT & FORECASTING ENGINE • VISION ZERO 2030</span>
            <span class="hero-badge">🏛️ ΕΛΣΤΑΤ SDT04 Official Open Data</span>
        </div>
        <div class="hero-title">Παρατηρητήριο Οδικής Ασφάλειας Ελλάδας — ML & Πρόγνωση KSI</div>
        <div class="hero-subtitle">
            Διαδραστική πλατφόρμα Μηχανικής Μάθησης, Γεωχωρικής Ανάλυσης Κινδύνου και Προσομοίωσης Παρεμβάσεων
            στις <b>13 Περιφέρειες NUTS-2 της Ελλάδας</b>, με ενσωματωμένα υπομνήματα επεξήγησης όρων σε κάθε ενότητα.
        </div>
        <div class="badge-row">
            <span class="hero-badge">✅ 100% Επίσημα Δεδομένα (Zero Synthetic)</span>
            <span class="hero-badge">📍 13 Περιφέρειες × 60 Μήνες (780 Παρατηρήσεις)</span>
            <span class="hero-badge">🏆 RandomForest Out-of-Time ROC-AUC: 0.8567</span>
            <span class="hero-badge">📖 Ενσωματωμένο Λεξικό Όρων & Υπομνήματα</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Navigation & Summary Card
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sb-brand">
            <div class="sb-icon">🛡️</div>
            <div>
                <div class="sb-brand-title">GREECE ROAD SAFETY</div>
                <div class="sb-brand-sub"><span class="sb-dot"></span> ELSTAT SDT04 • v1.2</div>
            </div>
        </div>
        <div class="sb-section-label">ΚΥΡΙΟ ΜΕΝΟΥ ΑΝΑΛΥΣΗΣ</div>
        """,
        unsafe_allow_html=True,
    )
    section = st.radio(
        "Επιλέξτε Ενότητα:",
        [
            "🗺️ 1. Διαδραστικός Χάρτης & What-If Προσομοιωτής",
            "🚨 2. Πρόγνωση 2026 & Σχέδιο Δράσης ανά Περιφέρεια",
            "📊 3. Επίσημα Στοιχεία ΕΛΣΤΑΤ (Ελλάδα 2015–2025)",
            "🤖 4. Αξιολόγηση Μοντέλων ML & Ablation Study",
            "🔍 5. Ερμηνευσιμότητα (XAI) & Έλεγχος Περιφερειών",
            "🔬 6. Ανεξάρτητες Έρευνες & Ευρωπαϊκά KPIs (ΕΜΠ / ESRA3 / ERA5)",
            "🔭 7. Strategic Foresight 2026–2035 (Vision Zero)",
            "🧠 8. Κέντρο Λήψης Αποφάσεων (Human-in-the-Loop Copilot)",
        ],
        index=0,
        label_visibility="collapsed",
    )
    st.markdown(
        """
        <div class="sb-section-label" style="margin-top:1.1rem;">ΣΥΝΟΨΗ ΣΥΣΤΗΜΑΤΟΣ</div>
        <div class="sb-kpi-grid">
            <div class="sb-kpi-card" title="Οι 13 επίσημες διοικητικές Περιφέρειες της Ελλάδας κατά Eurostat (NUTS-2)">
                <div class="sb-kpi-val">13 NUTS-2</div>
                <div class="sb-kpi-lbl">Περιφέρειες Ελλάδας</div>
            </div>
            <div class="sb-kpi-card" title="Ικανότητα διαχωρισμού επικίνδυνων μηνών (0.50 = τυχαίο, 1.00 = τέλειο)">
                <div class="sb-kpi-val">0.8567</div>
                <div class="sb-kpi-lbl">RF Test ROC-AUC</div>
            </div>
            <div class="sb-kpi-card" title="Αποτρεπόμενα σοβαρά/θανατηφόρα ατυχήματα με πλήρη εφαρμογή Vision Zero">
                <div class="sb-kpi-val">-566 KSI</div>
                <div class="sb-kpi-lbl">Vision Zero Όφελος</div>
            </div>
            <div class="sb-kpi-card" title="Εξοικονόμηση κοινωνικού κόστους (νοσηλείες, ΜΕΘ, απώλεια παραγωγικότητας)">
                <div class="sb-kpi-val">€650.9M</div>
                <div class="sb-kpi-lbl">Ετήσιο Κοιν. Όφελος</div>
            </div>
        </div>
        <div class="sb-glass-card">
            <b style="color:#93c5fd;">📖 Τι σημαίνει KSI;</b><br>
            <b>KSI (Killed or Seriously Injured):</b> Τροχαίο ατύχημα με τουλάχιστον 1 <b>νεκρό</b> (εντός 30 ημερών)
            ή 1 <b>βαριά τραυματία</b>.<br><br>
            <b style="color:#93c5fd;">📌 Το Ελληνικό Παράδοξο:</b><br>
            Η <b>Αττική</b> έχει το <b>53.5%</b> των ατυχημάτων αλλά χαμηλό ποσοστό KSI (~6.5%), ενώ τα νησιά και η περιφέρεια
            έχουν <b>3×–4× βαρύτητα KSI (20%–25%+)</b> λόγω υψηλών ταχυτήτων και δικύκλων.
        </div>
        <div class="sb-footer">
            Πηγή: Ελληνική Στατιστική Αρχή (ΕΛΣΤΑΤ SDT04)<br>Zero Synthetic Data Policy
        </div>
        """,
        unsafe_allow_html=True,
    )
    pdf_guide_path = REPORTS_DIR / "Presentation_Guide_Road_Safety_Dashboard_GR.pdf"
    if pdf_guide_path.exists():
        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        st.download_button(
            label="📥 Λήψη Οδηγού Παρουσίασης (PDF)",
            data=pdf_guide_path.read_bytes(),
            file_name="Οδηγός_Παρουσίασης_Dashboard_Οδικής_Ασφάλειας.pdf",
            mime="application/pdf",
            use_container_width=True,
            help="Κατεβάστε το πλήρες εγχειρίδιο παρουσίασης (PDF) με επεξηγήσεις, πίνακες, διαγράμματα και έτοιμα σημεία ομιλίας για όλες τις 7 καρτέλες.",
        )


# ===========================================================================
# VIEW 1: INTERACTIVE MAP & WHAT-IF POLICY SIMULATOR (Default Landing View!)
# ===========================================================================
if section == "🗺️ 1. Διαδραστικός Χάρτης & What-If Προσομοιωτής":
    st.markdown(
        """
        <div class="section-header-bar">
            <div>
                <h3>🗺️ Γεωχωρικός Χάρτης Κινδύνου NUTS-2 & What-If Προσομοιωτής Μέτρων (2026)</h3>
                <p>Επιλέξτε ένα έτοιμο πακέτο πολιτικής ή ρυθμίστε τους 5 άξονες του Safe System για δυναμική εκτίμηση αποτρεπόμενων KSI.</p>
            </div>
            <span class="card-title-tag">ERSO / SafetyCube CMF Engine</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Plain-Language Legend Box for View 1
    st.markdown(
        """
        <div class="legend-banner">
            <div class="legend-header">📖 Υπόμνημα & Επεξήγηση Όρων (Τι σημαίνει ο κάθε δείκτης χωρίς αναδρομή σε βιβλιογραφία)</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-term"><span>1. Ατύχημα KSI & Ποσοστό KSI (%)</span><span class="legend-tag">Βασικός Δείκτης</span></div>
                    <p class="legend-desc"><b>KSI = Killed or Seriously Injured (Νεκροί ή Βαριά Τραυματίες).</b> Το <i>Ποσοστό KSI (%)</i> δείχνει πόσα στα 100 τροχαία ατυχήματα καταλήγουν σε θάνατο ή βαρύ τραυματισμό (π.χ. 25% = 1 στα 4 ατυχήματα είναι πολύ σοβαρό).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>2. Όριο Υψηλού Κινδύνου (21%)</span><span class="legend-tag">Όριο Συναγερμού</span></div>
                    <p class="legend-desc">Όταν το ποσοστό KSI μιας Περιφέρειας ξεπερνά το <b>21%</b> (περίπου 1 στα 5 ατυχήματα), η Περιφέρεια εισέρχεται σε <b>Καθεστώς Υψηλής Βαρύτητας</b> (κόκκινη διακεκομμένη γραμμή στο γράφημα).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>3. Συντελεστές CMF & What-If</span><span class="legend-tag">Μεθοδολογία ΕΕ</span></div>
                    <p class="legend-desc"><b>CMF (Crash Modification Factor):</b> Εμπειρικοί συντελεστές της ΕΕ (ERSO / SafetyCube) και του ΕΜΠ που υπολογίζουν πόσο μειώνονται τα ατυχήματα όταν εφαρμόζεται ένα μέτρο (0% = καμία νέα παρέμβαση, 100% = πλήρης ανάπτυξη).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>4. Κοινωνικό Όφελος (€1.15M / KSI)</span><span class="legend-tag">Οικονομική Αποτίμηση</span></div>
                    <p class="legend-desc">Σύμφωνα με το ευρωπαϊκό πρότυπο <b>SafetyCube</b>, κάθε σοβαρό/θανατηφόρο ατύχημα (KSI) που αποτρέπεται στην Ελλάδα εξοικονομεί κατά μέσο όρο <b>€1.15 εκατ.</b> (δαπάνες ΜΕΘ, περίθαλψη, απώλεια ΑΕΠ και ανθρώπινο κόστος).</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not fc_df.empty:
        if "sim_speed" not in st.session_state:
            st.session_state.sim_speed = 60
            st.session_state.sim_helmet = 75
            st.session_state.sim_barriers = 50
            st.session_state.sim_urban30 = 65
            st.session_state.sim_ems = 60

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">⚡ Κέντρο Ελέγχου Παρεμβάσεων (5 Άξονες Vision Zero 2030)</span>
                    <span class="card-title-tag">Πατήστε ένα έτοιμο σενάριο ή σύρετε τους ρυθμιστές</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            b1, b2, b3, b4, b5 = st.columns(5)
            if b1.button("🔄 Baseline (0% Νέα Μέτρα)", use_container_width=True, help="Μηδενισμός όλων των νέων μέτρων — δείχνει τι θα συμβεί το 2026 αν δεν ληφθεί καμία επιπλέον δράση."):
                st.session_state.sim_speed = 0
                st.session_state.sim_helmet = 0
                st.session_state.sim_barriers = 0
                st.session_state.sim_urban30 = 0
                st.session_state.sim_ems = 0
            if b2.button("🏙️ Αστική Ασπίδα 30 km/h", use_container_width=True, help="Έμφαση σε Αττική & Θεσσαλονίκη: ζώνες 30 km/h, προστατευμένες διαβάσεις πεζών και κάμερες ερυθρού σηματοδότη."):
                st.session_state.sim_speed = 50
                st.session_state.sim_helmet = 40
                st.session_state.sim_barriers = 0
                st.session_state.sim_urban30 = 85
                st.session_state.sim_ems = 20
            if b3.button("☀️ Θερινό Blitz Νησιών & Μοτο", use_container_width=True, help="Έμφαση σε Νότιο Αιγαίο, Κρήτη & Ιόνιο: εντατικοί έλεγχοι κράνους/αλκοόλ και προ-τοποθέτηση ασθενοφόρων ΕΚΑΒ."):
                st.session_state.sim_speed = 35
                st.session_state.sim_helmet = 90
                st.session_state.sim_barriers = 25
                st.session_state.sim_urban30 = 20
                st.session_state.sim_ems = 75
            if b4.button("🛣️ Υπεραστικοί Άξονες (2+1)", use_container_width=True, help="Έμφαση σε Πελοπόννησο, Στερεά & Δυτική Ελλάδα: διαχωριστικά στηθαία 2+1 και έλεγχος μέσης ταχύτητας."):
                st.session_state.sim_speed = 80
                st.session_state.sim_helmet = 30
                st.session_state.sim_barriers = 85
                st.session_state.sim_urban30 = 15
                st.session_state.sim_ems = 60
            if b5.button("🛡️ Πλήρες Vision Zero 2030", use_container_width=True, type="primary", help="Συνδυαστική εφαρμογή και των 5 αξόνων οδικής ασφάλειας σε όλη την επικράτεια."):
                st.session_state.sim_speed = 85
                st.session_state.sim_helmet = 85
                st.session_state.sim_barriers = 80
                st.session_state.sim_urban30 = 80
                st.session_state.sim_ems = 80

            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                speed_pct = st.slider(
                    "📷 1. Κάμερες Ταχύτητας (%)",
                    min_value=0,
                    max_value=100,
                    key="sim_speed",
                    step=5,
                    help="Section Control: Κάμερες που μετρούν τη μέση ταχύτητα σε ολόκληρο οδικό τμήμα (μειώνουν τη βαρύτητα KSI έως 16% στις υπεραστικές οδούς).",
                )
            with s2:
                helmet_pct = st.slider(
                    "🪖 2. Κράνος & Αλκοτέστ (%)",
                    min_value=0,
                    max_value=100,
                    key="sim_helmet",
                    step=5,
                    help="Έλεγχοι χρήσης κράνους και νυχτερινού αλκοτέστ (22:00-05:00). Μέγιστη απόδοση σε περιφέρειες με υψηλό μερίδιο δικύκλων και θερινό τουρισμό.",
                )
            with s3:
                barriers_pct = st.slider(
                    "🛣️ 3. Στηθαία 2+1 Οδών (%)",
                    min_value=0,
                    max_value=100,
                    key="sim_barriers",
                    step=5,
                    help="Διαχωριστικά στηθαία ασφαλείας ανάμεσα στα ρεύματα κυκλοφορίας επαρχιακών/εθνικών οδών ώστε να εξαλείφονται οι μετωπικές συγκρούσεις.",
                )
            with s4:
                urban30_pct = st.slider(
                    "🚶 4. Ζώνες 30 km/h (%)",
                    min_value=0,
                    max_value=100,
                    key="sim_urban30",
                    step=5,
                    help="Όριο 30 km/h σε κατοικημένες περιοχές και φωτιζόμενες διαβάσεις πεζών (σώζει πεζούς και αναβάτες δικύκλων στις μεγαλουπόλεις).",
                )
            with s5:
                ems_pct = st.slider(
                    "🚑 5. Κινητές Μονάδες ΕΚΑΒ (%)",
                    min_value=0,
                    max_value=100,
                    key="sim_ems",
                    step=5,
                    help="Golden Hour (Χρυσή Ώρα): Στάθμευση ασθενοφόρων κοντά σε επικίνδυνους κόμβους/νησιά ώστε η βοήθεια να φτάνει στα πρώτα κρίσιμα λεπτά.",
                )

            fcol1, fcol2 = st.columns(2)
            with fcol1:
                period_mode = st.selectbox(
                    "📅 Χρονικός Ορίζοντας Προσομοίωσης (2026):",
                    options=[
                        "Ολόκληρο το Έτος 2026 (12 Μήνες)",
                        "Θερινή Περίοδος Αιχμής (Ιούνιος–Σεπτέμβριος 2026)",
                        "Αύγουστος 2026 (Μήνας Κορύφωσης Κινδύνου)",
                    ],
                    help="Επιλέξτε αν θέλετε να δείτε τις επιπτώσεις σε όλο το έτος 2026, μόνο στο τετράμηνο του καλοκαιριού, ή ειδικά τον Αύγουστο.",
                )
            with fcol2:
                map_metric = st.selectbox(
                    "🎨 Μετρική Χρωματισμού Διαδραστικού Χάρτη:",
                    options=[
                        "simulated_ksi_rate_pct",
                        "ksi_accidents_prevented",
                        "baseline_ksi_rate_pct",
                    ],
                    format_func=lambda m: {
                        "simulated_ksi_rate_pct": "Ποσοστό Βαρύτητας KSI μετά τα Μέτρα (%)",
                        "ksi_accidents_prevented": "Αποτρεπόμενα Σοβαρά/Θανατηφόρα Ατυχήματα (KSI)",
                        "baseline_ksi_rate_pct": "Αρχικό Προβλεπόμενο Ποσοστό KSI Baseline (%)",
                    }[m],
                    help="Αλλάζει το χρώμα των κύκλων στον χάρτη της Ελλάδας.",
                )

        sim_full = simulate_policy_interventions(
            fc_df,
            speed_cameras_intensity=speed_pct / 100.0,
            helmet_sobriety_intensity=helmet_pct / 100.0,
            rural_barriers_intensity=barriers_pct / 100.0,
            urban_30kmh_intensity=urban30_pct / 100.0,
            ems_golden_hour_intensity=ems_pct / 100.0,
        )

        if period_mode == "Θερινή Περίοδος Αιχμής (Ιούνιος–Σεπτέμβριος 2026)":
            sim_view = sim_full[sim_full["is_summer"] == 1].copy()
        elif period_mode == "Αύγουστος 2026 (Μήνας Κορύφωσης Κινδύνου)":
            sim_view = sim_full[sim_full["month"] == 8].copy()
        else:
            sim_view = sim_full.copy()

        base_ksi_total = int(sim_view["baseline_ksi_accidents"].sum())
        sim_ksi_total = int(sim_view["simulated_ksi_accidents"].sum())
        prevented_total = int(sim_view["ksi_accidents_prevented"].sum())
        rel_reduction_pct = (prevented_total / max(1, base_ksi_total)) * 100.0
        benefit_total_m = float(sim_view["socioeconomic_benefit_eur_m"].sum())
        mean_sim_rate = float(sim_view["simulated_ksi_rate_pct"].mean())
        mean_base_rate = float(sim_view["baseline_ksi_rate_pct"].mean())

        # Modern Bento KPI Cards Row with Built-In Hints
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                render_bento_kpi(
                    icon="🛡️",
                    icon_bg="#ecfdf5",
                    label="Αποτρεπόμενα Ατυχήματα KSI",
                    value=f"{prevented_total:,}",
                    pill_text=f"-{rel_reduction_pct:.1f}% vs Baseline",
                    pill_class="pill-emerald",
                    hint="Πόσα θανατηφόρα ή σοβαρά ατυχήματα γλιτώνουμε χάρη στα επιλεγμένα μέτρα.",
                ),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                render_bento_kpi(
                    icon="📉",
                    icon_bg="#eff6ff",
                    label=f"Αναμενόμενα KSI (από {base_ksi_total:,})",
                    value=f"{sim_ksi_total:,}",
                    pill_text=f"-{prevented_total:,} Ατυχήματα KSI",
                    pill_class="pill-blue",
                    hint="Ο εκτιμώμενος αριθμός σοβαρών/θανατηφόρων ατυχημάτων μετά την εφαρμογή των μέτρων.",
                ),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                render_bento_kpi(
                    icon="📊",
                    icon_bg="#fffbeb",
                    label=f"Μέσο Περιφερειακό KSI (από {mean_base_rate:.2f}%)",
                    value=f"{mean_sim_rate:.2f}%",
                    pill_text=f"{mean_sim_rate - mean_base_rate:+.2f}% Μονάδες",
                    pill_class="pill-amber",
                    hint="Πόσα στα 100 ατυχήματα καταλήγουν σε θάνατο/βαρύ τραυματισμό κατά μέσο όρο στις 13 Περιφέρειες.",
                ),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                render_bento_kpi(
                    icon="💶",
                    icon_bg="#f0fdf4",
                    label="Κοινωνικοοικονομικό Όφελος",
                    value=f"€{benefit_total_m:,.1f}M",
                    pill_text="€1.15M / Αποτρεπόμενο KSI",
                    pill_class="pill-emerald",
                    hint="Εξοικονόμηση σε νοσοκομειακή περίθαλψη, ΜΕΘ και παραγωγικότητα (πρότυπο ΕΕ SafetyCube).",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

        reg_map_df = (
            sim_view.groupby(["region", "nuts2_code", "lat", "lon", "zone_type"], as_index=False)
            .agg(
                baseline_ksi_rate_pct=("baseline_ksi_rate_pct", "mean"),
                simulated_ksi_rate_pct=("simulated_ksi_rate_pct", "mean"),
                baseline_ksi_accidents=("baseline_ksi_accidents", "sum"),
                simulated_ksi_accidents=("simulated_ksi_accidents", "sum"),
                ksi_accidents_prevented=("ksi_accidents_prevented", "sum"),
                socioeconomic_benefit_eur_m=("socioeconomic_benefit_eur_m", "sum"),
            )
            .sort_values("baseline_ksi_rate_pct", ascending=False)
        )
        reg_map_df["baseline_ksi_rate_pct"] = reg_map_df["baseline_ksi_rate_pct"].round(2)
        reg_map_df["simulated_ksi_rate_pct"] = reg_map_df["simulated_ksi_rate_pct"].round(2)
        reg_map_df["socioeconomic_benefit_eur_m"] = reg_map_df["socioeconomic_benefit_eur_m"].round(2)

        # Dynamic Step-by-Step Operational Roadmap for the Selected What-If Scenario
        scenario_tuple = (speed_pct, helmet_pct, barriers_pct, urban30_pct, ems_pct)
        scenario_names_lookup = {
            (0, 0, 0, 0, 0): "🔄 Baseline (0% Νέα Μέτρα — Σενάριο Αδράνειας)",
            (50, 40, 0, 85, 20): "🏙️ Αστική Ασπίδα 30 km/h (Μητροπολιτικά Κέντρα)",
            (35, 90, 25, 20, 75): "☀️ Θερινό Blitz Νησιών & Μοτοσικλετών",
            (80, 30, 85, 15, 60): "🛣️ Υπεραστικοί Άξονες & Στηθαία (2+1)",
            (85, 85, 80, 80, 80): "🛡️ Πλήρες Vision Zero 2030 (Και οι 5 Άξονες)",
        }
        active_scenario_label = scenario_names_lookup.get(
            scenario_tuple,
            f"🎛️ Προσαρμοσμένο Σενάριο ({speed_pct}% Κάμερες | {helmet_pct}% Κράνος | {barriers_pct}% Στηθαία | {urban30_pct}% 30km/h | {ems_pct}% ΕΚΑΒ)",
        )

        top_saved_df = reg_map_df.sort_values("ksi_accidents_prevented", ascending=False).head(3)
        top_saved_str = ", ".join(
            [
                f"<b>{r['region'].split('(')[-1].replace(')', '')}</b> (-{int(r['ksi_accidents_prevented'])} KSI)"
                for _, r in top_saved_df.iterrows()
                if int(r["ksi_accidents_prevented"]) > 0
            ]
        ) or "Καμία αποτροπή (επιλέξτε ένταση > 0%)"

        high_risk_remaining = int((reg_map_df["simulated_ksi_rate_pct"] >= 21.0).sum())
        high_risk_initial = int((reg_map_df["baseline_ksi_rate_pct"] >= 21.0).sum())

        if scenario_tuple == (0, 0, 0, 0, 0):
            step1_desc = "Δεν έχει ενεργοποιηθεί καμία νέα επιχειρησιακή δράση (0%). Πατήστε ένα από τα <b>4 σενάρια παρέμβασης</b> παραπάνω ή αυξήστε τους ρυθμιστές για να παραχθεί το σχέδιο δράσης."
            step2_desc = "Χωρίς κάμερες μέσης ταχύτητας (0%) και ζώνες 30 km/h (0%), η βαρύτητα KSI παραμένει σταθερή στα επίπεδα αναφοράς."
            step3_desc = f"Χωρίς στηθαία 2+1 (0%), <b>{high_risk_initial} στις 13 Περιφέρειες</b> παραμένουν πάνω από το κρίσιμο όριο του 21% KSI."
            step4_desc = f"Αναμένονται <b>{base_ksi_total:,} ατυχήματα KSI</b> (μηδενική αποτροπή). Επιλέξτε σενάριο για να δείτε τις Περιφέρειες προτεραιότητας."
        else:
            step1_parts = []
            if helmet_pct > 0:
                step1_parts.append(
                    f"<b>ΕΛ.ΑΣ. / Τροχαία ({helmet_pct}% ένταση):</b> Συγκρότηση μικτών κλιμακίων ελέγχου κράνους & νυχτερινού αλκοτέστ (22:00–05:00) στις τουριστικές/νησιωτικές εισόδους."
                )
            if ems_pct > 0:
                step1_parts.append(
                    f"<b>ΕΚΑΒ ({ems_pct}% κάλυψη):</b> Προ-τοποθέτηση κινητών μονάδων & μοτοσικλετών ταχείας απόκρισης σε κόμβους υψηλού KSI για μείωση χρόνου Χρυσής Ώρας."
                )
            step1_desc = " ".join(step1_parts) if step1_parts else "Δεν απαιτείται άμεση κινητοποίηση περιπολιών/ΕΚΑΒ (ρυθμιστές Κράνους & ΕΚΑΒ στο 0%)."

            step2_parts = []
            if speed_pct > 0:
                step2_parts.append(
                    f"<b>Ψηφιακή Τροχαία ({speed_pct}% δίκτυο):</b> Ενεργοποίηση καμερών μέσης ταχύτητας (Section Control) και αυτόματης βεβαίωσης παραβάσεων στους εθνικούς/επαρχιακούς άξονες."
                )
            if urban30_pct > 0:
                step2_parts.append(
                    f"<b>Δήμοι & Περιφέρειες ({urban30_pct}% κάλυψη):</b> Θεσμοθέτηση ζωνών 30 km/h, έξυπνων φωτιζόμενων διαβάσεων LED και καμερών ερυθρού σηματοδότη στα αστικά κέντρα."
                )
            step2_desc = " ".join(step2_parts) if step2_parts else "Δεν περιλαμβάνονται παρεμβάσεις καμερών ταχύτητας ή αστικών ζωνών 30 km/h σε αυτό το σενάριο."

            if barriers_pct > 0:
                step3_desc = (
                    f"<b>Υπ. Υποδομών & Περιφέρειες ({barriers_pct}% ανάπτυξη):</b> Εγκατάσταση μεσαίων διαχωριστικών στηθαίων ασφαλείας (2+1) και αντιολισθηρής ασφάλτου "
                    f"σε υπεραστικά τμήματα, μειώνοντας τις Περιφέρειες υψηλού κινδύνου (&ge;21% KSI) από <b>{high_risk_initial} σε {high_risk_remaining}</b>."
                )
            else:
                step3_desc = (
                    f"Ο ρυθμιστής Στηθαίων 2+1 είναι στο <b>0%</b>. Προτείνεται αύξηση &ge;50% για την εξάλειψη των μετωπικών συγκρούσεων στις <b>{high_risk_remaining} Περιφέρειες</b> που παραμένουν άνω του 21% KSI."
                )

            step4_desc = (
                f"<b>Πρώτες Περιφέρειες Εφαρμογής:</b> {top_saved_str}. "
                f"<b>Στόχος KPI:</b> Μείωση μέσου KSI από <b>{mean_base_rate:.2f}% σε {mean_sim_rate:.2f}%</b>, αποτροπή <b>{prevented_total:,} KSI</b> και εξοικονόμηση <b>€{benefit_total_m:,.1f}M</b>."
            )

        st.markdown(
            f"""
            <div class="legend-banner" style="background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 100%); border-color: #86efac;">
                <div class="legend-header" style="color: #065f46; justify-content: space-between; flex-wrap: wrap;">
                    <span>🧭 Επόμενα Επιχειρησιακά Βήματα Υλοποίησης (Step-by-Step Οδικός Χάρτης)</span>
                    <span class="bento-pill pill-emerald">{active_scenario_label}</span>
                </div>
                <div class="legend-grid">
                    <div class="legend-item" style="border-color: #bbf7d0;">
                        <div class="legend-term"><span>1. Άμεση Κινητοποίηση (0–30 Ημέρες)</span><span class="legend-tag">ΕΛ.ΑΣ. & ΕΚΑΒ</span></div>
                        <p class="legend-desc">{step1_desc}</p>
                    </div>
                    <div class="legend-item" style="border-color: #bfdbfe;">
                        <div class="legend-term"><span>2. Τακτική Ανάπτυξη (1–3 Μήνες)</span><span class="legend-tag">Κάμερες & Δήμοι</span></div>
                        <p class="legend-desc">{step2_desc}</p>
                    </div>
                    <div class="legend-item" style="border-color: #fde68a;">
                        <div class="legend-term"><span>3. Δομικά Έργα (3–12 Μήνες)</span><span class="legend-tag">Υποδομές 2+1</span></div>
                        <p class="legend-desc">{step3_desc}</p>
                    </div>
                    <div class="legend-item" style="border-color: #c7d2fe;">
                        <div class="legend-term"><span>4. Προτεραιοποίηση & Έλεγχος KPI</span><span class="legend-tag">Audit & Στόχοι</span></div>
                        <p class="legend-desc">{step4_desc}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_map, col_chart = st.columns([1.1, 1.0])
        with col_map:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">🗺️ Διαδραστικός Χάρτης NUTS-2 Ελλάδας</span>
                        <span class="card-title-tag">Μέγεθος Κύκλου = Αριθμός KSI | Χρώμα = Επιλεγμένη Μετρική</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                color_scale = "Tealgrn" if map_metric == "ksi_accidents_prevented" else "YlOrRd"
                hover_cols = {
                    "nuts2_code": True,
                    "zone_type": True,
                    "baseline_ksi_rate_pct": True,
                    "simulated_ksi_rate_pct": True,
                    "baseline_ksi_accidents": True,
                    "simulated_ksi_accidents": True,
                    "ksi_accidents_prevented": True,
                    "socioeconomic_benefit_eur_m": True,
                    "lat": False,
                    "lon": False,
                }
                if hasattr(px, "scatter_map"):
                    fig_map = px.scatter_map(
                        reg_map_df,
                        lat="lat",
                        lon="lon",
                        size="baseline_ksi_accidents",
                        color=map_metric,
                        hover_name="region",
                        hover_data=hover_cols,
                        color_continuous_scale=color_scale,
                        size_max=36,
                        zoom=5.35,
                        center={"lat": 38.3, "lon": 23.8},
                        map_style="carto-positron",
                        height=485,
                    )
                else:
                    fig_map = px.scatter_mapbox(
                        reg_map_df,
                        lat="lat",
                        lon="lon",
                        size="baseline_ksi_accidents",
                        color=map_metric,
                        hover_name="region",
                        hover_data=hover_cols,
                        color_continuous_scale=color_scale,
                        size_max=36,
                        zoom=5.35,
                        center={"lat": 38.3, "lon": 23.8},
                        mapbox_style="carto-positron",
                        height=485,
                    )
                fig_map.update_layout(
                    margin={"r": 0, "t": 38, "l": 0, "b": 0},
                    paper_bgcolor="#ffffff",
                    font=dict(family="Inter, sans-serif", color="#0f172a", size=12),
                    title=dict(
                        text="<b>🗺️ Χάρτης Κινδύνου NUTS-2 Ελλάδας (2026)</b>",
                        x=0.01,
                        y=0.98,
                        font=dict(size=13, color="#0f172a"),
                    ),
                    coloraxis_colorbar=dict(
                        title_font=dict(color="#0f172a", size=11),
                        tickfont=dict(color="#0f172a", size=11),
                    ),
                )
                st.plotly_chart(fig_map, use_container_width=True, theme=None)
                st.caption(
                    "💡 **Πώς διαβάζεται ο χάρτης:** Μεγαλύτερος κύκλος σημαίνει περισσότερα απόλυτα ατυχήματα KSI (π.χ. Αττική, Κεντρική Μακεδονία), "
                    "ενώ πιο σκούρο κόκκινο/πορτοκαλί σημαίνει ότι κάθε ατύχημα έχει πολύ μεγαλύτερη πιθανότητα να είναι θανατηφόρο (π.χ. Νότιο Αιγαίο, Κρήτη, Πελοπόννησος)."
                )

        with col_chart:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">📊 Μείωση Βαρύτητας KSI (%) ανά Περιφέρεια</span>
                        <span class="card-title-tag">Κόκκινο = Πριν τα Μέτρα | Πράσινο = Μετά τα Μέτρα</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                chart_df = reg_map_df.sort_values("baseline_ksi_rate_pct", ascending=True).copy()
                chart_df["short_region"] = chart_df["region"].apply(
                    lambda s: s.split("(")[-1].replace(")", "") if "(" in s else s
                )
                fig_bar = go.Figure()
                fig_bar.add_trace(
                    go.Bar(
                        y=chart_df["short_region"],
                        x=chart_df["baseline_ksi_rate_pct"],
                        name="Baseline 2026 KSI (%)",
                        orientation="h",
                        marker_color="#f43f5e",
                        marker_cornerradius=5,
                    )
                )
                fig_bar.add_trace(
                    go.Bar(
                        y=chart_df["short_region"],
                        x=chart_df["simulated_ksi_rate_pct"],
                        name="Μετά τα Μέτρα KSI (%)",
                        orientation="h",
                        marker_color="#10b981",
                        marker_cornerradius=5,
                    )
                )
                fig_bar.add_vline(
                    x=21.0,
                    line_dash="dash",
                    line_color="#991b1b",
                    annotation_text="Όριο Υψηλού KSI (21%)",
                    annotation_font_color="#991b1b",
                )
                fig_bar.update_layout(
                    barmode="group",
                    xaxis=dict(title="Μέσο Ποσοστό Βαρύτητας KSI (%)"),
                    yaxis=dict(title="Περιφέρεια NUTS-2"),
                )
                style_plotly_chart(
                    fig_bar,
                    title_text="📊 Σύγκριση Ποσοστού KSI (%) Πριν & Μετά τα Μέτρα (13 NUTS-2)",
                    height=485,
                    show_legend=True,
                )
                st.plotly_chart(fig_bar, use_container_width=True, theme=None)
                st.caption(
                    "💡 **Πώς διαβάζεται το γράφημα:** Στόχος μας είναι η **πράσινη μπάρα** κάθε Περιφέρειας να υποχωρήσει αριστερά από την κόκκινη διακεκομμένη γραμμή του **21%**."
                )

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">📋 Αναλυτικός Πίνακας Προσομοίωσης ανά Περιφέρεια (13 NUTS-2)</span>
                    <span class="card-title-tag">Περάστε το ποντίκι πάνω από τις επικεφαλίδες για επεξήγηση</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                reg_map_df[
                    [
                        "region",
                        "nuts2_code",
                        "zone_type",
                        "baseline_ksi_rate_pct",
                        "simulated_ksi_rate_pct",
                        "baseline_ksi_accidents",
                        "simulated_ksi_accidents",
                        "ksi_accidents_prevented",
                        "socioeconomic_benefit_eur_m",
                    ]
                ],
                column_config={
                    "region": st.column_config.TextColumn(
                        "Περιφέρεια (NUTS-2)",
                        help="Η διοικητική Περιφέρεια της Ελλάδας.",
                    ),
                    "nuts2_code": st.column_config.TextColumn(
                        "Κωδικός ΕΕ",
                        help="Επίσημος γεωγραφικός κωδικός Eurostat NUTS-2 (π.χ. EL30 = Αττική).",
                    ),
                    "zone_type": st.column_config.TextColumn(
                        "Τύπος Δικτύου",
                        help="Κυρίαρχος χαρακτήρας οδικού δικτύου (Μητροπολιτικό Αστικό, Νησιωτικό/Τουριστικό, ή Υπεραστικός Άξονας).",
                    ),
                    "baseline_ksi_rate_pct": st.column_config.ProgressColumn(
                        "Baseline KSI (%)",
                        format="%.2f%%",
                        min_value=0,
                        max_value=30,
                        help="Προβλεπόμενο ποσοστό θανατηφόρων/σοβαρών ατυχημάτων ΧΩΡΙΣ νέα μέτρα.",
                    ),
                    "simulated_ksi_rate_pct": st.column_config.ProgressColumn(
                        "KSI Μετά τα Μέτρα (%)",
                        format="%.2f%%",
                        min_value=0,
                        max_value=30,
                        help="Νέο μειωμένο ποσοστό KSI ΜΕΤΑ την εφαρμογή των επιλεγμένων μέτρων.",
                    ),
                    "baseline_ksi_accidents": st.column_config.NumberColumn(
                        "Αρχικά KSI",
                        format="%d",
                        help="Αναμενόμενος αριθμός σοβαρών/θανατηφόρων ατυχημάτων χωρίς παρεμβάσεις.",
                    ),
                    "simulated_ksi_accidents": st.column_config.NumberColumn(
                        "Τελικά KSI",
                        format="%d",
                        help="Αναμενόμενος αριθμός σοβαρών/θανατηφόρων ατυχημάτων μετά τις παρεμβάσεις.",
                    ),
                    "ksi_accidents_prevented": st.column_config.NumberColumn(
                        "Αποτρεπόμενα KSI",
                        format="🛡️ %d",
                        help="Αριθμός σοβαρών/θανατηφόρων ατυχημάτων που αποτρέπονται (Αρχικά μείον Τελικά).",
                    ),
                    "socioeconomic_benefit_eur_m": st.column_config.NumberColumn(
                        "Κοινωνικό Όφελος (€M)",
                        format="€%.2fM",
                        help="Κοινωνικοοικονομική εξοικονόμηση σε εκατομμύρια ευρώ (€1.15M ανά αποτρεπόμενο ατύχημα KSI).",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )
            st.download_button(
                label="📥 Λήψη Πίνακα Προσομοίωσης (CSV)",
                data=reg_map_df.to_csv(index=False).encode("utf-8"),
                file_name="greece_2026_what_if_simulation.csv",
                mime="text/csv",
            )


# ===========================================================================
# VIEW 2: 2026 FORECAST & REGIONAL POLICY ACTION PLAN
# ===========================================================================
elif section == "🚨 2. Πρόγνωση 2026 & Σχέδιο Δράσης ανά Περιφέρεια":
    st.markdown(
        """
        <div class="section-header-bar">
            <div>
                <h3>🚨 Μηνιαία Πρόγνωση Επικινδυνότητας KSI 2026 & Σχέδιο Δράσης ανά Περιφέρεια</h3>
                <p>Επιλέξτε μήνα και Περιφέρεια για εξειδικευμένη διάγνωση κινδύνου και επιχειρησιακές συστάσεις (Τροχαία, ΕΚΑΒ, Υποδομές).</p>
            </div>
            <span class="card-title-tag">12-Month Early Warning</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Legend for Risk Tiers in View 2
    st.markdown(
        """
        <div class="legend-banner">
            <div class="legend-header">📖 Υπόμνημα Επιπέδων Συναγερμού (Risk Tiers) & Προγνωστικών Δεικτών</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-term"><span>🔴 CRITICAL SUMMER ALERT</span><span class="legend-tag">Κρίσιμος Κίνδυνος</span></div>
                    <p class="legend-desc">Ενεργοποιείται τους θερινούς μήνες (Ιούν–Σεπ) σε νησιωτικές/τουριστικές περιφέρειες με <b>KSI ≥ 22%</b> και υψηλή κυκλοφορία δικύκλων (≥42%). Απαιτεί προ-τοποθέτηση ΕΚΑΒ και νυχτερινά μπλόκα αλκοτέστ/κράνους.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>🟠 HIGH SEVERITY RISK</span><span class="legend-tag">Υψηλή Βαρύτητα</span></div>
                    <p class="legend-desc">Περιφέρειες με <b>KSI ≥ 21%</b> στο υπεραστικό/επαρχιακό δίκτυο (π.χ. Πελοπόννησος, Στερεά Ελλάδα). Ο κίνδυνος οφείλεται στις υψηλές ταχύτητες και στις μετωπικές συγκρούσεις.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>🔵 URBAN VOLUME FOCUS</span><span class="legend-tag">Μητροπολιτικός Όγκος</span></div>
                    <p class="legend-desc">Αφορά <b>Αττική & Κεντρική Μακεδονία</b>: Χαμηλό ποσοστό KSI (7%–11%) λόγω χαμηλότερων αστικών ταχυτήτων, αλλά <b>τεράστιος απόλυτος αριθμός ατυχημάτων</b> (έμφαση σε πεζούς, διασταυρώσεις & ζώνες 30 km/h).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>🟡 MODERATE / REGIONAL WATCH</span><span class="legend-tag">Τακτική Επιτήρηση</span></div>
                    <p class="legend-desc">Περιφέρειες με <b>KSI &lt; 21%</b>. Απαιτείται προληπτική αστυνόμευση ταχύτητας/ζώνης/κράνους και συντήρηση αντιολισθηρότητας οδοστρώματος.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not fc_df.empty:
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
        with st.container(border=True):
            col_m, col_r = st.columns(2)
            with col_m:
                sel_month = st.selectbox(
                    "📅 Επιλογή Μήνα Πρόγνωσης (2026):",
                    options=list(month_names.keys()),
                    format_func=lambda k: month_names[k],
                    index=7,
                )
            with col_r:
                sel_reg = st.selectbox(
                    "📍 Επιλογή Περιφέρειας για Εξειδικευμένη Διάγνωση:",
                    options=sorted(fc_df["region"].unique()),
                    index=sorted(fc_df["region"].unique()).index("South Aegean (Νότιο Αιγαίο)")
                    if "South Aegean (Νότιο Αιγαίο)" in fc_df["region"].unique()
                    else 0,
                )

        month_fc = fc_df[fc_df["month"] == sel_month].sort_values(
            "predicted_ksi_rate_pct", ascending=False
        )
        row_sel = month_fc[month_fc["region"] == sel_reg].iloc[0]

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                render_bento_kpi(
                    icon="🎯",
                    icon_bg="#fef2f2",
                    label="Προβλεπόμενο Ποσοστό Βαρύτητας KSI",
                    value=f"{row_sel['predicted_ksi_rate_pct']:.2f}%",
                    pill_text=month_names[sel_month].split(" - ")[-1],
                    pill_class="pill-rose" if row_sel["predicted_ksi_rate_pct"] >= 21.0 else "pill-blue",
                    hint="Πόσα στα 100 τροχαία ατυχήματα αυτού του μήνα αναμένεται να είναι θανατηφόρα ή βαριά.",
                ),
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                render_bento_kpi(
                    icon="⚠️",
                    icon_bg="#fffbeb",
                    label="Πιθανότητα Καθεστώτος Υψηλού KSI (>=21%)",
                    value=f"{row_sel['severe_regime_prob_pct']:.1f}%",
                    pill_text="RandomForest Calibrated",
                    pill_class="pill-amber",
                    hint="Η πιθανότητα που δίνει το μοντέλο ML να ξεπεράσει η Περιφέρεια το κρίσιμο όριο του 21%.",
                ),
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                render_bento_kpi(
                    icon="🚑",
                    icon_bg="#eff6ff",
                    label="Αναμενόμενα Σοβαρά/Θανατηφόρα (KSI)",
                    value=f"{int(row_sel['expected_ksi_accidents'])}",
                    pill_text="Μηνιαία Εκτίμηση",
                    pill_class="pill-blue",
                    hint="Ο απόλυτος αριθμός σοβαρών/θανατηφόρων τροχαίων που αναμένονται τον επιλεγμένο μήνα.",
                ),
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                render_bento_kpi(
                    icon="🚗",
                    icon_bg="#f0fdf4",
                    label="Συνολικά Ατυχήματα με Παθόντες",
                    value=f"{int(row_sel['expected_total_accidents'])}",
                    pill_text=f"Δίκυκλα: {row_sel['motorcycle_share']*100:.0f}%",
                    pill_class="pill-emerald",
                    hint="Σύνολο ατυχημάτων (ελαφρά + βαριά + θανατηφόρα) και ποσοστό εμπλοκής μοτοσικλετών.",
                ),
                unsafe_allow_html=True,
            )

        tier_str = str(row_sel["risk_tier"])
        pill_cls = (
            "risk-critical"
            if "CRITICAL" in tier_str
            else "risk-high"
            if "HIGH" in tier_str
            else "risk-urban"
            if "URBAN" in tier_str
            else "risk-moderate"
        )

        st.markdown(
            f"""
            <div class="policy-box">
                <span class="risk-pill {pill_cls}">🚨 {tier_str}</span>
                <p style="margin:0.35rem 0; color:#1e293b; font-size:0.94rem;"><b>🔍 Διάγνωση Μηχανισμού Κινδύνου:</b> {row_sel['primary_risk_mechanism']}</p>
                <p style="margin:0.45rem 0 0 0; color:#065f46; font-size:0.94rem;"><b>🛠️ Προτεινόμενη Εισήγηση Παρεμβάσεων:</b> {row_sel['recommended_interventions']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">🔥 Χάρτης Θερμότητας (Heatmap) Πρόγνωσης KSI 2026 (13 Περιφέρειες × 12 Μήνες)</span>
                    <span class="card-title-tag">Σκούρο Κόκκινο = Υψηλότερη Βαρύτητα KSI (%)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            piv_df = fc_df.copy()
            piv_df["short_reg"] = piv_df["region"].apply(
                lambda s: s.split("(")[-1].replace(")", "") if "(" in s else s
            )
            piv = piv_df.pivot(index="short_reg", columns="month", values="predicted_ksi_rate_pct")
            piv = piv.loc[piv.mean(axis=1).sort_values(ascending=True).index]
            m_labels = ["Ιαν", "Φεβ", "Μαρ", "Απρ", "Μάι", "Ιούν", "Ιούλ", "Αύγ", "Σεπ", "Οκτ", "Νοέ", "Δεκ"]
            fig_hm_interactive = px.imshow(
                piv,
                labels=dict(x="Μήνας (2026)", y="Περιφέρεια", color="KSI Rate (%)"),
                x=m_labels[: len(piv.columns)],
                y=piv.index,
                text_auto=".1f",
                aspect="auto",
                color_continuous_scale="YlOrRd",
                height=460,
            )
            style_plotly_chart(
                fig_hm_interactive,
                title_text="🔥 Μηνιαίος Χάρτης Θερμότητας Πρόγνωσης KSI (%) ανά Περιφέρεια (2026)",
                height=470,
                show_legend=False,
            )
            st.plotly_chart(fig_hm_interactive, use_container_width=True, theme=None)
            st.caption(
                "💡 **Πώς διαβάζεται το Heatmap:** Κάθε κελί δείχνει το προβλεπόμενο ποσοστό KSI (%) για τη συγκεκριμένη Περιφέρεια και μήνα. "
                "Παρατηρήστε πώς το καλοκαίρι (Ιούνιος–Σεπτέμβριος) «κοκκινίζουν» οι τουριστικές και νησιωτικές περιφέρειες."
            )

        with st.container(border=True):
            st.markdown(
                f"""
                <div class="card-title-row">
                    <span class="card-title-text">📋 Συγκριτικός Πίνακας Πρόγνωσης — {month_names[sel_month]} 2026</span>
                    <span class="card-title-tag">13 Περιφέρειες NUTS-2</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
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
                column_config={
                    "region": st.column_config.TextColumn("Περιφέρεια", help="Διοικητική Περιφέρεια NUTS-2."),
                    "predicted_ksi_rate_pct": st.column_config.ProgressColumn(
                        "Πρόβλεψη KSI (%)",
                        format="%.2f%%",
                        min_value=0,
                        max_value=30,
                        help="Αναμενόμενο ποσοστό θανατηφόρων/σοβαρών ατυχημάτων επί του συνόλου των ατυχημάτων.",
                    ),
                    "severe_regime_prob_pct": st.column_config.ProgressColumn(
                        "Πιθανότητα Υψηλού KSI",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                        help="Πιθανότητα P(KSI >= 21%) σύμφωνα με τον ταξινομητή RandomForest.",
                    ),
                    "expected_total_accidents": st.column_config.NumberColumn(
                        "Σύνολο Ατυχημάτων",
                        help="Εκτιμώμενος συνολικός αριθμός τροχαίων ατυχημάτων με παθόντες τον συγκεκριμένο μήνα.",
                    ),
                    "expected_ksi_accidents": st.column_config.NumberColumn(
                        "Αναμενόμενα KSI",
                        help="Εκτιμώμενος αριθμός θανατηφόρων και σοβαρών τροχαίων ατυχημάτων.",
                    ),
                    "risk_tier": st.column_config.TextColumn(
                        "Επίπεδο Συναγερμού",
                        help="Κατηγορία επιχειρησιακής προτεραιότητας.",
                    ),
                    "recommended_interventions": st.column_config.TextColumn(
                        "Προτεινόμενες Δράσεις",
                        help="Συστάσεις πολιτικής βάσει του προφίλ κινδύνου της Περιφέρειας.",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

    if policy_doc_path.exists():
        with st.expander("📖 Πλήρης Στρατηγική Εισήγηση Αντιμετώπισης (5 Άξονες Οδικής Ασφάλειας — Vision Zero)"):
            st.markdown(policy_doc_path.read_text(encoding="utf-8"))


# ===========================================================================
# VIEW 3: OFFICIAL ELSTAT OVERVIEW (2015-2025)
# ===========================================================================
elif section == "📊 3. Επίσημα Στοιχεία ΕΛΣΤΑΤ (Ελλάδα 2015–2025)":
    st.markdown(
        """
        <div class="section-header-bar">
            <div>
                <h3>📊 Επίσημη Στατιστική Εικόνα Οδικής Ασφάλειας Ελλάδας (ΕΛΣΤΑΤ SDT04)</h3>
                <p>Διαχρονική εξέλιξη των οδικών τροχαίων ατυχημάτων στην Ελλάδα (2015–2024) και ανάλυση του περιφερειακού πάνελ (2021–2025).</p>
            </div>
            <span class="card-title-tag">100% Επίσημα Στοιχεία ΕΛΣΤΑΤ</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Legend for ELSTAT Definitions in View 3
    st.markdown(
        """
        <div class="legend-banner">
            <div class="legend-header">📖 Υπόμνημα Στατιστικών Ορισμών ΕΛΣΤΑΤ (Έρευνα SDT04)</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-term"><span>1. Ατύχημα με Παθόντες</span><span class="legend-tag">Ορισμός ΕΛΣΤΑΤ</span></div>
                    <p class="legend-desc">Κάθε οδικό τροχαίο ατύχημα στο οποίο υπάρχει τουλάχιστον ένας τραυματίας ή νεκρός. <i>Τα ατυχήματα αποκλειστικά με υλικές ζημιές (λαμαρίνας) δεν περιλαμβάνονται στην έρευνα SDT04.</i></p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>2. Νεκρός (Ορισμός 30 Ημερών)</span><span class="legend-tag">Διεθνές Πρότυπο</span></div>
                    <p class="legend-desc">Κάθε πρόσωπο που σκοτώθηκε ακαριαία ή υπέκυψε στα τραύματά του εντός <b>30 ημερών</b> από την ημέρα του τροχαίου ατυχήματος (εναρμονισμένος ορισμός ΕΕ / UNECE).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>3. Βαριά vs. Ελαφρά Τραυματίας</span><span class="legend-tag">Βαρύτητα Τραύματος</span></div>
                    <p class="legend-desc"><b>Βαριά τραυματίας:</b> Πρόσωπο που υπέστη σοβαρή σωματική βλάβη (π.χ. κρανιοεγκεφαλική κάκωση, ακρωτηριασμό, πολυτραυματισμό ή μακρά νοσηλεία). Οι υπόλοιποι τραυματισμοί καταγράφονται ως <b>Ελαφρά</b>.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>4. Το Ελληνικό Χωρικό Παράδοξο</span><span class="legend-tag">Βασικό Εύρημα</span></div>
                    <p class="legend-desc">Στην <b>Αττική</b> συμβαίνουν τα περισσότερα ατυχήματα λόγω κυκλοφοριακού φόρτου, αλλά με χαμηλές ταχύτητες. Στην <b>Περιφέρεια & τα Νησιά</b> τα ατυχήματα είναι λιγότερα σε αριθμό αλλά <b>πολύ πιο θανατηφόρα</b>.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not nat_df.empty:
        latest = nat_df.iloc[-1]
        prev = nat_df.iloc[-2]
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(
                render_bento_kpi(
                    icon="🚗",
                    icon_bg="#eff6ff",
                    label="Ατυχήματα με Παθόντες (2024)",
                    value=f"{int(latest['total_injury_accidents']):,}",
                    pill_text=f"{int(latest['total_injury_accidents'] - prev['total_injury_accidents']):+d} vs 2023",
                    pill_class="pill-amber",
                    hint="Σύνολο τροχαίων ατυχημάτων με σωματικές βλάβες στην Ελλάδα το 2024.",
                ),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                render_bento_kpi(
                    icon="🖤",
                    icon_bg="#fef2f2",
                    label="Νεκροί (Ορισμός 30 Ημερών, 2024)",
                    value=f"{int(latest['killed_30d']):,}",
                    pill_text=f"{int(latest['killed_30d'] - prev['killed_30d']):+d} vs 2023",
                    pill_class="pill-rose",
                    hint="Άνθρωποι που έχασαν τη ζωή τους εντός 30 ημερών από το τροχαίο ατύχημα.",
                ),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                render_bento_kpi(
                    icon="🏥",
                    icon_bg="#fffbeb",
                    label="Βαριά Τραυματίες (2024)",
                    value=f"{int(latest['seriously_injured']):,}",
                    pill_text=f"{int(latest['seriously_injured'] - prev['seriously_injured']):+d} vs 2023",
                    pill_class="pill-emerald",
                    hint="Άτομα που υπέστησαν σοβαρό τραυματισμό/νοσηλεία σε τροχαίο το 2024.",
                ),
                unsafe_allow_html=True,
            )
        with c4:
            st.markdown(
                render_bento_kpi(
                    icon="📈",
                    icon_bg="#f0fdf4",
                    label="Δείκτης KSI ανά 100 Ατυχήματα",
                    value=f"{latest['ksi_per_100_accidents']:.2f}%",
                    pill_text=f"{latest['ksi_per_100_accidents'] - prev['ksi_per_100_accidents']:+.2f}% vs 2023",
                    pill_class="pill-blue",
                    hint="Αναλογία (Νεκροί + Βαριά Τραυματίες) ανά 100 τροχαία ατυχήματα πανελλαδικά.",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

        col_nat1, col_nat2 = st.columns(2)
        with col_nat1:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">📈 Εξέλιξη Νεκρών & Βαριά Τραυματιών (2015–2024)</span>
                        <span class="card-title-tag">Εθνική Σειρά ΕΛΣΤΑΤ</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_nat = go.Figure()
                fig_nat.add_trace(
                    go.Scatter(
                        x=nat_df["year"],
                        y=nat_df["killed_30d"],
                        mode="lines+markers",
                        name="Νεκροί (30 ημερών)",
                        line=dict(color="#e11d48", width=3.2),
                    )
                )
                fig_nat.add_trace(
                    go.Scatter(
                        x=nat_df["year"],
                        y=nat_df["seriously_injured"],
                        mode="lines+markers",
                        name="Βαριά Τραυματίες",
                        line=dict(color="#f59e0b", width=3.2),
                    )
                )
                fig_nat.update_layout(
                    xaxis=dict(title="Έτος (2015–2024)"),
                    yaxis=dict(title="Αριθμός Παθόντων"),
                )
                style_plotly_chart(
                    fig_nat,
                    title_text="📈 Διαχρονική Εξέλιξη Νεκρών & Βαριά Τραυματιών (2015–2024)",
                    height=410,
                    show_legend=True,
                )
                st.plotly_chart(fig_nat, use_container_width=True, theme=None)

        with col_nat2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">⚖️ Μέση Βαρύτητα KSI (%) ανά Περιφέρεια (2021–2025)</span>
                        <span class="card-title-tag">Όριο Υψηλού KSI: 21%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                reg_agg_all = (
                    reg_df.groupby("region", as_index=False)
                    .agg(
                        total_accidents=("total_accidents", "sum"),
                        ksi_accidents=("ksi_accidents", "sum"),
                        mean_ksi_rate=("ksi_severity_rate", "mean"),
                    )
                    .sort_values("mean_ksi_rate", ascending=True)
                )
                reg_agg_all["short_region"] = reg_agg_all["region"].apply(
                    lambda s: s.split("(")[-1].replace(")", "") if "(" in s else s
                )
                reg_agg_all["mean_ksi_rate_pct"] = (reg_agg_all["mean_ksi_rate"] * 100.0).round(2)
                fig_par = px.bar(
                    reg_agg_all,
                    x="mean_ksi_rate_pct",
                    y="short_region",
                    orientation="h",
                    color="mean_ksi_rate_pct",
                    color_continuous_scale="YlOrRd",
                    labels={"mean_ksi_rate_pct": "Μέσο KSI (%)", "short_region": "Περιφέρεια"},
                    height=410,
                )
                fig_par.add_vline(x=21.0, line_dash="dash", line_color="#991b1b")
                fig_par.update_layout(coloraxis_showscale=False)
                style_plotly_chart(
                    fig_par,
                    title_text="⚖️ Μέσο Ποσοστό Βαρύτητας KSI (%) ανά Περιφέρεια (2021–2025)",
                    height=410,
                    show_legend=False,
                )
                st.plotly_chart(fig_par, use_container_width=True, theme=None)

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">📋 Επίσημος Εθνικός Πίνακας ΕΛΣΤΑΤ (2015–2024)</span>
                    <span class="card-title-tag">Έρευνα SDT04</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                nat_df,
                column_config={
                    "year": st.column_config.NumberColumn("Έτος", format="%d"),
                    "total_injury_accidents": st.column_config.NumberColumn(
                        "Σύνολο Ατυχημάτων με Παθόντες",
                        help="Ετήσιος αριθμός τροχαίων ατυχημάτων με τουλάχιστον έναν τραυματία ή νεκρό.",
                    ),
                    "killed_30d": st.column_config.NumberColumn(
                        "Νεκροί (30 Ημερών)",
                        help="Αριθμός θανάτων εντός 30 ημερών από το ατύχημα.",
                    ),
                    "seriously_injured": st.column_config.NumberColumn(
                        "Βαριά Τραυματίες",
                        help="Αριθμός ατόμων με σοβαρό τραυματισμό.",
                    ),
                    "slightly_injured": st.column_config.NumberColumn(
                        "Ελαφρά Τραυματίες",
                        help="Αριθμός ατόμων με ελαφρύ τραυματισμό.",
                    ),
                    "total_ksi_casualties": st.column_config.NumberColumn(
                        "Σύνολο Παθόντων KSI",
                        help="Άθροισμα Νεκρών + Βαριά Τραυματιών.",
                    ),
                    "ksi_per_100_accidents": st.column_config.NumberColumn(
                        "Παθόντες KSI ανά 100 Ατυχήματα (%)",
                        format="%.2f%%",
                        help="(Νεκροί + Βαριά Τραυματίες) / Σύνολο Ατυχημάτων × 100.",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )


# ===========================================================================
# VIEW 4: 2025 MODEL BENCHMARK & FEATURE ABLATION
# ===========================================================================
elif section == "🤖 4. Αξιολόγηση Μοντέλων ML & Ablation Study":
    st.markdown(
        """
        <div class="section-header-bar">
            <div>
                <h3>🤖 Out-of-Time Αξιολόγηση Μοντέλων Μηχανικής Μάθησης (Test Set: 2025)</h3>
                <p>Αυστηρός χρονολογικός διαχωρισμός (2021–2023 Train | 2024 Val | 2025 Test) με 95% Bootstrap CIs.</p>
            </div>
            <span class="card-title-tag">Zero Data Leakage</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Plain-Language Legend for ML Metrics in View 4
    st.markdown(
        """
        <div class="legend-banner">
            <div class="legend-header">📖 Υπόμνημα Μετρικών Μηχανικής Μάθησης (Πώς αξιολογούνται τα μοντέλα με απλά λόγια)</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-term"><span>1. ROC-AUC & PR-AUC</span><span class="legend-tag">0.50 έως 1.00 (Υψηλότερο = Καλύτερο)</span></div>
                    <p class="legend-desc"><b>ROC-AUC:</b> Η ικανότητα του μοντέλου να ξεχωρίζει τους επικίνδυνους μήνες από τους ασφαλείς (0.50 = τυχαίο νόμισμα, &gt;0.85 = άριστο). <b>PR-AUC:</b> Εστιάζει ειδικά στην ακρίβεια εντοπισμού των περιπτώσεων υψηλού KSI.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>2. Recall vs. Precision & F1</span><span class="legend-tag">Ευαισθησία & Ακρίβεια</span></div>
                    <p class="legend-desc"><b>Recall (Ευαισθησία):</b> Τι ποσοστό των πραγματικά επικίνδυνων μηνών «έπιασε» το μοντέλο (π.χ. 97.3% = δεν του ξέφυγε σχεδόν κανένας). <b>Precision:</b> Πόσοι από τους συναγερμούς ήταν αληθινοί. <b>F1:</b> Ο μέσος όρος τους.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>3. Brier Score (Βαθμονόμηση)</span><span class="legend-tag">0.00 έως 0.25 (Χαμηλότερο = Καλύτερο)</span></div>
                    <p class="legend-desc">Μετρά το μέσο τετραγωνικό σφάλμα των πιθανοτήτων. Αν το μοντέλο δώσει 80% πιθανότητα κινδύνου, το <b>χαμηλό Brier Score (0.1552)</b> εγγυάται ότι η πιθανότητα αυτή είναι ρεαλιστική και αξιόπιστη.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>4. Ablation Study & 95% CI</span><span class="legend-tag">Στατιστική Αυστηρότητα</span></div>
                    <p class="legend-desc"><b>Ablation Study:</b> Αφαιρούμε/προσθέτουμε ομάδες μεταβλητών μία-μία για να δούμε πόσο συνεισφέρει η καθεμία. <b>95% CI:</b> Το εύρος μέσα στο οποίο βρίσκεται η αληθινή επίδοση με 95% βεβαιότητα.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not bench_df.empty:
        best_row = bench_df.sort_values("roc_auc", ascending=False).iloc[0]
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            st.markdown(
                render_bento_kpi(
                    icon="🏆",
                    icon_bg="#eff6ff",
                    label="Κορυφαίο Μοντέλο Ταξινόμησης",
                    value=str(best_row["model"]),
                    pill_text="Winner 2025 Test",
                    pill_class="pill-blue",
                    hint="Σύνολο 200 δέντρων απόφασης (Random Forest) που ψηφίζουν συνδυαστικά.",
                ),
                unsafe_allow_html=True,
            )
        with b2:
            st.markdown(
                render_bento_kpi(
                    icon="📈",
                    icon_bg="#ecfdf5",
                    label="2025 Out-of-Time ROC-AUC",
                    value=f"{best_row['roc_auc']:.4f}",
                    pill_text=f"CI: {best_row['roc_auc_ci_95']}",
                    pill_class="pill-emerald",
                    hint="Πολύ υψηλή διαχωριστική ικανότητα σε δεδομένα του 2025 που δεν είδε στην εκπαίδευση.",
                ),
                unsafe_allow_html=True,
            )
        with b3:
            st.markdown(
                render_bento_kpi(
                    icon="🎯",
                    icon_bg="#fffbeb",
                    label="2025 Out-of-Time PR-AUC",
                    value=f"{best_row['pr_auc']:.4f}",
                    pill_text=f"CI: {best_row['pr_auc_ci_95']}",
                    pill_class="pill-amber",
                    hint="Υπερδιπλάσια ακρίβεια εντοπισμού υψηλού κινδύνου σε σχέση με το τυχαίο Baseline (0.4679).",
                ),
                unsafe_allow_html=True,
            )
        with b4:
            st.markdown(
                render_bento_kpi(
                    icon="⚖️",
                    icon_bg="#f0fdf4",
                    label="Brier Probability Calibration",
                    value=f"{best_row['brier_score']:.4f}",
                    pill_text="Χαμηλότερο = Καλύτερο",
                    pill_class="pill-emerald",
                    hint="Το χαμηλότερο σφάλμα πιθανοτήτων μεταξύ όλων των 5 εξεταζόμενων αλγορίθμων.",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">🏆 Συγκριτικός Πίνακας 5 Μοντέλων στο Held-Out Test Set (2025)</span>
                    <span class="card-title-tag">Περάστε το ποντίκι πάνω από τις στήλες για επεξήγηση</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
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
                    ]
                ],
                column_config={
                    "model": st.column_config.TextColumn("Αλγόριθμος ML", help="Όνομα του μοντέλου μηχανικής μάθησης."),
                    "roc_auc": st.column_config.ProgressColumn(
                        "ROC-AUC",
                        format="%.4f",
                        min_value=0.0,
                        max_value=1.0,
                        help="Εμβαδόν κάτω από την καμπύλη ROC (0.50 = τυχαίο, 1.00 = τέλειος διαχωρισμός).",
                    ),
                    "roc_auc_ci_95": st.column_config.TextColumn(
                        "95% CI (ROC-AUC)",
                        help="Διάστημα εμπιστοσύνης 95% με 400 επαναλήψεις Bootstrap.",
                    ),
                    "pr_auc": st.column_config.ProgressColumn(
                        "PR-AUC",
                        format="%.4f",
                        min_value=0.0,
                        max_value=1.0,
                        help="Εμβαδόν κάτω από την καμπύλη Ακρίβειας-Ανάκλησης (Precision-Recall).",
                    ),
                    "pr_auc_ci_95": st.column_config.TextColumn(
                        "95% CI (PR-AUC)",
                        help="Διάστημα εμπιστοσύνης 95% για το PR-AUC.",
                    ),
                    "f1": st.column_config.NumberColumn(
                        "F1-Score",
                        format="%.4f",
                        help="Αρμονικός μέσος όρος των Precision και Recall.",
                    ),
                    "precision": st.column_config.NumberColumn(
                        "Precision (Ακρίβεια)",
                        format="%.4f",
                        help="Ποσοστό προβλέψεων υψηλού κινδύνου που επαληθεύτηκαν.",
                    ),
                    "recall": st.column_config.NumberColumn(
                        "Recall (Ευαισθησία)",
                        format="%.4f",
                        help="Ποσοστό των πραγματικά επικίνδυνων μηνών που εντόπισε το μοντέλο.",
                    ),
                    "brier_score": st.column_config.NumberColumn(
                        "Brier Score",
                        format="%.4f",
                        help="Σφάλμα βαθμονόμησης πιθανοτήτων (όσο πιο κοντά στο 0, τόσο πιο αξιόπιστες οι πιθανότητες).",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

    if not abl_df.empty:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">🧪 Μελέτη Αφαίρεσης Ομάδων Χαρακτηριστικών (Feature Group Ablation Study)</span>
                    <span class="card-title-tag">Τι προσφέρει κάθε ομάδα δεδομένων στην πρόβλεψη</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                abl_df[["model", "num_features", "roc_auc", "roc_auc_ci_95", "pr_auc", "f1", "precision", "recall"]],
                column_config={
                    "model": st.column_config.TextColumn(
                        "Σύνολο Χαρακτηριστικών (Feature Subset)",
                        help="Ποιες ομάδες μεταβλητών χρησιμοποιήθηκαν στην εκπαίδευση.",
                    ),
                    "num_features": st.column_config.NumberColumn(
                        "Πλήθος Μεταβλητών",
                        help="Αριθμός στηλών εισόδου (features) που δόθηκαν στο μοντέλο.",
                    ),
                    "roc_auc": st.column_config.ProgressColumn("ROC-AUC", format="%.4f", min_value=0.0, max_value=1.0),
                    "roc_auc_ci_95": st.column_config.TextColumn("95% CI (ROC-AUC)"),
                    "pr_auc": st.column_config.ProgressColumn("PR-AUC", format="%.4f", min_value=0.0, max_value=1.0),
                    "f1": st.column_config.NumberColumn("F1-Score", format="%.4f"),
                    "precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                    "recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                },
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "💡 **Συμπέρασμα Ablation Study:** Μόνο με τον μήνα (`1. Calendar Only`) το μοντέλο πετυχαίνει μόλις `0.5805` ROC-AUC. "
                "Μόλις προστεθεί η ταυτότητα της Περιφέρειας (`2. + NUTS-2 Region`) ή τα δομικά χαρακτηριστικά (`3. + Urban & Moto Share`), "
                "η απόδοση εκτοξεύεται πάνω από **`0.83–0.84`**!"
            )

    fig_roc = FIGURES_DIR / "benchmark_roc_pr_curves.png"
    if fig_roc.exists():
        with st.container(border=True):
            st.image(str(fig_roc), caption="Καμπύλες ROC και Precision-Recall στο Held-Out Test Set 2025 της ΕΛΣΤΑΤ")


# ===========================================================================
# VIEW 5: EXPLAINABLE AI (XAI) & 13-REGION SUBGROUP AUDIT
# ===========================================================================
elif section == "🔍 5. Ερμηνευσιμότητα (XAI) & Έλεγχος Περιφερειών":
    st.markdown(
        """
        <div class="section-header-bar">
            <div>
                <h3>🔍 Ερμηνευσιμότητα Μοντέλου (Explainable AI) & Έλεγχος στις 13 Περιφέρειες</h3>
                <p>Ανάλυση σπουδαιότητας χαρακτηριστικών (Permutation Importance) και έλεγχος βαθμονόμησης ανά Περιφέρεια NUTS-2.</p>
            </div>
            <span class="card-title-tag">XAI & Fairness Audit</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Plain-Language Legend for Feature Names & XAI in View 5
    st.markdown(
        """
        <div class="legend-banner">
            <div class="legend-header">📖 Λεξικό Μεταβλητών του Μοντέλου (Τι σημαίνει το όνομα κάθε χαρακτηριστικού στο γράφημα)</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-term"><span>1. lag_12m_ksi_rate</span><span class="legend-tag">#1 Σημαντικότερο</span></div>
                    <p class="legend-desc">Το ποσοστό σοβαρών/θανατηφόρων ατυχημάτων (KSI) της ίδιας Περιφέρειας <b>τον ίδιο μήνα του προηγούμενου έτους</b> (π.χ. Αύγουστος 2024 για πρόβλεψη Αυγούστου 2025).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>2. urban_ratio</span><span class="legend-tag">#2 Δομή Οδικού Δικτύου</span></div>
                    <p class="legend-desc">Το ποσοστό των ατυχημάτων της Περιφέρειας που συμβαίνουν <b>εντός κατοικημένης/αστικής περιοχής</b> (υψηλό στην Αττική ~88%, χαμηλό στην επαρχία/νησιά).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>3. motorcycle_share</span><span class="legend-tag">Ευάλωτοι Χρήστες</span></div>
                    <p class="legend-desc">Το ποσοστό συμμετοχής <b>μοτοσικλετών και μοτοποδηλάτων</b> στα τροχαία ατυχήματα της Περιφέρειας (καθοριστικός παράγοντας βαρύτητας τραυματισμών).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>4. lag_1m / lag_12m_total_accidents</span><span class="legend-tag">Όγκος Κυκλοφορίας</span></div>
                    <p class="legend-desc">Ο συνολικός αριθμός τροχαίων ατυχημάτων στην Περιφέρεια <b>1 μήνα πριν</b> (βραχυπρόθεσμη τάση) και <b>12 μήνες πριν</b> (εποχικός όγκος).</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if not imp_df.empty:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">🎯 Σπουδαιότητα Χαρακτηριστικών (Permutation Importance)</span>
                        <span class="card-title-tag">Μέση Πτώση PR-AUC όταν αφαιρείται η μεταβλητή</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                feature_greek_labels = {
                    "lag_12m_ksi_rate": "lag_12m_ksi_rate (Περσινό KSI ίδιου μήνα)",
                    "urban_ratio": "urban_ratio (Ποσοστό Αστικού Δικτύου)",
                    "lag_12m_total_accidents": "lag_12m_total_accidents (Περσινά Ατυχήματα)",
                    "lag_1m_total_accidents": "lag_1m_total_accidents (Ατυχήματα προηγ. μήνα)",
                    "motorcycle_share": "motorcycle_share (Μερίδιο Δικύκλων)",
                    "lag_1m_ksi_rate": "lag_1m_ksi_rate (KSI προηγ. μήνα)",
                    "region": "region (Ταυτότητα Περιφέρειας NUTS-2)",
                    "month_sin": "month_sin (Εποχικότητα - Ημίτονο Μήνα)",
                    "month_cos": "month_cos (Εποχικότητα - Συνημίτονο Μήνα)",
                    "rolling_3m_ksi_rate": "rolling_3m_ksi_rate (Μέσο KSI 3μήνου)",
                    "lag_2m_ksi_rate": "lag_2m_ksi_rate (KSI προ 2 μηνών)",
                    "yoy_ksi_rate_diff": "yoy_ksi_rate_diff (Ετήσια Μεταβολή KSI)",
                    "is_summer": "is_summer (Θερινή Περίοδος Ιούν-Σεπ)",
                }
                top_imp = imp_df.head(10).iloc[::-1].copy()
                top_imp["display_feature"] = top_imp["feature"].map(
                    lambda f: feature_greek_labels.get(str(f), str(f))
                )
                fig_imp_bar = px.bar(
                    top_imp,
                    x="importance_mean",
                    y="display_feature",
                    error_x="importance_std",
                    orientation="h",
                    color="importance_mean",
                    color_continuous_scale="Blues",
                    labels={"importance_mean": "Μέση Πτώση PR-AUC", "display_feature": ""},
                    height=430,
                )
                fig_imp_bar.update_layout(coloraxis_showscale=False)
                style_plotly_chart(
                    fig_imp_bar,
                    title_text="🎯 Σπουδαιότητα Χαρακτηριστικών (Permutation Importance — PR-AUC)",
                    height=450,
                    show_legend=False,
                )
                st.plotly_chart(fig_imp_bar, use_container_width=True, theme=None)
                st.caption(
                    "💡 **Πώς διαβάζεται:** Όσο μεγαλύτερη είναι η μπλε μπάρα, τόσο πιο απαραίτητη είναι η συγκεκριμένη μεταβλητή για να προβλέψει σωστά το μοντέλο."
                )

    with col2:
        if not sub_df.empty:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">⚖️ Πραγματικό vs. Προβλεπόμενο Υψηλό KSI ανά Περιφέρεια</span>
                        <span class="card-title-tag">Έλεγχος Αξιοπιστίας (Calibration 2025)</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                sub_plot = sub_df.sort_values("actual_ksi_rate", ascending=True).copy()
                sub_plot["short_reg"] = sub_plot["region"].apply(
                    lambda s: s.split("(")[-1].replace(")", "") if "(" in s else s
                )
                fig_cal_bar = go.Figure()
                fig_cal_bar.add_trace(
                    go.Bar(
                        y=sub_plot["short_reg"],
                        x=sub_plot["actual_ksi_rate"] * 100.0,
                        name="Πραγματικό 2025 (%)",
                        orientation="h",
                        marker_color="#0f172a",
                        marker_cornerradius=5,
                    )
                )
                fig_cal_bar.add_trace(
                    go.Bar(
                        y=sub_plot["short_reg"],
                        x=sub_plot["mean_predicted_prob"] * 100.0,
                        name="Προβλεπόμενη Πιθανότητα (%)",
                        orientation="h",
                        marker_color="#3b82f6",
                        marker_cornerradius=5,
                    )
                )
                fig_cal_bar.update_layout(
                    barmode="group",
                    xaxis=dict(title="Ποσοστό Μηνών σε Καθεστώς Υψηλού KSI (%)"),
                    yaxis=dict(title="Περιφέρεια NUTS-2"),
                )
                style_plotly_chart(
                    fig_cal_bar,
                    title_text="⚖️ Πραγματικό vs. Προβλεπόμενο Υψηλό KSI ανά Περιφέρεια (2025)",
                    height=450,
                    show_legend=True,
                )
                st.plotly_chart(fig_cal_bar, use_container_width=True, theme=None)
                st.caption(
                    "💡 **Πώς διαβάζεται:** Συγκρίνει πόσους μήνες του 2025 μια Περιφέρεια είχε όντως υψηλό KSI (μαύρη μπάρα) σε σχέση με το τι προέβλεψε το μοντέλο (μπλε μπάρα)."
                )

    if not sub_df.empty:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">📋 Αναλυτικός Πίνακας Απόδοσης & Βαθμονόμησης στις 13 Περιφέρειες (2025)</span>
                    <span class="card-title-tag">Περάστε το ποντίκι πάνω από τις στήλες για επεξήγηση</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                sub_df,
                column_config={
                    "region": st.column_config.TextColumn("Περιφέρεια (NUTS-2)"),
                    "sample_size": st.column_config.NumberColumn(
                        "Πλήθος Μηνών (2025)",
                        help="Αριθμός μηνιαίων παρατηρήσεων στο Test Set του 2025 (12 μήνες ανά Περιφέρεια).",
                    ),
                    "actual_ksi_rate": st.column_config.NumberColumn(
                        "Πραγματική Συχνότητα Υψηλού KSI",
                        format="%.4f",
                        help="Ποσοστό μηνών του 2025 όπου το πραγματικό KSI ξεπέρασε το 21%.",
                    ),
                    "mean_predicted_prob": st.column_config.NumberColumn(
                        "Μέση Προβλεπόμενη Πιθανότητα",
                        format="%.4f",
                        help="Μέση πιθανότητα που έδωσε το μοντέλο για τη συγκεκριμένη Περιφέρεια το 2025.",
                    ),
                    "calibration_gap": st.column_config.NumberColumn(
                        "Απόκλιση Βαθμονόμησης (Gap)",
                        format="%.4f",
                        help="Απόλυτη διαφορά μεταξύ πραγματικής συχνότητας και προβλεπόμενης πιθανότητας (όσο πιο κοντά στο 0, τόσο πιο ακριβές).",
                    ),
                    "f1": st.column_config.NumberColumn("F1-Score", format="%.4f"),
                    "recall": st.column_config.NumberColumn("Recall (Ευαισθησία)", format="%.4f"),
                    "brier_score": st.column_config.NumberColumn(
                        "Brier Score",
                        format="%.4f",
                        help="Μέσο τετραγωνικό σφάλμα πιθανοτήτων στην Περιφέρεια (χαμηλότερο = καλύτερο).",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )


# ===========================================================================
# VIEW 6: INDEPENDENT EXTERNAL RESEARCH & EU KPIs (NON-ELSTAT SOURCES)
# ===========================================================================
elif section == "🔬 6. Ανεξάρτητες Έρευνες & Ευρωπαϊκά KPIs (ΕΜΠ / ESRA3 / ERA5)":
    st.markdown(
        """
        <div class="section-header-bar">
            <div>
                <h3>🔬 Ανεξάρτητες Επιστημονικές Έρευνες & Ευρωπαϊκοί Δείκτες Οδικής Ασφάλειας (Non-ELSTAT)</h3>
                <p>Διασταύρωση των ευρημάτων μας με 100% ανεξάρτητες πηγές πεδίου: ΕΜΠ (NTUA), Ευρωπαϊκό Πρόγραμμα Trendline, Έρευνα ESRA3, Τηλεματική, Copernicus ERA5 & ΕΚΑΒ.</p>
            </div>
            <span class="card-title-tag">Zero Circularity • Independent Sources</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="legend-banner">
            <div class="legend-header">📖 Υπόμνημα Ανεξάρτητων Πηγών (Πώς συλλέγονται αυτά τα δεδομένα ανεξάρτητα από την ΕΛΣΤΑΤ)</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-term"><span>1. EU Trendline / Baseline & ΕΜΠ</span><span class="legend-tag">Μετρήσεις Πεδίου</span></div>
                    <p class="legend-desc">Ανεξάρτητοι παρατηρητές και ραντάρ σε 100+ σημεία του οδικού δικτύου καταγράφουν τι ποσοστό οδηγών φοράει <b>κράνος</b>, <b>ζώνη</b> ή τηρεί τα <b>όρια ταχύτητας</b> στην καθημερινή κυκλοφορία (πριν συμβεί ατύχημα).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>2. Διεθνής Έρευνα ESRA3 (VIAS & ΕΜΠ)</span><span class="legend-tag">39 Χώρες</span></div>
                    <p class="legend-desc">Παγκόσμια έρευνα συμπεριφοράς χρηστών οδού που μετρά την οδήγηση υπό την επήρεια <b>αλκοόλ</b>, την <b>υπνηλία/κόπωση</b> και τη χρήση κινητού στην Ελλάδα έναντι του μέσου όρου της ΕΕ-27.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>3. Τηλεματική & Κλίμα (OSeven / ERA5)</span><span class="legend-tag">Αισθητήρες & Μετεωρολογία</span></div>
                    <p class="legend-desc">Δεδομένα επιταχυνσιομέτρου/GPS από έξυπνα κινητά (απότομα φρεναρίσματα ανά 100 km), τουριστικές αφίξεις (ΥΠΑ/Fraport) και κλιματικά στοιχεία (Copernicus ERA5 / ΕΜΥ).</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>4. Κλινικά Μητρώα MAIS3+ & ΕΚΑΒ</span><span class="legend-tag">Νοσοκομεία & Golden Hour</span></div>
                    <p class="legend-desc">Δεδομένα από ΜΕΘ/ΤΕΠ (κλίμακα σοβαρού τραύματος <b>MAIS3+</b>) και διάμεσος χρόνος απόκρισης ασθενοφόρων ΕΚΑΒ ανά Περιφέρεια (11.5 λεπτά στην Αττική έναντι 21–24.5 λεπτών στα νησιά).</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            render_bento_kpi(
                icon="🪖",
                icon_bg="#fef2f2",
                label="Χρήση Κράνους Συνεπιβάτη (Ελλάδα)",
                value="65.2%",
                pill_text="-29.6% vs ΕΕ-27 (94.8%)",
                pill_class="pill-rose",
                hint="Πηγή: EU Trendline / ΕΜΠ. Στα νησιά το καλοκαίρι υποχωρεί κάτω από 60%.",
            ),
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            render_bento_kpi(
                icon="💺",
                icon_bg="#fffbeb",
                label="Χρήση Ζώνης Πίσω Καθισμάτων",
                value="42.8%",
                pill_text="-41.7% vs ΕΕ-27 (84.5%)",
                pill_class="pill-amber",
                hint="Πηγή: EU Trendline / ΕΜΠ. Μόλις 4 στους 10 επιβάτες πίσω καθισμάτων φορούν ζώνη.",
            ),
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            render_bento_kpi(
                icon="📡",
                icon_bg="#eff6ff",
                label="Τήρηση Ορίου Ταχύτητας (Υπεραστικές)",
                value="61.8%",
                pill_text="-17.7% vs ΕΕ-27 (79.5%)",
                pill_class="pill-blue",
                hint="Πηγή: Μετρήσεις Ραντάρ Πεδίου. Το 38.2% των οχημάτων υπερβαίνει το όριο στις επαρχιακές οδούς.",
            ),
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            render_bento_kpi(
                icon="🚑",
                icon_bg="#f0fdf4",
                label="Χρόνος Άφιξης ΕΚΑΒ (Νησιά vs Αττική)",
                value="24.5 min",
                pill_text="έναντι 11.5 min στην Αττική",
                pill_class="pill-emerald",
                hint="Ο διπλάσιος χρόνος προνοσοκομειακής απόκρισης αυξάνει τη μετατροπή τραυματισμών σε KSI.",
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

    col_ext1, col_ext2 = st.columns(2)
    with col_ext1:
        if not ext_kpi_df.empty:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">🇪🇺 Ελλάδα vs. Μέσος Όρος ΕΕ-27 (Ανεξάρτητα KPIs Πεδίου & ESRA3)</span>
                        <span class="card-title-tag">Μπλε = Ελλάδα | Γκρι = ΕΕ-27</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                kpi_comp = ext_kpi_df[ext_kpi_df["indicator_id"] != "CLINICAL_MAIS3_MOTO"].iloc[::-1].copy()
                fig_kpi = go.Figure()
                fig_kpi.add_trace(
                    go.Bar(
                        y=kpi_comp["indicator_name_el"],
                        x=kpi_comp["greece_value_pct"],
                        name="Ελλάδα (%)",
                        orientation="h",
                        marker_color="#2563eb",
                        marker_cornerradius=5,
                    )
                )
                fig_kpi.add_trace(
                    go.Bar(
                        y=kpi_comp["indicator_name_el"],
                        x=kpi_comp["eu27_benchmark_pct"],
                        name="Μέσος Όρος ΕΕ-27 (%)",
                        orientation="h",
                        marker_color="#94a3b8",
                        marker_cornerradius=5,
                    )
                )
                fig_kpi.update_layout(
                    barmode="group",
                    xaxis=dict(title="Ποσοστό Συμμόρφωσης / Ασφαλούς Συμπεριφοράς (%)"),
                    yaxis=dict(title="Δείκτης Οδικής Ασφάλειας (KPI)"),
                )
                style_plotly_chart(
                    fig_kpi,
                    title_text="🇪🇺 Ελλάδα vs. Μέσος Όρος ΕΕ-27 σε Ανεξάρτητα KPIs Πεδίου & ESRA3",
                    height=470,
                    show_legend=True,
                )
                st.plotly_chart(fig_kpi, use_container_width=True, theme=None)
                st.caption(
                    "💡 **Τι αποδεικνύει το γράφημα:** Η υστέρηση της Ελλάδας έναντι της ΕΕ-27 στη χρήση κράνους, ζώνης και τήρησης ταχύτητας εξηγεί ανεξάρτητα την υψηλή βαρύτητα των τροχαίων."
                )

    with col_ext2:
        if not ext_reg_df.empty and not reg_df.empty:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">🔍 Χρήση Κράνους Πεδίου vs. Χρόνος ΕΚΑΒ & Βαρύτητα KSI (13 NUTS-2)</span>
                        <span class="card-title-tag">Μέγεθος = Θερινή Τουριστική Πίεση</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                elstat_reg_mean = (
                    reg_df.groupby("region", as_index=False)["ksi_severity_rate"]
                    .mean()
                    .rename(columns={"ksi_severity_rate": "elstat_ksi_rate"})
                )
                merged_ext = pd.merge(ext_reg_df, elstat_reg_mean, on="region", how="left")
                merged_ext["elstat_ksi_rate_pct"] = (merged_ext["elstat_ksi_rate"] * 100.0).round(2)
                merged_ext["short_region"] = merged_ext["region"].apply(
                    lambda s: s.split("(")[-1].replace(")", "") if "(" in s else s
                )

                fig_sc = px.scatter(
                    merged_ext,
                    x="field_helmet_compliance_pct",
                    y="ems_median_response_min",
                    size="summer_tourism_pressure_index",
                    color="elstat_ksi_rate_pct",
                    text="short_region",
                    color_continuous_scale="YlOrRd",
                    size_max=32,
                    labels={
                        "field_helmet_compliance_pct": "Επιτόπια Χρήση Κράνους (%) [ΕΜΠ / Trendline]",
                        "ems_median_response_min": "Διάμεσος Χρόνος Απόκρισης ΕΚΑΒ (Λεπτά)",
                        "elstat_ksi_rate_pct": "Βαρύτητα KSI (%)",
                        "summer_tourism_pressure_index": "Δείκτης Τουρισμού (×)",
                    },
                    height=470,
                )
                fig_sc.update_traces(
                    textposition="top center",
                    textfont=dict(color="#0f172a", size=11, family="Inter, sans-serif"),
                    marker=dict(line=dict(width=1.2, color="#0f172a")),
                )
                fig_sc.update_yaxes(range=[9.5, 27.5])
                fig_sc.update_xaxes(range=[54.0, 88.0])
                style_plotly_chart(
                    fig_sc,
                    title_text="🔍 Συσχέτιση Κράνους Πεδίου, Χρόνου ΕΚΑΒ & Βαρύτητας KSI (13 NUTS-2)",
                    height=470,
                    show_legend=False,
                )
                st.plotly_chart(fig_sc, use_container_width=True, theme=None)
                st.caption(
                    "💡 **Η Αιτιώδης Αλυσίδα:** Όσο μειώνεται η χρήση κράνους (αριστερά στον οριζόντιο άξονα) και αυξάνεται ο χρόνος άφιξης του ΕΚΑΒ (πάνω στον κάθετο άξονα), "
                    "το ποσοστό KSI εκτοξεύεται πάνω από 22%–25% (Νότιο Αιγαίο, Κρήτη, Ιόνια Νησιά)."
                )

    if not ext_kpi_df.empty:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">📋 Πίνακας 1: Εθνικοί Δείκτες Ανεξάρτητων Ερευνών (EU Trendline / ΕΜΠ / ESRA3 / Κλινικά Μητρώα)</span>
                    <span class="card-title-tag">100% Ανεξάρτητα από την ΕΛΣΤΑΤ</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                ext_kpi_df[
                    [
                        "category",
                        "indicator_name_el",
                        "greece_value_pct",
                        "eu27_benchmark_pct",
                        "gap_vs_eu_pct",
                        "primary_source",
                        "road_safety_impact",
                    ]
                ],
                column_config={
                    "category": st.column_config.TextColumn("Κατηγορία Έρευνας"),
                    "indicator_name_el": st.column_config.TextColumn("Δείκτης Οδικής Ασφάλειας (KPI)"),
                    "greece_value_pct": st.column_config.ProgressColumn(
                        "Ελλάδα (%)", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "eu27_benchmark_pct": st.column_config.ProgressColumn(
                        "Μέσος Όρος ΕΕ-27 (%)", format="%.1f%%", min_value=0, max_value=100
                    ),
                    "gap_vs_eu_pct": st.column_config.NumberColumn("Απόκλιση vs ΕΕ-27", format="%+.1f%%"),
                    "primary_source": st.column_config.TextColumn("Πρωτογενής Πηγή Δεδομένων"),
                    "road_safety_impact": st.column_config.TextColumn("Επίδραση στη Βαρύτητα Τροχαίων (KSI)"),
                },
                hide_index=True,
                use_container_width=True,
            )

    if not ext_reg_df.empty:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">📋 Πίνακας 2: Περιφερειακοί Δείκτες Συμπεριφοράς, Τηλεματικής, Κλίματος & ΕΚΑΒ (13 NUTS-2)</span>
                    <span class="card-title-tag">Πολυπλευρική Τεκμηρίωση Περιφερειών</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                ext_reg_df,
                column_config={
                    "region": st.column_config.TextColumn("Περιφέρεια (NUTS-2)"),
                    "nuts2_code": st.column_config.TextColumn("Κωδικός"),
                    "field_helmet_compliance_pct": st.column_config.ProgressColumn(
                        "Χρήση Κράνους Πεδίου (%)",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                        help="Ποσοστό χρήσης κράνους από επιτόπιες παρατηρήσεις πεδίου.",
                    ),
                    "rural_speed_compliance_pct": st.column_config.ProgressColumn(
                        "Τήρηση Ταχύτητας Υπεραστικών (%)",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                        help="Ποσοστό οχημάτων που κινούνται εντός ορίου ταχύτητας στο επαρχιακό/υπεραστικό δίκτυο.",
                    ),
                    "telematics_harsh_braking_per_100km": st.column_config.NumberColumn(
                        "Απότομα Φρεναρίσματα / 100 km",
                        format="%.1f",
                        help="Δείκτης επιθετικής οδήγησης από δεδομένα τηλεματικής (smartphone sensors).",
                    ),
                    "summer_tourism_pressure_index": st.column_config.NumberColumn(
                        "Δείκτης Θερινής Πίεσης (×)",
                        format="%.2fx",
                        help="Πολλαπλασιαστής τουριστικών διανυκτερεύσεων/αφίξεων το καλοκαίρι σε σχέση με τον χειμώνα.",
                    ),
                    "annual_sunshine_hours": st.column_config.NumberColumn(
                        "Ετήσιες Ώρες Ηλιοφάνειας",
                        format="%d h",
                        help="Κλιματικός δείκτης (Copernicus ERA5 / ΕΜΥ) που ευνοεί την ολόχρονη κυκλοφορία δικύκλων.",
                    ),
                    "winter_rainfall_mm": st.column_config.NumberColumn(
                        "Χειμερινή Βροχόπτωση (mm)",
                        format="%d mm",
                        help="Ύψος βροχόπτωσης φθινοπώρου-χειμώνα (επηρεάζει την ολισθηρότητα οδοστρώματος).",
                    ),
                    "ems_median_response_min": st.column_config.NumberColumn(
                        "Διάμεσος Χρόνος ΕΚΑΒ (min)",
                        format="%.1f min",
                        help="Εκτιμώμενος διάμεσος χρόνος προνοσοκομειακής απόκρισης (Golden Hour).",
                    ),
                    "primary_external_sources": st.column_config.TextColumn("Ανεξάρτητες Πηγές"),
                },
                hide_index=True,
                use_container_width=True,
            )


# ===========================================================================
# VIEW 7: STRATEGIC FORESIGHT 2026–2035 & EU VISION ZERO BACKCASTING
# ===========================================================================
elif section == "🔭 7. Strategic Foresight 2026–2035 (Vision Zero)":
    st.markdown(
        """
        <div class="section-header-bar">
            <div>
                <h3>🔭 Στρατηγική Προόραση Οδικής Ασφάλειας (Strategic Foresight 2026–2035) & Vision Zero</h3>
                <p>Διερεύνηση εναλλακτικών μελλόντων 10ετίας (Κώνος Αληθοφάνειας), Megatrends και Ανάδρομος Σχεδιασμός (Backcasting) για τον στόχο ΕΕ -50% KSI.</p>
            </div>
            <span class="card-title-tag">10-Year Horizon Scanning</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="legend-banner">
            <div class="legend-header">📖 Υπόμνημα Strategic Foresight (Πώς διαφέρει η Προόραση 10ετίας από την απλή Πρόγνωση)</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-term"><span>1. Forecasting vs. Foresight</span><span class="legend-tag">Έννοια</span></div>
                    <p class="legend-desc">Το <b>Forecasting</b> προβλέπει τους επόμενους 12 μήνες (2026). Το <b>Strategic Foresight (2026–2035)</b> χαρτογραφεί <b>4 εναλλακτικά σενάρια 10ετίας</b> ανάλογα με τις τεχνολογικές, κλιματικές και θεσμικές επιλογές της χώρας.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>2. Κώνος Αληθοφάνειας (4 Σενάρια)</span><span class="legend-tag">Cone of Plausibility</span></div>
                    <p class="legend-desc">Από το δυσμενές σενάριο <b>S1 (Αδράνεια & Τουριστική/Κλιματική Πίεση)</b> έως το <b>S4 (Πλήρες Vision Zero)</b>, ο κώνος δείχνει πόσο ανοίγει η «ψαλίδα» των θυμάτων μέσα στη δεκαετία.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>3. Megatrends (Μεγάλες Τάσεις)</span><span class="legend-tag">Horizon Scanning</span></div>
                    <p class="legend-desc">Εξωγενείς δυνάμεις 10ετίας: <b>(+) Αυξητικές:</b> e-scooters/delivery, κλιματική επιμήκυνση τουρισμού, γήρανση οδηγών. <b>(-) Μειωτικές:</b> συστήματα ADAS/AEB οχημάτων, ΒΟΑΚ/αυτοκινητόδρομοι, κάμερες AI.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>4. Backcasting (Στόχος ΕΕ -50% KSI)</span><span class="legend-tag">Οδικός Χάρτης</span></div>
                    <p class="legend-desc">Ο ευρωπαϊκός στόχος απαιτεί <b>μείωση των KSI κατά 50%</b>. Το Backcasting υπολογίζει <b>σε ποιο ακριβώς έτος</b> κάθε Περιφέρεια πετυχαίνει τον στόχο αυτόν και πόσες ζωές σώζονται στη 10ετία.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not fc_df.empty:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">🎛️ Παραμετροποίηση Μακρο-Τάσεων 10ετίας (Megatrend Sensitivity Shocks 2026–2035)</span>
                    <span class="card-title-tag">Δοκιμάστε την ευαισθησία των σεναρίων</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            f1, f2, f3, f4 = st.columns([1.3, 1.0, 1.0, 1.0])
            with f1:
                sel_foresight_reg = st.selectbox(
                    "📍 Επιλογή Εμβέλειας Προόρασης (2026–2035):",
                    options=["Όλη η Ελλάδα (Εθνική Τροχιά 13 Περιφερειών)"] + sorted(fc_df["region"].unique()),
                )
            with f2:
                shock_tourism = st.slider(
                    "☀️ Ένταση Τουριστικής & Κλιματικής Πίεσης",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="1.0 = Κανονική τάση (+1.8%/έτος στα νησιά). >1.0 = Ισχυρότερη επιμήκυνση τουριστικής περιόδου και μικροκινητικότητας.",
                )
            with f3:
                shock_adas = st.slider(
                    "🚗 Ταχύτητα Ανανέωσης Στόλου (ADAS / GSR2)",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Ρυθμός απόσυρσης παλαιών ΙΧ και διείσδυσης αυτόματου φρεναρίσματος (AEB) & Intelligent Speed Assistance.",
                )
            with f4:
                shock_infra = st.slider(
                    "🛣️ Ρυθμός Ολοκλήρωσης Έργων & Καμερών AI",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Ταχύτητα ολοκλήρωσης ΒΟΑΚ/αυτοκινητοδρόμων, στηθαίων 2+1 και ψηφιακών καμερών Τροχαίας.",
                )

        traj_df = generate_foresight_trajectories(
            fc_df,
            start_year=2026,
            end_year=2035,
            tourism_climate_shock=shock_tourism,
            adas_fleet_renewal_speed=shock_adas,
            infra_enforcement_speed=shock_infra,
        )
        backcast_df = compute_regional_backcasting_table(traj_df)
        megatrends_df = get_megatrends_matrix()

        if sel_foresight_reg == "Όλη η Ελλάδα (Εθνική Τροχιά 13 Περιφερειών)":
            view_traj = (
                traj_df.groupby(["year", "scenario_id", "scenario_name", "short_scenario"], as_index=False)
                .agg(
                    baseline_2026_ksi=("baseline_2026_ksi", "sum"),
                    eu_target_50pct_ksi=("eu_target_50pct_ksi", "sum"),
                    projected_ksi_accidents=("projected_ksi_accidents", "sum"),
                    annual_ksi_saved_vs_2026=("annual_ksi_saved_vs_2026", "sum"),
                    annual_ksi_saved_vs_stress=("annual_ksi_saved_vs_stress", "sum"),
                    annual_socioeconomic_benefit_eur_m=("annual_socioeconomic_benefit_eur_m", "sum"),
                )
            )
            scope_title = "Ελλάδα (Σύνολο 13 Περιφερειών)"
        else:
            view_traj = traj_df[traj_df["region"] == sel_foresight_reg].copy()
            scope_title = sel_foresight_reg

        s4_view = view_traj[view_traj["scenario_id"] == "S4_VISION_ZERO"].sort_values("year")
        s1_view = view_traj[view_traj["scenario_id"] == "S1_STRESS"].sort_values("year")

        base_2026_val = int(s4_view.iloc[0]["baseline_2026_ksi"])
        target_50_val = int(s4_view.iloc[0]["eu_target_50pct_ksi"])
        s4_2035_val = int(s4_view.iloc[-1]["projected_ksi_accidents"])
        s1_2035_val = int(s1_view.iloc[-1]["projected_ksi_accidents"])
        cum_10yr_saved = int(s4_view["annual_ksi_saved_vs_2026"].sum())
        cum_10yr_vs_stress = int(s4_view["annual_ksi_saved_vs_stress"].sum())
        cum_10yr_eur_m = float(s4_view["annual_socioeconomic_benefit_eur_m"].sum())

        hit_yrs = s4_view[s4_view["projected_ksi_accidents"] <= target_50_val]["year"].tolist()
        hit_yr_label = f"Έτος {hit_yrs[0]}" if hit_yrs else "Μετά το 2035"

        fk1, fk2, fk3, fk4 = st.columns(4)
        with fk1:
            st.markdown(
                render_bento_kpi(
                    icon="🎯",
                    icon_bg="#ecfdf5",
                    label=f"Στόχος ΕΕ -50% KSI ({scope_title[:18]})",
                    value=f"{target_50_val:,} KSI/έτος",
                    pill_text=f"Επίτευξη: {hit_yr_label}",
                    pill_class="pill-emerald",
                    hint=f"Υποδιπλασιασμός από τα {base_2026_val:,} ετήσια KSI του 2026. Στο Vision Zero επιτυγχάνεται το {hit_yr_label}.",
                ),
                unsafe_allow_html=True,
            )
        with fk2:
            st.markdown(
                render_bento_kpi(
                    icon="🔭",
                    icon_bg="#eff6ff",
                    label="Ετήσια KSI το 2035 (Vision Zero vs Αδράνεια)",
                    value=f"{s4_2035_val:,} vs {s1_2035_val:,}",
                    pill_text=f"-{((base_2026_val - s4_2035_val)/max(1, base_2026_val))*100:.1f}% vs 2026",
                    pill_class="pill-blue",
                    hint="Διαφορά μεταξύ του σεναρίου Vision Zero (S4) και του σεναρίου Αδράνειας/Πίεσης (S1) το έτος 2035.",
                ),
                unsafe_allow_html=True,
            )
        with fk3:
            st.markdown(
                render_bento_kpi(
                    icon="🛡️",
                    icon_bg="#fffbeb",
                    label="Σωρευτικά Αποτρεπόμενα KSI 10ετίας (2026–2035)",
                    value=f"{cum_10yr_saved:,}",
                    pill_text=f"{cum_10yr_vs_stress:,} vs Σενάριο Αδράνειας",
                    pill_class="pill-amber",
                    hint="Συνολικός αριθμός σοβαρών/θανατηφόρων ατυχημάτων που αποτρέπονται αθροιστικά στη δεκαετία 2026–2035.",
                ),
                unsafe_allow_html=True,
            )
        with fk4:
            st.markdown(
                render_bento_kpi(
                    icon="💶",
                    icon_bg="#f0fdf4",
                    label="Σωρευτικό Κοινωνικό Όφελος 10ετίας",
                    value=f"€{cum_10yr_eur_m/1000.0:.2f} Δισ." if cum_10yr_eur_m >= 1000 else f"€{cum_10yr_eur_m:,.1f}M",
                    pill_text="Ορίζοντας 2026–2035",
                    pill_class="pill-emerald",
                    hint="Συνολική κοινωνικοοικονομική εξοικονόμηση της δεκαετίας 2026–2035 (€1.15M ανά KSI).",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

        col_f1, col_f2 = st.columns([1.15, 1.0])
        with col_f1:
            with st.container(border=True):
                st.markdown(
                    f"""
                    <div class="card-title-row">
                        <span class="card-title-text">🔭 Κώνος Αληθοφάνειας 10ετίας (2026–2035) — {scope_title}</span>
                        <span class="card-title-tag">4 Εναλλακτικά Σενάρια Μέλλοντος</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_cone = go.Figure()
                sc_colors = {
                    "S1_STRESS": "#ef4444",
                    "S2_DIGITAL": "#f59e0b",
                    "S3_TECH_INFRA": "#2563eb",
                    "S4_VISION_ZERO": "#10b981",
                }
                for sc_id in ["S1_STRESS", "S2_DIGITAL", "S3_TECH_INFRA", "S4_VISION_ZERO"]:
                    sub_sc = view_traj[view_traj["scenario_id"] == sc_id].sort_values("year")
                    fig_cone.add_trace(
                        go.Scatter(
                            x=sub_sc["year"],
                            y=sub_sc["projected_ksi_accidents"],
                            mode="lines+markers",
                            name=str(sub_sc["short_scenario"].iloc[0]),
                            line=dict(color=sc_colors[sc_id], width=3.4),
                            marker=dict(size=8),
                        )
                    )
                fig_cone.add_hline(
                    y=target_50_val,
                    line_dash="dash",
                    line_color="#047857",
                    annotation_text=f"Στόχος ΕΕ -50% KSI ({target_50_val:,} ατυχήματα/έτος)",
                    annotation_position="bottom right",
                    annotation_font_color="#047857",
                )
                fig_cone.update_layout(
                    xaxis=dict(title="Έτος Ορίζοντα Προόρασης (2026–2035)", dtick=1),
                    yaxis=dict(title="Ετήσια Ατυχήματα KSI (Νεκροί/Βαριά Τραυματίες)"),
                )
                style_plotly_chart(
                    fig_cone,
                    title_text=f"🔭 Κώνος Αληθοφάνειας 10ετίας (2026–2035) — {scope_title}",
                    height=470,
                    show_legend=True,
                )
                st.plotly_chart(fig_cone, use_container_width=True, theme=None)
                st.caption(
                    "💡 **Πώς διαβάζεται ο Κώνος Προόρασης:** Η κόκκινη καμπύλη (**S1**) δείχνει την επιδείνωση λόγω τουριστικής/κλιματικής πίεσης και γήρανσης στόλου. "
                    "Η πράσινη καμπύλη (**S4: Vision Zero**) τέμνει την πράσινη διακεκομμένη γραμμή του ευρωπαϊκού στόχου **-50% KSI**!"
                )

        with col_f2:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">🛡️ Σωρευτικά Αποτρεπόμενα KSI 10ετίας (2026–2035) ανά Περιφέρεια</span>
                        <span class="card-title-tag">Σενάριο Vision Zero</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                bc_plot = backcast_df.sort_values("cumulative_10yr_ksi_saved", ascending=True).copy()
                bc_plot["short_reg"] = bc_plot["region"].apply(
                    lambda s: s.split("(")[-1].replace(")", "") if "(" in s else s
                )
                fig_bc = px.bar(
                    bc_plot,
                    x="cumulative_10yr_ksi_saved",
                    y="short_reg",
                    orientation="h",
                    color="cumulative_10yr_ksi_saved",
                    color_continuous_scale="Tealgrn",
                    text="year_eu_target_achieved",
                    labels={
                        "cumulative_10yr_ksi_saved": "Αποτρεπόμενα Ατυχήματα KSI (2026–2035)",
                        "short_reg": "Περιφέρεια NUTS-2",
                    },
                    height=470,
                )
                fig_bc.update_traces(texttemplate="Στόχος -50%: %{text}", textposition="inside")
                fig_bc.update_layout(coloraxis_showscale=False)
                style_plotly_chart(
                    fig_bc,
                    title_text="🛡️ Σωρευτικά Αποτρεπόμενα KSI 10ετίας & Έτος Στόχου -50%",
                    height=470,
                    show_legend=False,
                )
                st.plotly_chart(fig_bc, use_container_width=True, theme=None)
                st.caption(
                    "💡 **Πώς διαβάζεται:** Δείχνει πόσα σοβαρά/θανατηφόρα ατυχήματα γλιτώνει κάθε Περιφέρεια μέσα στη 10ετία, καθώς και **το έτος που πιάνει τον στόχο -50% της ΕΕ**."
                )

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">📋 Πίνακας 1: Οδικός Χάρτης Backcasting 2026–2035 ανά Περιφέρεια NUTS-2</span>
                    <span class="card-title-tag">Πότε επιτυγχάνεται ο στόχος -50% της ΕΕ</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                backcast_df,
                column_config={
                    "region": st.column_config.TextColumn("Περιφέρεια (NUTS-2)"),
                    "nuts2_code": st.column_config.TextColumn("Κωδικός"),
                    "zone_type": st.column_config.TextColumn("Τύπος Δικτύου"),
                    "baseline_2026_ksi": st.column_config.NumberColumn(
                        "2026 Baseline KSI",
                        help="Αρχικός ετήσιος αριθμός ατυχημάτων KSI το 2026.",
                    ),
                    "eu_target_50pct_ksi": st.column_config.NumberColumn(
                        "Στόχος ΕΕ (-50% KSI)",
                        help="Ο μέγιστος ετήσιος αριθμός KSI που αντιστοιχεί σε μείωση 50%.",
                    ),
                    "stress_2035_ksi": st.column_config.NumberColumn(
                        "2035 Αδράνεια (S1)",
                        help="Ετήσια KSI το 2035 αν δεν ληφθούν μέτρα (λόγω τουριστικής/κλιματικής πίεσης).",
                    ),
                    "vision_zero_2030_ksi": st.column_config.NumberColumn(
                        "2030 Vision Zero (S4)",
                        help="Ετήσια KSI το 2030 στο σενάριο Vision Zero.",
                    ),
                    "vision_zero_2035_ksi": st.column_config.NumberColumn(
                        "2035 Vision Zero (S4)",
                        help="Ετήσια KSI το 2035 στο σενάριο Vision Zero.",
                    ),
                    "reduction_by_2035_pct": st.column_config.ProgressColumn(
                        "Μείωση KSI έως 2035 (%)",
                        format="%.1f%%",
                        min_value=0,
                        max_value=80,
                        help="Ποσοστιαία μείωση των ετήσιων KSI το 2035 σε σχέση με το 2026.",
                    ),
                    "year_eu_target_achieved": st.column_config.TextColumn(
                        "Έτος Επίτευξης -50%",
                        help="Το πρώτο έτος κατά το οποίο η Περιφέρεια πετυχαίνει τον στόχο μείωσης 50%.",
                    ),
                    "cumulative_10yr_ksi_saved": st.column_config.NumberColumn(
                        "Αποτρεπόμενα KSI 10ετίας",
                        format="🛡️ %d",
                        help="Αθροιστικός αριθμός KSI που αποτρέπονται την περίοδο 2026–2035 έναντι του 2026.",
                    ),
                    "cumulative_10yr_saved_vs_stress": st.column_config.NumberColumn(
                        "Αποτρεπόμενα vs Αδράνεια",
                        format="%d",
                        help="Αθροιστικός αριθμός KSI που αποτρέπονται σε σύγκριση με το δυσμενές σενάριο Αδράνειας (S1).",
                    ),
                    "cumulative_10yr_benefit_eur_m": st.column_config.NumberColumn(
                        "Όφελος 10ετίας (€M)",
                        format="€%.1fM",
                        help="Σωρευτικό κοινωνικοοικονομικό όφελος 2026–2035 σε εκατ. ευρώ.",
                    ),
                },
                hide_index=True,
                use_container_width=True,
            )

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">🌐 Πίνακας 2: Μήτρα Μακρο-Τάσεων 10ετίας (Horizon Scanning Megatrends 2026–2035)</span>
                    <span class="card-title-tag">6 Κρίσιμες Δυνάμεις Μετασχηματισμού</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                megatrends_df[
                    [
                        "category",
                        "megatrend_name_el",
                        "direction",
                        "annual_ksi_drift_pct",
                        "primary_affected_regions",
                        "mitigation_lever",
                    ]
                ],
                column_config={
                    "category": st.column_config.TextColumn("Θεματικός Πυλώνας"),
                    "megatrend_name_el": st.column_config.TextColumn("Μακρο-Τάση (Megatrend 2026–2035)"),
                    "direction": st.column_config.TextColumn("Κατεύθυνση Επίδρασης"),
                    "annual_ksi_drift_pct": st.column_config.NumberColumn(
                        "Ετήσια Επίδραση KSI (%)",
                        format="%+.1f%%/έτος",
                        help="Εκτιμώμενη ετήσια ποσοστιαία μεταβολή στη βαρύτητα/συχνότητα KSI.",
                    ),
                    "primary_affected_regions": st.column_config.TextColumn("Περιφέρειες Αιχμής"),
                    "mitigation_lever": st.column_config.TextColumn("Στρατηγικός Μοχλός Αντιμετώπισης / Αξιοποίησης"),
                },
                hide_index=True,
                use_container_width=True,
            )



# ===========================================================================
# VIEW 8: HUMAN-IN-THE-LOOP (HITL) PRESCRIPTIVE DECISION SUPPORT COPILOT
# ===========================================================================
elif section == "🧠 8. Κέντρο Λήψης Αποφάσεων (Human-in-the-Loop Copilot)":
    st.markdown(
        """
        <div class="section-header-bar">
            <div>
                <h3>🧠 Κέντρο Υποστήριξης Επιχειρησιακών Αποφάσεων (Human-in-the-Loop AI Copilot)</h3>
                <p>Ο αλγόριθμος συνθέτει προβλέψεις ML, δεδομένα πεδίου (ΕΜΠ/ΕΚΑΒ/ERA5) και επιχειρησιακούς περιορισμούς για να εισηγηθεί τις 3 βέλτιστες επιλογές — η τελική απόφαση ανήκει στον άνθρωπο (EU AI Act Art. 14).</p>
            </div>
            <span class="card-title-tag">EU AI Act Art. 14 • Human Oversight</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="legend-banner">
            <div class="legend-header">📖 Υπόμνημα Λειτουργίας Human-in-the-Loop (Πώς συνεργάζεται η Τεχνητή Νοημοσύνη με τον Άνθρωπο σε Κρίσιμη Στιγμή)</div>
            <div class="legend-grid">
                <div class="legend-item">
                    <div class="legend-term"><span>1. Ανίχνευση Κρίσιμης Στιγμής</span><span class="legend-tag">Βήμα 1 (Διάγνωση)</span></div>
                    <p class="legend-desc">Το σύστημα συνδυάζει την πρόβλεψη KSI της ΕΛΣΤΑΤ με τους ανεξάρτητους δείκτες πεδίου (κράνος, ταχύτητα, βροχόπτωση, χρόνος ΕΚΑΒ) και εντοπίζει τις ακριβείς αιτίες κινδύνου της Περιφέρειας.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>2. Περιορισμοί Πεδίου (Constraints)</span><span class="legend-tag">Βήμα 2 (Όρια)</span></div>
                    <p class="legend-desc">Ο επικεφαλής ορίζει τον <b>διαθέσιμο χρόνο αντίδρασης</b> (π.χ. 24–72 ώρες για επείγουσα κρίση) και τον <b>διαθέσιμο προϋπολογισμό (€K)</b> ώστε οι προτάσεις να είναι 100% ρεαλιστικές.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>3. Τρεις Εισηγήσεις AI (Options A/B/C)</span><span class="legend-tag">Βήμα 3 (Εισήγηση)</span></div>
                    <p class="legend-desc">Ο αλγόριθμος αξιολογεί τον επιχειρησιακό κατάλογο 10 παρεμβάσεων και προτείνει 3 πακέτα: <b>Επιλογή Α (Μέγιστη Ασφάλεια)</b>, <b>Επιλογή Β (Μέγιστο ROI)</b> και <b>Επιλογή Γ (Άμεση Κινητοποίηση 24–72h)</b>.</p>
                </div>
                <div class="legend-item">
                    <div class="legend-term"><span>4. Ανθρώπινη Απόφαση & Audit Trail</span><span class="legend-tag">Βήμα 4 (Έγκριση)</span></div>
                    <p class="legend-desc">Ο άνθρωπος έχει τον τελικό λόγο: εγκρίνει αυτούσια μία πρόταση ή προσθαφαιρεί μέτρα, καταγράφει την αιτιολόγησή του και εκδίδει το <b>Πρακτικό Επιχειρησιακής Εντολής</b>.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not fc_df.empty and not ext_reg_df.empty and not cat_df.empty:
        month_names_hitl = {
            1: "1 - Ιανουάριος (Χειμερινό Δίκτυο)",
            2: "2 - Φεβρουάριος (Χειμερινό Δίκτυο)",
            3: "3 - Μάρτιος",
            4: "4 - Απρίλιος (Έξοδος Πάσχα)",
            5: "5 - Μάιος",
            6: "6 - Ιούνιος (Έναρξη Θερινής Αιχμής)",
            7: "7 - Ιούλιος (Θερινή Αιχμή)",
            8: "8 - Αύγουστος (Κορύφωση Κινδύνου)",
            9: "9 - Σεπτέμβριος",
            10: "10 - Οκτώβριος (Πρώτες Βροχές)",
            11: "11 - Νοέμβριος (Φθινοπωρινές Βροχοπτώσεις)",
            12: "12 - Δεκέμβριος (Εορταστική Έξοδος & Κακοκαιρία)",
        }

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">🎛️ Βήμα 1: Ορισμός Κρίσιμης Στιγμής & Επιχειρησιακών Περιορισμών Διοίκησης</span>
                    <span class="card-title-tag">Ρυθμίστε τις συνθήκες της απόφασης</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            c_reg, c_mon, c_crisis, c_hor, c_bud = st.columns([1.3, 1.1, 1.2, 1.15, 1.0])
            with c_reg:
                hitl_reg = st.selectbox(
                    "📍 Περιφέρεια Ευθύνης (NUTS-2):",
                    options=sorted(fc_df["region"].unique()),
                    index=sorted(fc_df["region"].unique()).index("South Aegean (Νότιο Αιγαίο)")
                    if "South Aegean (Νότιο Αιγαίο)" in fc_df["region"].unique()
                    else 0,
                )
            with c_mon:
                hitl_month = st.selectbox(
                    "📅 Μήνας Επιχειρησιακής Απόφασης:",
                    options=list(month_names_hitl.keys()),
                    format_func=lambda k: month_names_hitl[k],
                    index=7,
                )
            with c_crisis:
                crisis_map = {
                    "AUTO": "🔍 Αυτόματη Διάγνωση από τα Δεδομένα",
                    "TOURISM_MOTO": "☀️ Θερινή Τουριστική Αιχμή & Δίκυκλα",
                    "WEATHER_RAIN": "🌧️ Έκτακτη Κακοκαιρία / Ολισθηρότητα",
                    "RURAL_SPEED": "🛣️ Υψηλές Ταχύτητες Επαρχιακού Δικτύου",
                    "URBAN_VRU": "🏙️ Αστικός Φόρτος & Ευάλωτοι Πεζοί",
                }
                hitl_crisis = st.selectbox(
                    "⚡ Είδος Έκτακτης Συνθήκης / Κρίσης:",
                    options=list(crisis_map.keys()),
                    format_func=lambda k: crisis_map[k],
                )
            with c_hor:
                horizon_map = {
                    "ALL": "🏗️ Όλοι οι Ορίζοντες (Άμεσα + Δομικά)",
                    "UP_TO_SEASONAL": "📅 Έως 1–3 Μήνες (Άμεσα + Εποχικά)",
                    "24H_ONLY": "⚡ Μόνο Άμεση Κρίση (24–72 Ώρες)",
                }
                hitl_horizon = st.selectbox(
                    "⏱️ Διαθέσιμος Χρόνος Αντίδρασης:",
                    options=list(horizon_map.keys()),
                    format_func=lambda k: horizon_map[k],
                )
            with c_bud:
                hitl_budget = st.slider(
                    "💰 Ανώτατο Όριο Προϋπολογισμού (€K):",
                    min_value=50,
                    max_value=950,
                    value=320,
                    step=25,
                    help="Μέγιστος διαθέσιμος προϋπολογισμός σε χιλιάδες ευρώ (€K) για τη συγκεκριμένη Περιφέρεια.",
                )

        fc_row_hitl = fc_df[(fc_df["region"] == hitl_reg) & (fc_df["month"] == hitl_month)].iloc[0]
        ext_row_hitl = ext_reg_df[ext_reg_df["region"] == hitl_reg].iloc[0]

        diagnosis = diagnose_critical_moment(fc_row_hitl, ext_row_hitl, crisis_override=hitl_crisis)
        scored_df = score_interventions_for_context(cat_df, fc_row_hitl, diagnosis, horizon_filter=hitl_horizon)
        options_dict = generate_three_hitl_options(scored_df, fc_row_hitl, max_budget_eur_k=float(hitl_budget))

        triggers_html = " • ".join([f"<b>{t}</b>" for t in diagnosis["triggers"]])
        st.markdown(
            f"""
            <div class="policy-box" style="border-left-color:#2563eb;">
                <span class="risk-pill risk-critical">{diagnosis['urgency_level']} — Δείκτης Κρισιμότητας: {diagnosis['criticality_index']}/100</span>
                <p style="margin:0.3rem 0; color:#0f172a; font-size:0.93rem;">
                    <b>📡 Αυτόματη Διάγνωση Αισθητήρων & Μοντέλου ML ({hitl_reg}, Μήνας {hitl_month}):</b>
                    Προβλεπόμενο KSI <b>{fc_row_hitl['predicted_ksi_rate_pct']:.2f}%</b> (Πιθανότητα Έξαρσης: <b>{fc_row_hitl['severe_regime_prob_pct']:.1f}%</b>).
                </p>
                <p style="margin:0.35rem 0 0 0; color:#1e3a8a; font-size:0.9rem;">
                    <b>🔎 Εντοπισθέντες Παράγοντες Κινδύνου Πεδίου:</b> {triggers_html}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Step 2: Display the 3 AI Prescriptive Options Side-by-Side
        st.markdown(
            """
            <div class="card-title-row" style="margin-top:0.4rem;">
                <span class="card-title-text">🤖 Βήμα 2: Οι 3 Εισηγήσεις της Τεχνητής Νοημοσύνης (Ranked Prescriptive Options)</span>
                <span class="card-title-tag">Συγκρίνετε τις 3 προτάσεις πριν πάρετε την τελική απόφαση</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        opt_cols = st.columns(3)
        opt_keys = ["OPTION_A", "OPTION_B", "OPTION_C"]
        opt_badges = ["pill-emerald", "pill-blue", "pill-amber"]
        opt_icons = ["🛡️", "⚖️", "⚡"]

        for idx, okey in enumerate(opt_keys):
            opt_info = options_dict[okey]
            m_info = opt_info["metrics"]
            names_bullets = "".join(
                [f"<li style='margin-bottom:0.2rem;'>{nm}</li>" for nm in m_info["selected_names"]]
            )
            with opt_cols[idx]:
                st.markdown(
                    f"""
                    <div class="bento-card" style="border-top: 4px solid {'#10b981' if idx==0 else '#2563eb' if idx==1 else '#f59e0b'};">
                        <div class="bento-top">
                            <div class="bento-icon" style="background:#f8fafc;">{opt_icons[idx]}</div>
                            <span class="bento-pill {opt_badges[idx]}">{opt_info['code']}</span>
                        </div>
                        <div style="font-size:1.02rem; font-weight:800; color:#0f172a; margin-bottom:0.25rem;">
                            {opt_info['title']}
                        </div>
                        <div style="font-size:0.79rem; color:#475569; margin-bottom:0.65rem; line-height:1.35;">
                            {opt_info['subtitle']}
                        </div>
                        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.6rem 0.75rem; font-size:0.81rem; color:#0f172a; margin-bottom:0.6rem;">
                            <b>📉 Μείωση KSI:</b> -{m_info['combined_ksi_reduction_pct']:.1f}% (Νέο KSI: {m_info['new_ksi_rate_pct']:.2f}%)<br>
                            <b>🛡️ Αποτρεπόμενα KSI (Ετήσιο Ισοδ.):</b> {m_info['annual_equiv_ksi_saved']:.1f} ατυχήματα<br>
                            <b>💰 Κόστος Πακέτου:</b> €{m_info['total_cost_eur_k']:.0f}K | <b>ROI:</b> {m_info['portfolio_roi']:.1f}×<br>
                            <b>👮 Προσωπικό Πεδίου:</b> {m_info['total_personnel']} μονάδες
                        </div>
                        <div style="font-size:0.78rem; color:#1e293b;">
                            <b>Προτεινόμενες Δράσεις ({m_info['selected_count']}):</b>
                            <ul style="margin:0.3rem 0 0 1.1rem; padding:0;">{names_bullets}</ul>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:0.9rem;'></div>", unsafe_allow_html=True)

        # Step 3: Human-in-the-Loop Decision Gate (Final Human Choice & Customization)
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">👨‍✈️ Βήμα 3: Τελική Απόφαση Ανθρώπου (Human-in-the-Loop Decision Gate & Προσαρμογή)</span>
                    <span class="card-title-tag">Ο άνθρωπος επιλέγει, τροποποιεί και εγκρίνει το τελικό σχέδιο</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            hcol1, hcol2 = st.columns([1.05, 1.35])
            with hcol1:
                human_base_choice = st.radio(
                    "1️⃣ Επιλέξτε Βάση Εισήγησης AI:",
                    options=["OPTION_A", "OPTION_B", "OPTION_C"],
                    format_func=lambda k: {
                        "OPTION_A": "🛡️ Επιλογή Α — Μέγιστη Αποτροπή KSI (Recommended)",
                        "OPTION_B": "⚖️ Επιλογή Β — Βέλτιστο Κόστος/Όφελος (ROI)",
                        "OPTION_C": "⚡ Επιλογή Γ — Άμεση Κινητοποίηση 24–72 Ωρών",
                    }[k],
                )
                commander_role = st.selectbox(
                    "2️⃣ Ρόλος Υπεύθυνου Λήψης Απόφασης:",
                    options=[
                        "Διοικητής Τροχαίας / ΕΛ.ΑΣ. Περιφέρειας",
                        "Συντονιστής Επιχειρήσεων ΕΚΑΒ & Υπ. Υγείας",
                        "Περιφερειάρχης / Διεύθυνση Τεχνικών Έργων",
                        "Γενικός Γραμματέας Πολιτικής Προστασίας / Οδικής Ασφάλειας",
                    ],
                )
                decision_status = st.selectbox(
                    "3️⃣ Κατάσταση Ανθρώπινης Έγκρισης:",
                    options=[
                        "✅ ΕΓΚΡΙΘΗΚΕ ΑΥΤΟΥΣΙΑ Η ΕΙΣΗΓΗΣΗ AI",
                        "🛠️ ΕΓΚΡΙΘΗΚΕ ΜΕ ΑΝΘΡΩΠΙΝΕΣ ΤΡΟΠΟΠΟΙΗΣΕΙΣ (Custom Override)",
                        "⏸️ ΣΕ ΕΠΙΦΥΛΑΚΗ / ΑΠΑΙΤΕΙΤΑΙ ΕΠΙΠΛΕΟΝ ΕΛΕΓΧΟΣ ΠΕΔΙΟΥ",
                    ],
                )

            with hcol2:
                default_ids = options_dict[human_base_choice]["ids"]
                id_to_label = {
                    str(r["intervention_id"]): f"{r['intervention_id']} — {r['name_el']} (€{r['unit_cost_eur_k']:.0f}K | -{r['contextual_ksi_reduction_pct']:.1f}% KSI | ROI {r['roi_benefit_cost_ratio']:.0f}×)"
                    for _, r in scored_df.iterrows()
                }
                final_selected_ids = st.multiselect(
                    "4️⃣ Προσαρμογή Παρεμβάσεων από τον Άνθρωπο (μπορείτε να προσθέσετε ή να αφαιρέσετε μέτρα):",
                    options=list(id_to_label.keys()),
                    default=[i for i in default_ids if i in id_to_label],
                    format_func=lambda k: id_to_label.get(k, k),
                    help="Ως τελικός λήπτης απόφασης, μπορείτε να τροποποιήσετε ελεύθερα τη λίστα μέτρων της Τεχνητής Νοημοσύνης.",
                )
                human_notes = st.text_area(
                    "5️⃣ Αιτιολόγηση Ανθρώπινης Απόφασης / Ειδικές Εντολές Πεδίου (Audit Trail):",
                    value=(
                        f"Εγκρίνεται η επιχειρησιακή ανάπτυξη για την Περιφέρεια {hitl_reg} (Μήνας {hitl_month}) "
                        f"με προτεραιότητα στην προστασία ανθρώπινης ζωής και τον έλεγχο των τοπικών παραγόντων κινδύνου."
                    ),
                    height=88,
                )

        final_eval = evaluate_intervention_portfolio(scored_df, final_selected_ids, fc_row_hitl)

        # Final Approved Portfolio KPI Row
        fk1, fk2, fk3, fk4 = st.columns(4)
        with fk1:
            st.markdown(
                render_bento_kpi(
                    icon="✅",
                    icon_bg="#ecfdf5",
                    label="Τελική Μείωση Βαρύτητας KSI",
                    value=f"-{final_eval['combined_ksi_reduction_pct']:.1f}%",
                    pill_text=f"Νέο KSI: {final_eval['new_ksi_rate_pct']:.2f}% (από {fc_row_hitl['predicted_ksi_rate_pct']:.2f}%)",
                    pill_class="pill-emerald",
                    hint="Συνδυαστική ποσοστιαία μείωση KSI του εγκεκριμένου ανθρώπινου σχεδίου.",
                ),
                unsafe_allow_html=True,
            )
        with fk2:
            st.markdown(
                render_bento_kpi(
                    icon="🛡️",
                    icon_bg="#eff6ff",
                    label="Αποτρεπόμενα KSI (Ετήσιο Ισοδύναμο)",
                    value=f"{final_eval['annual_equiv_ksi_saved']:.1f}",
                    pill_text=f"{final_eval['monthly_ksi_saved']:.1f} KSI / μήνα αιχμής",
                    pill_class="pill-blue",
                    hint="Αριθμός σοβαρών/θανατηφόρων ατυχημάτων που αποτρέπονται στην Περιφέρεια.",
                ),
                unsafe_allow_html=True,
            )
        with fk3:
            st.markdown(
                render_bento_kpi(
                    icon="💰",
                    icon_bg="#fffbeb",
                    label="Συνολικό Κόστος Εγκεκριμένων Μέτρων",
                    value=f"€{final_eval['total_cost_eur_k']:.0f}K",
                    pill_text=f"Όριο: €{hitl_budget}K ({final_eval['total_personnel']} μονάδες)",
                    pill_class="pill-amber" if final_eval["total_cost_eur_k"] <= hitl_budget else "pill-rose",
                    hint="Άθροισμα κόστους και απαιτούμενου προσωπικού των επιλεγμένων παρεμβάσεων.",
                ),
                unsafe_allow_html=True,
            )
        with fk4:
            st.markdown(
                render_bento_kpi(
                    icon="📈",
                    icon_bg="#f0fdf4",
                    label="Κοινωνικό Όφελος & Δείκτης ROI",
                    value=f"€{final_eval['socioeconomic_benefit_eur_m']:.2f}M",
                    pill_text=f"Απόδοση ROI: {final_eval['portfolio_roi']:.1f}×",
                    pill_class="pill-emerald",
                    hint="Κάθε €1 που επενδύεται στα επιλεγμένα μέτρα επιστρέφει πολλαπλάσιο κοινωνικό όφελος.",
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

        # Step 4: Concrete Chronological Execution Plan (Next Steps for Selected HITL Portfolio)
        chosen_rows = scored_df[scored_df["intervention_id"].isin(final_selected_ids)].copy()
        imm_rows = chosen_rows[chosen_rows["deployment_horizon"] == "24H_CRITICAL"]
        seas_rows = chosen_rows[chosen_rows["deployment_horizon"] == "SEASONAL_30D"]
        struct_rows = chosen_rows[chosen_rows["deployment_horizon"].isin(["ANNUAL_INFRA", "STRUCTURAL_6M"])]

        def _fmt_phase_items(df_sub: pd.DataFrame, fallback_txt: str) -> str:
            if df_sub.empty:
                return f"<i>{fallback_txt}</i>"
            items = [
                f"• <b>{r['name_el']}</b> ({r['lead_agency_el']} | €{r['unit_cost_eur_k']:.0f}K | {int(r['personnel_units'])} μον. | -{r['contextual_ksi_reduction_pct']:.1f}% KSI)"
                for _, r in df_sub.iterrows()
            ]
            return "<br>".join(items)

        phase1_html = _fmt_phase_items(
            imm_rows, "Δεν επιλέχθηκαν δράσεις άμεσης κινητοποίησης 24–72 ωρών."
        )
        phase2_html = _fmt_phase_items(
            seas_rows, "Δεν επιλέχθηκαν εποχικές/τακτικές δράσεις 1–3 μηνών."
        )
        phase3_html = _fmt_phase_items(
            struct_rows, "Δεν επιλέχθηκαν δομικά έργα υποδομής 3–12 μηνών."
        )
        phase4_html = (
            f"• <b>Στόχος Βαρύτητας KSI:</b> Πτώση από <b>{fc_row_hitl['predicted_ksi_rate_pct']:.2f}% σε {final_eval['new_ksi_rate_pct']:.2f}%</b> (-{final_eval['combined_ksi_reduction_pct']:.1f}%).<br>"
            f"• <b>Έλεγχος Πεδίου:</b> Συμμόρφωση κράνους &gt;85% (από {diagnosis['helmet_pct']:.1f}%) & διάμεσος χρόνος ΕΚΑΒ &lt;14 λεπτά (από {diagnosis['ems_min']:.1f} λεπτά).<br>"
            f"• <b>Επανεξέταση (Review):</b> Εβδομαδιαία αναφορά αποτελεσμάτων από: <b>{', '.join(final_eval['agencies_involved']) if final_eval['agencies_involved'] else 'Διοίκηση'}</b>."
        )

        st.markdown(
            f"""
            <div class="legend-banner" style="background: linear-gradient(135deg, #f0fdf4 0%, #eff6ff 100%); border-color: #86efac;">
                <div class="legend-header" style="color: #065f46; justify-content: space-between; flex-wrap: wrap;">
                    <span>🚀 Βήμα 4: Επόμενα Επιχειρησιακά Βήματα Εκτέλεσης (Step-by-Step Οδικός Χάρτης Απόφασης)</span>
                    <span class="bento-pill pill-emerald">{final_eval['selected_count']} Εγκεκριμένες Δράσεις • €{final_eval['total_cost_eur_k']:.0f}K</span>
                </div>
                <div class="legend-grid">
                    <div class="legend-item" style="border-color: #bbf7d0;">
                        <div class="legend-term"><span>Φάση 1: Άμεση Εντολή (0–72 Ώρες)</span><span class="legend-tag">{len(imm_rows)} Δράσεις</span></div>
                        <p class="legend-desc">{phase1_html}</p>
                    </div>
                    <div class="legend-item" style="border-color: #bfdbfe;">
                        <div class="legend-term"><span>Φάση 2: Τακτική Ανάπτυξη (1–3 Μήνες)</span><span class="legend-tag">{len(seas_rows)} Δράσεις</span></div>
                        <p class="legend-desc">{phase2_html}</p>
                    </div>
                    <div class="legend-item" style="border-color: #fde68a;">
                        <div class="legend-term"><span>Φάση 3: Δομικά Έργα (3–12 Μήνες)</span><span class="legend-tag">{len(struct_rows)} Δράσεις</span></div>
                        <p class="legend-desc">{phase3_html}</p>
                    </div>
                    <div class="legend-item" style="border-color: #c7d2fe;">
                        <div class="legend-term"><span>Φάση 4: Έλεγχος KPIs & Αναφορά</span><span class="legend-tag">Audit Πεδίου</span></div>
                        <p class="legend-desc">{phase4_html}</p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_hitl_chart, col_hitl_proto = st.columns([1.15, 1.0])
        with col_hitl_chart:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">📊 Χάρτης Κόστους vs. Αποτελεσματικότητας Παρεμβάσεων (AI Decision Matrix)</span>
                        <span class="card-title-tag">Μέγεθος = Καταλληλότητα AI | Πράσινο Περίγραμμα = Επιλογή Ανθρώπου</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                plot_cat = scored_df.copy()
                plot_cat["Κατάσταση"] = plot_cat["intervention_id"].apply(
                    lambda i: "✅ Εγκεκριμένο από Άνθρωπο" if i in final_selected_ids else "⚪ Διαθέσιμη Εναλλακτική"
                )
                fig_hitl = px.scatter(
                    plot_cat,
                    x="unit_cost_eur_k",
                    y="contextual_ksi_reduction_pct",
                    size="suitability_score",
                    color="Κατάσταση",
                    text="intervention_id",
                    hover_name="name_el",
                    color_discrete_map={
                        "✅ Εγκεκριμένο από Άνθρωπο": "#10b981",
                        "⚪ Διαθέσιμη Εναλλακτική": "#94a3b8",
                    },
                    size_max=28,
                    labels={
                        "unit_cost_eur_k": "Κόστος Παρέμβασης (€ Χιλιάδες)",
                        "contextual_ksi_reduction_pct": "Μείωση Βαρύτητας KSI (%)",
                        "suitability_score": "Καταλληλότητα AI (0-100)",
                    },
                    height=430,
                )
                fig_hitl.update_traces(
                    textposition="top center",
                    textfont=dict(color="#0f172a", size=11, family="Inter, sans-serif"),
                    marker=dict(line=dict(width=1.5, color="#0f172a")),
                )
                style_plotly_chart(
                    fig_hitl,
                    title_text="📊 Κόστος (€K) vs. Μείωση KSI (%) ανά Επιχειρησιακή Δράση",
                    height=430,
                    show_legend=True,
                )
                st.plotly_chart(fig_hitl, use_container_width=True, theme=None)

        with col_hitl_proto:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="card-title-row">
                        <span class="card-title-text">📜 Πρακτικό Επιχειρησιακής Εντολής (EU AI Act Art. 14 Audit Trail)</span>
                        <span class="card-title-tag">Έτοιμο προς Υπογραφή & Εκτέλεση</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                agencies_str = ", ".join(final_eval["agencies_involved"]) if final_eval["agencies_involved"] else "—"
                actions_md = "\n".join([f"- **{nm}** ({hz} — {ag})" for nm, hz, ag in zip(chosen_rows["name_el"], chosen_rows["horizon_label_el"], chosen_rows["lead_agency_el"])]) or "- *(Καμία επιλεγμένη δράση)*"
                actions_html_items = "".join(
                    [
                        f"<li style='margin-bottom:0.32rem; color:#0f172a;'><b>{nm}</b> "
                        f"<span style='background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; padding:0.08rem 0.45rem; border-radius:6px; font-size:0.75rem; font-weight:700;'>{hz}</span> "
                        f"<span style='color:#475569; font-size:0.8rem;'>— {ag}</span></li>"
                        for nm, hz, ag in zip(
                            chosen_rows["name_el"],
                            chosen_rows["horizon_label_el"],
                            chosen_rows["lead_agency_el"],
                        )
                    ]
                ) or "<li style='color:#64748b;'><i>Καμία επιλεγμένη δράση</i></li>"

                protocol_html = f"""
                <div style="background:#f8fafc; border:1px solid #cbd5e1; border-radius:12px; padding:1rem 1.15rem; color:#0f172a; font-size:0.86rem; line-height:1.55; margin-bottom:0.75rem;">
                    <div style="font-size:0.94rem; font-weight:800; color:#0f172a; border-bottom:2px solid #cbd5e1; padding-bottom:0.45rem; margin-bottom:0.65rem;">
                        ΠΡΑΚΤΙΚΟ ΛΗΨΗΣ ΑΠΟΦΑΣΗΣ ΟΔΙΚΗΣ ΑΣΦΑΛΕΙΑΣ (HITL)
                    </div>
                    <div style="margin-bottom:0.25rem;"><b>• Περιφέρεια & Μήνας:</b> <span style="color:#1e3a8a; font-weight:700;">{hitl_reg}</span> | <span style="color:#1e3a8a; font-weight:700;">{month_names_hitl[hitl_month]}</span></div>
                    <div style="margin-bottom:0.25rem;"><b>• Επίπεδο Κρισιμότητας:</b> <span style="color:#b91c1c; font-weight:700;">{diagnosis['urgency_level']} ({diagnosis['criticality_index']}/100)</span></div>
                    <div style="margin-bottom:0.25rem;"><b>• Υπεύθυνος Διοίκησης (HITL):</b> <span style="color:#0f172a; font-weight:600;">{commander_role}</span></div>
                    <div style="margin-bottom:0.25rem;"><b>• Κατάσταση Απόφασης:</b> <span style="color:#047857; font-weight:700;">{decision_status}</span></div>
                    <div style="margin-bottom:0.25rem;"><b>• Εμπλεκόμενοι Φορείς Εκτέλεσης:</b> <span style="color:#1e293b;">{agencies_str}</span></div>
                    <div style="margin-bottom:0.25rem;"><b>• Προϋπολογισμός & Προσωπικό:</b> <span style="background:#fffbeb; color:#b45309; border:1px solid #fde68a; padding:0.08rem 0.45rem; border-radius:6px; font-weight:700;">€{final_eval['total_cost_eur_k']:.0f}K</span> | <span style="font-weight:700;">{final_eval['total_personnel']} μονάδες πεδίου</span></div>
                    <div style="margin-bottom:0.65rem;"><b>• Αναμενόμενο Αποτέλεσμα:</b> Μείωση KSI κατά <b style="color:#047857;">-{final_eval['combined_ksi_reduction_pct']:.1f}%</b> (<b>{final_eval['annual_equiv_ksi_saved']:.1f}</b> αποτρεπόμενα KSI/έτος, Όφελος <b style="color:#047857;">€{final_eval['socioeconomic_benefit_eur_m']:.2f}M</b>, ROI <b>{final_eval['portfolio_roi']:.1f}×</b>)</div>
                    <div style="font-weight:800; color:#0f172a; margin-top:0.5rem; margin-bottom:0.25rem;">Εγκεκριμένες Δράσεις Πεδίου & Χρονικός Ορίζοντας:</div>
                    <ul style="margin:0 0 0.65rem 1.15rem; padding:0;">{actions_html_items}</ul>
                    <div style="font-weight:800; color:#0f172a; margin-bottom:0.25rem;">Επόμενα Επιχειρησιακά Βήματα (Action Plan):</div>
                    <ol style="margin:0 0 0.65rem 1.15rem; padding:0; color:#0f172a;">
                        <li><b>Φάση 1 (0–72 Ώρες):</b> Άμεση ανάπτυξη <b>{len(imm_rows)}</b> επειγουσών δράσεων πεδίου.</li>
                        <li><b>Φάση 2 (1–3 Μήνες):</b> Ενεργοποίηση <b>{len(seas_rows)}</b> εποχικών/τακτικών παρεμβάσεων.</li>
                        <li><b>Φάση 3 (3–12 Μήνες):</b> Υλοποίηση <b>{len(struct_rows)}</b> δομικών έργων υποδομής.</li>
                        <li><b>Φάση 4 (Έλεγχος KPI):</b> Μηνιαία επαλήθευση πτώσης KSI σε <b>{final_eval['new_ksi_rate_pct']:.2f}%</b>.</li>
                    </ol>
                    <div style="font-weight:800; color:#0f172a; margin-bottom:0.25rem;">Αιτιολόγηση Διοίκησης:</div>
                    <div style="background:#ffffff; border-left:4px solid #2563eb; border-radius:6px; padding:0.55rem 0.8rem; color:#1e293b; font-style:italic;">
                        {human_notes}
                    </div>
                </div>
                """

                protocol_md = f"""ΠΡΑΚΤΙΚΟ ΛΗΨΗΣ ΑΠΟΦΑΣΗΣ ΟΔΙΚΗΣ ΑΣΦΑΛΕΙΑΣ (HITL)
==================================================
- Περιφέρεια & Μήνας: {hitl_reg} | {month_names_hitl[hitl_month]}
- Επίπεδο Κρισιμότητας: {diagnosis['urgency_level']} ({diagnosis['criticality_index']}/100)
- Υπεύθυνος Διοίκησης (Human-in-the-Loop): {commander_role}
- Κατάσταση Απόφασης: {decision_status}
- Εμπλεκόμενοι Φορείς Εκτέλεσης: {agencies_str}
- Προϋπολογισμός & Προσωπικό: €{final_eval['total_cost_eur_k']:.0f}K | {final_eval['total_personnel']} μονάδες πεδίου
- Αναμενόμενο Αποτέλεσμα: Μείωση KSI κατά -{final_eval['combined_ksi_reduction_pct']:.1f}% ({final_eval['annual_equiv_ksi_saved']:.1f} αποτρεπόμενα KSI / έτος, Όφελος €{final_eval['socioeconomic_benefit_eur_m']:.2f}M, ROI {final_eval['portfolio_roi']:.1f}×)

Εγκεκριμένες Δράσεις Πεδίου & Χρονικός Ορίζοντας:
{actions_md}

Επόμενα Επιχειρησιακά Βήματα (Action Plan):
1. Φάση 1 (0–72 Ώρες): Άμεση ανάπτυξη {len(imm_rows)} επειγουσών δράσεων πεδίου.
2. Φάση 2 (1–3 Μήνες): Ενεργοποίηση {len(seas_rows)} εποχικών/τακτικών παρεμβάσεων.
3. Φάση 3 (3–12 Μήνες): Υλοποίηση {len(struct_rows)} δομικών έργων υποδομής.
4. Φάση 4 (Έλεγχος KPI): Μηνιαία επαλήθευση πτώσης KSI σε {final_eval['new_ksi_rate_pct']:.2f}%.

Αιτιολόγηση Διοίκησης:
{human_notes}
"""
                st.markdown(protocol_html, unsafe_allow_html=True)
                st.download_button(
                    label="📥 Λήψη Εγκεκριμένου Πρακτικού Απόφασης (.md / .txt)",
                    data=protocol_md.encode("utf-8"),
                    file_name=f"HITL_Decision_Protocol_{hitl_reg[:12].strip()}_M{hitl_month}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        with st.container(border=True):
            st.markdown(
                """
                <div class="card-title-row">
                    <span class="card-title-text">📋 Πλήρης Πίνακας Αξιολόγησης & Ιεράρχησης Παρεμβάσεων για την Επιλεγμένη Στιγμή</span>
                    <span class="card-title-tag">Operational Interventions Catalog (SafetyCube / ERSO / ΕΜΠ)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(
                scored_df[
                    [
                        "intervention_id",
                        "name_el",
                        "horizon_label_el",
                        "lead_agency_el",
                        "suitability_score",
                        "contextual_ksi_reduction_pct",
                        "unit_cost_eur_k",
                        "personnel_units",
                        "roi_benefit_cost_ratio",
                        "ai_rationale_el",
                    ]
                ],
                column_config={
                    "intervention_id": st.column_config.TextColumn("Κωδικός"),
                    "name_el": st.column_config.TextColumn("Επιχειρησιακή Παρέμβαση"),
                    "horizon_label_el": st.column_config.TextColumn("Χρόνος Ανάπτυξης"),
                    "lead_agency_el": st.column_config.TextColumn("Φορέας Εκτέλεσης"),
                    "suitability_score": st.column_config.ProgressColumn(
                        "Καταλληλότητα AI (0-100)", format="%.1f", min_value=0, max_value=100
                    ),
                    "contextual_ksi_reduction_pct": st.column_config.NumberColumn(
                        "Μείωση KSI (%)", format="-%.1f%%"
                    ),
                    "unit_cost_eur_k": st.column_config.NumberColumn("Κόστος (€K)", format="€%.0fK"),
                    "personnel_units": st.column_config.NumberColumn("Προσωπικό", format="%d μον."),
                    "roi_benefit_cost_ratio": st.column_config.NumberColumn("Δείκτης ROI", format="%.1f×"),
                    "ai_rationale_el": st.column_config.TextColumn("Αιτιολόγηση Εισήγησης AI (Γιατί προτείνεται)"),
                },
                hide_index=True,
                use_container_width=True,
            )
