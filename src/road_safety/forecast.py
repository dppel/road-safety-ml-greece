"""
Prospective Regional KSI Forecasting & Policy Recommendation Engine (`forecast.py`)
===================================================================================
Trains calibrated classification and regression ensembles on the full verified
ELSTAT Regional Monthly Panel (2021-2025) to generate forward-looking 12-month
forecasts for all 13 Greek NUTS-2 Regions:
  1. `predicted_ksi_rate_pct`: Expected % of injury collisions resulting in death or serious injury.
  2. `severe_regime_prob_pct`: Probability P(KSI Rate >= 21%) in that region-month.
  3. `expected_ksi_collisions`: Expected monthly count of fatal + serious injury collisions.
  4. `risk_tier`: Actionable operational alert level (`CRITICAL`, `HIGH`, `MODERATE`, `URBAN_VOLUME_FOCUS`).
  5. `recommended_interventions`: Evidence-based policy, enforcement, and infrastructure actions
     tailored to each region's empirical risk profile (urban vs. rural/island, motorcycle share, seasonality).
"""

from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline

from road_safety.features import (
    REGIONAL_CATEGORICAL_FEATURES,
    REGIONAL_NUMERIC_FEATURES,
    build_regional_panel_features,
    extract_regional_X_y,
)
from road_safety.preprocessing import build_preprocessing_transformer


def assign_policy_recommendation(
    region: str,
    month: int,
    urban_ratio: float,
    motorcycle_share: float,
    pred_ksi_rate: float,
) -> Tuple[str, str, str]:
    """
    Maps a region's forecasted KSI severity rate and structural exposure profile
    to a Risk Tier, Primary Risk Mechanism, and Targeted Policy Intervention
    following the EU Safe System (Vision Zero) framework.
    """
    is_summer = month in (6, 7, 8, 9)

    if urban_ratio >= 0.60:
        # Metropolitan centers (Attica, Central Macedonia): High collision volume, lower KSI %
        tier = "URBAN VOLUME FOCUS (Υψηλός Όγκος / Ευάλωτοι Χρήστες)"
        mechanism = (
            f"Metropolitan high-density network (Urban: {urban_ratio:.0%}, Two-Wheelers: {motorcycle_share:.0%}); "
            "high absolute crash frequency at urban intersections."
        )
        intervention = (
            "1) Enforce 30 km/h urban residential zones & red-light/speed cameras on main arterials; "
            "2) Protected pedestrian crossings & high-contrast night lighting; "
            "3) Targeted helmet & lane-filtering safety campaigns for urban two-wheelers."
        )
    elif pred_ksi_rate >= 0.22 and is_summer and motorcycle_share >= 0.42:
        # High-severity summer island / tourist regions (South Aegean, Crete, Ionian, North Aegean)
        tier = "CRITICAL SUMMER ALERT (Κρίσιμη Θερινή Επικινδυνότητα)"
        mechanism = (
            f"Peak summer tourism inflow + high two-wheeler share ({motorcycle_share:.0%}) "
            f"on rural/island roads (Urban: {urban_ratio:.0%})."
        )
        intervention = (
            "1) Pre-deploy Traffic Police & EMS (ΕΚΑΒ) mobile units before June 1st; "
            "2) Strict nighttime sobriety (alcohol) & helmet checkpoints (22:00-05:00) and rental moto/ATV inspections; "
            "3) Roadside barrier upgrades & optical delineation on unlit coastal/provincial curves."
        )
    elif pred_ksi_rate >= 0.21:
        # High-severity rural mainland corridors (Peloponnese, Central Greece, Crete, South Aegean off-peak)
        tier = "HIGH SEVERITY RISK (Υψηλός Κίνδυνος Βαρύτητας KSI)"
        mechanism = (
            f"High-speed rural & interurban network (Rural share: {1.0 - urban_ratio:.0%}); "
            "elevated kinetic energy in single-carriageway & nighttime collisions."
        )
        intervention = (
            "1) Median separation (2+1 barriers) & forgiving roadsides on provincial/national highways; "
            "2) Average-speed enforcement (section control) & LED lighting at rural junctions; "
            "3) Dynamic EMS (ΕΚΑΒ) staging along interurban corridors to reduce Golden-Hour response time."
        )
    else:
        tier = "MODERATE / REGIONAL WATCH (Μέτρια Επιτήρηση)"
        mechanism = (
            f"Mixed regional network (Urban: {urban_ratio:.0%}, Two-Wheelers: {motorcycle_share:.0%}); "
            "moderate seasonal KSI severity."
        )
        intervention = (
            "1) Preventative speed & seatbelt/helmet enforcement on provincial connectors; "
            "2) Road surface skid-resistance maintenance & weather hazard signage prior to winter/autumn rains."
        )

    return tier, mechanism, intervention


def generate_prospective_12m_forecast(
    panel_df: pd.DataFrame,
    forecast_year: int = 2026,
    high_ksi_threshold: float = 0.21,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Trains on the full historical ELSTAT regional panel (2021-2025) and generates
    a 12-month prospective forecast for `forecast_year` across all 13 Greek NUTS-2 regions,
    using strictly verified historical observations from the prior year as lags.
    """
    fe_hist = build_regional_panel_features(panel_df, high_ksi_threshold=high_ksi_threshold)
    X_hist, y_clf_hist = extract_regional_X_y(fe_hist)
    y_reg_hist = fe_hist["ksi_severity_rate"].astype(float).copy()

    # 1. Fit Classifier for P(Severe KSI Regime >= 21%)
    prep_clf = build_preprocessing_transformer(
        REGIONAL_NUMERIC_FEATURES, REGIONAL_CATEGORICAL_FEATURES
    )
    clf_pipe = Pipeline(
        [
            ("prep", prep_clf),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=6,
                    min_samples_leaf=8,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    clf_pipe.fit(X_hist, y_clf_hist)

    # 2. Fit Regressor for Expected KSI Severity Rate (%)
    prep_reg = build_preprocessing_transformer(
        REGIONAL_NUMERIC_FEATURES, REGIONAL_CATEGORICAL_FEATURES
    )
    reg_pipe = Pipeline(
        [
            ("prep", prep_reg),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=6,
                    min_samples_leaf=8,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    reg_pipe.fit(X_hist, y_reg_hist)

    # 3. Construct forward 12-month feature matrix from latest verified year (2025)
    latest_year = int(fe_hist["year"].max())
    latest_df = fe_hist[fe_hist["year"] == latest_year].sort_values(["region", "month"]).copy()

    future_rows: List[Dict[str, object]] = []
    for _, row in latest_df.iterrows():
        m = int(row["month"])
        reg = str(row["region"])
        u_ratio = float(row["urban_ratio"])
        m_share = float(row["motorcycle_share"])

        # For prospective month (forecast_year, m), the 12-month lag is the observed (latest_year, m)
        lag_12_rate = float(row["ksi_severity_rate"])
        lag_12_acc = float(row["total_accidents"])
        lag_1_rate = float(row["lag_1m_ksi_rate"])
        lag_2_rate = float(row["lag_2m_ksi_rate"])
        roll_3_rate = float(row["rolling_3m_ksi_rate"])
        lag_1_acc = float(row["lag_1m_total_accidents"])

        future_rows.append(
            {
                "forecast_year": forecast_year,
                "month": m,
                "region": reg,
                "month_sin": float(np.round(np.sin(2.0 * np.pi * m / 12.0), 6)),
                "month_cos": float(np.round(np.cos(2.0 * np.pi * m / 12.0), 6)),
                "is_summer": int(m in (6, 7, 8, 9)),
                "urban_ratio": u_ratio,
                "motorcycle_share": m_share,
                "lag_1m_ksi_rate": lag_1_rate,
                "lag_2m_ksi_rate": lag_2_rate,
                "lag_12m_ksi_rate": lag_12_rate,
                "rolling_3m_ksi_rate": roll_3_rate,
                "yoy_ksi_rate_diff": float(np.round(lag_1_rate - lag_12_rate, 6)),
                "lag_1m_total_accidents": lag_1_acc,
                "lag_12m_total_accidents": lag_12_acc,
            }
        )

    future_df = pd.DataFrame(future_rows)
    X_future = future_df[REGIONAL_NUMERIC_FEATURES + REGIONAL_CATEGORICAL_FEATURES].copy()

    pred_rates = reg_pipe.predict(X_future)
    pred_probs = clf_pipe.predict_proba(X_future)[:, 1]

    future_df["predicted_ksi_rate"] = np.round(pred_rates, 4)
    future_df["predicted_ksi_rate_pct"] = np.round(pred_rates * 100.0, 2)
    future_df["severe_regime_prob"] = np.round(pred_probs, 4)
    future_df["severe_regime_prob_pct"] = np.round(pred_probs * 100.0, 1)
    future_df["expected_total_accidents"] = future_df["lag_12m_total_accidents"].astype(int)
    future_df["expected_ksi_accidents"] = np.maximum(
        1, np.round(future_df["expected_total_accidents"] * future_df["predicted_ksi_rate"]).astype(int)
    )

    tiers, mechanisms, interventions = [], [], []
    for _, r in future_df.iterrows():
        t, mech, inter = assign_policy_recommendation(
            region=str(r["region"]),
            month=int(r["month"]),
            urban_ratio=float(r["urban_ratio"]),
            motorcycle_share=float(r["motorcycle_share"]),
            pred_ksi_rate=float(r["predicted_ksi_rate"]),
        )
        tiers.append(t)
        mechanisms.append(mech)
        interventions.append(inter)

    future_df["risk_tier"] = tiers
    future_df["primary_risk_mechanism"] = mechanisms
    future_df["recommended_interventions"] = interventions

    return future_df.sort_values(
        ["month", "predicted_ksi_rate"], ascending=[True, False]
    ).reset_index(drop=True)


def save_forecast_heatmap(forecast_df: pd.DataFrame, output_path: Path) -> None:
    """
    Generates a 13 Regions x 12 Months heatmap of forecasted KSI Severity Rate (%) across Greece.
    """
    df_piv = forecast_df.copy()
    df_piv["short_region"] = df_piv["region"].apply(lambda s: s.split(" (")[0])
    piv = df_piv.pivot(index="short_region", columns="month", values="predicted_ksi_rate_pct")
    # Order regions from highest mean forecasted KSI rate to lowest
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    piv.columns = month_names[: len(piv.columns)]

    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.heatmap(
        piv,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "Forecasted KSI Severity Rate (%)"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(
        "Prospective 12-Month Regional KSI Severity Rate Forecast (%) Across Greece (2026 Early-Warning)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("Forecast Month (2026)")
    ax.set_ylabel("Greek NUTS-2 Region")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=220)
    plt.close(fig)
