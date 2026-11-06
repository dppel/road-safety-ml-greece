"""
Unit Tests for Counterfactual Policy Simulator & Geospatial NUTS-2 Metadata (`test_simulator.py`)
=================================================================================================
"""

from pathlib import Path
import pandas as pd
import pytest

from road_safety.forecast import generate_prospective_12m_forecast
from road_safety.foresight import (
    compute_regional_backcasting_table,
    generate_foresight_trajectories,
    get_megatrends_matrix,
)
from road_safety.simulator import (
    GREEK_REGIONS_GEODATA,
    attach_geospatial_metadata,
    evaluate_policy_packages,
    simulate_policy_interventions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGIONAL_DATA_PATH = PROJECT_ROOT / "data" / "elstat_regional_monthly_panel_2021_2025.csv"


@pytest.fixture(scope="module")
def sample_forecast_df() -> pd.DataFrame:
    panel_df = pd.read_csv(REGIONAL_DATA_PATH)
    return generate_prospective_12m_forecast(panel_df, forecast_year=2026, seed=42)


def test_geospatial_metadata_covers_all_13_regions(sample_forecast_df: pd.DataFrame) -> None:
    enriched = attach_geospatial_metadata(sample_forecast_df)
    assert len(GREEK_REGIONS_GEODATA) == 13
    assert set(enriched["region"].unique()) == set(GREEK_REGIONS_GEODATA.keys())
    assert "ELXX" not in enriched["nuts2_code"].values
    assert enriched["lat"].between(34.5, 42.0).all()
    assert enriched["lon"].between(19.0, 27.5).all()


def test_zero_intensity_preserves_baseline(sample_forecast_df: pd.DataFrame) -> None:
    sim_zero = simulate_policy_interventions(
        sample_forecast_df,
        speed_cameras_intensity=0.0,
        helmet_sobriety_intensity=0.0,
        rural_barriers_intensity=0.0,
        urban_30kmh_intensity=0.0,
        ems_golden_hour_intensity=0.0,
    )
    assert (sim_zero["simulated_ksi_accidents"] == sim_zero["baseline_ksi_accidents"]).all()
    assert int(sim_zero["ksi_accidents_prevented"].sum()) == 0
    assert float(sim_zero["socioeconomic_benefit_eur_m"].sum()) == 0.0


def test_policy_monotonicity_and_bounds(sample_forecast_df: pd.DataFrame) -> None:
    sim_mod = simulate_policy_interventions(
        sample_forecast_df,
        speed_cameras_intensity=0.4,
        helmet_sobriety_intensity=0.4,
        rural_barriers_intensity=0.4,
        urban_30kmh_intensity=0.4,
        ems_golden_hour_intensity=0.4,
    )
    sim_high = simulate_policy_interventions(
        sample_forecast_df,
        speed_cameras_intensity=0.9,
        helmet_sobriety_intensity=0.9,
        rural_barriers_intensity=0.9,
        urban_30kmh_intensity=0.9,
        ems_golden_hour_intensity=0.9,
    )
    assert (sim_mod["simulated_ksi_rate"] <= sim_mod["predicted_ksi_rate"] + 1e-6).all()
    assert (sim_high["simulated_ksi_rate"] <= sim_mod["simulated_ksi_rate"] + 1e-6).all()
    assert int(sim_high["ksi_accidents_prevented"].sum()) > int(sim_mod["ksi_accidents_prevented"].sum()) > 0


def test_invalid_policy_intensity_raises_error(sample_forecast_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must be in"):
        simulate_policy_interventions(sample_forecast_df, speed_cameras_intensity=1.2)
    with pytest.raises(ValueError, match="must be in"):
        simulate_policy_interventions(sample_forecast_df, helmet_sobriety_intensity=-0.1)


def test_evaluate_policy_packages_structure(sample_forecast_df: pd.DataFrame) -> None:
    pkgs = evaluate_policy_packages(sample_forecast_df)
    assert len(pkgs) == 5
    assert pkgs.iloc[0]["annual_ksi_accidents_prevented"] == 0
    assert (pkgs.iloc[1:]["annual_ksi_accidents_prevented"] > 0).all()
    assert pkgs.iloc[-1]["annual_ksi_accidents_prevented"] == pkgs["annual_ksi_accidents_prevented"].max()


def test_independent_external_datasets_integrity() -> None:
    ext_kpi_path = PROJECT_ROOT / "data" / "independent_road_safety_kpis_greece.csv"
    ext_reg_path = PROJECT_ROOT / "data" / "regional_climate_telematics_kpis_nuts2.csv"
    assert ext_kpi_path.exists()
    assert ext_reg_path.exists()

    kpi_df = pd.read_csv(ext_kpi_path)
    reg_df = pd.read_csv(ext_reg_path)

    assert len(kpi_df) >= 10
    assert (kpi_df["independently_collected_from_elstat"] == True).all()  # noqa: E712
    assert len(reg_df) == 13
    assert set(reg_df["region"].unique()) == set(GREEK_REGIONS_GEODATA.keys())


def test_foresight_trajectories_and_backcasting(sample_forecast_df: pd.DataFrame) -> None:
    traj = generate_foresight_trajectories(sample_forecast_df, start_year=2026, end_year=2035)
    # 13 regions x 10 years x 4 scenarios = 520 rows
    assert len(traj) == 13 * 10 * 4
    assert set(traj["scenario_id"].unique()) == {
        "S1_STRESS",
        "S2_DIGITAL",
        "S3_TECH_INFRA",
        "S4_VISION_ZERO",
    }

    backcast = compute_regional_backcasting_table(traj)
    assert len(backcast) == 13
    assert (backcast["vision_zero_2035_ksi"] < backcast["baseline_2026_ksi"]).all()
    assert (backcast["stress_2035_ksi"] >= backcast["baseline_2026_ksi"]).all()
    assert (backcast["cumulative_10yr_ksi_saved"] > 0).all()

    mt = get_megatrends_matrix()
    assert len(mt) == 6




def test_hitl_decision_support_engine(sample_forecast_df: pd.DataFrame) -> None:
    from road_safety.decision_support import (
        diagnose_critical_moment,
        evaluate_intervention_portfolio,
        generate_three_hitl_options,
        score_interventions_for_context,
    )

    cat_path = PROJECT_ROOT / "data" / "operational_interventions_catalog.csv"
    ext_reg_path = PROJECT_ROOT / "data" / "regional_climate_telematics_kpis_nuts2.csv"
    assert cat_path.exists()
    cat_df = pd.read_csv(cat_path)
    ext_reg_df = pd.read_csv(ext_reg_path)
    assert len(cat_df) == 10

    fc_row = sample_forecast_df[
        (sample_forecast_df["region"] == "South Aegean (Νότιο Αιγαίο)") & (sample_forecast_df["month"] == 8)
    ].iloc[0]
    ext_row = ext_reg_df[ext_reg_df["region"] == "South Aegean (Νότιο Αιγαίο)"].iloc[0]

    diag = diagnose_critical_moment(fc_row, ext_row, crisis_override="AUTO")
    assert diag["criticality_index"] >= 60.0
    assert len(diag["triggers"]) >= 1

    scored = score_interventions_for_context(cat_df, fc_row, diag, horizon_filter="ALL")
    assert len(scored) == 10
    assert (scored["suitability_score"] >= 35.0).all()

    opts = generate_three_hitl_options(scored, fc_row, max_budget_eur_k=300.0)
    assert set(opts.keys()) == {"OPTION_A", "OPTION_B", "OPTION_C"}
    for okey, odata in opts.items():
        assert len(odata["ids"]) >= 1
        assert odata["metrics"]["combined_ksi_reduction_pct"] > 0.0
        assert odata["metrics"]["total_cost_eur_k"] <= 300.0
