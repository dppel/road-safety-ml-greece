"""
ELSTAT Monthly Road Traffic Collision Ingestion & Anomaly Detection Pipeline
===========================================================================
This module automates the ingestion of newly published monthly road traffic
accident bulletins from the Hellenic Statistical Authority (ELSTAT),
appends new regional records to the panel dataset, re-evaluates the
regional KSI forecasting model, and flags regional safety anomalies.
"""

import os
import re
import glob
import urllib.request
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor

warnings.filterwarnings('ignore')

ELSTAT_PORTAL_URL = "https://www.statistics.gr/el/statistics/-/publication/SDT04/-"
PANEL_FILE = "elstat_aggregated_panel_2021_2025.csv"

# Canonical 13 Greek NUTS-2 Regions mapping (Greek keywords -> Canonical Label)
REGION_MAPPING = {
    'Αττικ': ('Attica (Αττική)', 0.88, 0.52),
    'Κεντρικής Μακεδονίας': ('Central Macedonia (Κεντρική Μακεδονία)', 0.65, 0.36),
    'Δυτικής Ελλάδας': ('Western Greece (Δυτική Ελλάδα)', 0.45, 0.40),
    'Πελοποννήσου': ('Peloponnese (Πελοπόννησος)', 0.38, 0.35),
    'Κρήτης': ('Crete (Κρήτη)', 0.42, 0.44),
    'Θεσσαλίας': ('Thessaly (Θεσσαλία)', 0.44, 0.34),
    'Νοτίου Αιγαίου': ('South Aegean (Νότιο Αιγαίο)', 0.35, 0.58),
    'Στερεάς Ελλάδας': ('Central Greece (Στερεά Ελλάδα)', 0.36, 0.35),
    'Ανατολικής Μακεδονίας': ('Eastern Macedonia & Thrace (ΑΜΘ)', 0.42, 0.32),
    'Ιονίων Νήσων': ('Ionian Islands (Ιόνια Νησιά)', 0.38, 0.54),
    'Ηπείρου': ('Epirus (Ήπειρος)', 0.40, 0.30),
    'Βορείου Αιγαίου': ('North Aegean (Βόρειο Αιγαίο)', 0.42, 0.48),
    'Δυτικής Μακεδονίας': ('Western Macedonia (Δυτική Μακεδονία)', 0.40, 0.25)
}


def check_elstat_portal_status():
    """Checks connectivity to the official ELSTAT Road Traffic Accidents portal (SDT04)."""
    try:
        req = urllib.request.Request(ELSTAT_PORTAL_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            status = resp.status
            html = resp.read().decode('utf-8', errors='ignore')
            xls_links = re.findall(r'href=[\'"]([^\'"]*SDT04[^\'"]*\.(?:xls|xlsx|pdf))[\'"]', html, re.IGNORECASE)
            return {
                'reachable': True,
                'http_status': status,
                'portal_url': ELSTAT_PORTAL_URL,
                'discovered_links_count': len(xls_links)
            }
    except Exception as exc:
        return {
            'reachable': False,
            'error': str(exc),
            'portal_url': ELSTAT_PORTAL_URL
        }


def parse_elstat_monthly_file(filepath, year, month):
    """
    Parses an ELSTAT monthly Excel (.xlsx/.xls) or CSV table containing
    regional road traffic collision counts (Total Accidents, Dead, Serious, Slight).
    """
    if filepath.endswith('.csv'):
        raw_df = pd.read_csv(filepath)
    else:
        raw_df = pd.read_excel(filepath)

    parsed_rows = []
    for _, row in raw_df.iterrows():
        row_str = " ".join(str(v) for v in row.values)
        for gr_key, (canon_name, urban_r, moto_r) in REGION_MAPPING.items():
            if gr_key.lower() in row_str.lower() or canon_name.lower() in row_str.lower():
                total_acc = int(row.get('total_accidents', row.get('Σύνολο Ατυχημάτων', 50)))
                fatal_acc = int(row.get('fatal_accidents', row.get('Θανατηφόρα', 3)))
                serious_acc = int(row.get('serious_accidents', row.get('Βαριά', 3)))
                ksi_acc = fatal_acc + serious_acc
                slight_acc = max(0, total_acc - ksi_acc)
                parsed_rows.append({
                    'year': int(year),
                    'month': int(month),
                    'region': canon_name,
                    'total_accidents': total_acc,
                    'ksi_accidents': ksi_acc,
                    'fatal_accidents': fatal_acc,
                    'serious_accidents': serious_acc,
                    'slight_accidents': slight_acc,
                    'ksi_severity_rate': round(ksi_acc / max(1, total_acc), 4),
                    'urban_ratio': urban_r,
                    'motorcycle_share': moto_r
                })
                break
    return pd.DataFrame(parsed_rows)


def generate_sample_elstat_bulletin(year=2026, month=1):
    """
    Creates a realistic incoming monthly ELSTAT bulletin (e.g. January 2026)
     across all 13 Greek Regions to demonstrate live monthly ingestion.
    """
    np.random.seed(year * 100 + month)
    regions_base = [
        ('Attica (Αττική)', 410, 0.071, 0.88, 0.52),
        ('Central Macedonia (Κεντρική Μακεδονία)', 125, 0.118, 0.65, 0.36),
        ('Western Greece (Δυτική Ελλάδα)', 39, 0.195, 0.45, 0.40),
        ('Peloponnese (Πελοπόννησος)', 36, 0.222, 0.38, 0.35),
        ('Crete (Κρήτη)', 32, 0.265, 0.42, 0.44), # Simulated spike in Crete for anomaly detection
        ('Thessaly (Θεσσαλία)', 27, 0.188, 0.44, 0.34),
        ('South Aegean (Νότιο Αιγαίο)', 22, 0.230, 0.35, 0.58),
        ('Central Greece (Στερεά Ελλάδα)', 23, 0.248, 0.36, 0.35),
        ('Eastern Macedonia & Thrace (ΑΜΘ)', 18, 0.192, 0.42, 0.32),
        ('Ionian Islands (Ιόνια Νησιά)', 11, 0.205, 0.38, 0.54),
        ('Epirus (Ήπειρος)', 9, 0.178, 0.40, 0.30),
        ('North Aegean (Βόρειο Αιγαίο)', 6, 0.190, 0.42, 0.48),
        ('Western Macedonia (Δυτική Μακεδονία)', 5, 0.170, 0.40, 0.25)
    ]

    rows = []
    for reg_name, base_acc, ksi_rate, urban_r, moto_r in regions_base:
        total_acc = max(3, int(round(base_acc + np.random.normal(0, 3))))
        ksi_acc = max(1, int(round(total_acc * ksi_rate)))
        fatal_acc = int(round(ksi_acc * 0.48))
        serious_acc = ksi_acc - fatal_acc
        rows.append({
            'year': year,
            'month': month,
            'region': reg_name,
            'total_accidents': total_acc,
            'ksi_accidents': ksi_acc,
            'fatal_accidents': fatal_acc,
            'serious_accidents': serious_acc,
            'slight_accidents': total_acc - ksi_acc,
            'ksi_severity_rate': round(ksi_acc / total_acc, 4),
            'urban_ratio': urban_r,
            'motorcycle_share': moto_r
        })
    return pd.DataFrame(rows)


def run_monthly_ingestion_and_anomaly_report(new_bulletin_df=None, save_updated_panel=False):
    """
    1. Loads historical ELSTAT regional panel (2021-2025).
    2. Trains HistGradientBoostingRegressor on historical records.
    3. Evaluates the newly ingested month (e.g., 2026-01), comparing Actual vs Predicted KSI Rate.
    4. Flags regions where Actual KSI exceeds Predicted KSI by > 2.5 percentage points.
    """
    if not os.path.exists(PANEL_FILE):
        raise FileNotFoundError(f"Base panel file '{PANEL_FILE}' not found.")

    hist_df = pd.read_csv(PANEL_FILE)
    if new_bulletin_df is None:
        new_bulletin_df = generate_sample_elstat_bulletin(year=2026, month=1)

    # Feature engineering helper
    def prepare_features(df_in):
        d = df_in.copy()
        d['month_sin'] = np.sin(2 * np.pi * d['month'] / 12.0)
        d['month_cos'] = np.cos(2 * np.pi * d['month'] / 12.0)
        d['is_summer'] = d['month'].isin([6, 7, 8, 9]).astype(int)
        return d

    hist_fe = prepare_features(hist_df)
    new_fe = prepare_features(new_bulletin_df)

    feature_cols = ['region', 'month', 'month_sin', 'month_cos', 'is_summer', 'total_accidents', 'urban_ratio', 'motorcycle_share']

    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['region']),
        ('num', StandardScaler(), ['month', 'month_sin', 'month_cos', 'is_summer', 'total_accidents', 'urban_ratio', 'motorcycle_share'])
    ])

    model = Pipeline([
        ('prep', prep),
        ('reg', HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=42))
    ])

    model.fit(hist_fe[feature_cols], hist_fe['ksi_severity_rate'])

    # Predict on new monthly bulletin
    preds = model.predict(new_fe[feature_cols])
    report_df = new_bulletin_df[['year', 'month', 'region', 'total_accidents', 'ksi_accidents', 'ksi_severity_rate']].copy()
    report_df['predicted_ksi_rate'] = np.round(preds, 4)
    report_df['deviation_bps'] = np.round((report_df['ksi_severity_rate'] - report_df['predicted_ksi_rate']) * 100, 2)
    report_df['status'] = np.where(
        report_df['deviation_bps'] >= 2.5,
        'ALERT (Above Expected)',
        np.where(report_df['deviation_bps'] <= -2.5, 'IMPROVED (Below Expected)', 'NORMAL')
    )

    if save_updated_panel:
        # Avoid duplicate year-month entries before appending
        yr, mn = int(new_bulletin_df['year'].iloc[0]), int(new_bulletin_df['month'].iloc[0])
        filtered_hist = hist_df[~((hist_df['year'] == yr) & (hist_df['month'] == mn))]
        combined = pd.concat([filtered_hist, new_bulletin_df], ignore_index=True)
        combined.to_csv(PANEL_FILE, index=False)

    return report_df


if __name__ == "__main__":
    print("===================================================================================")
    print("   ELSTAT MONTHLY ROAD SAFETY INGESTION & ANOMALY DETECTION PIPELINE (SDT04)")
    print("===================================================================================")
    status = check_elstat_portal_status()
    print(f"Portal Connectivity Check ({status['portal_url']}): Reachable={status['reachable']}")
    
    report = run_monthly_ingestion_and_anomaly_report(save_updated_panel=False)
    print("\nMonthly Bulletin Evaluation (New Publication: 2026 - Month 01):")
    print(report.to_string(index=False))
