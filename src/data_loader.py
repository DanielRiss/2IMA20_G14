"""
data_loader.py — Parse Eurostat intra-EU trade CSV files.

Produces:
  - export_matrix: DataFrame[reporter, partner] = gross exports in million EUR
  - net_matrix:    DataFrame[reporter, partner] = net exports (export - import) in million EUR
  - countries:     sorted list of EU27 short country names
"""

import pandas as pd

EU27_ISO = [
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
    'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK'
]

ISO_SHORT = {
    'AT': 'Austria',     'BE': 'Belgium',    'BG': 'Bulgaria',
    'CY': 'Cyprus',      'CZ': 'Czechia',    'DE': 'Germany',
    'DK': 'Denmark',     'EE': 'Estonia',    'ES': 'Spain',
    'FI': 'Finland',     'FR': 'France',     'GR': 'Greece',
    'HR': 'Croatia',     'HU': 'Hungary',    'IE': 'Ireland',
    'IT': 'Italy',       'LT': 'Lithuania',  'LU': 'Luxembourg',
    'LV': 'Latvia',      'MT': 'Malta',      'NL': 'Netherlands',
    'PL': 'Poland',      'PT': 'Portugal',   'RO': 'Romania',
    'SE': 'Sweden',      'SI': 'Slovenia',   'SK': 'Slovakia',
}


def parse_label_file(label_path):
    """
    Parse the Eurostat label CSV to build a verbose-name -> ISO code mapping.
    Only parses the REPORTER section (same names appear in PARTNER section).
    """
    verbose_to_iso = {}
    in_reporter = False
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == 'REPORTER':
                in_reporter = True
                continue
            # Stop at next section header
            if line in ('PARTNER', 'PRODUCT', 'FLOW', 'STAT_PROCEDURE', 'PERIOD', 'INDICATORS'):
                in_reporter = False
                continue
            if in_reporter and ',' in line:
                iso, verbose = line.split(',', 1)
                iso = iso.strip()
                verbose = verbose.strip().strip('"')
                if iso in EU27_ISO:
                    verbose_to_iso[verbose] = iso
    return verbose_to_iso


def load_trade_data(data_path, label_path, year=2024):
    """
    Load and clean bilateral trade data.

    Filters to: PERIOD='Jan.-Dec. {year}', FLOW='EXPORT', INDICATORS='VALUE_IN_EUR'.

    Returns
    -------
    export_matrix : pd.DataFrame  (27×27, values in million EUR)
    net_matrix    : pd.DataFrame  (27×27, net(i,j) = export(i->j) - export(j->i))
    countries     : list[str]     sorted EU27 short country names
    """
    verbose_to_iso = parse_label_file(label_path)
    verbose_to_short = {v: ISO_SHORT[iso] for v, iso in verbose_to_iso.items()}

    df = pd.read_csv(data_path)

    # Filter to the requested year's gross exports in EUR
    df = df[
        (df['PERIOD_LAB'] == f'Jan.-Dec. {year}') &
        (df['FLOW_LAB'] == 'EXPORT') &
        (df['INDICATORS_LAB'] == 'VALUE_IN_EUR')
    ].copy()

    # Map verbose country names to short names
    df['reporter'] = df['REPORTER_LAB'].map(verbose_to_short)
    df['partner'] = df['PARTNER_LAB'].map(verbose_to_short)
    df = df.dropna(subset=['reporter', 'partner'])

    # Convert to millions EUR
    df['value_meur'] = df['INDICATOR_VALUE'] / 1e6

    countries = sorted(ISO_SHORT.values())

    # Build export matrix
    export_matrix = pd.DataFrame(0.0, index=countries, columns=countries)
    for _, row in df.iterrows():
        r, p = row['reporter'], row['partner']
        if r != p and r in export_matrix.index and p in export_matrix.columns:
            export_matrix.loc[r, p] = row['value_meur']

    # Net-flow matrix: net(i,j) = export(i->j) - export(j->i)
    net_matrix = export_matrix - export_matrix.T

    return export_matrix, net_matrix, countries
