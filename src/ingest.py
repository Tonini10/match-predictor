import urllib.request
from pathlib import Path

import pandas as pd

LEAGUES = {
    'E0': 'Premier League',
    'E1': 'Championship',
    'E2': 'League One',
    'SC0': 'Scottish Premiership',
    'SP1': 'La Liga',
    'SP2': 'La Liga 2',
    'D1': 'Bundesliga',
    'D2': '2. Bundesliga',
    'I1': 'Serie A',
    'I2': 'Serie B',
    'F1': 'Ligue 1',
    'F2': 'Ligue 2',
    'N1': 'Eredivisie',
    'B1': 'Pro League',
    'P1': 'Primeira Liga',
    'T1': 'Super Lig',
    'G1': 'Super League Greece',
}

SEASONS = [
    '9394', '9495', '9596', '9697', '9798', '9899', '9900',
    '0001', '0102', '0203', '0304', '0405', '0506', '0607',
    '0708', '0809', '0910', '1011', '1112', '1213', '1314',
    '1415', '1516', '1617', '1718', '1819', '1920', '2021',
    '2122', '2223', '2324', '2425',
]

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{season}/{code}.csv'


def normalize_csv(raw_df, league_name, competition_type='club'):
    """Normalize a football-data.co.uk raw DataFrame to the unified schema."""
    col_map = {
        'Date': 'date',
        'HomeTeam': 'home_team',
        'AwayTeam': 'away_team',
        'FTHG': 'home_score',
        'FTAG': 'away_score',
        'FTR': 'result',
    }
    df = raw_df.rename(columns=col_map).copy()
    required = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    df = df[df[required].notna().all(axis=1)].copy()

    keep = required + (['result'] if 'result' in df.columns else [])
    df = df[keep].copy()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    if 'result' not in df.columns:
        df['result'] = df.apply(
            lambda r: 'H' if r['home_score'] > r['away_score']
            else ('D' if r['home_score'] == r['away_score'] else 'A'),
            axis=1,
        )

    df['neutral'] = False
    df['tournament'] = league_name
    df['league'] = league_name
    df['competition_type'] = competition_type
    result = df.reset_index(drop=True)
    # Convert neutral to Python bool objects (required for 'is False' identity checks)
    result['neutral'] = result['neutral'].astype('object')
    return result
