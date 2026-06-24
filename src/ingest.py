import urllib.request
from pathlib import Path

import pandas as pd

from src.ingest_api import fetch_wc_matches

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
    '2122', '2223', '2324', '2425', '2526',
]

BASE_URL = 'https://www.football-data.co.uk/mmz4281/{season}/{code}.csv'

STAT_COL_MAP = {
    'HS': 'home_shots', 'AS': 'away_shots',
    'HST': 'home_shots_on_target', 'AST': 'away_shots_on_target',
    'HC': 'home_corners', 'AC': 'away_corners',
    'HY': 'home_yellow', 'AY': 'away_yellow',
    'HR': 'home_red', 'AR': 'away_red',
}


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
    df = raw_df.rename(columns={**col_map, **STAT_COL_MAP}).copy()
    required = ['date', 'home_team', 'away_team', 'home_score', 'away_score']
    df = df[df[required].notna().all(axis=1)].copy()

    stat_cols = list(STAT_COL_MAP.values())
    keep = required + (['result'] if 'result' in df.columns else []) \
           + [c for c in stat_cols if c in df.columns]
    df = df[keep].copy()

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])
    df['home_score'] = df['home_score'].astype(int)
    df['away_score'] = df['away_score'].astype(int)

    for c in stat_cols:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors='coerce')

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


def download_csv(league_code, season, raw_dir):
    """Download a league/season CSV. Returns path if successful, None on error."""
    dest = Path(raw_dir) / f'{league_code}_{season}.csv'
    if dest.exists():
        return dest
    url = BASE_URL.format(season=season, code=league_code)
    tmp = Path(raw_dir) / f'{league_code}_{season}.tmp'
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)
        return dest
    except Exception:
        tmp.unlink(missing_ok=True)
        return None


def build_clubs_dataset(raw_dir, leagues=None, seasons=None):
    """Download all league CSVs and return a unified normalized DataFrame."""
    if leagues is None:
        leagues = LEAGUES
    if seasons is None:
        seasons = SEASONS
    Path(raw_dir).mkdir(parents=True, exist_ok=True)
    frames = []
    for code, name in leagues.items():
        for season in seasons:
            path = Path(raw_dir) / f'{code}_{season}.csv'
            if not path.exists():
                path = download_csv(code, season, raw_dir)
            if path is None or not path.exists():
                continue
            try:
                raw = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
                normalized = normalize_csv(raw, name)
                frames.append(normalized)
            except Exception as exc:
                print(f"Warning: skipping {code} {season}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def combine_datasets(clubs_df, international_path='data/results.csv'):
    """Merge clubs DataFrame with existing international results.csv."""
    intl = pd.read_csv(international_path)
    intl['date'] = pd.to_datetime(intl['date'])
    intl['league'] = intl.get('tournament', pd.Series('International', index=intl.index)).fillna('International')
    intl['competition_type'] = 'international'
    combined = pd.concat([intl, clubs_df], ignore_index=True)
    return combined.sort_values('date').reset_index(drop=True)


def merge_wc_data(combined_df, api_key):
    """Append finished WC 2026 matches from football-data.org to combined_df."""
    wc = fetch_wc_matches(api_key)
    if len(wc) == 0:
        return combined_df
    merged = pd.concat([combined_df, wc], ignore_index=True).sort_values('date').reset_index(drop=True)
    return merged.drop_duplicates(subset=['date', 'home_team', 'away_team'], keep='last').reset_index(drop=True)


if __name__ == '__main__':
    import sys
    import os
    raw_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    print(f'Downloading club data to {raw_dir}...')
    clubs = build_clubs_dataset(raw_dir)
    print(f'Downloaded {len(clubs)} club matches.')
    clubs.to_csv('data/clubs.csv', index=False)
    print('Saved data/clubs.csv')
    combined = combine_datasets(clubs)
    api_key = os.environ.get('FOOTBALL_DATA_API_KEY')
    if api_key:
        print('Fetching FIFA World Cup 2026 results...')
        combined = merge_wc_data(combined, api_key)
        print(f'Dataset after WC merge: {len(combined)} total matches')
    else:
        print('FOOTBALL_DATA_API_KEY not set — skipping WC 2026 data.')
    combined.to_csv('data/all_matches.csv', index=False)
    print(f'Saved data/all_matches.csv ({len(combined)} total matches)')
