import json
import urllib.request
import pandas as pd

WC_URL = 'https://api.football-data.org/v4/competitions/WC/matches'

# WC 2026 host nations — have home advantage when playing in their country
_WC_HOSTS = {'United States', 'Mexico', 'Canada'}

# Maps football-data.org team names to the canonical names used in results.csv
_TEAM_NAME_MAP = {
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    'Cape Verde Islands': 'Cape Verde',
    'Czechia': 'Czech Republic',
    'Congo DR': 'DR Congo',
}

# Mirrors the values of STAT_COL_MAP in src/ingest.py — update both if columns change
_STAT_COLS = [
    'home_shots', 'away_shots',
    'home_shots_on_target', 'away_shots_on_target',
    'home_corners', 'away_corners',
    'home_yellow', 'away_yellow',
    'home_red', 'away_red',
]


def fetch_wc_matches(api_key):
    """Fetch finished FIFA World Cup matches from football-data.org.

    Returns a DataFrame in the unified all_matches schema.
    Returns an empty DataFrame on any network or API error.
    """
    req = urllib.request.Request(WC_URL, headers={'X-Auth-Token': api_key})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as exc:
        print(f"Warning: could not fetch WC matches: {exc}")
        return pd.DataFrame()

    rows = []
    for m in data.get('matches', []):
        if m.get('status') not in ('FINISHED', 'TIMED'):
            continue
        score = m.get('score', {}).get('fullTime', {})
        home_goals = score.get('home')
        away_goals = score.get('away')
        if home_goals is None or away_goals is None:
            continue
        home_name = _TEAM_NAME_MAP.get(m['homeTeam']['name'], m['homeTeam']['name'])
        away_name = _TEAM_NAME_MAP.get(m['awayTeam']['name'], m['awayTeam']['name'])
        is_neutral = home_name not in _WC_HOSTS
        rows.append({
            'date': pd.to_datetime(m['utcDate']).normalize(),
            'home_team': home_name,
            'away_team': away_name,
            'home_score': int(home_goals),
            'away_score': int(away_goals),
            'tournament': 'FIFA World Cup 2026',
            'league': 'FIFA World Cup 2026',
            'competition_type': 'international',
            'neutral': is_neutral,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    for col in _STAT_COLS:
        df[col] = float('nan')
    # Convert neutral to Python bool objects (required for 'is True' identity checks)
    df['neutral'] = df['neutral'].astype('object')
    return df
