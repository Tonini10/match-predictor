# -*- coding: utf-8 -*-
import json
import urllib.request
import pandas as pd

# API-Football v3 — FIFA World Cup 2026
# league_id = 1 (FIFA World Cup), season = 2026
_AF_URL = 'https://v3.football.api-sports.io/fixtures?league=1&season=2026'

_FINISHED = {'FT', 'AET', 'PEN'}

_WC_HOSTS = {'United States', 'Mexico', 'Canada'}

_TEAM_NAME_MAP = {
    'Czech Republic': 'Czech Republic',
    'Czechia': 'Czech Republic',
    'Bosnia And Herzegovina': 'Bosnia and Herzegovina',
    'Cape Verde': 'Cape Verde',
    'Congo DR': 'DR Congo',
    'DR Congo': 'DR Congo',
    'United States': 'United States',
    "Côte D'Ivoire": 'Ivory Coast',
    "Cote D'Ivoire": 'Ivory Coast',
    'Korea Republic': 'South Korea',
    'Republic Of Ireland': 'Republic of Ireland',
}

_STAT_COLS = [
    'home_shots', 'away_shots',
    'home_shots_on_target', 'away_shots_on_target',
    'home_corners', 'away_corners',
    'home_yellow', 'away_yellow',
    'home_red', 'away_red',
]


def fetch_wc_matches_apifootball(api_key):
    """Fetch finished FIFA World Cup 2026 matches from api-football.com.

    Returns a DataFrame in the unified all_matches schema.
    Returns an empty DataFrame on any error.
    """
    req = urllib.request.Request(
        _AF_URL,
        headers={'x-apisports-key': api_key},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except Exception as exc:
        print(f"Warning: could not fetch WC matches from API-Football: {exc}")
        return pd.DataFrame()

    errors = data.get('errors', {})
    if errors:
        print(f"Warning: API-Football errors: {errors}")
        return pd.DataFrame()

    rows = []
    for fixture in data.get('response', []):
        status = fixture.get('fixture', {}).get('status', {}).get('short', '')
        if status not in _FINISHED:
            continue

        goals = fixture.get('goals', {})
        home_goals = goals.get('home')
        away_goals = goals.get('away')
        if home_goals is None or away_goals is None:
            # fallback to fulltime score
            ft = fixture.get('score', {}).get('fulltime', {})
            home_goals = ft.get('home')
            away_goals = ft.get('away')
        if home_goals is None or away_goals is None:
            continue

        teams = fixture.get('teams', {})
        home_raw = teams.get('home', {}).get('name', '')
        away_raw = teams.get('away', {}).get('name', '')
        home_name = _TEAM_NAME_MAP.get(home_raw, home_raw)
        away_name = _TEAM_NAME_MAP.get(away_raw, away_raw)

        date_str = fixture.get('fixture', {}).get('date', '')
        try:
            date = pd.to_datetime(date_str, utc=True).normalize().tz_localize(None)
        except Exception:
            continue

        is_neutral = home_name not in _WC_HOSTS
        rows.append({
            'date': date,
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
    df['neutral'] = df['neutral'].astype('object')
    return df
