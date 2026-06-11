import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.player_features import get_team_player_features

# feature suffix -> (columna del equipo local, columna del visitante)
MATCH_STAT_SOURCES = {
    'avg_shots': ('home_shots', 'away_shots'),
    'avg_shots_on_target': ('home_shots_on_target', 'away_shots_on_target'),
    'avg_corners': ('home_corners', 'away_corners'),
    'avg_yellow': ('home_yellow', 'away_yellow'),
    'avg_red': ('home_red', 'away_red'),
}

FEATURE_COLS = [
    'home_avg_goals_scored',
    'home_avg_goals_conceded',
    'home_win_rate',
    'away_avg_goals_scored',
    'away_avg_goals_conceded',
    'away_win_rate',
    'home_is_neutral',
    'league_encoded',
    'is_international',
    'home_league_win_rate',
    'away_league_win_rate',
    'home_team_rating',
    'away_team_rating',
    'home_team_attack',
    'away_team_attack',
    'home_team_defense',
    'away_team_defense',
    'rating_diff',
    'home_avg_shots',
    'home_avg_shots_on_target',
    'home_avg_corners',
    'home_avg_yellow',
    'home_avg_red',
    'away_avg_shots',
    'away_avg_shots_on_target',
    'away_avg_corners',
    'away_avg_yellow',
    'away_avg_red',
]


def build_training_data(df, n=5, players_df=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    if 'league' not in df.columns:
        df['league'] = df.get('tournament', pd.Series('International', index=df.index)).fillna('International')
    if 'competition_type' not in df.columns:
        df['competition_type'] = 'international'

    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['away_win'] = (df['away_score'] > df['home_score']).astype(int)

    def rolling_mean(group_col, val_col):
        return df.groupby(group_col)[val_col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean().fillna(0)
        )

    def rolling_mean_by_league(team_col, val_col):
        return df.groupby([team_col, 'league'])[val_col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean().fillna(0)
        )

    df['home_avg_goals_scored']   = rolling_mean('home_team', 'home_score')
    df['home_avg_goals_conceded'] = rolling_mean('home_team', 'away_score')
    df['home_win_rate']           = rolling_mean('home_team', 'home_win')
    df['away_avg_goals_scored']   = rolling_mean('away_team', 'away_score')
    df['away_avg_goals_conceded'] = rolling_mean('away_team', 'home_score')
    df['away_win_rate']           = rolling_mean('away_team', 'away_win')
    df['home_is_neutral']         = df['neutral'].astype(int)

    for feat, (h_col, a_col) in MATCH_STAT_SOURCES.items():
        if h_col in df.columns and a_col in df.columns:
            df[f'home_{feat}'] = rolling_mean('home_team', h_col)
            df[f'away_{feat}'] = rolling_mean('away_team', a_col)
        else:
            df[f'home_{feat}'] = 0.0
            df[f'away_{feat}'] = 0.0

    df['home_league_win_rate'] = rolling_mean_by_league('home_team', 'home_win')
    df['away_league_win_rate'] = rolling_mean_by_league('away_team', 'away_win')
    df['is_international'] = (df['competition_type'] == 'international').astype(int)

    le = LabelEncoder()
    df['league_encoded'] = le.fit_transform(df['league'].fillna('Unknown'))

    df['result'] = df.apply(
        lambda r: 'H' if r['home_score'] > r['away_score']
                  else ('D' if r['home_score'] == r['away_score'] else 'A'),
        axis=1,
    )

    # Player features — vectorized over unique teams for performance
    if players_df is not None:
        unique_teams = list(set(df['home_team'].unique()) | set(df['away_team'].unique()))
        team_cache = {t: get_team_player_features(players_df, t) for t in unique_teams}
        df['home_team_rating']  = df['home_team'].map(lambda t: team_cache[t]['team_rating'])
        df['away_team_rating']  = df['away_team'].map(lambda t: team_cache[t]['team_rating'])
        df['home_team_attack']  = df['home_team'].map(lambda t: team_cache[t]['team_attack'])
        df['away_team_attack']  = df['away_team'].map(lambda t: team_cache[t]['team_attack'])
        df['home_team_defense'] = df['home_team'].map(lambda t: team_cache[t]['team_defense'])
        df['away_team_defense'] = df['away_team'].map(lambda t: team_cache[t]['team_defense'])
        df['rating_diff'] = df['home_team_rating'] - df['away_team_rating']
    else:
        for col in ['home_team_rating', 'away_team_rating', 'home_team_attack',
                    'away_team_attack', 'home_team_defense', 'away_team_defense', 'rating_diff']:
            df[col] = 0.0

    return df[FEATURE_COLS + ['result']].copy(), le


def build_features_for_prediction(df, home_team, away_team, is_neutral=False, n=5,
                                   before_date=None, league=None, label_encoder=None,
                                   competition_type=None, players_df=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if before_date is not None:
        df = df[df['date'] < pd.Timestamp(before_date)]

    hm = df[df['home_team'] == home_team].tail(n)
    am = df[df['away_team'] == away_team].tail(n)

    def safe_mean(series):
        return 0.0 if len(series) == 0 or series.isna().all() else float(series.mean())

    if league and 'league' in df.columns:
        hm_l = df[(df['home_team'] == home_team) & (df['league'] == league)].tail(n)
        am_l = df[(df['away_team'] == away_team) & (df['league'] == league)].tail(n)
        home_league_win_rate = safe_mean((hm_l['home_score'] > hm_l['away_score']).astype(float))
        away_league_win_rate = safe_mean((am_l['away_score'] > am_l['home_score']).astype(float))
    else:
        home_league_win_rate = 0.0
        away_league_win_rate = 0.0

    if label_encoder is not None and league:
        try:
            league_enc = int(label_encoder.transform([league])[0])
        except ValueError:
            league_enc = 0
    else:
        league_enc = 0

    is_international = 1 if competition_type == 'international' else 0

    if players_df is not None:
        h_p = get_team_player_features(players_df, home_team)
        a_p = get_team_player_features(players_df, away_team)
    else:
        h_p = {'team_rating': 0.0, 'team_attack': 0.0, 'team_defense': 0.0}
        a_p = {'team_rating': 0.0, 'team_attack': 0.0, 'team_defense': 0.0}

    def stat_avg(matches, col):
        if col not in matches.columns:
            return 0.0
        return safe_mean(pd.to_numeric(matches[col], errors='coerce').dropna())

    match_stat_features = {}
    for feat, (h_col, a_col) in MATCH_STAT_SOURCES.items():
        match_stat_features[f'home_{feat}'] = stat_avg(hm, h_col)
        match_stat_features[f'away_{feat}'] = stat_avg(am, a_col)

    return {
        'home_avg_goals_scored':   safe_mean(hm['home_score']),
        'home_avg_goals_conceded': safe_mean(hm['away_score']),
        'home_win_rate':           safe_mean((hm['home_score'] > hm['away_score']).astype(float)),
        'away_avg_goals_scored':   safe_mean(am['away_score']),
        'away_avg_goals_conceded': safe_mean(am['home_score']),
        'away_win_rate':           safe_mean((am['away_score'] > am['home_score']).astype(float)),
        'home_is_neutral':         int(is_neutral),
        'league_encoded':          league_enc,
        'is_international':        is_international,
        'home_league_win_rate':    home_league_win_rate,
        'away_league_win_rate':    away_league_win_rate,
        'home_team_rating':        h_p['team_rating'],
        'away_team_rating':        a_p['team_rating'],
        'home_team_attack':        h_p['team_attack'],
        'away_team_attack':        a_p['team_attack'],
        'home_team_defense':       h_p['team_defense'],
        'away_team_defense':       a_p['team_defense'],
        'rating_diff':             h_p['team_rating'] - a_p['team_rating'],
        **match_stat_features,
    }
