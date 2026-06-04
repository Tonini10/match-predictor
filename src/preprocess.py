import pandas as pd

FEATURE_COLS = [
    'home_avg_goals_scored',
    'home_avg_goals_conceded',
    'home_win_rate',
    'away_avg_goals_scored',
    'away_avg_goals_conceded',
    'away_win_rate',
    'home_is_neutral',
]


def build_training_data(df, n=5):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['away_win'] = (df['away_score'] > df['home_score']).astype(int)

    def rolling_mean(group_col, val_col):
        return df.groupby(group_col)[val_col].transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean().fillna(0)
        )

    df['home_avg_goals_scored']   = rolling_mean('home_team', 'home_score')
    df['home_avg_goals_conceded'] = rolling_mean('home_team', 'away_score')
    df['home_win_rate']           = rolling_mean('home_team', 'home_win')
    df['away_avg_goals_scored']   = rolling_mean('away_team', 'away_score')
    df['away_avg_goals_conceded'] = rolling_mean('away_team', 'home_score')
    df['away_win_rate']           = rolling_mean('away_team', 'away_win')
    df['home_is_neutral']         = df['neutral'].astype(int)

    df['result'] = df.apply(
        lambda r: 'H' if r['home_score'] > r['away_score']
                  else ('D' if r['home_score'] == r['away_score'] else 'A'),
        axis=1,
    )

    return df[FEATURE_COLS + ['result']].copy()


def build_features_for_prediction(df, home_team, away_team, is_neutral=False, n=5, before_date=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if before_date is not None:
        df = df[df['date'] < pd.Timestamp(before_date)]

    hm = df[df['home_team'] == home_team].tail(n)
    am = df[df['away_team'] == away_team].tail(n)

    def safe_mean(series):
        return 0.0 if series.isna().all() else float(series.mean())

    return {
        'home_avg_goals_scored':   safe_mean(hm['home_score']),
        'home_avg_goals_conceded': safe_mean(hm['away_score']),
        'home_win_rate':           safe_mean((hm['home_score'] > hm['away_score']).astype(float)),
        'away_avg_goals_scored':   safe_mean(am['away_score']),
        'away_avg_goals_conceded': safe_mean(am['home_score']),
        'away_win_rate':           safe_mean((am['away_score'] > am['home_score']).astype(float)),
        'home_is_neutral':         int(is_neutral),
    }
