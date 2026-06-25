import math
import functools
import numpy as np
import pandas as pd
import joblib
from src.preprocess import build_features_for_prediction, FEATURE_COLS

_RESULT_LABELS = {
    'H': '{home} wins',
    'D': 'Draw',
    'A': '{away} wins',
}


@functools.lru_cache(maxsize=None)
def _load_artifact(model_path):
    return joblib.load(model_path)


def predict_match(home_team, away_team, df, model_path='model.pkl', is_neutral=False,
                  league=None, competition_type=None):
    artifact = _load_artifact(model_path)
    clf = artifact['model']
    clf_ou = artifact.get('model_ou')
    n = artifact.get('n', 5)
    feature_cols = artifact['feature_cols']
    label_encoder = artifact.get('league_encoder')
    players_df = artifact.get('players_df')

    features = build_features_for_prediction(
        df, home_team, away_team,
        is_neutral=is_neutral, n=n,
        league=league, label_encoder=label_encoder,
        competition_type=competition_type,
        players_df=players_df,
    )
    X = pd.DataFrame([features])[feature_cols]

    result_encoder = artifact.get('result_encoder')

    predicted_raw = clf.predict(X)[0]
    probas = clf.predict_proba(X)[0]

    # Decode integer prediction back to string label ('H', 'D', 'A')
    if result_encoder is not None:
        predicted = str(result_encoder.inverse_transform([predicted_raw])[0])
        classes = [str(c) for c in result_encoder.classes_]
    else:
        predicted = str(predicted_raw)
        classes = [str(c) for c in clf.classes_]

    label = _RESULT_LABELS[predicted].replace('{home}', home_team).replace('{away}', away_team)

    def _binary_prob(model_key):
        m = artifact.get(model_key)
        if m is None:
            return None
        p = m.predict_proba(X)[0]
        cls = list(m.classes_)
        return float(p[cls.index(1)]) if 1 in cls else None

    over_1_5_prob = _binary_prob('model_ou15')
    over_2_5_prob = _binary_prob('model_ou')
    over_3_5_prob = _binary_prob('model_ou35')
    btts_prob     = _binary_prob('model_btts')

    return {
        'result': predicted,
        'result_label': label,
        'probabilities': {cls: float(p) for cls, p in zip(classes, probas)},
        'home_team': home_team,
        'away_team': away_team,
        'over_1_5_prob': over_1_5_prob,
        'over_2_5_prob': over_2_5_prob,
        'over_3_5_prob': over_3_5_prob,
        'btts_prob':     btts_prob,
        'over_under_prob': over_2_5_prob,  # backward compat
    }


def get_team_match_count(df, team):
    return int(((df['home_team'] == team) | (df['away_team'] == team)).sum())


def _poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    try:
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    except OverflowError:
        return 0.0


def predict_scoreline(df, home_team, away_team, is_neutral=False, top_n=5, max_goals=8):
    """Return (scorelines, lam_home, lam_away) using a Poisson strength model.

    scorelines: list of (home_goals, away_goals, probability) sorted by probability desc.
    lam_home / lam_away: expected goals per team.
    """
    # Prefer international matches; fall back to full dataset if too small
    if 'competition_type' in df.columns:
        base = df[df['competition_type'] == 'international']
        if len(base) < 200:
            base = df
    else:
        base = df

    base = base.dropna(subset=['home_score', 'away_score'])
    avg_h = base['home_score'].mean()
    avg_a = base['away_score'].mean()
    if avg_h == 0 or avg_a == 0:
        avg_h = avg_a = 1.3

    def _scored(team):
        as_h = base.loc[base['home_team'] == team, 'home_score'].tolist()
        as_a = base.loc[base['away_team'] == team, 'away_score'].tolist()
        return as_h + as_a

    def _conceded(team):
        as_h = base.loc[base['home_team'] == team, 'away_score'].tolist()
        as_a = base.loc[base['away_team'] == team, 'home_score'].tolist()
        return as_h + as_a

    def _strength(vals, baseline):
        return np.mean(vals) / baseline if vals and baseline > 0 else 1.0

    base_avg = (avg_h + avg_a) / 2
    atk_h = _strength(_scored(home_team), base_avg)
    def_h = _strength(_conceded(home_team), base_avg)
    atk_a = _strength(_scored(away_team), base_avg)
    def_a = _strength(_conceded(away_team), base_avg)

    lam_h = float(np.clip(avg_h * atk_h * def_a, 0.1, 7.0))
    lam_a = float(np.clip(avg_a * atk_a * def_h, 0.1, 7.0))

    # Neutral venue: remove home advantage by averaging lambdas slightly
    if is_neutral:
        mid = (lam_h + lam_a) / 2
        lam_h = lam_h * 0.85 + mid * 0.15
        lam_a = lam_a * 0.85 + mid * 0.15

    scorelines = []
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            p = _poisson_pmf(h, lam_h) * _poisson_pmf(a, lam_a)
            scorelines.append((h, a, p))

    scorelines.sort(key=lambda x: x[2], reverse=True)
    return scorelines[:top_n], lam_h, lam_a
