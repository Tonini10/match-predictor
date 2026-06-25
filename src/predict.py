import functools
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
