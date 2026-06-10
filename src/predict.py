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
    n = artifact.get('n', 5)
    feature_cols = artifact['feature_cols']
    label_encoder = artifact.get('league_encoder')

    features = build_features_for_prediction(
        df, home_team, away_team,
        is_neutral=is_neutral, n=n,
        league=league, label_encoder=label_encoder,
        competition_type=competition_type,
    )
    X = pd.DataFrame([features])[feature_cols]

    predicted = clf.predict(X)[0]
    probas = clf.predict_proba(X)[0]

    label = _RESULT_LABELS[predicted].replace('{home}', home_team).replace('{away}', away_team)

    return {
        'result': predicted,
        'result_label': label,
        'probabilities': {cls: float(p) for cls, p in zip(clf.classes_, probas)},
        'home_team': home_team,
        'away_team': away_team,
    }


def get_team_match_count(df, team):
    return int(((df['home_team'] == team) | (df['away_team'] == team)).sum())
