import pandas as pd
import joblib
from src.preprocess import build_features_for_prediction, FEATURE_COLS

_RESULT_LABELS = {
    'H': '{home} gana',
    'D': 'Empate',
    'A': '{away} gana',
}


def predict_match(home_team, away_team, df, model_path='model.pkl'):
    artifact = joblib.load(model_path)
    clf = artifact['model']

    features = build_features_for_prediction(df, home_team, away_team)
    X = pd.DataFrame([features])[FEATURE_COLS]

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
