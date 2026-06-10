import os
import joblib
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
from src.train import train
from src.preprocess import FEATURE_COLS


@pytest.fixture
def sample_csv(tmp_path):
    dates = pd.date_range(start='2000-01-01', periods=20, freq='30D')
    df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d').tolist(),
        'home_team': ['Brazil', 'Argentina'] * 10,
        'away_team': ['Germany', 'France'] * 10,
        'home_score': [2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3, 0],
        'away_score': [1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1],
        'neutral': [False] * 20,
        'tournament': ['Friendly'] * 20,
    })
    path = tmp_path / 'results.csv'
    df.to_csv(path, index=False)
    return str(path)


def test_train_saves_model_pkl(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert os.path.exists(model_path)


def test_train_artifact_contains_required_keys(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    artifact = joblib.load(model_path)
    assert 'model' in artifact
    assert artifact['feature_cols'] == FEATURE_COLS
    assert artifact['n'] == 5
    assert 'league_encoder' in artifact
    assert isinstance(artifact['league_encoder'], LabelEncoder)


def test_train_model_has_predict_proba(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    clf, _ = train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert hasattr(clf, 'predict_proba')


def test_train_returns_accuracy_in_valid_range(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    _, accuracy = train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert 0.0 <= accuracy <= 1.0


def test_train_uses_xgboost(sample_csv, tmp_path):
    from xgboost import XGBClassifier
    model_path = str(tmp_path / 'model.pkl')
    clf, _ = train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    assert isinstance(clf, XGBClassifier)


def test_train_artifact_contains_players_df_key(sample_csv, tmp_path):
    model_path = str(tmp_path / 'model.pkl')
    train(data_path=sample_csv, model_path=model_path, n_estimators=5)
    artifact = joblib.load(model_path)
    # players_df can be None (no data/players.csv in test env) but key must exist
    assert 'players_df' in artifact
