import pandas as pd
import joblib
import pytest
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from src.predict import predict_match, get_team_match_count
from src.preprocess import FEATURE_COLS


class XGBClassifierWrapper:
    """Wrapper around XGBClassifier that handles string labels"""
    def __init__(self, base_clf, label_encoder):
        self.base_clf = base_clf
        self.label_encoder = label_encoder
        self.classes_ = label_encoder.classes_

    def predict(self, X):
        return self.label_encoder.inverse_transform(self.base_clf.predict(X))

    def predict_proba(self, X):
        return self.base_clf.predict_proba(X)

    def __getstate__(self):
        return {'base_clf': self.base_clf, 'label_encoder': self.label_encoder, 'classes_': self.classes_}

    def __setstate__(self, state):
        self.base_clf = state['base_clf']
        self.label_encoder = state['label_encoder']
        self.classes_ = state['classes_']

    def __getattr__(self, name):
        return getattr(self.base_clf, name)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01',
                 '2020-04-01', '2020-05-01'],
        'home_team': ['Brazil', 'Brazil', 'Brazil', 'Argentina', 'Brazil'],
        'away_team': ['Germany', 'France',  'Spain',  'Brazil',   'Italy'],
        'home_score': [3, 2, 1, 0, 2],
        'away_score': [1, 2, 0, 1, 0],
        'neutral': [False] * 5,
        'tournament': ['Friendly'] * 5,
        'league': ['International'] * 5,
        'competition_type': ['international'] * 5,
    })


@pytest.fixture
def mock_model_path(tmp_path):
    le = LabelEncoder()
    le.fit(['International', 'Premier League'])

    # Train XGBoost with numeric labels
    outcome_le = LabelEncoder()
    outcome_le.fit(['H', 'D', 'A'])

    clf = XGBClassifier(n_estimators=2, random_state=42, eval_metric='mlogloss', verbosity=0)
    X = pd.DataFrame([[0.5] * len(FEATURE_COLS)] * 6, columns=FEATURE_COLS)
    y = pd.Series(['H', 'D', 'A', 'H', 'D', 'A'])
    y_encoded = outcome_le.transform(y)
    clf.fit(X, y_encoded)

    # Wrap it to handle string labels
    wrapped_clf = XGBClassifierWrapper(clf, outcome_le)

    path = str(tmp_path / 'model.pkl')
    joblib.dump({'model': wrapped_clf, 'feature_cols': FEATURE_COLS, 'n': 5, 'league_encoder': le}, path)
    return path


def test_predict_match_returns_required_keys(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path)
    assert {'result', 'result_label', 'probabilities', 'home_team', 'away_team'}.issubset(result.keys())


def test_predict_match_result_is_valid_class(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path)
    assert result['result'] in ['H', 'D', 'A']


def test_predict_match_probabilities_sum_to_one(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path)
    assert abs(sum(result['probabilities'].values()) - 1.0) < 1e-6


def test_predict_match_teams_stored_in_result(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path)
    assert result['home_team'] == 'Brazil'
    assert result['away_team'] == 'Germany'


def test_predict_match_with_league_param(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path,
                           league='International', competition_type='international')
    assert result['result'] in ['H', 'D', 'A']


def test_predict_match_unknown_league_does_not_raise(sample_df, mock_model_path):
    result = predict_match('Brazil', 'Germany', sample_df, mock_model_path,
                           league='Unknown League', competition_type='club')
    assert result['result'] in ['H', 'D', 'A']


def test_get_team_match_count_counts_home_and_away(sample_df):
    assert get_team_match_count(sample_df, 'Brazil') == 5


def test_get_team_match_count_returns_zero_for_unknown(sample_df):
    assert get_team_match_count(sample_df, 'Unknown') == 0
