import pandas as pd
import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier
from src.predict import predict_match, get_team_match_count
from src.preprocess import FEATURE_COLS


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
    })


@pytest.fixture
def mock_model_path(tmp_path):
    clf = RandomForestClassifier(n_estimators=2, random_state=42)
    X = pd.DataFrame([[0.5] * len(FEATURE_COLS)] * 6, columns=FEATURE_COLS)
    y = ['H', 'D', 'A', 'H', 'D', 'A']
    clf.fit(X, y)
    path = str(tmp_path / 'model.pkl')
    joblib.dump({'model': clf, 'feature_cols': FEATURE_COLS}, path)
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


def test_get_team_match_count_counts_home_and_away(sample_df):
    # Brazil: home in rows 0,1,2,4 and away in row 3 = 5 total
    assert get_team_match_count(sample_df, 'Brazil') == 5


def test_get_team_match_count_returns_zero_for_unknown(sample_df):
    assert get_team_match_count(sample_df, 'Unknown') == 0
