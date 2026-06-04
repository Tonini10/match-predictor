import pandas as pd
import pytest
from src.preprocess import build_training_data, build_features_for_prediction, FEATURE_COLS


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01',
                 '2020-04-01', '2020-05-01', '2020-06-01'],
        'home_team': ['Brazil', 'Brazil', 'Brazil', 'Argentina', 'Brazil', 'Germany'],
        'away_team': ['Germany', 'France',  'Spain',  'Brazil',   'Italy',  'Brazil'],
        'home_score': [3, 2, 1, 0, 2, 1],
        'away_score': [1, 2, 0, 1, 0, 0],
        'neutral': [False] * 6,
        'tournament': ['Friendly'] * 6,
    })


def test_build_training_data_returns_expected_columns(sample_df):
    result = build_training_data(sample_df)
    assert set(FEATURE_COLS + ['result']).issubset(set(result.columns))


def test_build_training_data_length_matches_input(sample_df):
    result = build_training_data(sample_df)
    assert len(result) == len(sample_df)


def test_build_training_data_result_home_win(sample_df):
    result = build_training_data(sample_df)
    assert result.iloc[0]['result'] == 'H'  # Brazil 3-1 Germany


def test_build_training_data_result_draw(sample_df):
    result = build_training_data(sample_df)
    assert result.iloc[1]['result'] == 'D'  # Brazil 2-2 France


def test_build_training_data_result_away_win(sample_df):
    result = build_training_data(sample_df)
    assert result.iloc[3]['result'] == 'A'  # Argentina 0-1 Brazil


def test_build_training_data_first_match_stats_are_zero(sample_df):
    result = build_training_data(sample_df)
    assert result.iloc[0]['home_avg_goals_scored'] == 0.0
    assert result.iloc[0]['home_win_rate'] == 0.0


def test_build_training_data_second_match_uses_first(sample_df):
    # Row 1 is Brazil's second home match; prior home match (row 0) scored 3
    result = build_training_data(sample_df)
    assert result.iloc[1]['home_avg_goals_scored'] == pytest.approx(3.0)


def test_build_features_for_prediction_returns_all_keys(sample_df):
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany')
    assert set(features.keys()) == set(FEATURE_COLS)


def test_build_features_for_prediction_neutral_flag(sample_df):
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', is_neutral=True)
    assert features['home_is_neutral'] == 1


def test_build_features_for_prediction_unknown_team_returns_zeros(sample_df):
    features = build_features_for_prediction(sample_df, 'Unknown', 'Brazil')
    assert features['home_avg_goals_scored'] == 0.0
    assert features['home_win_rate'] == 0.0
