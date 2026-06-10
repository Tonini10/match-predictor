import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder
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


@pytest.fixture
def sample_df_with_leagues():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01'],
        'home_team': ['Arsenal', 'Arsenal', 'Barcelona', 'Arsenal'],
        'away_team': ['Chelsea', 'Liverpool', 'Madrid', 'Man City'],
        'home_score': [2, 1, 3, 0],
        'away_score': [1, 2, 0, 1],
        'neutral': [False] * 4,
        'tournament': ['Premier League', 'Premier League', 'La Liga', 'Premier League'],
        'league': ['Premier League', 'Premier League', 'La Liga', 'Premier League'],
        'competition_type': ['club', 'club', 'club', 'club'],
    })


def test_build_training_data_returns_tuple(sample_df):
    result = build_training_data(sample_df)
    assert isinstance(result, tuple) and len(result) == 2


def test_build_training_data_returns_expected_columns(sample_df):
    df, le = build_training_data(sample_df)
    assert set(FEATURE_COLS + ['result']) == set(df.columns)


def test_build_training_data_returns_label_encoder(sample_df):
    df, le = build_training_data(sample_df)
    assert isinstance(le, LabelEncoder)


def test_build_training_data_length_matches_input(sample_df):
    df, _ = build_training_data(sample_df)
    assert len(df) == len(sample_df)


def test_build_training_data_result_home_win(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[0]['result'] == 'H'


def test_build_training_data_result_draw(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[1]['result'] == 'D'


def test_build_training_data_result_away_win(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[3]['result'] == 'A'


def test_build_training_data_first_match_stats_are_zero(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[0]['home_avg_goals_scored'] == 0.0
    assert df.iloc[0]['home_win_rate'] == 0.0


def test_build_training_data_second_match_uses_first(sample_df):
    df, _ = build_training_data(sample_df)
    assert df.iloc[1]['home_avg_goals_scored'] == pytest.approx(3.0)


def test_build_training_data_has_new_feature_cols(sample_df):
    df, _ = build_training_data(sample_df)
    assert 'league_encoded' in df.columns
    assert 'is_international' in df.columns
    assert 'home_league_win_rate' in df.columns
    assert 'away_league_win_rate' in df.columns


def test_build_training_data_backward_compat_no_league_col(sample_df):
    df, le = build_training_data(sample_df)
    assert len(df) == len(sample_df)
    assert 'league_encoded' in df.columns


def test_build_training_data_league_win_rate_per_league(sample_df_with_leagues):
    df, _ = build_training_data(sample_df_with_leagues)
    assert df.iloc[0]['home_league_win_rate'] == 0.0
    assert df.iloc[1]['home_league_win_rate'] == pytest.approx(1.0)


def test_build_training_data_is_international_flag(sample_df_with_leagues):
    df, _ = build_training_data(sample_df_with_leagues)
    assert (df['is_international'] == 0).all()


def test_build_features_for_prediction_returns_all_keys(sample_df):
    df, le = build_training_data(sample_df)
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', label_encoder=le)
    assert set(features.keys()) == set(FEATURE_COLS)


def test_build_features_for_prediction_neutral_flag(sample_df):
    df, le = build_training_data(sample_df)
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', is_neutral=True, label_encoder=le)
    assert features['home_is_neutral'] == 1


def test_build_features_for_prediction_unknown_team_returns_zeros(sample_df):
    features = build_features_for_prediction(sample_df, 'Unknown', 'Brazil')
    assert features['home_avg_goals_scored'] == 0.0
    assert features['home_win_rate'] == 0.0


def test_build_features_for_prediction_league_encoded_zero_without_encoder(sample_df):
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', league='Friendly')
    assert features['league_encoded'] == 0


def test_build_features_for_prediction_is_international_from_param(sample_df):
    features = build_features_for_prediction(
        sample_df, 'Brazil', 'Germany', competition_type='international'
    )
    assert features['is_international'] == 1

    features_club = build_features_for_prediction(
        sample_df, 'Brazil', 'Germany', competition_type='club'
    )
    assert features_club['is_international'] == 0


def test_build_training_data_away_stats_rolling_correctness():
    df = pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01'],
        'home_team': ['Brazil', 'France'],
        'away_team': ['Germany', 'Germany'],
        'home_score': [3, 2],
        'away_score': [1, 2],
        'neutral': [False, False],
        'tournament': ['Friendly', 'Friendly'],
    })
    result, _ = build_training_data(df)
    assert result.iloc[0]['away_avg_goals_scored'] == 0.0
    assert result.iloc[1]['away_avg_goals_scored'] == pytest.approx(1.0)


from src.player_features import get_team_player_features


@pytest.fixture
def players_df_simple():
    return pd.DataFrame({
        'club_name': ['Brazil', 'Germany', 'Argentina'],
        'nationality_name': ['Brazil', 'Germany', 'Argentina'],
        'overall': [88, 85, 90],
        'shooting': [85, 80, 88],
        'dribbling': [90, 78, 87],
        'defending': [70, 80, 72],
        'physic': [75, 82, 74],
        'player_positions': ['ST', 'CM', 'LW'],
    })


def test_build_training_data_with_players_df_adds_player_cols(sample_df, players_df_simple):
    df, _ = build_training_data(sample_df, players_df=players_df_simple)
    for col in ['home_team_rating', 'away_team_rating', 'home_team_attack',
                'away_team_attack', 'home_team_defense', 'away_team_defense', 'rating_diff']:
        assert col in df.columns, f"Missing column: {col}"


def test_build_training_data_without_players_df_player_cols_are_zero(sample_df):
    df, _ = build_training_data(sample_df, players_df=None)
    assert df['home_team_rating'].sum() == 0.0
    assert df['away_team_rating'].sum() == 0.0
    assert df['rating_diff'].sum() == 0.0


def test_build_features_for_prediction_with_players_df(sample_df, players_df_simple):
    features = build_features_for_prediction(
        sample_df, 'Brazil', 'Germany', players_df=players_df_simple
    )
    assert set(features.keys()) == set(FEATURE_COLS)
    assert features['home_team_rating'] > 0.0
    assert features['away_team_rating'] > 0.0


def test_build_features_for_prediction_without_players_df_zeros(sample_df):
    features = build_features_for_prediction(sample_df, 'Brazil', 'Germany', players_df=None)
    assert features['home_team_rating'] == 0.0
    assert features['rating_diff'] == 0.0
