import pandas as pd
import pytest
from src.player_features import load_players, get_team_player_features, TEAM_NAME_MAP


@pytest.fixture
def players_df():
    return pd.DataFrame({
        'club_name': ['Arsenal', 'Arsenal', 'Arsenal', 'Barcelona', 'Barcelona'],
        'nationality_name': ['England', 'England', 'France', 'Spain', 'Spain'],
        'overall': [85, 80, 75, 90, 88],
        'shooting': [80, 70, 60, 88, 75],
        'dribbling': [75, 65, 55, 92, 80],
        'defending': [60, 70, 80, 55, 65],
        'physic': [70, 75, 80, 65, 72],
        'player_positions': ['ST', 'CM', 'CB', 'LW', 'CDM'],
    })


def test_load_players_returns_none_if_missing(tmp_path):
    result = load_players(str(tmp_path / 'nonexistent.csv'))
    assert result is None


def test_load_players_returns_dataframe_if_exists(tmp_path):
    path = tmp_path / 'players.csv'
    pd.DataFrame({'overall': [80]}).to_csv(path, index=False)
    result = load_players(str(path))
    assert isinstance(result, pd.DataFrame)


def test_get_team_player_features_returns_three_keys(players_df):
    result = get_team_player_features(players_df, 'Arsenal')
    assert set(result.keys()) == {'team_rating', 'team_attack', 'team_defense'}


def test_get_team_player_features_rating_is_normalized(players_df):
    result = get_team_player_features(players_df, 'Arsenal')
    assert 0.0 <= result['team_rating'] <= 1.0


def test_get_team_player_features_rating_correct_value(players_df):
    result = get_team_player_features(players_df, 'Arsenal')
    expected = (85 + 80 + 75) / 3 / 100.0
    assert result['team_rating'] == pytest.approx(expected, abs=0.01)


def test_get_team_player_features_returns_zeros_for_unknown_team(players_df):
    result = get_team_player_features(players_df, 'Unknown FC')
    assert result == {'team_rating': 0.0, 'team_attack': 0.0, 'team_defense': 0.0}


def test_get_team_player_features_normalizes_alias(players_df):
    players_df_mc = players_df.copy()
    players_df_mc['club_name'] = ['Manchester City'] * 5
    result_alias = get_team_player_features(players_df_mc, 'Man City')
    result_full = get_team_player_features(players_df_mc, 'Manchester City')
    assert result_alias['team_rating'] == pytest.approx(result_full['team_rating'])


def test_get_team_player_features_uses_nationality_for_international(players_df):
    result = get_team_player_features(players_df, 'England')
    assert result['team_rating'] > 0.0


def test_team_name_map_has_common_aliases():
    assert 'Man City' in TEAM_NAME_MAP
    assert 'Man United' in TEAM_NAME_MAP
