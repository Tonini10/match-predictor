import pandas as pd
import pytest
from src.player_features import (load_players, get_team_player_features,
                                 get_team_squad, TEAM_NAME_MAP)


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


@pytest.fixture
def squad_players_df():
    return pd.DataFrame({
        'short_name': ['Saka', 'Odegaard', 'Saliba', 'Lewandowski', 'Pedri'],
        'club_name': ['Arsenal', 'Arsenal', 'Arsenal', 'Barcelona', 'Barcelona'],
        'nationality_name': ['England', 'Norway', 'France', 'Poland', 'Spain'],
        'age': [22, 25, 23, 35, 21],
        'overall': [86, 87, 84, 91, 85],
        'shooting': [80, 78, 50, 92, 74],
        'attacking_finishing': [82, 76, 45, 94, 70],
        'pace': [85, 72, 80, 75, 79],
        'player_positions': ['RW', 'CAM', 'CB', 'ST', 'CM'],
    })


def test_get_team_squad_returns_top_n_by_overall(squad_players_df):
    squad = get_team_squad(squad_players_df, 'Arsenal', n=2)
    assert len(squad) == 2
    assert squad.iloc[0]['short_name'] == 'Odegaard'   # overall 87
    assert squad.iloc[1]['short_name'] == 'Saka'        # overall 86


def test_get_team_squad_includes_offensive_attributes(squad_players_df):
    squad = get_team_squad(squad_players_df, 'Barcelona')
    for col in ['short_name', 'player_positions', 'age', 'overall',
                'shooting', 'attacking_finishing', 'pace']:
        assert col in squad.columns


def test_get_team_squad_falls_back_to_nationality(squad_players_df):
    squad = get_team_squad(squad_players_df, 'Poland')
    assert len(squad) == 1
    assert squad.iloc[0]['short_name'] == 'Lewandowski'


def test_get_team_squad_unknown_team_returns_empty(squad_players_df):
    squad = get_team_squad(squad_players_df, 'Real Madrid')
    assert len(squad) == 0


def test_get_team_squad_none_players_returns_empty():
    squad = get_team_squad(None, 'Arsenal')
    assert len(squad) == 0
