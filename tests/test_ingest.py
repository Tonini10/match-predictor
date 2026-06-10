import pandas as pd
import pytest
from src.ingest import normalize_csv


def _raw(date='01/08/2024', home='Arsenal', away='Chelsea', fthg=2, ftag=1, ftr='H'):
    return pd.DataFrame({
        'Date': [date], 'HomeTeam': [home], 'AwayTeam': [away],
        'FTHG': [fthg], 'FTAG': [ftag], 'FTR': [ftr],
    })


def test_normalize_csv_renames_columns():
    result = normalize_csv(_raw(), 'Premier League')
    expected_cols = {'date', 'home_team', 'away_team', 'home_score', 'away_score',
                     'result', 'neutral', 'tournament', 'league', 'competition_type'}
    assert expected_cols == set(result.columns)


def test_normalize_csv_sets_league_name():
    result = normalize_csv(_raw(), 'La Liga')
    assert result['league'].iloc[0] == 'La Liga'


def test_normalize_csv_sets_competition_type_club_by_default():
    result = normalize_csv(_raw(), 'Premier League')
    assert result['competition_type'].iloc[0] == 'club'


def test_normalize_csv_parses_date():
    result = normalize_csv(_raw(date='15/03/2023'), 'Serie A')
    assert str(result['date'].iloc[0].date()) == '2023-03-15'


def test_normalize_csv_drops_rows_with_missing_scores():
    raw = pd.DataFrame({
        'Date': ['01/08/2024', '08/08/2024'],
        'HomeTeam': ['Arsenal', 'Man City'],
        'AwayTeam': ['Chelsea', 'Liverpool'],
        'FTHG': [2, None], 'FTAG': [1, 1], 'FTR': ['H', None],
    })
    result = normalize_csv(raw, 'Premier League')
    assert len(result) == 1


def test_normalize_csv_infers_result_when_ftr_missing():
    raw = pd.DataFrame({
        'Date': ['01/08/2024'], 'HomeTeam': ['Arsenal'], 'AwayTeam': ['Chelsea'],
        'FTHG': [0], 'FTAG': [0],
    })
    result = normalize_csv(raw, 'La Liga')
    assert result['result'].iloc[0] == 'D'


def test_normalize_csv_infers_home_win():
    raw = pd.DataFrame({
        'Date': ['01/08/2024'], 'HomeTeam': ['Arsenal'], 'AwayTeam': ['Chelsea'],
        'FTHG': [3], 'FTAG': [1],
    })
    result = normalize_csv(raw, 'La Liga')
    assert result['result'].iloc[0] == 'H'


def test_normalize_csv_infers_away_win():
    raw = pd.DataFrame({
        'Date': ['01/08/2024'], 'HomeTeam': ['Arsenal'], 'AwayTeam': ['Chelsea'],
        'FTHG': [0], 'FTAG': [2],
    })
    result = normalize_csv(raw, 'La Liga')
    assert result['result'].iloc[0] == 'A'


def test_normalize_csv_neutral_is_false():
    result = normalize_csv(_raw(), 'Bundesliga')
    assert result['neutral'].iloc[0] is False
