import pandas as pd
import pytest
from unittest.mock import patch
from src.ingest import normalize_csv, download_csv, build_clubs_dataset, combine_datasets


def _raw(date='01/08/2024', home='Arsenal', away='Chelsea', fthg=2, ftag=1, ftr='H'):
    return pd.DataFrame({
        'Date': [date], 'HomeTeam': [home], 'AwayTeam': [away],
        'FTHG': [fthg], 'FTAG': [ftag], 'FTR': [ftr],
    })


def test_normalize_csv_renames_columns():
    result = normalize_csv(_raw(), 'Premier League')
    expected_cols = {'date', 'home_team', 'away_team', 'home_score', 'away_score',
                     'result', 'neutral', 'tournament', 'league', 'competition_type',
                     'home_shots', 'away_shots', 'home_shots_on_target', 'away_shots_on_target',
                     'home_corners', 'away_corners', 'home_yellow', 'away_yellow',
                     'home_red', 'away_red'}
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


def test_download_csv_returns_none_on_http_error(tmp_path):
    with patch('urllib.request.urlretrieve', side_effect=Exception('404')):
        result = download_csv('E0', '2425', str(tmp_path))
    assert result is None


def test_download_csv_skips_if_file_exists(tmp_path):
    dest = tmp_path / 'E0_2425.csv'
    dest.write_text('existing')
    with patch('urllib.request.urlretrieve') as mock_dl:
        result = download_csv('E0', '2425', str(tmp_path))
    mock_dl.assert_not_called()
    assert str(result) == str(dest)


def test_build_clubs_dataset_normalizes_multiple_csvs(tmp_path):
    raw_dir = tmp_path / 'raw'
    raw_dir.mkdir()
    csv1 = raw_dir / 'E0_2324.csv'
    csv1.write_text('Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/08/2023,Arsenal,Chelsea,2,1,H\n')
    csv2 = raw_dir / 'SP1_2324.csv'
    csv2.write_text('Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n05/08/2023,Barcelona,Madrid,1,0,H\n')
    leagues = {'E0': 'Premier League', 'SP1': 'La Liga'}
    seasons = ['2324']
    result = build_clubs_dataset(str(raw_dir), leagues=leagues, seasons=seasons)
    assert len(result) == 2
    assert set(result['league'].tolist()) == {'Premier League', 'La Liga'}
    assert 'competition_type' in result.columns
    assert (result['competition_type'] == 'club').all()


def test_combine_datasets_adds_league_to_international(tmp_path):
    intl_path = tmp_path / 'results.csv'
    intl_df = pd.DataFrame({
        'date': ['2020-01-01'],
        'home_team': ['Brazil'],
        'away_team': ['Germany'],
        'home_score': [2],
        'away_score': [1],
        'neutral': [False],
        'tournament': ['Friendly'],
    })
    intl_df.to_csv(intl_path, index=False)

    clubs_df = pd.DataFrame({
        'date': pd.to_datetime(['2020-02-01']),
        'home_team': ['Arsenal'],
        'away_team': ['Chelsea'],
        'home_score': [1],
        'away_score': [0],
        'neutral': [False],
        'tournament': ['Premier League'],
        'league': ['Premier League'],
        'competition_type': ['club'],
        'result': ['H'],
    })

    combined = combine_datasets(clubs_df, str(intl_path))
    assert len(combined) == 2
    assert 'league' in combined.columns
    assert 'competition_type' in combined.columns
    assert set(combined['competition_type'].tolist()) == {'club', 'international'}


@pytest.fixture
def raw_df_with_stats():
    return pd.DataFrame({
        'Date': ['16/08/2024', '17/08/2024'],
        'HomeTeam': ['Man United', 'Arsenal'],
        'AwayTeam': ['Fulham', 'Wolves'],
        'FTHG': [1, 2], 'FTAG': [0, 0], 'FTR': ['H', 'H'],
        'HS': [14, 18], 'AS': [10, 6],
        'HST': [5, 9], 'AST': [2, 1],
        'HC': [7, 8], 'AC': [8, 2],
        'HY': [2, 1], 'AY': [3, 2],
        'HR': [0, 0], 'AR': [0, 1],
    })


def test_normalize_csv_keeps_match_stats(raw_df_with_stats):
    result = normalize_csv(raw_df_with_stats, 'Premier League')
    assert result.iloc[0]['home_shots'] == 14
    assert result.iloc[0]['away_shots'] == 10
    assert result.iloc[0]['home_shots_on_target'] == 5
    assert result.iloc[0]['home_corners'] == 7
    assert result.iloc[0]['home_yellow'] == 2
    assert result.iloc[1]['away_red'] == 1


def test_normalize_csv_stats_nan_when_missing():
    raw = pd.DataFrame({
        'Date': ['16/08/2024'],
        'HomeTeam': ['Man United'], 'AwayTeam': ['Fulham'],
        'FTHG': [1], 'FTAG': [0], 'FTR': ['H'],
    })
    result = normalize_csv(raw, 'Premier League')
    assert 'home_shots' in result.columns
    assert pd.isna(result.iloc[0]['home_shots'])
    assert pd.isna(result.iloc[0]['away_red'])
