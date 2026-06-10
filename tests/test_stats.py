import pandas as pd
import pytest
from src.stats import (
    get_team_overall_stats,
    get_radar_stats,
    get_head_to_head,
    get_recent_matches,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'date': [
            '2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01',
            '2020-05-01', '2020-06-01', '2020-07-01', '2020-08-01',
        ],
        'home_team': ['Brazil', 'Brazil', 'Argentina', 'Brazil', 'Argentina', 'Brazil', 'Mexico', 'Argentina'],
        'away_team': ['Germany', 'Argentina', 'Germany', 'Mexico', 'Brazil', 'Argentina', 'Brazil', 'Mexico'],
        'home_score': [3, 2, 1, 2, 0, 1, 1, 2],
        'away_score': [1, 1, 0, 0, 2, 1, 2, 0],
        'neutral': [False] * 8,
        'tournament': ['Friendly'] * 8,
    })


# ── get_team_overall_stats ────────────────────────────────────────────────

def test_overall_stats_counts_home_and_away_matches(sample_df):
    # Brazil: home rows 0,1,3,5 + away rows 4,6 = 6 total
    stats = get_team_overall_stats(sample_df, 'Brazil')
    assert stats['total_matches'] == 6


def test_overall_stats_win_draw_loss_counts(sample_df):
    # Brazil: W(row0), W(row1), W(row3), W(row4 away), D(row5), W(row6 away) = 5W 1D 0L
    stats = get_team_overall_stats(sample_df, 'Brazil')
    assert stats['wins'] == 5
    assert stats['draws'] == 1
    assert stats['losses'] == 0


def test_overall_stats_goals_scored_per_game(sample_df):
    # Brazil goals_for: 3+2+2+2+1+2 = 12 across 6 matches → 2.0
    stats = get_team_overall_stats(sample_df, 'Brazil')
    assert stats['goals_scored_per_game'] == pytest.approx(2.0, rel=0.01)


def test_overall_stats_form_length_respects_n_recent(sample_df):
    stats = get_team_overall_stats(sample_df, 'Brazil', n_recent=3)
    assert len(stats['form']) == 3


def test_overall_stats_unknown_team_returns_zero_matches(sample_df):
    stats = get_team_overall_stats(sample_df, 'Unknown')
    assert stats['total_matches'] == 0
    assert stats['win_rate'] == 0.0
    assert stats['form'] == []


# ── get_radar_stats ───────────────────────────────────────────────────────

def test_radar_stats_returns_six_dimensions(sample_df):
    radar = get_radar_stats(sample_df, 'Brazil', 'Argentina')
    assert len(radar['dimensions']) == 6
    assert len(radar['home']) == 6
    assert len(radar['away']) == 6


def test_radar_stats_all_values_between_0_and_1(sample_df):
    radar = get_radar_stats(sample_df, 'Brazil', 'Argentina')
    for v in radar['home'] + radar['away']:
        assert 0.0 <= v <= 1.0, f"Value {v} out of range [0,1]"


# ── get_head_to_head ──────────────────────────────────────────────────────

def test_h2h_total_encounters(sample_df):
    # Brazil vs Argentina: row1 (Brazil H 2-1), row4 (Argentina H 0-2), row5 (Brazil H 1-1) = 3
    h2h = get_head_to_head(sample_df, 'Brazil', 'Argentina')
    assert h2h['home_wins'] + h2h['draws'] + h2h['away_wins'] == 3


def test_h2h_correct_win_counts(sample_df):
    # Brazil wins: row1 (2>1), row4 (away: 2>0) = 2; Draws: row5 (1==1) = 1; Argentina wins: 0
    h2h = get_head_to_head(sample_df, 'Brazil', 'Argentina')
    assert h2h['home_wins'] == 2
    assert h2h['draws'] == 1
    assert h2h['away_wins'] == 0


def test_h2h_no_encounters_returns_empty(sample_df):
    h2h = get_head_to_head(sample_df, 'Germany', 'Mexico')
    assert h2h['home_wins'] == 0
    assert h2h['last_matches'] == []


# ── get_recent_matches ────────────────────────────────────────────────────

def test_recent_matches_returns_dataframe_with_required_columns(sample_df):
    result = get_recent_matches(sample_df, 'Brazil')
    assert isinstance(result, pd.DataFrame)
    for col in ['date', 'opponent', 'home_away', 'goals_for', 'goals_against', 'result', 'tournament']:
        assert col in result.columns


def test_recent_matches_result_values_are_valid(sample_df):
    result = get_recent_matches(sample_df, 'Brazil')
    assert set(result['result'].unique()).issubset({'W', 'D', 'L'})
    assert (result['result'] == 'W').sum() == 5


# ── get_league_stats ──────────────────────────────────────────────────────

from src.stats import get_league_stats


@pytest.fixture
def league_df():
    return pd.DataFrame({
        'date': ['2020-01-01', '2020-02-01', '2020-03-01',
                 '2020-04-01', '2020-05-01'],
        'home_team': ['Arsenal', 'Arsenal', 'Arsenal', 'Arsenal', 'Chelsea'],
        'away_team': ['Chelsea', 'Liverpool', 'Man City', 'Spurs', 'Arsenal'],
        'home_score': [2, 1, 0, 3, 1],
        'away_score': [1, 1, 2, 0, 0],
        'neutral': [False] * 5,
        'tournament': ['Premier League'] * 5,
        'league': ['Premier League'] * 5,
        'competition_type': ['club'] * 5,
    })


def test_get_league_stats_total_matches(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'Premier League')
    # Arsenal: 4 home + 1 away = 5
    assert stats['total_matches'] == 5


def test_get_league_stats_wins(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'Premier League')
    # Home wins: rows 0 (2-1), 3 (3-0) = 2. Away win: row 4 (Chelsea 1-0 Arsenal) = Arsenal LOSES as away
    # Let me recalculate: row 4: Chelsea(home) 1, Arsenal(away) 0 -> Arsenal loses
    # Home wins for Arsenal: row0(2>1=W), row1(1=1=D), row2(0<2=L), row3(3>0=W) = 2 wins
    # Away match for Arsenal: row4 Chelsea 1 vs Arsenal 0 -> Arsenal away_score=0 < home_score=1 -> Loss
    # Total wins = 2
    assert stats['wins'] == 2


def test_get_league_stats_returns_league_name(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'Premier League')
    assert stats['league'] == 'Premier League'


def test_get_league_stats_unknown_team_returns_zeros(league_df):
    stats = get_league_stats(league_df, 'Unknown FC', 'Premier League')
    assert stats['total_matches'] == 0
    assert stats['wins'] == 0


def test_get_league_stats_unknown_league_returns_zeros(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'La Liga')
    assert stats['total_matches'] == 0


def test_get_league_stats_win_rate_is_ratio(league_df):
    stats = get_league_stats(league_df, 'Arsenal', 'Premier League')
    assert 0.0 <= stats['win_rate'] <= 1.0


def test_get_league_stats_no_league_col_falls_back(league_df):
    df_no_league = league_df.drop(columns=['league'])
    stats = get_league_stats(df_no_league, 'Arsenal', 'Premier League')
    assert stats['total_matches'] == 0
