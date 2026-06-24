import json
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest
from src.ingest_api import fetch_wc_matches


SAMPLE_RESPONSE = {
    "matches": [
        {
            "status": "FINISHED",
            "utcDate": "2026-06-15T18:00:00Z",
            "homeTeam": {"name": "Mexico"},
            "awayTeam": {"name": "USA"},
            "score": {"fullTime": {"home": 1, "away": 2}},
        },
        {
            "status": "SCHEDULED",
            "utcDate": "2026-06-30T18:00:00Z",
            "homeTeam": {"name": "Brazil"},
            "awayTeam": {"name": "Germany"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
    ]
}


def _mock_urlopen(data_dict):
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = json.dumps(data_dict).encode()
    return cm


def test_fetch_wc_matches_normalizes_response():
    with patch('src.ingest_api.urllib.request.urlopen', return_value=_mock_urlopen(SAMPLE_RESPONSE)):
        df = fetch_wc_matches('test-key')
    assert len(df) == 1  # only FINISHED
    row = df.iloc[0]
    assert row['home_team'] == 'Mexico'
    assert row['away_team'] == 'USA'
    assert row['home_score'] == 1
    assert row['away_score'] == 2
    assert row['tournament'] == 'FIFA World Cup 2026'
    assert row['competition_type'] == 'international'
    assert row['neutral'] is True


def test_fetch_wc_matches_skips_unplayed():
    with patch('src.ingest_api.urllib.request.urlopen', return_value=_mock_urlopen(SAMPLE_RESPONSE)):
        df = fetch_wc_matches('test-key')
    teams = set(df['home_team'].tolist()) | set(df['away_team'].tolist())
    assert 'Brazil' not in teams
    assert 'Germany' not in teams


def test_fetch_wc_matches_includes_stat_columns():
    with patch('src.ingest_api.urllib.request.urlopen', return_value=_mock_urlopen(SAMPLE_RESPONSE)):
        df = fetch_wc_matches('test-key')
    for col in ['home_shots', 'away_shots', 'home_corners', 'away_corners']:
        assert col in df.columns
        assert pd.isna(df.iloc[0][col])


def test_fetch_wc_matches_api_error_returns_empty():
    with patch('src.ingest_api.urllib.request.urlopen', side_effect=Exception("network error")):
        df = fetch_wc_matches('bad-key')
    assert len(df) == 0
    assert isinstance(df, pd.DataFrame)
