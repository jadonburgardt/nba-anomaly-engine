# tests/test_data_ingestion.py
# Tests for the data ingestion layer. Verifies that player lookup,
# API calls, and dataframe structure behave correctly under both
# normal and edge case conditions.
# Uses mocking to avoid hitting the real NBA API during testing.

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data_ingestion import get_player_id, get_game_log

# ── get_player_id ──────────────────────────────────────────────────────────────

def test_get_player_id_valid_player():
    """LeBron James is a known player and should return a valid integer ID."""
    player_id = get_player_id("LeBron James")
    assert isinstance(player_id, int)
    assert player_id > 0

def test_get_player_id_case_insensitive():
    """Player lookup should work regardless of casing."""
    id_lower = get_player_id("lebron james")
    id_upper = get_player_id("LEBRON JAMES")
    id_mixed = get_player_id("LeBron James")
    assert id_lower == id_upper == id_mixed

def test_get_player_id_invalid_player():
    """An unknown player name should raise a ValueError."""
    with pytest.raises(ValueError, match="not found"):
        get_player_id("Fake Player XYZ")

# ── get_game_log ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_game_log():
    """Mock DataFrame mimicking the structure returned by the NBA API."""
    data = {
        'GAME_DATE': pd.date_range(start='2025-10-01', periods=15, freq='2D'),
        'MATCHUP': [
            'LAL vs. GSW', 'LAL @ BOS', 'LAL vs. MIA', 'LAL @ CHI', 'LAL vs. HOU',
            'LAL @ IND', 'LAL vs. NOP', 'LAL @ DAL', 'LAL vs. OKC', 'LAL @ DEN',
            'LAL vs. PHX', 'LAL @ MEM', 'LAL vs. SAC', 'LAL @ POR', 'LAL vs. MIN'
        ],
        'WL':  ['W','L','W','W','L','W','W','L','W','W','L','W','W','L','W'],
        'MIN': [36, 38, 34, 37, 35, 36, 38, 34, 37, 35, 36, 38, 34, 37, 35],
        'PTS': [28, 15, 22, 31, 18, 25, 30, 12, 27, 20, 24, 33, 19, 26, 21],
        'REB': [8,  4,  6,  9,  5,  7,  8,  3,  6,  5,  7,  9,  4,  6,  5],
        'AST': [7,  5,  8,  6,  4,  9,  7,  5,  8,  6,  7,  8,  5,  7,  6],
        'STL': [1,  0,  2,  1,  0,  1,  2,  0,  1,  1,  0,  2,  1,  0,  1],
        'BLK': [1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1,  0,  1],
        'TOV': [2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1,  2,  3,  1],
        'FG_PCT':  [0.55, 0.40, 0.50, 0.60, 0.42, 0.53, 0.58, 0.38, 0.52, 0.47,
                    0.51, 0.62, 0.44, 0.55, 0.48],
        'FG3_PCT': [0.40, 0.25, 0.33, 0.50, 0.20, 0.38, 0.44, 0.20, 0.36, 0.30,
                    0.35, 0.50, 0.28, 0.40, 0.32],
        'FT_PCT':  [0.80, 0.70, 0.85, 0.90, 0.75, 0.82, 0.88, 0.65, 0.83, 0.78,
                    0.81, 0.92, 0.72, 0.85, 0.79],
        'PLAYER_NAME': ['LeBron James'] * 15
    }
    return pd.DataFrame(data)

def test_get_game_log_returns_dataframe(mock_game_log):
    """get_game_log should return a pandas DataFrame."""
    with patch('data_ingestion.playergamelog.PlayerGameLog') as mock_api:
        mock_api.return_value.get_data_frames.return_value = [mock_game_log]
        df = get_game_log("LeBron James")
        assert isinstance(df, pd.DataFrame)

def test_get_game_log_expected_columns(mock_game_log):
    """Game log must contain all required columns for downstream processing."""
    with patch('data_ingestion.playergamelog.PlayerGameLog') as mock_api:
        mock_api.return_value.get_data_frames.return_value = [mock_game_log]
        df = get_game_log("LeBron James")
        required_cols = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS',
                         'REB', 'AST', 'STL', 'BLK', 'TOV',
                         'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLAYER_NAME']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

def test_get_game_log_not_empty(mock_game_log):
    """A current season game log for an active player should not be empty."""
    with patch('data_ingestion.playergamelog.PlayerGameLog') as mock_api:
        mock_api.return_value.get_data_frames.return_value = [mock_game_log]
        df = get_game_log("LeBron James")
        assert len(df) > 0

def test_get_game_log_sorted_by_date(mock_game_log):
    """Games should be sorted chronologically for rolling feature correctness."""
    with patch('data_ingestion.playergamelog.PlayerGameLog') as mock_api:
        mock_api.return_value.get_data_frames.return_value = [mock_game_log]
        df = get_game_log("LeBron James")
        assert df['GAME_DATE'].is_monotonic_increasing

def test_get_game_log_no_null_dates(mock_game_log):
    """GAME_DATE should never be null — it's a required index for time series."""
    with patch('data_ingestion.playergamelog.PlayerGameLog') as mock_api:
        mock_api.return_value.get_data_frames.return_value = [mock_game_log]
        df = get_game_log("LeBron James")
        assert df['GAME_DATE'].isnull().sum() == 0

def test_get_game_log_invalid_player():
    """Passing an invalid player name should raise a ValueError."""
    with pytest.raises(ValueError):
        get_game_log("Fake Player XYZ")

def test_get_game_log_stats_in_valid_range(mock_game_log):
    """Basic sanity check — stats should be within realistic NBA ranges."""
    with patch('data_ingestion.playergamelog.PlayerGameLog') as mock_api:
        mock_api.return_value.get_data_frames.return_value = [mock_game_log]
        df = get_game_log("LeBron James")
        assert df['PTS'].between(0, 100).all()
        assert df['REB'].between(0, 50).all()
        assert df['FG_PCT'].between(0, 1).all()