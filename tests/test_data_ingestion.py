# tests/test_data_ingestion.py
# Tests for the data ingestion layer. Verifies that player lookup,
# API calls, and dataframe structure behave correctly under both
# normal and edge case conditions.

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

def test_get_game_log_returns_dataframe():
    """get_game_log should return a pandas DataFrame."""
    df = get_game_log("LeBron James")
    assert isinstance(df, pd.DataFrame)

def test_get_game_log_expected_columns():
    """Game log must contain all required columns for downstream processing."""
    df = get_game_log("LeBron James")
    required_cols = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS',
                     'REB', 'AST', 'STL', 'BLK', 'TOV',
                     'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLAYER_NAME']
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

def test_get_game_log_not_empty():
    """A current season game log for an active player should not be empty."""
    df = get_game_log("LeBron James")
    assert len(df) > 0

def test_get_game_log_sorted_by_date():
    """Games should be sorted chronologically for rolling feature correctness."""
    df = get_game_log("LeBron James")
    assert df['GAME_DATE'].is_monotonic_increasing

def test_get_game_log_no_null_dates():
    """GAME_DATE should never be null — it's a required index for time series."""
    df = get_game_log("LeBron James")
    assert df['GAME_DATE'].isnull().sum() == 0

def test_get_game_log_invalid_player():
    """Passing an invalid player name should raise a ValueError."""
    with pytest.raises(ValueError):
        get_game_log("Fake Player XYZ")

def test_get_game_log_stats_in_valid_range():
    """Basic sanity check — stats should be within realistic NBA ranges."""
    df = get_game_log("LeBron James")
    assert df['PTS'].between(0, 100).all(), "Points out of realistic range"
    assert df['REB'].between(0, 50).all(), "Rebounds out of realistic range"
    assert df['FG_PCT'].between(0, 1).all(), "FG% should be between 0 and 1"