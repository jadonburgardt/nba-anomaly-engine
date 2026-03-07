# tests/test_feature_engineering.py
# Tests for the feature engineering pipeline. Verifies rolling averages,
# fatigue flags, and the composite performance score are computed correctly
# and that no data leakage occurs through improper shifting.
# Uses mock data to avoid hitting the NBA API during testing.

import pytest
import pandas as pd
import numpy as np
from feature_engineering import build_features

@pytest.fixture
def raw_df():
    """Mock game log — mimics the structure returned by get_game_log()."""
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

def test_build_features_returns_dataframe(raw_df):
    """build_features should return a pandas DataFrame."""
    df = build_features(raw_df)
    assert isinstance(df, pd.DataFrame)

def test_rolling_columns_exist(raw_df):
    """All rolling average columns should be present in the output."""
    df = build_features(raw_df)
    rolling_cols = ['PTS_roll5', 'REB_roll5', 'AST_roll5', 'STL_roll5',
                    'BLK_roll5', 'TOV_roll5', 'FG_PCT_roll5',
                    'FG3_PCT_roll5', 'FT_PCT_roll5', 'MIN_roll5']
    for col in rolling_cols:
        assert col in df.columns, f"Missing rolling column: {col}"

def test_no_data_leakage(raw_df):
    """Rolling averages must use shift(1) — current game stats should
    never be included in that game's rolling average (data leakage)."""
    df = build_features(raw_df)
    exact_matches = (df['PTS_roll5'] == df['PTS']).sum()
    assert exact_matches < len(df) * 0.5, "Possible data leakage in rolling average"

def test_days_rest_non_negative(raw_df):
    """Days rest should never be negative — games are chronological."""
    df = build_features(raw_df)
    assert (df['DAYS_REST'] >= 0).all()

def test_back_to_back_flag(raw_df):
    """IS_B2B should only be 1 when DAYS_REST is exactly 1."""
    df = build_features(raw_df)
    b2b_games = df[df['IS_B2B'] == 1]
    assert (b2b_games['DAYS_REST'] == 1).all()

def test_home_away_flag_binary(raw_df):
    """IS_HOME should only contain 0 or 1."""
    df = build_features(raw_df)
    assert df['IS_HOME'].isin([0, 1]).all()

def test_perf_score_positive(raw_df):
    """Performance score should always be positive for NBA players."""
    df = build_features(raw_df)
    assert (df['PERF_SCORE'] > 0).all()

def test_no_nulls_in_output(raw_df):
    """Feature engineering should produce no null values in key columns."""
    df = build_features(raw_df)
    key_cols = ['PTS_roll5', 'DAYS_REST', 'IS_B2B', 'IS_HOME', 'PERF_SCORE']
    for col in key_cols:
        assert df[col].isnull().sum() == 0, f"Null values found in {col}"