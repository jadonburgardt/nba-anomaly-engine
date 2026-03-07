# tests/test_anomaly_detection.py
# Tests for the anomaly detection pipeline. Verifies that both detection
# methods produce valid output, that anomaly flags are binary, and that
# the combined flag correctly represents the union of both methods.
# Uses mock data to avoid hitting the real NBA API during testing.

import pytest
import pandas as pd
from unittest.mock import patch
from feature_engineering import build_features
from anomaly_detection import (
    detect_zscore_anomalies,
    detect_isolation_anomalies,
    run_anomaly_detection
)

@pytest.fixture
def mock_raw_df():
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

@pytest.fixture
def feature_df(mock_raw_df):
    return build_features(mock_raw_df)

def test_zscore_adds_required_columns(feature_df):
    df = detect_zscore_anomalies(feature_df)
    assert 'PERF_ZSCORE' in df.columns
    assert 'ZSCORE_ANOMALY' in df.columns
    assert 'ANOMALY_DIRECTION' in df.columns

def test_zscore_anomaly_is_boolean(feature_df):
    df = detect_zscore_anomalies(feature_df)
    assert df['ZSCORE_ANOMALY'].dtype == bool

def test_anomaly_direction_valid_values(feature_df):
    df = detect_zscore_anomalies(feature_df)
    valid = {'normal', 'overperform', 'underperform'}
    assert set(df['ANOMALY_DIRECTION'].unique()).issubset(valid)

def test_isolation_forest_adds_column(feature_df):
    df = detect_isolation_anomalies(feature_df)
    assert 'ISO_ANOMALY' in df.columns
    assert df['ISO_ANOMALY'].dtype == bool

def test_combined_anomaly_is_union(mock_raw_df):
    with patch('anomaly_detection.get_game_log', return_value=mock_raw_df):
        df = run_anomaly_detection("LeBron James")
        for _, row in df.iterrows():
            if row['ZSCORE_ANOMALY'] or row['ISO_ANOMALY']:
                assert row['IS_ANOMALY']

def test_anomaly_rate_reasonable(mock_raw_df):
    with patch('anomaly_detection.get_game_log', return_value=mock_raw_df):
        df = run_anomaly_detection("LeBron James")
        rate = df['IS_ANOMALY'].sum() / len(df)
        assert 0.05 <= rate <= 0.40, f"Anomaly rate {rate:.2%} is outside reasonable bounds"

def test_run_anomaly_detection_invalid_player():
    with pytest.raises(ValueError):
        run_anomaly_detection("Fake Player XYZ")