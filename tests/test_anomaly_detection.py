# tests/test_anomaly_detection.py
# Tests for the anomaly detection pipeline. Verifies that both detection
# methods produce valid output, that anomaly flags are binary, and that
# the combined flag correctly represents the union of both methods.

import pytest
import pandas as pd
from data_ingestion import get_game_log
from feature_engineering import build_features
from anomaly_detection import (
    detect_zscore_anomalies,
    detect_isolation_anomalies,
    run_anomaly_detection
)

@pytest.fixture
def feature_df():
    """Shared fixture — build features once and reuse."""
    raw = get_game_log("LeBron James")
    return build_features(raw)

def test_zscore_adds_required_columns(feature_df):
    """Z-score detection should add PERF_ZSCORE, ZSCORE_ANOMALY,
    and ANOMALY_DIRECTION columns."""
    df = detect_zscore_anomalies(feature_df)
    assert 'PERF_ZSCORE' in df.columns
    assert 'ZSCORE_ANOMALY' in df.columns
    assert 'ANOMALY_DIRECTION' in df.columns

def test_zscore_anomaly_is_boolean(feature_df):
    """ZSCORE_ANOMALY should be a boolean column."""
    df = detect_zscore_anomalies(feature_df)
    assert df['ZSCORE_ANOMALY'].dtype == bool

def test_anomaly_direction_valid_values(feature_df):
    """ANOMALY_DIRECTION should only contain the three expected labels."""
    df = detect_zscore_anomalies(feature_df)
    valid = {'normal', 'overperform', 'underperform'}
    assert set(df['ANOMALY_DIRECTION'].unique()).issubset(valid)

def test_isolation_forest_adds_column(feature_df):
    """Isolation Forest should add an ISO_ANOMALY boolean column."""
    df = detect_isolation_anomalies(feature_df)
    assert 'ISO_ANOMALY' in df.columns
    assert df['ISO_ANOMALY'].dtype == bool

def test_combined_anomaly_is_union(feature_df):
    """IS_ANOMALY should be True whenever either method flags a game."""
    df = run_anomaly_detection("LeBron James")
    for _, row in df.iterrows():
        if row['ZSCORE_ANOMALY'] or row['ISO_ANOMALY']:
            assert row['IS_ANOMALY'], \
                f"Game on {row['GAME_DATE']} should be flagged but isn't"

def test_anomaly_rate_reasonable():
    """Anomaly rate should be between 5% and 40% — too few or too many
    suggests the thresholds are misconfigured."""
    df = run_anomaly_detection("LeBron James")
    rate = df['IS_ANOMALY'].sum() / len(df)
    assert 0.05 <= rate <= 0.40, f"Anomaly rate {rate:.2%} is outside reasonable bounds"

def test_run_anomaly_detection_invalid_player():
    """Passing an invalid player name should raise a ValueError."""
    with pytest.raises(ValueError):
        run_anomaly_detection("Fake Player XYZ")