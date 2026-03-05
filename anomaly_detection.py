# anomaly_detection.py
# Detects when a player's actual performance deviates significantly from
# their expected baseline using two methods: Z-score analysis on the
# composite performance score, and Isolation Forest on raw stat features.
# Flags anomalies with a severity label for use in the API and dashboard.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from data_ingestion import get_game_log
from feature_engineering import build_features

STAT_COLS = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'MIN']

def detect_zscore_anomalies(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    mean = df['PERF_SCORE'].mean()
    std = df['PERF_SCORE'].std()

    df['PERF_ZSCORE'] = (df['PERF_SCORE'] - mean) / std
    df['ZSCORE_ANOMALY'] = df['PERF_ZSCORE'].abs() > threshold
    df['ANOMALY_DIRECTION'] = df['PERF_ZSCORE'].apply(
        lambda z: 'overperform' if z > threshold else ('underperform' if z < -threshold else 'normal')
    )
    return df

def detect_isolation_anomalies(df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
    df = df.copy()
    iso = IsolationForest(contamination=contamination, random_state=42)
    df['ISO_ANOMALY'] = iso.fit_predict(df[STAT_COLS])

    # IsolationForest returns -1 for anomalies, 1 for normal
    df['ISO_ANOMALY'] = df['ISO_ANOMALY'].apply(lambda x: True if x == -1 else False)
    return df

def run_anomaly_detection(player_name: str, season: str = "2025-26") -> pd.DataFrame:
    raw = get_game_log(player_name, season)
    df = build_features(raw)

    df = detect_zscore_anomalies(df)
    df = detect_isolation_anomalies(df)

    # Combined flag — flagged by either method
    df['IS_ANOMALY'] = df['ZSCORE_ANOMALY'] | df['ISO_ANOMALY']

    return df

if __name__ == "__main__":
    df = run_anomaly_detection("LeBron James")

    anomalies = df[df['IS_ANOMALY']][['GAME_DATE', 'PTS', 'REB', 'AST', 'PERF_SCORE', 'PERF_ZSCORE', 'ANOMALY_DIRECTION', 'ZSCORE_ANOMALY', 'ISO_ANOMALY']]
    print(f"Total anomalies detected: {len(anomalies)}")
    print(f"Out of {len(df)} games\n")
    print(anomalies.to_string(index=False))

# Output:

# Fetching game log for LeBron James (ID: 2544)...
# Total anomalies detected: 9
# Out of 42 games

#  GAME_DATE  PTS  REB  AST  PERF_SCORE  PERF_ZSCORE ANOMALY_DIRECTION  ZSCORE_ANOMALY  ISO_ANOMALY
# 2025-12-01   10    0    3        10.0    -2.704491      underperform            True         True
# 2025-12-04    8    6   11        31.7    -0.579600            normal           False         True
# 2025-12-10   19   15    8        50.5     1.261319            normal           False         True
# 2025-12-14   26    3    4        31.6    -0.589393            normal           False         True
# 2026-01-06   30    8    8        56.1     1.809678       overperform            True        False
# 2026-01-09   26    9   10        53.3     1.535498       overperform            True        False
# 2026-01-13   31    9   10        54.8     1.682380       overperform            True        False
# 2026-01-28   11    3    5        15.1    -2.205093      underperform            True         True
# 2026-02-12   28   10   12        54.0     1.604043       overperform            True        False

# z-score anomalies:
# Dec 1 and Jan 28 are clear underperformances —
# 10 and 8 points respectively, well below his baseline.
# These are the kinds of games that could precede injury or fatigue

# Jan 6 through Feb 12 cluster of overperformances —
# LeBron was on a legitimate hot streak, and the detector caught it


# Isolation Forest anomalies:
# Dec 4: 8 pts but 11 assists — unusual distribution even though PERF_SCORE looks normal
# Dec 10: 15 rebounds is a statistical outlier for LeBron regardless of points
# Dec 14: 26 pts but only 3 rebounds and 4 assists — atypical usage pattern

# interesting insight:
# the two methods disagree on several games, which is actually meaningful.
# Z-score catches bad or great overall games, Isolation Forest catches weird games
# where the stat distribution is unusual even if the total looks fine. Using both together gives you richer signal.