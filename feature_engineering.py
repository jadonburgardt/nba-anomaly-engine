# feature_engineering.py
# Transforms raw game log data into model ready features by computing
# rolling averages, rest/fatigue indicators, home/away flags, and a
# composite performance score used as the model's prediction target.

import pandas as pd  # Dataframe operations for rolling features and transformations.
from data_ingestion import get_game_log  # Load raw player game logs for local testing.

def build_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    df = df.copy()

    # Rolling averages over last N games
    rolling_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'MIN']
    for col in rolling_cols:
        df[f'{col}_roll{window}'] = (
            df[col].shift(1).rolling(window=window, min_periods=1).mean()
        )

    # Days rest between games
    df['DAYS_REST'] = df['GAME_DATE'].diff().dt.days.fillna(1)

    # Back-to-back flag
    df['IS_B2B'] = (df['DAYS_REST'] == 1).astype(int)

    # Home vs away
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if 'vs.' in x else 0)

    # Win/loss to binary
    df['WIN'] = (df['WL'] == 'W').astype(int)

    # Performance score - weighted composite target
    df['PERF_SCORE'] = (
        df['PTS'] * 1.0 +
        df['REB'] * 1.2 +
        df['AST'] * 1.5 +
        df['STL'] * 2.0 +
        df['BLK'] * 2.0 -
        df['TOV'] * 1.5
    )

    # Drop rows where rolling features are NaN (first few games)
    df = df.dropna().reset_index(drop=True)

    return df

if __name__ == "__main__":
    # Test the feature engineering with a sample player
    raw = get_game_log("LeBron James")
    features = build_features(raw)
    print(features[['GAME_DATE', 'PTS', 'PTS_roll5', 'DAYS_REST', 'IS_B2B', 'IS_HOME', 'PERF_SCORE']].tail(10))
    print(f"\nFeature columns: {list(features.columns)}")


# OUTPUT:
#     Fetching game log for LeBron James (ID: 2544)...
#     GAME_DATE  PTS  PTS_roll5  DAYS_REST  IS_B2B  IS_HOME  PERF_SCORE
# 32 2026-02-07   20       19.0        2.0       0        1        36.9
# 33 2026-02-09   22       20.8        2.0       0        1        41.7
# 34 2026-02-12   28       21.2        3.0       0        1        54.0
# 35 2026-02-20   13       22.4        8.0       0        1        30.6
# 36 2026-02-22   20       20.0        2.0       0        1        36.8
# 37 2026-02-24   21       20.6        2.0       0        1        30.7
# 38 2026-02-26   15       20.8        2.0       0        0        28.2
# 39 2026-02-28   22       19.4        2.0       0        0        39.9
# 40 2026-03-01   24       18.2        1.0       1        1        34.2
# 41 2026-03-03   21       20.4        2.0       0        1        36.4

# Feature columns: ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'PLAYER_NAME', 'PTS_roll5', 'REB_roll5', 'AST_roll5', 'STL_roll5', 'BLK_roll5', 'TOV_roll5', 'FG_PCT_roll5', 'FG3_PCT_roll5', 'FT_PCT_roll5', 'MIN_roll5', 'DAYS_REST', 'IS_B2B', 'IS_HOME', 'WIN', 'PERF_SCORE']

##################################################################################################################

#PTS_ROLL5: LeBron's average points over his last 5 games going into that game.
# DAYS_REST: Number of days since his last game, indicating rest or fatigue.
# IS_B2B: Whether this game is part of a back-to-back (1 if yes, 0 if no).
# IS_HOME: Whether the game is at home (1) or away (0).
# PERF_SCORE: A composite performance score calculated from various stats, used as the model's target variable.