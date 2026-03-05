# data_ingestion.py
# Handles all data retrieval from the NBA Stats API.
# Provides utilities to look up player IDs by name and fetch
# full game logs for a given player and season. This is the
# foundation layer that all other modules depend on for raw data.

import pandas as pd  # Dataframe formatting and datetime parsing for game logs.
from nba_api.stats.endpoints import playergamelog  # Endpoint client for per-player game logs.
from nba_api.stats.static import players  # Static NBA player directory for name-to-ID lookup.
import time  # Time utilities for optional request throttling/retry handling.

def get_player_id(player_name: str) -> int:
    all_players = players.get_players()
    match = [p for p in all_players if p['full_name'].lower() == player_name.lower()]
    if not match:
        raise ValueError(f"Player '{player_name}' not found.")
    return match[0]['id']

def get_game_log(player_name: str, season: str = "2025-26") -> pd.DataFrame:
    player_id = get_player_id(player_name)
    print(f"Fetching game log for {player_name} (ID: {player_id})...")

    log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = log.get_data_frames()[0]

    # Columns that we care about
    cols = ['GAME_DATE', 'MATCHUP', 'WL', 'MIN', 'PTS', 'REB', 'AST',
            'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    df = df[cols]
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['PLAYER_NAME'] = player_name
    df = df.sort_values('GAME_DATE').reset_index(drop=True)

    return df

if __name__ == "__main__":
    # Simple player test
    df = get_game_log("LeBron James")
    print(df.tail(10))
    print(f"\nTotal games: {len(df)}")

# OUTPUT:
#     Fetching game log for LeBron James (ID: 2544)...
#     GAME_DATE      MATCHUP WL  MIN  PTS  REB  AST  STL  BLK  TOV  FG_PCT  FG3_PCT  FT_PCT   PLAYER_NAME
# 33 2026-02-07  LAL vs. GSW  W   35   20    7   10    1    1    7   0.353    0.286   0.857  LeBron James
# 34 2026-02-09  LAL vs. OKC  L   36   22    6   10    1    0    3   0.529    0.000   0.667  LeBron James
# 35 2026-02-12  LAL vs. DAL  W   35   28   10   12    0    1    4   0.500    0.286   0.857  LeBron James
# 36 2026-02-20  LAL vs. LAC  W   33   13    3   11    1    0    3   0.385    0.333   1.000  LeBron James
# 37 2026-02-22  LAL vs. BOS  L   34   20    4    5    2    1    1   0.429    0.200   1.000  LeBron James
# 38 2026-02-24  LAL vs. ORL  L   32   21    6    4    0    2    5   0.615    0.333   0.750  LeBron James
# 39 2026-02-26    LAL @ PHX  L   35   15    6    5    0    0    1   0.438    0.000   0.500  LeBron James
# 40 2026-02-28    LAL @ GSW  W   28   22    7    9    1    0    4   0.538    0.667   0.800  LeBron James
# 41 2026-03-01  LAL vs. SAC  W   27   24    1    5    2    1    3   0.533    0.750   0.833  LeBron James
# 42 2026-03-03  LAL vs. NOP  W   33   21    7    7    0    2    5   0.667    0.200   0.500  LeBron James

# Total games: 43

#######################################################################################

# The output shows the last 10 games of LeBron James for the 2025-26 season,
# including key stats like points, rebounds, assists, and shooting percentages.
# This raw data will be used as input for the feature engineering step.