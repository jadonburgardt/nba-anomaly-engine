# model_training.py
# Trains an XGBoost model to forecast a player's composite performance
# score for their next game using engineered features. Logs all experiments,
# parameters, and metrics to MLflow for tracking and comparison.

import pandas as pd  # Dataframe utilities for tabular feature/target handling.
import numpy as np  # Numeric helpers (used for fold metric aggregation).
import mlflow  # Experiment tracking (params, metrics, and runs).
import mlflow.xgboost  # MLflow integration for logging XGBoost models.
from xgboost import XGBRegressor  # Gradient-boosted regressor used for training.
from sklearn.model_selection import TimeSeriesSplit  # Time-aware CV split strategy.
from sklearn.metrics import mean_absolute_error, r2_score  # Regression evaluation metrics.
from data_ingestion import get_game_log  # Pull raw player game logs from NBA API layer.
from feature_engineering import build_features  # Create model-ready features/target columns.

FEATURE_COLS = [
    'PTS_roll5', 'REB_roll5', 'AST_roll5', 'STL_roll5', 'BLK_roll5',
    'TOV_roll5', 'FG_PCT_roll5', 'FG3_PCT_roll5', 'FT_PCT_roll5',
    'MIN_roll5', 'DAYS_REST', 'IS_B2B', 'IS_HOME'
]
TARGET = 'PERF_SCORE'

def train(player_name: str, season: str = "2025-26"):
    raw = get_game_log(player_name, season)
    df = build_features(raw)

    X = df[FEATURE_COLS]
    y = df[TARGET]

    # Time series split — never train on future data
    tscv = TimeSeriesSplit(n_splits=5)

    mlflow.set_experiment("nba-performance-forecaster")

    with mlflow.start_run(run_name=player_name):
        mlflow.log_param("player", player_name)
        mlflow.log_param("season", season)
        mlflow.log_param("features", FEATURE_COLS)

        maes, r2s = [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                random_state=42,
                verbosity=0
            )
            model.fit(X_train, y_train)

            preds = model.predict(X_val)
            mae = mean_absolute_error(y_val, preds)
            r2 = r2_score(y_val, preds)
            maes.append(mae)
            r2s.append(r2)
            print(f"  Fold {fold+1} — MAE: {mae:.2f}, R²: {r2:.3f}")

        avg_mae = np.mean(maes)
        avg_r2 = np.mean(r2s)
        mlflow.log_metric("avg_mae", avg_mae)
        mlflow.log_metric("avg_r2", avg_r2)

        # Final model trained on all data
        final_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
            verbosity=0
        )
        final_model.fit(X, y)
        mlflow.xgboost.log_model(final_model, "model")

        print(f"\nAverage MAE: {avg_mae:.2f}")
        print(f"Average R²: {avg_r2:.3f}")

        return final_model, df

if __name__ == "__main__":
    model, df = train("LeBron James")

# OUTPUT:
#     Fetching game log for LeBron James (ID: 2544)...
# 2026/03/04 21:03:04 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
# 2026/03/04 21:03:04 INFO mlflow.store.db.utils: Updating database tables
# 2026/03/04 21:03:06 INFO mlflow.tracking.fluent: Experiment with name 'nba-performance-forecaster' does not exist. Creating a new experiment.
#   Fold 1 — MAE: 10.75, R²: -1.417
#   Fold 2 — MAE: 19.23, R²: -4.498
#   Fold 3 — MAE: 9.20, R²: -1.260
#   Fold 4 — MAE: 6.43, R²: 0.085
#   Fold 5 — MAE: 11.71, R²: -11.132
# 2026/03/04 21:03:08 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.

# Average MAE: 11.46
# Average R²: -3.644

################################################################################################################

# MAE of 11.46 — on average, the model's predictions
# are off by ~11.5 points on the PERF_SCORE scale. That's not great but not surprising given the small dataset.

# MAE of 11.46 — on average, the model's predictions are
# off by ~11.5 points on the PERF_SCORE scale. That's not great but not surprising given the small dataset

# Why?:
# NBA performance is inherently noisy, a player can score 13 one night and 33 the next.
# With only one player's season (~42 samples after windowing), the model doesn't have enough signal.
# The fix is training on multiple players,
# which gives the model generalizable patterns rather than overfitting to one player's variance.