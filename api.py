# api.py
# FastAPI server that exposes the NBA anomaly detection engine via HTTP endpoints.
# Provides routes for retrieving game logs, next-game performance predictions,
# and anomaly reports for any player in the current season.

## API Endpoints:
# GET /player/{player_name}/gamelog — returns the raw game log for any player

# GET /player/{player_name}/predict — runs the model and returns the predicted next-game
# performance score alongside their recent rolling average

# GET /player/{player_name}/anomalies — returns all flagged anomaly games for a player with direction
# (overperform/underperform) and which method caught it

from fastapi import FastAPI, HTTPException
from model_training import train, FEATURE_COLS
from anomaly_detection import run_anomaly_detection
from data_ingestion import get_game_log
from feature_engineering import build_features
import pandas as pd
import numpy as np

app = FastAPI(
    title="NBA Anomaly Detection Engine",
    description="Forecasts player performance and detects statistical anomalies.",
    version="1.0.0"
)

def serialize(df: pd.DataFrame) -> list:
    return df.where(pd.notnull(df), None).to_dict(orient="records")

@app.get("/player/{player_name}/gamelog")
def gamelog(player_name: str, season: str = "2025-26"):
    try:
        df = get_game_log(player_name, season)
        df['GAME_DATE'] = df['GAME_DATE'].astype(str)
        return {"player": player_name, "season": season, "games": serialize(df)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/player/{player_name}/predict")
def predict(player_name: str, season: str = "2025-26"):
    try:
        model, df = train(player_name, season)
        latest = df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
        prediction = float(model.predict(latest)[0])
        rolling_avg = float(df['PERF_SCORE'].tail(5).mean())

        return {
            "player": player_name,
            "predicted_next_game_perf_score": round(prediction, 2),
            "rolling_5game_avg": round(rolling_avg, 2),
            "delta": round(prediction - rolling_avg, 2)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/player/{player_name}/anomalies")
def anomalies(player_name: str, season: str = "2025-26"):
    try:
        df = run_anomaly_detection(player_name, season)
        flagged = df[df['IS_ANOMALY']][[
            'GAME_DATE', 'PTS', 'REB', 'AST', 'PERF_SCORE',
            'PERF_ZSCORE', 'ANOMALY_DIRECTION', 'ZSCORE_ANOMALY', 'ISO_ANOMALY'
        ]].copy()
        flagged['GAME_DATE'] = flagged['GAME_DATE'].astype(str)
        flagged['PERF_ZSCORE'] = flagged['PERF_ZSCORE'].round(3)

        return {
            "player": player_name,
            "total_games": len(df),
            "anomalies_detected": len(flagged),
            "anomalies": serialize(flagged)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))