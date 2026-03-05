**An NBA Player Performance Forecasting & Anomaly Detection API**

**To Start Virtual Environment:**

- run ".\venv\Scripts\activate" within the nba-anomaly-engine folder to load the venv and this projects packages

**Packages:**

- nba_api — provides access to the official NBA Stats website to pull player and game data
- pandas — handles all tabular data manipulation.
- numpy — provides fast math operations and array handling that pandas and ML libraries build on
- scikit-learn — standard toolkit for ML utilities like preprocessing, train/test splitting, and evaluation metrics
- xgboost — the model we'll train to forecast player performance.
- fastapi — builds the API server that exposes our model via HTTP endpoints
- uvicorn — runs the FastAPI server
- mlflow — tracks model experiments, parameters, and metrics to compare runs and version models
- streamlit — turns Python scripts into interactive web dashboards
- requests — makes HTTP calls from Python, used for hitting external APIs or services

**For individual testing:**

python (data_ingestion.py) , (feature_engineering.py), (model_training.py) , (.\anomaly_detection.py)

**To run FASTAPI server and expose HTTP endpoints:**

uvicorn api:app --reload
