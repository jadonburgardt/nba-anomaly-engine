# dashboard.py
# Streamlit dashboard for the NBA Anomaly Detection Engine.
# Provides an interactive interface to search any player, visualize
# their performance trend over the season, and inspect flagged anomaly
# games with supporting context.

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_ingestion import get_game_log
from feature_engineering import build_features
from anomaly_detection import run_anomaly_detection
from model_training import train, FEATURE_COLS

st.set_page_config(
    page_title="NBA Anomaly Detection Engine",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 NBA Anomaly Detection Engine")
st.markdown("Enter any active NBA player to analyze their performance trend and detect statistical anomalies.")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Player Selection")
    player_name = st.text_input("Player Name", value="LeBron James")
    season = st.selectbox("Season", ["2025-26", "2024-25"], index=0)
    run_button = st.button("Analyze", use_container_width=True)

if run_button and player_name:
    with st.spinner(f"Fetching data for {player_name}..."):
        try:
            # ── Load data ──────────────────────────────────────────────────────
            df = run_anomaly_detection(player_name, season)
            model, model_df = train(player_name, season)

            # ── Next game prediction ───────────────────────────────────────────
            latest = model_df[FEATURE_COLS].iloc[-1].values.reshape(1, -1)
            prediction = float(model.predict(latest)[0])
            rolling_avg = float(model_df['PERF_SCORE'].tail(5).mean())
            delta = prediction - rolling_avg

            # ── Top metrics ───────────────────────────────────────────────────
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Games Analyzed", len(df))
            col2.metric("Anomalies Detected", int(df['IS_ANOMALY'].sum()))
            col3.metric("Predicted Next Game Score", f"{prediction:.1f}")
            col4.metric("vs 5-Game Avg", f"{delta:+.1f}")

            st.divider()

            # ── Performance trend chart ────────────────────────────────────────
            st.subheader("Performance Score — Season Trend")

            normal = df[~df['IS_ANOMALY']]
            anomalies = df[df['IS_ANOMALY']]

            fig = go.Figure()

            # Trend line
            fig.add_trace(go.Scatter(
                x=df['GAME_DATE'], y=df['PERF_SCORE'],
                mode='lines',
                line=dict(color='steelblue', width=2),
                name='Performance Score'
            ))

            # Rolling average line
            fig.add_trace(go.Scatter(
                x=df['GAME_DATE'], y=df['PERF_SCORE'].rolling(5).mean(),
                mode='lines',
                line=dict(color='orange', width=2, dash='dash'),
                name='5-Game Rolling Avg'
            ))

            # Normal games
            fig.add_trace(go.Scatter(
                x=normal['GAME_DATE'], y=normal['PERF_SCORE'],
                mode='markers',
                marker=dict(color='steelblue', size=7),
                name='Normal Game'
            ))

            # Overperform anomalies
            over = anomalies[anomalies['ANOMALY_DIRECTION'] == 'overperform']
            fig.add_trace(go.Scatter(
                x=over['GAME_DATE'], y=over['PERF_SCORE'],
                mode='markers',
                marker=dict(color='green', size=12, symbol='star'),
                name='Overperformance'
            ))

            # Underperform anomalies
            under = anomalies[anomalies['ANOMALY_DIRECTION'] == 'underperform']
            fig.add_trace(go.Scatter(
                x=under['GAME_DATE'], y=under['PERF_SCORE'],
                mode='markers',
                marker=dict(color='red', size=12, symbol='x'),
                name='Underperformance'
            ))

            # Isolation forest only anomalies
            iso_only = anomalies[anomalies['ANOMALY_DIRECTION'] == 'normal']
            fig.add_trace(go.Scatter(
                x=iso_only['GAME_DATE'], y=iso_only['PERF_SCORE'],
                mode='markers',
                marker=dict(color='purple', size=12, symbol='diamond'),
                name='Unusual Stat Pattern'
            ))

            fig.update_layout(
                height=450,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02),
                xaxis_title='Date',
                yaxis_title='Performance Score'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ── Anomaly table ──────────────────────────────────────────────────
            st.subheader("Flagged Games")
            flagged = df[df['IS_ANOMALY']][[
                'GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST',
                'PERF_SCORE', 'PERF_ZSCORE', 'ANOMALY_DIRECTION',
                'ZSCORE_ANOMALY', 'ISO_ANOMALY'
            ]].copy()
            flagged['GAME_DATE'] = flagged['GAME_DATE'].astype(str)
            flagged['PERF_ZSCORE'] = flagged['PERF_ZSCORE'].round(3)
            st.dataframe(flagged, use_container_width=True)

            st.divider()

            # ── Raw game log ───────────────────────────────────────────────────
            with st.expander("View Full Game Log"):
                raw = get_game_log(player_name, season)
                st.dataframe(raw, use_container_width=True)

        except ValueError as e:
            st.error(f"Player not found: {e}")
        except Exception as e:
            st.error(f"Something went wrong: {e}")