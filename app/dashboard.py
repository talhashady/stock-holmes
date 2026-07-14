import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objects as go

# Import database and inference functions
from src.serving.db_utils import get_predictions_history, get_all_candles
from src.ingestion.fetcher import fetch_and_cache
from src.models.predict import predict_latest
from src.models.train import train_pipeline

# ----------------- PAGE CONFIG & DESIGN -----------------
st.set_page_config(
    page_title="Stock Holmes | XAUUSD Predictor",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Mode Theme Injection
st.markdown("""
<style>
    /* Dark glassmorphic styling */
    .reportview-container {
        background: #0f172a;
    }
    div.stButton > button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #2563eb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(4px);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.02);
        border-color: rgba(59, 130, 246, 0.3);
    }
    .glow-up {
        color: #10b981;
        text-shadow: 0 0 10px rgba(16, 185, 129, 0.3);
        font-weight: bold;
    }
    .glow-down {
        color: #ef4444;
        text-shadow: 0 0 10px rgba(239, 68, 68, 0.3);
        font-weight: bold;
    }
    .glow-flat {
        color: #94a3b8;
        text-shadow: 0 0 10px rgba(148, 163, 184, 0.3);
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.title("🕵️‍♂️ Stock Holmes: XAUUSD 5m-Ahead Predictor")
st.markdown("A Time-Series Intelligence System for Short-Horizon Gold Spot Price Direction.")

# Sidebar setup
st.sidebar.header("⚙️ Pipeline Management")

# Helper to load .env manually if present
def load_env_dashboard():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env_dashboard()
api_key_default = os.getenv("TWELVE_DATA_API_KEY", "")
api_key = st.sidebar.text_input("Twelve Data API Key", value=api_key_default, type="password")

if st.sidebar.button("🔄 Ingest Latest Data"):
    with st.spinner("Fetching live candles from Twelve Data..."):
        try:
            inserted = fetch_and_cache(api_key=api_key)
            st.sidebar.success(f"Fetched and cached {inserted} new candles!")
        except Exception as e:
            st.sidebar.error(f"Ingestion error: {e}")

if st.sidebar.button("🤖 Retrain LightGBM"):
    with st.spinner("Rebuilding features and training walk-forward pipeline..."):
        try:
            metrics = train_pipeline()
            st.sidebar.success(f"Trained! Test Acc: {metrics.get('accuracy', 0.0):.1%}")
        except Exception as e:
            st.sidebar.error(f"Training error: {e}")

if st.sidebar.button("🎯 Run Inference (Predict)"):
    with st.spinner("Generating fresh 5-minute predictions..."):
        try:
            res = predict_latest()
            if res:
                st.sidebar.success("Latest prediction saved!")
            else:
                st.sidebar.warning("Inference executed but returned no prediction (likely missing history).")
        except Exception as e:
            st.sidebar.error(f"Inference error: {e}")

# ----------------- DATA LOADING -----------------
@st.cache_data(ttl=10)
def load_dashboard_data():
    preds = get_predictions_history()
    candles = get_all_candles()
    
    # Load training metrics
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "models", "metrics.json")
    train_metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            train_metrics = json.load(f)
            
    return preds, candles, train_metrics

preds, candles, train_metrics = load_dashboard_data()

# Check if we have data to display
if candles.empty:
    st.warning("⚠️ Local database is empty. Please use the sidebar to 'Ingest Latest Data' to build the initial database.")
    st.stop()

# ----------------- SECTION 1: LIVE PREDICTION TERMINAL -----------------
st.markdown("### 🖥️ Live Prediction Terminal")
col1, col2, col3 = st.columns(3)

# Latest candle
latest_candle = candles.iloc[-1]
col1.markdown("""
<div class="metric-card">
    <h4>💵 Current Spot Price</h4>
    <h2 style='font-size: 36px; margin: 5px 0;'>${:,.2f}</h2>
    <p style='color: #64748b; font-size: 13px;'>Last Candle: {} UTC</p>
</div>
""".format(latest_candle["close"], latest_candle["timestamp"]), unsafe_allow_html=True)

# Latest prediction
if not preds.empty:
    latest_pred = preds.iloc[0]
    pred_dir = latest_pred["predicted_direction"]
    conf = latest_pred["confidence"]
    
    if pred_dir == 1:
        glow_class = "glow-up"
        dir_text = "📈 UP"
    elif pred_dir == -1:
        glow_class = "glow-down"
        dir_text = "📉 DOWN"
    else:
        glow_class = "glow-flat"
        dir_text = "➡️ FLAT"
        
    col2.markdown("""
    <div class="metric-card">
        <h4>🕵️‍♂️ Predicted Direction (5m ahead)</h4>
        <h2 class="{}" style='font-size: 36px; margin: 5px 0;'>{}</h2>
        <p style='color: #64748b; font-size: 13px;'>Model Confidence: {:.1%}</p>
    </div>
    """.format(glow_class, dir_text, conf), unsafe_allow_html=True)
    
    # Probs distribution
    col3.markdown("""
    <div class="metric-card">
        <h4>📊 Signal Confidence Weights</h4>
        <p style='margin: 4px 0;'><b>DOWN</b>: {:.1%}</p>
        <p style='margin: 4px 0;'><b>FLAT</b>: {:.1%}</p>
        <p style='margin: 4px 0;'><b>UP</b>: {:.1%}</p>
    </div>
    """.format(latest_pred["prob_down"], latest_pred["prob_flat"], latest_pred["prob_up"]), unsafe_allow_html=True)
else:
    col2.info("No predictions logged yet. Run Inference from sidebar.")
    col3.info("No probability distributions logged.")

# ----------------- SECTION 2: CHARTS & MOVEMENT -----------------
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📉 Price Action & Predictions", 
    "📈 Walk-Forward Backtesting (Skill vs Luck)", 
    "🎯 Predicted vs. Actual", 
    "📋 Predictions Log"
])

with tab1:
    st.markdown("#### XAUUSD Recent Price Activity")
    # Take last 100 candles
    recent_candles = candles.tail(100).copy()
    recent_candles["timestamp"] = pd.to_datetime(recent_candles["timestamp"])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent_candles["timestamp"], recent_candles["close"], label="Spot Gold Price", color="#fbbf24")
    ax.set_facecolor("#0f172a")
    fig.patch.set_facecolor("#0f172a")
    ax.spines['bottom'].set_color('#475569')
    ax.spines['top'].set_color('#475569')
    ax.spines['left'].set_color('#475569')
    ax.spines['right'].set_color('#475569')
    ax.tick_params(colors='#94a3b8')
    ax.grid(True, color='#1e293b', linestyle='--')
    ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    st.pyplot(fig)

with tab2:
    st.markdown("#### Performance Metrics")
    
    if train_metrics:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("LightGBM Accuracy", f"{train_metrics['accuracy']:.1%}", 
                      delta=f"{train_metrics['accuracy'] - train_metrics['naive_flat_accuracy']:.1%} vs Flat Baseline")
        m_col2.metric("Naive Directional Sign Accuracy", f"{train_metrics['naive_sign_accuracy']:.1%}")
        m_col3.metric("High-Confidence Accuracy (>45%)", f"{train_metrics['high_confidence_accuracy']:.1%}",
                      delta=f"{train_metrics['high_confidence_count']} samples")
        
        # Cumulative P&L chart
        st.markdown("#### Cumulative Returns (Walk-Forward Test Window)")
        backtest_data = train_metrics.get("backtest", {})
        if backtest_data:
            bt_df = pd.DataFrame({
                "Timestamp": pd.to_datetime(backtest_data["timestamps"]),
                "Model Strategy": backtest_data["model_cumulative_returns"],
                "Naive Sign Baseline": backtest_data["baseline_cumulative_returns"]
            })
            
            fig_bt, ax_bt = plt.subplots(figsize=(10, 4.5))
            ax_bt.plot(bt_df["Timestamp"], bt_df["Model Strategy"], label="Stock Holmes Strategy", color="#10b981", linewidth=2)
            ax_bt.plot(bt_df["Timestamp"], bt_df["Naive Sign Baseline"], label="Naive Sign Carry-Forward", color="#ef4444", linestyle="--")
            ax_bt.set_facecolor("#0f172a")
            fig_bt.patch.set_facecolor("#0f172a")
            ax_bt.spines['bottom'].set_color('#475569')
            ax_bt.spines['top'].set_color('#475569')
            ax_bt.spines['left'].set_color('#475569')
            ax_bt.spines['right'].set_color('#475569')
            ax_bt.tick_params(colors='#94a3b8')
            ax_bt.grid(True, color='#1e293b', linestyle='--')
            ax_bt.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
            st.pyplot(fig_bt)
        else:
            st.warning("No walk-forward backtesting metrics found. Please Retrain model to generate backtest logs.")
    else:
        st.info("No training metrics registered yet. Please click 'Retrain LightGBM' in the sidebar to build the model and log performance.")

with tab3:
    st.markdown("#### Predicted vs. Actual Price Overlay")
    
    if preds.empty or "status" not in preds.columns:
        st.info("No predictions logged yet.")
    else:
        # Filter to RESOLVED predictions
        resolved_preds = preds[preds["status"] == "RESOLVED"].copy()
        
        if len(resolved_preds) < 2:
            st.info("ℹ️ Not enough resolved predictions yet to plot (need at least 2). Run ingestion and inference to resolve past predictions.")
        else:
            # Timeframe filter controls
            timeframe = st.selectbox("Time Range Filter", ["Last 1 Hour", "Last 6 Hours", "Last 24 Hours", "All Available"], index=3)
            
            # Sort chronologically for charting
            resolved_preds["target_dt"] = pd.to_datetime(resolved_preds["target_timestamp"])
            resolved_preds = resolved_preds.sort_values(by="target_dt")
            
            # Apply time filter
            if timeframe == "Last 1 Hour":
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=1)
                resolved_preds = resolved_preds[resolved_preds["target_dt"] >= cutoff]
            elif timeframe == "Last 6 Hours":
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=6)
                resolved_preds = resolved_preds[resolved_preds["target_dt"] >= cutoff]
            elif timeframe == "Last 24 Hours":
                cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
                resolved_preds = resolved_preds[resolved_preds["target_dt"] >= cutoff]
                
            if len(resolved_preds) < 2:
                st.warning("No resolved predictions in selected time range.")
            else:
                # Summary stats above the chart
                last_20 = resolved_preds.tail(20)
                correct_count = (last_20["predicted_direction"] == last_20["actual_direction"]).sum()
                rolling_acc = correct_count / len(last_20) if len(last_20) > 0 else 0.0
                
                sc1, sc2 = st.columns(2)
                sc1.metric("Rolling Accuracy (Last 20)", f"{rolling_acc:.1%}")
                sc2.metric("Total Resolved Predictions", len(resolved_preds))
                
                # Plotly Chart Setup
                fig_pa = go.Figure()
                
                # 1. Line chart of actual prices
                fig_pa.add_trace(go.Scatter(
                    x=resolved_preds["target_timestamp"],
                    y=resolved_preds["actual_close"],
                    mode="lines",
                    name="Actual Spot Close",
                    line=dict(color="#fbbf24", width=2)
                ))
                
                # 2. Scatter overlay for predictions
                colors = {"UP": "#10b981", "DOWN": "#ef4444", "FLAT": "#94a3b8"}
                symbols = {
                    ("UP", True): "triangle-up",
                    ("UP", False): "triangle-up-open",
                    ("DOWN", True): "triangle-down",
                    ("DOWN", False): "triangle-down-open",
                    ("FLAT", True): "circle",
                    ("FLAT", False): "circle-open"
                }
                
                for (pred_val, is_correct), group in resolved_preds.groupby(
                    [resolved_preds["predicted"], resolved_preds["predicted_direction"] == resolved_preds["actual_direction"]]
                ):
                    marker_symbol = symbols.get((pred_val, is_correct), "circle")
                    color = colors.get(pred_val, "#94a3b8")
                    
                    # Size based on confidence
                    sizes = group["confidence"].map(lambda c: 10 + 10 * c).tolist()
                    
                    fig_pa.add_trace(go.Scatter(
                        x=group["target_timestamp"],
                        y=group["actual_close"],
                        mode="markers",
                        name=f"Predicted {pred_val} ({'Correct' if is_correct else 'Incorrect'})",
                        marker=dict(
                            symbol=marker_symbol,
                            color=color,
                            size=sizes,
                            opacity=0.9,
                            line=dict(color=color, width=1.5)
                        ),
                        hoverinfo="text",
                        hovertext=group.apply(
                            lambda r: f"Time: {r['target_timestamp']}<br>Pred: {r['predicted']}<br>Actual Close: ${r['actual_close']:.2f}<br>Base Price: ${r['spot_price_at_prediction']:.2f}<br>Conf: {r['confidence']:.1%}",
                            axis=1
                        )
                    ))
                    
                fig_pa.update_layout(
                    paper_bgcolor="#0f172a",
                    plot_bgcolor="#0f172a",
                    xaxis=dict(gridcolor="#1e293b", tickcolor="#94a3b8", tickfont=dict(color="#94a3b8")),
                    yaxis=dict(gridcolor="#1e293b", tickcolor="#94a3b8", tickfont=dict(color="#94a3b8"), title=dict(text="Price (USD)", font=dict(color="#94a3b8"))),
                    legend=dict(font=dict(color="#94a3b8"), bgcolor="rgba(15,23,42,0.8)"),
                    margin=dict(l=40, r=40, t=20, b=40)
                )
                
                st.plotly_chart(fig_pa, use_container_width=True)

with tab4:
    st.markdown("#### Predictions History Log")
    if not preds.empty:
        # Format prediction direction
        def format_dir(x):
            return "📈 UP" if x == 1 else "📉 DOWN" if x == -1 else "➡️ FLAT"
            
        display_preds = preds.copy()
        display_preds["predicted"] = display_preds["predicted_direction"].map(format_dir)
        display_preds["actual"] = display_preds["actual_direction"].map(lambda x: format_dir(x) if pd.notna(x) else "⏳ PENDING")
        
        # Calculate correctness
        display_preds["status"] = np.where(
            display_preds["actual_direction"].isna(), "⏳ PENDING",
            np.where(display_preds["predicted_direction"] == display_preds["actual_direction"], "✅ CORRECT", "❌ WRONG")
        )
        
        # Columns to display
        display_cols = ["timestamp", "predicted", "confidence", "actual", "actual_close", "status"]
        st.dataframe(display_preds[display_cols].head(50), use_container_width=True)
    else:
        st.info("No predictions logged to database yet.")
