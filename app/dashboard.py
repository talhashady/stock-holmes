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
from src.ingestion.fetcher import fetch_and_cache, fetch_and_cache_multi
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
    with st.spinner("Fetching live candles from Twelve Data (XAUUSD + EURUSD + USDJPY)..."):
        try:
            results = fetch_and_cache_multi(api_key=api_key)
            total_inserted = sum(v for v in results.values() if v > 0)
            st.sidebar.success(f"Fetched and cached {total_inserted} new candles across {len(results)} symbols!")
            for sym, count in results.items():
                if count >= 0:
                    st.sidebar.text(f"  {sym}: {count} new candles")
                else:
                    st.sidebar.warning(f"  {sym}: fetch failed (non-critical)")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Ingestion error: {e}")

if st.sidebar.button("🤖 Retrain LightGBM"):
    with st.spinner("Rebuilding features and training walk-forward pipeline..."):
        try:
            metrics = train_pipeline()
            st.sidebar.success(f"Trained! Test Acc: {metrics.get('accuracy', 0.0):.1%}")
            st.cache_data.clear()
        except Exception as e:
            st.sidebar.error(f"Training error: {e}")

if st.sidebar.button("🎯 Run Inference (Predict)"):
    with st.spinner("Generating fresh 5-minute predictions..."):
        try:
            res = predict_latest()
            if res:
                st.sidebar.success("Latest prediction saved!")
                st.cache_data.clear()
            else:
                st.sidebar.warning("Inference executed but returned no prediction (likely missing history).")
        except Exception as e:
            st.sidebar.error(f"Inference error: {e}")

def sync_data_from_github():
    import urllib.request
    import os
    
    # Only run sync on Streamlit Cloud (check path prefix to handle casing /mount/src/Stock-Holmes)
    abs_path = os.path.abspath(__file__).replace("\\", "/")
    if not (abs_path.startswith("/app/") or abs_path.startswith("/mount/")):
        return
        
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(base_dir, "data", "stock_holmes.db")
    log_path = os.path.join(base_dir, "data", "predictions_log.jsonl")
    metrics_path = os.path.join(base_dir, "src", "models", "metrics.json")
    
    urls = {
        db_path: "https://raw.githubusercontent.com/talhashady/stock-holmes/main/data/stock_holmes.db",
        log_path: "https://raw.githubusercontent.com/talhashady/stock-holmes/main/data/predictions_log.jsonl",
        metrics_path: "https://raw.githubusercontent.com/talhashady/stock-holmes/main/src/models/metrics.json"
    }
    
    for path, url in urls.items():
        temp_path = path + ".tmp"
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read()
                if len(content) > 0:
                    with open(temp_path, "wb") as f:
                        f.write(content)
                    if os.path.exists(temp_path):
                        if os.path.exists(path):
                            os.remove(path)
                        os.rename(temp_path, path)
        except Exception as e:
            print(f"[SYNC ERROR] Failed to sync {url} to {path}: {e}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

# ----------------- DATA LOADING -----------------
@st.cache_data(ttl=10)
def load_dashboard_data():
    # If running on Streamlit Cloud, fetch directly from GitHub to bypass dirty/conflicted local checkout files
    abs_path = os.path.abspath(__file__).replace("\\", "/")
    if abs_path.startswith("/app/") or abs_path.startswith("/mount/"):
        try:
            # 1. Fetch predictions history log directly from GitHub
            preds_url = "https://raw.githubusercontent.com/talhashady/stock-holmes/main/data/predictions_log.jsonl"
            preds = pd.read_json(preds_url, lines=True)
            
            if not preds.empty:
                preds["timestamp"] = preds["timestamp"].astype(str)
                if "predicted" in preds.columns:
                    preds["predicted_direction"] = preds["predicted"].map({"UP": 1, "DOWN": -1, "FLAT": 0})
                else:
                    preds["predicted_direction"] = 0
                    
                if "actual_close" in preds.columns and "spot_price_at_prediction" in preds.columns:
                    actual_dir = []
                    for _, row in preds.iterrows():
                        ac = row["actual_close"]
                        sp = row["spot_price_at_prediction"]
                        if pd.isna(ac) or ac is None or pd.isna(sp) or sp is None or sp == 0:
                            actual_dir.append(np.nan)
                        else:
                            change = (ac - sp) / sp
                            if change > 0.0001:
                                actual_dir.append(1)
                            elif change < -0.0001:
                                actual_dir.append(-1)
                            else:
                                actual_dir.append(0)
                    preds["actual_direction"] = actual_dir
                else:
                    preds["actual_direction"] = np.nan
                    
                if "spot_price_at_prediction" in preds.columns:
                    preds["current_close"] = preds["spot_price_at_prediction"]
                    
                preds = preds.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
            
            # 2. Download sqlite database to /tmp/ to bypass file lock on container's local db
            import urllib.request
            import sqlite3
            
            db_url = "https://raw.githubusercontent.com/talhashady/stock-holmes/main/data/stock_holmes.db"
            temp_db = "/tmp/stock_holmes_cloud.db"
            
            os.makedirs(os.path.dirname(temp_db), exist_ok=True)
            req = urllib.request.Request(db_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                with open(temp_db, "wb") as f:
                    f.write(response.read())
            
            conn = sqlite3.connect(temp_db)
            candles = pd.read_sql_query("SELECT * FROM candles ORDER BY timestamp ASC", conn)
            conn.close()
            
            # 3. Fetch training metrics directly from GitHub
            metrics_url = "https://raw.githubusercontent.com/talhashady/stock-holmes/main/src/models/metrics.json"
            train_metrics = None
            try:
                req_metrics = urllib.request.Request(metrics_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req_metrics, timeout=5) as resp:
                    train_metrics = json.loads(resp.read().decode("utf-8"))
            except Exception:
                pass
                
            return preds, candles, train_metrics
        except Exception as cloud_err:
            st.sidebar.warning(f"Cloud load fallback: {cloud_err}")
            
    # Local fallback
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

# ----------------- SIDEBAR DIAGNOSTICS -----------------
st.sidebar.markdown("---")
st.sidebar.markdown("**🔍 Environment Diagnostics**")
st.sidebar.text(f"File Path: {os.path.abspath(__file__)}")
st.sidebar.text(f"CWD: {os.getcwd()}")
st.sidebar.text(f"Loaded Predictions: {len(preds)}")
st.sidebar.text(f"Loaded Candles: {len(candles)}")

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
        m_col1.metric("Combined Accuracy", f"{train_metrics.get('accuracy', 0.0):.1%}", 
                      delta=f"{train_metrics.get('accuracy', 0.0) - train_metrics.get('naive_flat_accuracy', 0.0):.1%} vs Flat Baseline")
        m_col2.metric("Naive Directional Sign Accuracy", f"{train_metrics.get('naive_sign_accuracy', 0.0):.1%}")
        m_col3.metric("High-Confidence Accuracy (>55%)", f"{train_metrics.get('high_confidence_accuracy', 0.0):.1%}",
                      delta=f"{train_metrics.get('high_confidence_count', 0)} samples")
        
        # --- Binary Model Per-Detector Metrics ---
        st.markdown("#### 🎯 Binary Detector Metrics (UP / DOWN)")
        det_col1, det_col2 = st.columns(2)
        
        with det_col1:
            st.markdown("""
            <div class="metric-card">
                <h4>📈 UP-Detector</h4>
                <p>Precision: <b>{:.1%}</b></p>
                <p>Recall: <b>{:.1%}</b></p>
                <p>F1 Score: <b>{:.1%}</b></p>
                <p style='color: #64748b;'>Threshold: {:.2f}</p>
            </div>
            """.format(
                train_metrics.get("up_precision", 0.0),
                train_metrics.get("up_recall", 0.0),
                train_metrics.get("up_f1", 0.0),
                train_metrics.get("up_threshold", 0.5),
            ), unsafe_allow_html=True)
        
        with det_col2:
            st.markdown("""
            <div class="metric-card">
                <h4>📉 DOWN-Detector</h4>
                <p>Precision: <b>{:.1%}</b></p>
                <p>Recall: <b>{:.1%}</b></p>
                <p>F1 Score: <b>{:.1%}</b></p>
                <p style='color: #64748b;'>Threshold: {:.2f}</p>
            </div>
            """.format(
                train_metrics.get("down_precision", 0.0),
                train_metrics.get("down_recall", 0.0),
                train_metrics.get("down_f1", 0.0),
                train_metrics.get("down_threshold", 0.5),
            ), unsafe_allow_html=True)
        
        # --- Class Distribution ---
        test_dist = train_metrics.get("test_class_distribution", {})
        if test_dist:
            st.markdown("#### 📊 Combined Test Set Class Distribution")
            dist_labels = {"1": "UP", "0": "FLAT", "-1": "DOWN"}
            dist_data = {dist_labels.get(k, k): v for k, v in test_dist.items()}
            
            fig_dist, ax_dist = plt.subplots(figsize=(6, 3))
            colors_dist = {"UP": "#10b981", "FLAT": "#94a3b8", "DOWN": "#ef4444"}
            bars = ax_dist.bar(
                dist_data.keys(), 
                dist_data.values(),
                color=[colors_dist.get(k, "#3b82f6") for k in dist_data.keys()]
            )
            ax_dist.set_ylabel("Proportion", color="#94a3b8")
            ax_dist.set_facecolor("#0f172a")
            fig_dist.patch.set_facecolor("#0f172a")
            ax_dist.tick_params(colors='#94a3b8')
            for spine in ax_dist.spines.values():
                spine.set_color('#475569')
            for bar, val in zip(bars, dist_data.values()):
                ax_dist.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{val:.1%}', ha='center', va='bottom', color='#94a3b8', fontsize=10)
            st.pyplot(fig_dist)
        
        # --- Feature Importance ---
        for model_name, display_name in [("up", "📈 UP-Detector"), ("down", "📉 DOWN-Detector")]:
            feat_imp = train_metrics.get(f"{model_name}_feature_importance", {})
            if feat_imp:
                st.markdown(f"#### {display_name} — Top 10 Feature Importance")
                sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:10]
                feat_names = [f[0] for f in sorted_feats]
                feat_vals = [f[1] for f in sorted_feats]
                
                fig_fi, ax_fi = plt.subplots(figsize=(8, 3.5))
                ax_fi.barh(feat_names[::-1], feat_vals[::-1], color="#3b82f6")
                ax_fi.set_xlabel("Gain", color="#94a3b8")
                ax_fi.set_facecolor("#0f172a")
                fig_fi.patch.set_facecolor("#0f172a")
                ax_fi.tick_params(colors='#94a3b8')
                for spine in ax_fi.spines.values():
                    spine.set_color('#475569')
                st.pyplot(fig_fi)
        
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
            ax_bt.plot(bt_df["Timestamp"], bt_df["Model Strategy"], label="Stock Holmes v2 Strategy", color="#10b981", linewidth=2)
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
        
        if len(resolved_preds) < 1:
            st.info("ℹ️ No resolved predictions yet to plot. Run ingestion and inference to resolve past predictions.")
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
        # Use the predicted string from JSONL directly if available, else derive from int
        if "predicted" not in display_preds.columns or display_preds["predicted"].isna().all():
            display_preds["predicted"] = display_preds["predicted_direction"].map(format_dir)
        else:
            display_preds["predicted"] = display_preds["predicted"].map(
                lambda x: "📈 UP" if x == "UP" else "📉 DOWN" if x == "DOWN" else "➡️ FLAT" if x == "FLAT" else format_dir(x)
            )
        display_preds["actual"] = display_preds["actual_direction"].map(lambda x: format_dir(x) if pd.notna(x) else "⏳ PENDING")
        
        # Map status from JSONL, then override with correctness for RESOLVED ones
        if "status" in display_preds.columns:
            display_preds["result"] = np.where(
                display_preds["status"] != "RESOLVED", "⏳ PENDING",
                np.where(display_preds["predicted_direction"] == display_preds["actual_direction"], "✅ CORRECT", "❌ WRONG")
            )
        else:
            display_preds["result"] = np.where(
                display_preds["actual_direction"].isna(), "⏳ PENDING",
                np.where(display_preds["predicted_direction"] == display_preds["actual_direction"], "✅ CORRECT", "❌ WRONG")
            )
        
        # Columns to display
        display_cols = ["timestamp", "predicted", "confidence", "actual", "actual_close", "result"]
        st.dataframe(display_preds[display_cols].head(50), use_container_width=True)
    else:
        st.info("No predictions logged to database yet.")
