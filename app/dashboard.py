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
    page_title="Stock Holmes | Case Board",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Case File Terminal Theme Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
      --ink-black: #0E1015;
      --panel-surface: #1C2128;
      --brass-accent: #C9A227;
      --evidence-red: #8B2635;
      --muted-teal: #3A7D6E;
      --aged-paper: #E8DCC0;
    }

    /* Page background and general typography */
    .stApp {
      background-color: var(--ink-black) !important;
      color: var(--aged-paper) !important;
      font-family: 'IBM Plex Sans', sans-serif !important;
    }

    /* Sidebar overrides */
    [data-testid="stSidebar"] {
      background-color: #0b0c10 !important;
      border-right: 1px solid rgba(201, 162, 39, 0.15) !important;
    }

    /* Header typography */
    h1, h2, h3, h4, h5, h6, .case-header {
      font-family: 'Special Elite', monospace !important;
      color: var(--brass-accent) !important;
      font-weight: normal !important;
    }

    /* Evidence Card styles */
    .evidence-card {
      background: linear-gradient(135deg, #1d2128 0%, #161a20 100%);
      border: 1px solid rgba(201, 162, 39, 0.2);
      border-radius: 4px;
      padding: 20px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
      position: relative;
      transition: transform 0.3s cubic-bezier(0.165, 0.84, 0.44, 1), box-shadow 0.3s ease;
      margin-bottom: 20px;
    }
    .evidence-card::before {
      content: "📌";
      position: absolute;
      top: -12px;
      left: 50%;
      transform: translateX(-50%);
      font-size: 22px;
      filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));
      z-index: 10;
    }

    @media (prefers-reduced-motion: no-preference) {
      .evidence-card:hover {
        transform: translateY(-4px) scale(1.01);
        box-shadow: 0 15px 35px rgba(201, 162, 39, 0.25);
        border-color: var(--brass-accent);
      }
    }

    .evidence-card-1 {
      transform: rotate(-1.2deg);
    }
    .evidence-card-2 {
      transform: rotate(1deg);
    }
    .evidence-card-3 {
      transform: rotate(-0.5deg);
    }

    /* Polygraph Gauge pointer animation */
    @media (prefers-reduced-motion: no-preference) {
      .gauge-needle {
        transition: transform 1.2s cubic-bezier(0.25, 0.8, 0.25, 1);
      }
    }

    /* Rubber Stamp styles */
    .stamp-confirmed {
      color: var(--muted-teal);
      border: 2px dashed var(--muted-teal);
      padding: 2px 8px;
      font-family: 'Special Elite', monospace;
      font-weight: bold;
      text-transform: uppercase;
      display: inline-block;
      transform: rotate(-3deg);
      border-radius: 4px;
      font-size: 11px;
      letter-spacing: 1px;
      background-color: rgba(58, 125, 110, 0.05);
    }
    .stamp-wrong {
      color: var(--evidence-red);
      border: 2px dashed var(--evidence-red);
      padding: 2px 8px;
      font-family: 'Special Elite', monospace;
      font-weight: bold;
      text-transform: uppercase;
      display: inline-block;
      transform: rotate(3deg);
      border-radius: 4px;
      font-size: 11px;
      letter-spacing: 1px;
      background-color: rgba(139, 38, 53, 0.05);
    }
    .stamp-pending {
      color: var(--brass-accent);
      border: 2px dashed var(--brass-accent);
      padding: 2px 8px;
      font-family: 'Special Elite', monospace;
      font-weight: bold;
      text-transform: uppercase;
      display: inline-block;
      border-radius: 4px;
      font-size: 11px;
      letter-spacing: 1px;
      background-color: rgba(201, 162, 39, 0.05);
    }

    /* Manila Folder Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
      gap: 8px !important;
      background-color: transparent !important;
      border-bottom: 2px solid var(--panel-surface) !important;
    }
    .stTabs [data-baseweb="tab"] {
      font-family: 'Special Elite', monospace !important;
      background-color: var(--panel-surface) !important;
      color: #718096 !important;
      border-top-left-radius: 6px !important;
      border-top-right-radius: 6px !important;
      border-bottom-left-radius: 0px !important;
      border-bottom-right-radius: 0px !important;
      padding: 10px 22px !important;
      border: 1px solid rgba(255,255,255,0.03) !important;
      border-bottom: none !important;
      font-size: 13px !important;
      letter-spacing: 1px !important;
      transition: all 0.2s ease !important;
    }
    .stTabs [aria-selected="true"] {
      background-color: var(--aged-paper) !important;
      color: var(--ink-black) !important;
      border-color: var(--aged-paper) !important;
      font-weight: bold !important;
    }

    /* General metric overrides */
    [data-testid="stMetricValue"] {
      color: var(--brass-accent) !important;
      font-family: 'IBM Plex Mono', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown('<div class="case-header" style="color: var(--brass-accent); font-size: 14px; letter-spacing: 2px; margin-top: 10px;">CASE #XAUUSD-001 · STATUS: ACTIVE</div>', unsafe_allow_html=True)
st.title("🕵️‍♂️ Stock Holmes: Case Board")
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

# Detect Streamlit Cloud to set default API key value to blank
abs_path = os.path.abspath(__file__).replace("\\", "/")
is_streamlit_cloud = abs_path.startswith("/app/") or abs_path.startswith("/mount/")

if is_streamlit_cloud:
    api_key_default = ""
else:
    api_key_default = os.getenv("TWELVE_DATA_API_KEY", "")

# Render a text input for Twelve Data API Key
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
            import sys
            print(f"Ingestion error: {e}", file=sys.stderr)
            st.sidebar.error("Ingestion failed. (Operation blocked or invalid key)")

if st.sidebar.button("🤖 Retrain LightGBM"):
    with st.spinner("Rebuilding features and training walk-forward pipeline..."):
        try:
            metrics = train_pipeline()
            st.sidebar.success(f"Trained! Test Acc: {metrics.get('accuracy', 0.0):.1%}")
            st.cache_data.clear()
        except Exception as e:
            import sys
            print(f"Training error: {e}", file=sys.stderr)
            st.sidebar.error("Training failed. (Check database/log files)")

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
            import sys
            print(f"Inference error: {e}", file=sys.stderr)
            st.sidebar.error("Inference failed. (Insufficient history or model missing)")

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
st.sidebar.markdown("**🔍 Case Board Statistics**")
st.sidebar.text(f"Total Leads Logged: {len(preds)}")
st.sidebar.text(f"Total Evidence Candles: {len(candles)}")

# Check if we have data to display
if candles.empty:
    st.warning("⚠️ Local database is empty. Please use the sidebar to 'Ingest Latest Data' to build the initial database.")
    st.stop()

# ----------------- SECTION 1: LIVE CASE FILE TERMINAL -----------------
st.markdown("### 🖥️ Live Case File Terminal")
col1, col2, col3 = st.columns(3)

# Latest candle
latest_candle = candles.iloc[-1]
col1.markdown(f'<div class="evidence-card evidence-card-1"><h4 style="margin: 0; font-size: 13px; letter-spacing: 1px; color: var(--brass-accent);">💵 CURRENT GOLD SPOT</h4><h2 style="font-family: \'IBM Plex Mono\', monospace; font-size: 32px; margin: 10px 0; color: var(--aged-paper);">${latest_candle["close"]:,.2f}</h2><p style="color: #64748b; font-size: 12px; margin: 0;">Case Log: {latest_candle["timestamp"]} UTC</p></div>', unsafe_allow_html=True)

# Latest prediction
if not preds.empty:
    latest_pred = preds.iloc[0]
    pred_dir = latest_pred["predicted_direction"]
    conf = latest_pred["confidence"]
    
    meta_conf_val = latest_pred.get("meta_confidence")
    meta_info = ""
    if pd.notna(meta_conf_val) and meta_conf_val is not None:
        meta_info = f" | Trust: {float(meta_conf_val):.1%}"
    
    if pred_dir == 1:
        color = "var(--muted-teal)"
        dir_text = "📈 UP LEAD"
    elif pred_dir == -1:
        color = "var(--evidence-red)"
        dir_text = "📉 DOWN LEAD"
    else:
        color = "var(--brass-accent)"
        dir_text = "➡️ FLAT LEAD"
        
    col2.markdown(f'<div class="evidence-card evidence-card-2"><h4 style="margin: 0; font-size: 13px; letter-spacing: 1px; color: var(--brass-accent);">🕵️‍♂️ FORECAST LEAD (5M)</h4><h2 style="font-family: \'Special Elite\', monospace; font-size: 32px; margin: 10px 0; color: {color};">{dir_text}</h2><p style="color: #64748b; font-size: 12px; margin: 0;">Certainty: {conf:.1%}{meta_info}</p></div>', unsafe_allow_html=True)
    
    # SVG Certainty Polygraph Gauge
    angle = (conf - 0.5) * 140.0
    gauge_html = (
        f'<div class="evidence-card evidence-card-3" style="height: 100%;">'
        f'<h4 style="margin: 0; font-size: 13px; letter-spacing: 1px; text-align: center; color: var(--brass-accent);">📊 CERTAINTY GAUGE</h4>'
        f'<div style="display: flex; flex-direction: column; align-items: center; justify-content: center; margin-top: 10px;">'
        f'<svg width="150" height="55" viewBox="0 0 160 80">'
        f'<path d="M 20 70 A 60 60 0 0 1 140 70" fill="none" stroke="#2a303c" stroke-width="8" stroke-linecap="round"/>'
        f'<path d="M 20 70 A 60 60 0 0 1 140 70" fill="none" stroke="var(--brass-accent)" stroke-width="2" stroke-dasharray="2 4"/>'
        f'<circle cx="80" cy="70" r="6" fill="var(--brass-accent)"/>'
        f'<line class="gauge-needle" x1="80" y1="70" x2="80" y2="20" stroke="var(--evidence-red)" stroke-width="4" stroke-linecap="round" transform="rotate({angle}, 80, 70)"/>'
        f'</svg>'
        f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 11px; color: #64748b; margin-top: 2px;">'
        f'DOWN: {latest_pred["prob_down"]:.0%} | FLAT: {latest_pred["prob_flat"]:.0%} | UP: {latest_pred["prob_up"]:.0%}'
        f'</div>'
        f'</div>'
        f'</div>'
    )
    col3.markdown(gauge_html, unsafe_allow_html=True)
else:
    col2.markdown('<div class="evidence-card evidence-card-2"><h4 style="margin: 0; font-size: 13px; letter-spacing: 1px; color: var(--brass-accent);">🕵️‍♂️ FORECAST LEAD (5M)</h4><h2 style="font-family: \'Special Elite\', monospace; font-size: 20px; margin: 15px 0; color: var(--brass-accent);">INSUFFICIENT EVIDENCE</h2></div>', unsafe_allow_html=True)
    col3.markdown('<div class="evidence-card evidence-card-3"><h4 style="margin: 0; font-size: 13px; letter-spacing: 1px; color: var(--brass-accent); text-align: center;">📊 CERTAINTY GAUGE</h4><h2 style="font-family: \'Special Elite\', monospace; font-size: 20px; margin: 15px 0; color: var(--brass-accent); text-align: center;">NO LOGS YET</h2></div>', unsafe_allow_html=True)

# ----------------- SECTION 2: CHARTS & MOVEMENT -----------------
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📂 Evidence Timeline", 
    "📈 Case Backtests", 
    "🎯 Leads vs Outcomes", 
    "📋 Case File Ledger"
])

with tab1:
    st.markdown("#### XAUUSD Recent Price Activity")
    # Take last 100 candles
    recent_candles = candles.tail(100).copy()
    recent_candles["timestamp"] = pd.to_datetime(recent_candles["timestamp"])
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(recent_candles["timestamp"], recent_candles["close"], label="Spot Gold Price", color="#E8DCC0")
    ax.set_facecolor("#0E1015")
    fig.patch.set_facecolor("#0E1015")
    
    # Use RGBA tuples for matplotlib colors to prevent ValueError
    spine_color = (232/255, 220/255, 192/255, 0.15)
    grid_color = (232/255, 220/255, 192/255, 0.05)
    
    ax.spines['bottom'].set_color(spine_color)
    ax.spines['top'].set_color(spine_color)
    ax.spines['left'].set_color(spine_color)
    ax.spines['right'].set_color(spine_color)
    ax.tick_params(colors='#E8DCC0')
    ax.grid(True, color=grid_color, linestyle='--')
    ax.legend(facecolor='#1C2128', edgecolor='none', labelcolor='#E8DCC0')
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
        st.markdown("#### 🎯 Binary Detector Metrics (UP / DOWN / META)")
        det_col1, det_col2, det_col3 = st.columns(3)
        
        with det_col1:
            st.markdown("""
            <div class="evidence-card" style="height: 100%;">
                <h4 style="margin: 0; font-size: 13px; color: var(--brass-accent);">📈 UP-Detector</h4>
                <p style="margin: 8px 0 4px 0;">Precision: <b>{:.1%}</b></p>
                <p style="margin: 4px 0;">Recall: <b>{:.1%}</b></p>
                <p style="margin: 4px 0;">F1 Score: <b>{:.1%}</b></p>
                <p style='color: #64748b; margin: 4px 0; font-size: 11px;'>Threshold: {:.2f}</p>
            </div>
            """.format(
                train_metrics.get("up_precision", 0.0),
                train_metrics.get("up_recall", 0.0),
                train_metrics.get("up_f1", 0.0),
                train_metrics.get("up_threshold", 0.5),
            ), unsafe_allow_html=True)
        
        with det_col2:
            st.markdown("""
            <div class="evidence-card" style="height: 100%;">
                <h4 style="margin: 0; font-size: 13px; color: var(--brass-accent);">📉 DOWN-Detector</h4>
                <p style="margin: 8px 0 4px 0;">Precision: <b>{:.1%}</b></p>
                <p style="margin: 4px 0;">Recall: <b>{:.1%}</b></p>
                <p style="margin: 4px 0;">F1 Score: <b>{:.1%}</b></p>
                <p style='color: #64748b; margin: 4px 0; font-size: 11px;'>Threshold: {:.2f}</p>
            </div>
            """.format(
                train_metrics.get("down_precision", 0.0),
                train_metrics.get("down_recall", 0.0),
                train_metrics.get("down_f1", 0.0),
                train_metrics.get("down_threshold", 0.5),
            ), unsafe_allow_html=True)
            
        with det_col3:
            st.markdown("""
            <div class="evidence-card" style="height: 100%;">
                <h4 style="margin: 0; font-size: 13px; color: var(--brass-accent);">🛡️ Meta-Model Filter</h4>
                <p style="margin: 8px 0 4px 0;">Precision (Trust Acc): <b>{:.1%}</b></p>
                <p style="margin: 4px 0;">Filter Rate: <b>{:.1%}</b></p>
                <p style="margin: 4px 0;">Acted-upon Acc: <b>{:.1%}</b></p>
                <p style='color: #64748b; margin: 4px 0; font-size: 11px;'>Trust Threshold: {:.2f}</p>
            </div>
            """.format(
                train_metrics.get("meta_precision", 0.0),
                train_metrics.get("meta_filter_rate", 0.0),
                train_metrics.get("meta_acted_accuracy", 0.0),
                train_metrics.get("meta_trust_threshold", 0.5),
            ), unsafe_allow_html=True)
        
        # --- Class Distribution ---
        test_dist = train_metrics.get("test_class_distribution", {})
        if test_dist:
            st.markdown("#### 📊 Combined Test Set Class Distribution")
            dist_labels = {"1": "UP", "0": "FLAT", "-1": "DOWN"}
            dist_data = {dist_labels.get(k, k): v for k, v in test_dist.items()}
            
            fig_dist, ax_dist = plt.subplots(figsize=(6, 3))
            colors_dist = {"UP": "#3A7D6E", "FLAT": "#718096", "DOWN": "#8B2635"}
            bars = ax_dist.bar(
                dist_data.keys(), 
                dist_data.values(),
                color=[colors_dist.get(k, "#C9A227") for k in dist_data.keys()]
            )
            ax_dist.set_ylabel("Proportion", color="#E8DCC0", family="IBM Plex Sans")
            ax_dist.set_facecolor("#0E1015")
            fig_dist.patch.set_facecolor("#0E1015")
            ax_dist.tick_params(colors='#E8DCC0')
            
            spine_color = (232/255, 220/255, 192/255, 0.15)
            for spine in ax_dist.spines.values():
                spine.set_color(spine_color)
            for bar, val in zip(bars, dist_data.values()):
                ax_dist.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                           f'{val:.1%}', ha='center', va='bottom', color='#E8DCC0', fontsize=10)
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
                ax_fi.barh(feat_names[::-1], feat_vals[::-1], color="#C9A227")
                ax_fi.set_xlabel("Gain", color="#E8DCC0", family="IBM Plex Sans")
                ax_fi.set_facecolor("#0E1015")
                fig_fi.patch.set_facecolor("#0E1015")
                ax_fi.tick_params(colors='#E8DCC0')
                for spine in ax_fi.spines.values():
                    spine.set_color(spine_color)
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
            ax_bt.plot(bt_df["Timestamp"], bt_df["Model Strategy"], label="Stock Holmes v3 Strategy", color="#3A7D6E", linewidth=2)
            ax_bt.plot(bt_df["Timestamp"], bt_df["Naive Sign Baseline"], label="Naive Sign Carry-Forward", color="#8B2635", linestyle="--")
            ax_bt.set_facecolor("#0E1015")
            fig_bt.patch.set_facecolor("#0E1015")
            
            grid_color = (232/255, 220/255, 192/255, 0.05)
            ax_bt.spines['bottom'].set_color(spine_color)
            ax_bt.spines['top'].set_color(spine_color)
            ax_bt.spines['left'].set_color(spine_color)
            ax_bt.spines['right'].set_color(spine_color)
            ax_bt.tick_params(colors='#E8DCC0')
            ax_bt.grid(True, color=grid_color, linestyle='--')
            ax_bt.legend(facecolor='#1C2128', edgecolor='none', labelcolor='#E8DCC0')
            st.pyplot(fig_bt)
        else:
            st.warning("No walk-forward backtesting metrics found. Please Retrain model to generate backtest logs.")
    else:
        st.markdown('<div style="font-family: \'Special Elite\', monospace; padding: 20px; text-align: center; border: 1px dashed var(--brass-accent); color: var(--brass-accent);">INSUFFICIENT EVIDENCE — awaiting more resolved cases</div>', unsafe_allow_html=True)

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
                
                # 1. Line chart of actual prices (Aged Paper color)
                fig_pa.add_trace(go.Scatter(
                    x=resolved_preds["target_timestamp"],
                    y=resolved_preds["actual_close"],
                    mode="lines",
                    name="Actual Spot Close",
                    line=dict(color="#E8DCC0", width=1.5)
                ))
                
                # 2. Draw connecting string lines
                correct_x = []
                correct_y = []
                incorrect_x = []
                incorrect_y = []
                
                for _, row in resolved_preds.iterrows():
                    x0 = row["timestamp"]
                    y0 = row["spot_price_at_prediction"]
                    x1 = row["target_timestamp"]
                    y1 = row["actual_close"]
                    
                    if pd.isna(y0) or pd.isna(y1) or y0 == 0 or y1 == 0:
                        continue
                        
                    is_correct = row["predicted_direction"] == row["actual_direction"]
                    
                    if is_correct:
                        correct_x.extend([x0, x1, None])
                        correct_y.extend([y0, y1, None])
                    else:
                        incorrect_x.extend([x0, x1, None])
                        incorrect_y.extend([y0, y1, None])
                
                if correct_x:
                    fig_pa.add_trace(go.Scatter(
                        x=correct_x, y=correct_y,
                        mode="lines",
                        line=dict(color="#C9A227", width=1.0),
                        name="Confirmed Lead Thread",
                        hoverinfo="skip"
                    ))
                if incorrect_x:
                    fig_pa.add_trace(go.Scatter(
                        x=incorrect_x, y=incorrect_y,
                        mode="lines",
                        line=dict(color="#8B2635", width=1.0, dash="dash"),
                        name="Flawed Lead Thread (Red String)",
                        hoverinfo="skip"
                    ))
                
                # 3. Scatter overlay for predictions at their entry points
                colors = {"UP": "#3A7D6E", "DOWN": "#8B2635", "FLAT": "#718096"}
                
                for pred_val, group in resolved_preds.groupby("predicted"):
                    color = colors.get(pred_val, "#718096")
                    sizes = group["confidence"].map(lambda c: 8 + 8 * c).tolist()
                    
                    fig_pa.add_trace(go.Scatter(
                        x=group["timestamp"],
                        y=group["spot_price_at_prediction"],
                        mode="markers",
                        name=f"Lead: {pred_val} (Pin)",
                        marker=dict(
                            symbol="circle-dot",
                            color=color,
                            size=sizes,
                            opacity=0.9,
                            line=dict(color=color, width=1)
                        ),
                        hoverinfo="text",
                        hovertext=group.apply(
                            lambda r: f"Entry Time: {r['timestamp']}<br>Lead: {r['predicted']}<br>Base Price: ${r['spot_price_at_prediction']:.2f}<br>Resolution: {r['target_timestamp']}<br>Close: ${r['actual_close']:.2f}<br>Certainty: {r['confidence']:.1%}",
                            axis=1
                        )
                    ))
                    
                fig_pa.update_layout(
                    paper_bgcolor="#0E1015",
                    plot_bgcolor="#0E1015",
                    xaxis=dict(
                        gridcolor="rgba(232, 220, 192, 0.05)",
                        tickcolor="#E8DCC0",
                        tickfont=dict(color="#E8DCC0", family="IBM Plex Mono")
                    ),
                    yaxis=dict(
                        gridcolor="rgba(232, 220, 192, 0.05)",
                        tickcolor="#E8DCC0",
                        tickfont=dict(color="#E8DCC0", family="IBM Plex Mono"),
                        title=dict(text="Price (USD)", font=dict(color="#E8DCC0", family="Special Elite", size=12))
                    ),
                    legend=dict(
                        font=dict(color="#E8DCC0", family="IBM Plex Sans", size=10),
                        bgcolor="rgba(14,16,21,0.85)"
                    ),
                    margin=dict(l=40, r=40, t=20, b=40)
                )
                
                st.plotly_chart(fig_pa, use_container_width=True)

with tab4:
    st.markdown("#### Case File Ledger")
    if not preds.empty:
        # Format prediction direction
        def format_dir(x):
            return "UP" if x == 1 else "DOWN" if x == -1 else "FLAT"
            
        display_preds = preds.copy()
        if "predicted" not in display_preds.columns or display_preds["predicted"].isna().all():
            display_preds["predicted"] = display_preds["predicted_direction"].map(format_dir)
        else:
            display_preds["predicted"] = display_preds["predicted"].map(
                lambda x: "UP" if x == "UP" else "DOWN" if x == "DOWN" else "FLAT" if x == "FLAT" else format_dir(x)
            )
        display_preds["actual"] = display_preds["actual_direction"].map(lambda x: format_dir(x) if pd.notna(x) else "⏳ PENDING")
        
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
            
        # Custom HTML Case File Ledger Table (constructed as clean string list to avoid markdown code-block issues)
        html_lines = [
            '<div style="overflow-x: auto; margin-top: 15px;">',
            '<table style="width: 100%; border-collapse: collapse; font-family: \'IBM Plex Mono\', monospace; background-color: var(--panel-surface); color: var(--aged-paper); font-size: 13px;">',
            '<thead>',
            '<tr style="border-bottom: 2px solid var(--brass-accent); text-align: left;">',
            '<th style="font-family: \'Special Elite\', monospace; padding: 12px; color: var(--brass-accent);">TIMESTAMP</th>',
            '<th style="font-family: \'Special Elite\', monospace; padding: 12px; color: var(--brass-accent);">LEAD FORECAST</th>',
            '<th style="font-family: \'Special Elite\', monospace; padding: 12px; color: var(--brass-accent);">CERTAINTY</th>'
        ]
        if "meta_confidence" in display_preds.columns:
            html_lines.append('<th style="font-family: \'Special Elite\', monospace; padding: 12px; color: var(--brass-accent);">META TRUST</th>')
        html_lines.extend([
            '<th style="font-family: \'Special Elite\', monospace; padding: 12px; color: var(--brass-accent);">OUTCOME</th>',
            '<th style="font-family: \'Special Elite\', monospace; padding: 12px; color: var(--brass-accent);">ACTUAL CLOSE</th>',
            '<th style="font-family: \'Special Elite\', monospace; padding: 12px; color: var(--brass-accent);">STATUS STAMP</th>',
            '</tr>',
            '</thead>',
            '<tbody>'
        ])
        
        for idx, row in display_preds.reset_index(drop=True).head(50).iterrows():
            bg_style = "background-color: #151a21;" if idx % 2 == 1 else "background-color: var(--panel-surface);"
            
            res_val = row["result"]
            if res_val == "✅ CORRECT":
                stamp_html = '<span class="stamp-confirmed">CONFIRMED</span>'
            elif res_val == "❌ WRONG":
                stamp_html = '<span class="stamp-wrong">WRONG</span>'
            else:
                stamp_html = '<span class="stamp-pending">PENDING</span>'
                
            pred_str = row["predicted"]
            if pred_str == "UP":
                pred_color = "color: var(--muted-teal); font-weight: bold;"
            elif pred_str == "DOWN":
                pred_color = "color: var(--evidence-red); font-weight: bold;"
            else:
                pred_color = "color: #718096;"
                
            actual_str = row["actual"]
            if actual_str == "UP":
                actual_color = "color: var(--muted-teal);"
            elif actual_str == "DOWN":
                actual_color = "color: var(--evidence-red);"
            else:
                actual_color = "color: #718096;"
                
            close_val = f"${row['actual_close']:,.2f}" if pd.notna(row["actual_close"]) else "—"
            meta_conf_td = ""
            if "meta_confidence" in display_preds.columns:
                m_conf = f"{row['meta_confidence']:.1%}" if pd.notna(row.get("meta_confidence")) else "—"
                meta_conf_td = f'<td style="padding: 10px 12px;">{m_conf}</td>'
                
            html_lines.append(
                f'<tr style="{bg_style} border-bottom: 1px solid rgba(255,255,255,0.02);">'
                f'<td style="padding: 10px 12px;">{row["timestamp"]}</td>'
                f'<td style="padding: 10px 12px; {pred_color}">{pred_str}</td>'
                f'<td style="padding: 10px 12px;">{row["confidence"]:.1%}</td>'
                f'{meta_conf_td}'
                f'<td style="padding: 10px 12px; {actual_color}">{actual_str}</td>'
                f'<td style="padding: 10px 12px;">{close_val}</td>'
                f'<td style="padding: 10px 12px;">{stamp_html}</td>'
                f'</tr>'
            )
            
        html_lines.extend(['</tbody>', '</table>', '</div>'])
        html = "".join(html_lines)
        st.markdown(html, unsafe_allow_html=True)
    else:
        st.markdown('<div style="font-family: \'Special Elite\', monospace; padding: 20px; text-align: center; border: 1px dashed var(--brass-accent); color: var(--brass-accent);">INSUFFICIENT EVIDENCE — awaiting more resolved cases</div>', unsafe_allow_html=True)
