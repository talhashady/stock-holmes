import sqlite3
import os
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple
from contextlib import contextmanager

DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "stock_holmes.db"
)

@contextmanager
def get_connection(db_path: str = DEFAULT_DB_PATH):
    """Establishes connection to the SQLite database, yielding it and ensuring it is closed."""
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()

# Valid cross-asset table names for parameterized queries
VALID_CANDLE_TABLES = {"candles", "candles_dxy", "candles_eurusd", "candles_usdjpy"}

def _validate_table_name(table_name: str) -> str:
    """Validates table name against whitelist to prevent SQL injection."""
    if table_name not in VALID_CANDLE_TABLES:
        raise ValueError(f"Invalid table name '{table_name}'. Must be one of: {VALID_CANDLE_TABLES}")
    return table_name

def init_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Initializes the database schemas for candles and predictions."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        # Candles table (XAUUSD — primary)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                timestamp TEXT PRIMARY KEY,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                volume REAL NOT NULL
            );
        """)
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp TEXT PRIMARY KEY,
                predicted_direction INTEGER NOT NULL,  -- -1 (DOWN), 0 (FLAT), 1 (UP)
                confidence REAL NOT NULL,               -- Probability of predicted class (0.0 to 1.0)
                prob_down REAL NOT NULL,
                prob_flat REAL NOT NULL,
                prob_up REAL NOT NULL,
                actual_direction INTEGER,               -- Set when actual t+5 data arrives
                actual_close REAL                       -- Actual price at target time
            );
        """)
        conn.commit()
    # Also initialize cross-asset tables
    init_cross_asset_tables(db_path)

def init_cross_asset_tables(db_path: str = DEFAULT_DB_PATH) -> None:
    """Creates candle tables for cross-asset correlation symbols (EUR/USD, USDJPY)."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for table_name in ["candles_eurusd", "candles_usdjpy"]:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    timestamp TEXT PRIMARY KEY,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL
                );
            """)
        conn.commit()

def save_candles(df: pd.DataFrame, db_path: str = DEFAULT_DB_PATH, table_name: str = "candles") -> int:
    """Saves a pandas DataFrame of candles to SQLite. Ignores duplicates.
    
    Args:
        df: DataFrame with OHLCV columns.
        db_path: Path to SQLite database.
        table_name: Target table — 'candles' (XAUUSD), 'candles_dxy', or 'candles_usdjpy'.
    """
    if df.empty:
        return 0
    
    table_name = _validate_table_name(table_name)
    
    required_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in DataFrame: {col}")
            
    # Ensure timestamp is string format and correct types
    df = df[required_cols].copy()
    df["timestamp"] = df["timestamp"].astype(str)
    
    records = df.to_dict(orient="records")
    inserted = 0
    
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        for rec in records:
            try:
                cursor.execute(f"""
                    INSERT OR IGNORE INTO {table_name} (timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (rec["timestamp"], rec["open"], rec["high"], rec["low"], rec["close"], rec["volume"]))
                if cursor.rowcount > 0:
                    inserted += 1
            except sqlite3.Error:
                pass
        conn.commit()
    return inserted

def get_latest_candle_time(db_path: str = DEFAULT_DB_PATH, table_name: str = "candles") -> Optional[str]:
    """Returns the ISO timestamp of the latest candle stored in the database.
    
    Args:
        db_path: Path to SQLite database.
        table_name: Table to query — 'candles', 'candles_dxy', or 'candles_usdjpy'.
    """
    table_name = _validate_table_name(table_name)
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(f"SELECT MAX(timestamp) FROM {table_name}")
            res = cursor.fetchone()
            return res[0] if res else None
        except sqlite3.OperationalError:
            # Table may not exist yet for cross-asset symbols
            return None

def get_all_candles(db_path: str = DEFAULT_DB_PATH, table_name: str = "candles") -> pd.DataFrame:
    """Fetches all candles sorted chronologically.
    
    Args:
        db_path: Path to SQLite database.
        table_name: Table to query — 'candles', 'candles_dxy', or 'candles_usdjpy'.
    """
    table_name = _validate_table_name(table_name)
    with get_connection(db_path) as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY timestamp ASC", conn)
        except Exception:
            # Table may not exist yet — return empty DataFrame
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df

def save_prediction(timestamp: str, predicted_direction: int, confidence: float, 
                    probs: Tuple[float, float, float], db_path: str = DEFAULT_DB_PATH,
                    meta_confidence: Optional[float] = None) -> None:
    """Saves or updates a model prediction both in SQLite and predictions_log.jsonl."""
    import json
    prob_down, prob_flat, prob_up = probs
    
    # 1. Save to SQLite
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO predictions 
            (timestamp, predicted_direction, confidence, prob_down, prob_flat, prob_up, actual_direction, actual_close)
            VALUES (?, ?, ?, ?, ?, ?, 
                    (SELECT actual_direction FROM predictions WHERE timestamp = ?),
                    (SELECT actual_close FROM predictions WHERE timestamp = ?))
        """, (timestamp, predicted_direction, confidence, prob_down, prob_flat, prob_up, timestamp, timestamp))
        conn.commit()
        
    # 2. Save/Sync to predictions_log.jsonl
    log_path = os.path.join(os.path.dirname(db_path), "predictions_log.jsonl")
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        
    # Construct target timestamp (+5 mins)
    pred_dt = pd.to_datetime(timestamp)
    target_dt = pred_dt + pd.Timedelta(minutes=5)
    target_ts = target_dt.strftime("%Y-%m-%d %H:%M:%S")
    dir_str = "UP" if predicted_direction == 1 else "DOWN" if predicted_direction == -1 else "FLAT"
    
    new_record = {
        "timestamp": timestamp,
        "target_timestamp": target_ts,
        "predicted": dir_str,
        "confidence": confidence,
        "spot_price_at_prediction": 0.0,
        "actual_close": None,
        "status": "PENDING",
        "prob_down": float(prob_down),
        "prob_flat": float(prob_flat),
        "prob_up": float(prob_up)
    }
    
    if meta_confidence is not None:
        new_record["meta_confidence"] = float(meta_confidence)
    
    # Retrieve spot price at prediction from candles SQLite table if present
    try:
        with get_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT close FROM candles WHERE timestamp = ?", (timestamp,))
            res = cursor.fetchone()
            if res:
                new_record["spot_price_at_prediction"] = float(res[0])
    except Exception:
        pass
        
    # Read existing JSONL predictions
    lines = []
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        lines.append(json.loads(line_str))
                    except Exception:
                        pass
                        
    # Replace matching or append
    replaced = False
    updated_lines = []
    for pred in lines:
        if pred.get("timestamp") == timestamp:
            new_record["actual_close"] = pred.get("actual_close")
            new_record["status"] = pred.get("status", "PENDING")
            updated_lines.append(new_record)
            replaced = True
        else:
            updated_lines.append(pred)
            
    if not replaced:
        updated_lines.append(new_record)
        
    # Rewrite JSONL file
    with open(log_path, "w", encoding="utf-8") as f:
        for pred in updated_lines:
            f.write(json.dumps(pred) + "\n")

def resolve_pending_predictions(db_path: str = DEFAULT_DB_PATH, flat_threshold_pct: float = 0.0001) -> int:
    """
    Finds PENDING predictions in the JSONL log whose target_timestamp has passed,
    looks up their actual close price in the candles SQLite table, and updates them to RESOLVED.
    """
    import json
    log_path = os.path.join(os.path.dirname(db_path), "predictions_log.jsonl")
    if not os.path.exists(log_path):
        return 0
        
    # Read the JSONL file
    lines = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    lines.append(json.loads(line_str))
                except Exception:
                    pass
                    
    resolved_count = 0
    updated_lines = []
    
    # We fetch all candles from SQLite to check actual close prices
    df_candles = get_all_candles(db_path)
    if df_candles.empty:
        return 0
        
    candle_dict = dict(zip(df_candles["timestamp"], df_candles["close"]))
    
    for pred in lines:
        if pred.get("status") == "PENDING":
            target_ts = pred.get("target_timestamp")
            if target_ts in candle_dict:
                actual_close = candle_dict[target_ts]
                pred["actual_close"] = float(actual_close)
                pred["status"] = "RESOLVED"
                
                # Backfill spot_price_at_prediction if it was missing or zero
                base_ts = pred.get("timestamp")
                if base_ts in candle_dict and (pred.get("spot_price_at_prediction", 0.0) == 0.0):
                    pred["spot_price_at_prediction"] = float(candle_dict[base_ts])
                    
                resolved_count += 1
                
        updated_lines.append(pred)
        
    if resolved_count > 0:
        with open(log_path, "w", encoding="utf-8") as f:
            for pred in updated_lines:
                f.write(json.dumps(pred) + "\n")
                
    return resolved_count

def backfill_actuals(db_path: str = DEFAULT_DB_PATH, flat_threshold_pct: float = 0.0001) -> int:
    """
    Compares predictions with actual candles at prediction_time + 5 minutes.
    Updates actual_direction and actual_close in the predictions table.
    FLAT (0) is defined as price change within +/- flat_threshold_pct.
    """
    updated = 0
    with get_connection(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Select predictions that don't have actuals backfilled yet
        cursor.execute("""
            SELECT timestamp FROM predictions 
            WHERE actual_close IS NULL
        """)
        rows = cursor.fetchall()
        
        for row in rows:
            pred_time_str = row["timestamp"]
            try:
                pred_ts = pd.to_datetime(pred_time_str)
                target_ts = pred_ts + pd.Timedelta(minutes=5)
                target_time_str = target_ts.strftime("%Y-%m-%d %H:%M:%S")
                
                cursor.execute("""
                    SELECT c1.close as base_close, c2.close as target_close 
                    FROM candles c1
                    JOIN candles c2 ON c2.timestamp = ?
                    WHERE c1.timestamp = ?
                """, (target_time_str, pred_time_str))
                
                match = cursor.fetchone()
                if match:
                    base_close = match["base_close"]
                    target_close = match["target_close"]
                    
                    price_change = (target_close - base_close) / base_close
                    
                    if price_change > flat_threshold_pct:
                        actual_dir = 1
                    elif price_change < -flat_threshold_pct:
                        actual_dir = -1
                    else:
                        actual_dir = 0
                        
                    cursor.execute("""
                        UPDATE predictions 
                        SET actual_direction = ?, actual_close = ?
                        WHERE timestamp = ?
                    """, (actual_dir, target_close, pred_time_str))
                    updated += 1
            except Exception:
                pass
        conn.commit()
        
    # Synchronize actuals with the JSONL predictions log
    resolve_pending_predictions(db_path, flat_threshold_pct)
    
    return updated

def get_predictions_history(db_path: str = DEFAULT_DB_PATH) -> pd.DataFrame:
    """Returns the historical predictions from JSONL, mapped to the legacy SQL schema for backwards compatibility."""
    log_path = os.path.join(os.path.dirname(db_path), "predictions_log.jsonl")
    
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=[
            "timestamp", "predicted_direction", "confidence", 
            "prob_down", "prob_flat", "prob_up", 
            "actual_direction", "actual_close", "current_close", "status"
        ])
        
    try:
        df = pd.read_json(log_path, lines=True)
        if df.empty:
            return df
            
        df["timestamp"] = df["timestamp"].astype(str)
            
        if "predicted" in df.columns:
            df["predicted_direction"] = df["predicted"].map({"UP": 1, "DOWN": -1, "FLAT": 0})
        else:
            df["predicted_direction"] = 0
            
        if "actual_close" in df.columns and "spot_price_at_prediction" in df.columns:
            actual_dir = []
            for _, row in df.iterrows():
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
            df["actual_direction"] = actual_dir
        else:
            df["actual_direction"] = np.nan
            
        if "spot_price_at_prediction" in df.columns:
            df["current_close"] = df["spot_price_at_prediction"]
            
        df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame(columns=[
            "timestamp", "predicted_direction", "confidence", 
            "prob_down", "prob_flat", "prob_up", 
            "actual_direction", "actual_close", "current_close", "status"
        ])
