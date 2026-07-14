import os
import sys
import logging
import pickle
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, Optional

from src.serving.db_utils import (
    DEFAULT_DB_PATH,
    get_all_candles,
    save_prediction,
    backfill_actuals,
    resolve_pending_predictions
)
from src.features.builder import build_features_df
from src.models.train import train_pipeline

logger = logging.getLogger("models.predict")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.pkl"
)

def get_log_path() -> str:
    return os.path.join(os.path.dirname(DEFAULT_DB_PATH), "predictions_log.jsonl")

def append_prediction(prediction: dict, path: Optional[str] = None) -> None:
    """Appends a new prediction to the JSONL log file, creating parent directories if needed."""
    if path is None:
        path = get_log_path()
    db_dir = os.path.dirname(path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
        
    # Ensure UTC timestamp is set in prediction if not present
    if "timestamp" not in prediction:
        from datetime import datetime, timezone
        prediction["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        
    # Append as a single JSON line
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(prediction) + "\n")


def get_trained_model():
    """Loads the model, training it if it does not exist."""
    if not os.path.exists(MODEL_PATH):
        logger.warning("Trained model file not found. Running training pipeline...")
        train_pipeline()
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def check_live_regression(log_path: str, window_size: int = 30, threshold: float = 0.90) -> None:
    """Reads the latest predictions from predictions_log.jsonl and checks for model collapse."""
    if not os.path.exists(log_path):
        return
        
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                try:
                    records.append(json.loads(line_str))
                except Exception:
                    pass
                    
    if len(records) < 10:
        # Not enough history to check collapse yet
        return
        
    # Get last window_size predictions
    recent_records = records[-window_size:]
    recent_preds = [r.get("predicted") for r in recent_records if r.get("predicted")]
    
    if not recent_preds:
        return
        
    total = len(recent_preds)
    counts = {}
    for pred in recent_preds:
        counts[pred] = counts.get(pred, 0) + 1
        
    logger.info(f"Checking rolling predictions distribution over last {total} records: {counts}")
    
    warnings_path = os.path.join(os.path.dirname(log_path), "warnings.log")
    
    majority_class = None
    majority_pct = 0.0
    for pred, count in counts.items():
        pct = count / total
        if pct > majority_pct:
            majority_pct = pct
            majority_class = pred
            
    if majority_pct >= threshold:
        msg = f"[CRITICAL MODEL COLLAPSE] Over the last {total} predictions, {majority_class} accounts for {majority_pct:.1%} of all outcomes. Triggering regression warning."
        logger.error(msg)
        
        # Write to warnings.log
        with open(warnings_path, "a", encoding="utf-8") as wf:
            wf.write(msg + "\n")
            
        sys.stderr.write(msg + "\n")
    else:
        # Clean up warnings.log if model is healthy
        if os.path.exists(warnings_path):
            try:
                os.remove(warnings_path)
            except Exception:
                pass

def predict_latest(flat_threshold_pct: float = 0.0001) -> Optional[Dict[str, Any]]:
    """
    Builds features on the latest candles, loads the trained model,
    makes a 5-minute ahead direction prediction, saves it to SQLite,
    and runs the actuals backfill process.
    """
    logger.info("Fetching latest cached candles from database...")
    df_raw = get_all_candles()
    
    if len(df_raw) < 100:
        logger.warning(f"Insufficient candle history ({len(df_raw)} candles) to build features. Need at least 100.")
        return None
        
    # Build features (live mode)
    df_features, feature_cols = build_features_df(df_raw, is_training=False, flat_threshold_pct=flat_threshold_pct)
    
    if df_features.empty:
        logger.warning("Feature matrix is empty after preprocessing.")
        return None
        
    # Get the latest row for inference
    latest_row = df_features.iloc[-1]
    latest_ts = latest_row["timestamp"]
    
    logger.info(f"Generating prediction for latest candle at timestamp: {latest_ts}")
    
    # Load model
    try:
        model = get_trained_model()
    except Exception as e:
        logger.error(f"Error loading/training model: {e}")
        return None
        
    # Extract feature vector
    X_inference = latest_row[feature_cols].values.reshape(1, -1)
    
    # Model returns shape (1, 3) representing probabilities for [DOWN, FLAT, UP]
    probs = model.predict(X_inference)[0]
    
    # Predicted class: 0 (DOWN), 1 (FLAT), 2 (UP)
    pred_shifted = int(np.argmax(probs))
    pred_class = pred_shifted - 1 # Map back to [-1, 0, 1]
    
    confidence = float(probs[pred_shifted])
    
    # Save the prediction to the database
    probs_tuple = (float(probs[0]), float(probs[1]), float(probs[2]))
    save_prediction(latest_ts, pred_class, confidence, probs_tuple)
    logger.info(f"Saved prediction for {latest_ts}: Direction={pred_class}, Conf={confidence:.2%}")
    
    # Run backfill of actuals for past predictions in SQLite and JSONL
    backfilled_count = backfill_actuals(db_path=DEFAULT_DB_PATH, flat_threshold_pct=flat_threshold_pct)
    jsonl_resolved = resolve_pending_predictions(db_path=DEFAULT_DB_PATH, flat_threshold_pct=flat_threshold_pct)
    
    if backfilled_count > 0 or jsonl_resolved > 0:
        logger.info(f"Backfilled actual target performance for {backfilled_count} past SQL predictions and {jsonl_resolved} JSONL predictions (aligned to DB path).")
        
    # Run live regression distribution check
    try:
        check_live_regression(get_log_path())
    except Exception as e:
        logger.warning(f"Error executing live regression checks: {e}")
        
    return {
        "timestamp": latest_ts,
        "prediction": pred_class,
        "confidence": confidence,
        "prob_down": probs[0],
        "prob_flat": probs[1],
        "prob_up": probs[2],
        "close_price": float(latest_row["close"])
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    res = predict_latest()
    if res:
        print("\n--- Latest Prediction ---")
        print(f"Time: {res['timestamp']}")
        print(f"Close Price: {res['close_price']:.2f}")
        dir_str = "UP" if res['prediction'] == 1 else "DOWN" if res['prediction'] == -1 else "FLAT"
        print(f"Predicted direction (5 mins ahead): {dir_str} (Confidence: {res['confidence']:.2%})")
        print(f"Probability Distribution: Down={res['prob_down']:.1%}, Flat={res['prob_flat']:.1%}, Up={res['prob_up']:.1%}")
