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
from src.models.train import (
    train_pipeline, combine_binary_predictions,
    MODEL_UP_PATH, MODEL_DOWN_PATH, MODEL_META_PATH, MODEL_LEGACY_PATH,
    METRICS_SAVE_PATH
)

logger = logging.getLogger("models.predict")

# Configurable thresholds — can be tuned independently per model
DEFAULT_UP_THRESHOLD = 0.50
DEFAULT_DOWN_THRESHOLD = 0.50
DEFAULT_TRUST_THRESHOLD = 0.50


def load_tuned_thresholds() -> tuple:
    """
    Loads auto-tuned decision thresholds from metrics.json.
    Falls back to defaults if the file doesn't exist or is malformed.
    
    Returns:
        Tuple of (up_threshold, down_threshold, meta_trust_threshold).
    """
    if os.path.exists(METRICS_SAVE_PATH):
        try:
            with open(METRICS_SAVE_PATH, "r") as f:
                metrics = json.load(f)
            up_thresh = metrics.get("up_threshold", DEFAULT_UP_THRESHOLD)
            down_thresh = metrics.get("down_threshold", DEFAULT_DOWN_THRESHOLD)
            trust_thresh = metrics.get("meta_trust_threshold", DEFAULT_TRUST_THRESHOLD)
            logger.info(f"Loaded tuned thresholds from metrics.json: UP={up_thresh:.3f}, DOWN={down_thresh:.3f}, TRUST={trust_thresh:.3f}")
            return float(up_thresh), float(down_thresh), float(trust_thresh)
        except Exception as e:
            logger.warning(f"Could not load thresholds from metrics.json: {e}. Using defaults.")
    return DEFAULT_UP_THRESHOLD, DEFAULT_DOWN_THRESHOLD, DEFAULT_TRUST_THRESHOLD


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


def get_trained_models() -> tuple:
    """
    Loads primary binary classifiers (UP-detector and DOWN-detector) and secondary meta-model.
    If models don't exist, triggers the training pipeline first.
    
    Returns:
        Tuple of (model_up, model_down, model_meta).
    """
    if not os.path.exists(MODEL_UP_PATH) or not os.path.exists(MODEL_DOWN_PATH) or not os.path.exists(MODEL_META_PATH):
        logger.warning("Trained model files not found. Running training pipeline...")
        train_pipeline()
    
    with open(MODEL_UP_PATH, "rb") as f:
        model_up = pickle.load(f)
    with open(MODEL_DOWN_PATH, "rb") as f:
        model_down = pickle.load(f)
    with open(MODEL_META_PATH, "rb") as f:
        model_meta = pickle.load(f)
    
    return model_up, model_down, model_meta


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

def predict_latest(
    flat_threshold_pct: float = 0.0001,
    up_threshold: float = None,
    down_threshold: float = None,
    meta_trust_threshold: float = None
) -> Optional[Dict[str, Any]]:
    """
    Builds features on the latest candles, loads primary classifiers and meta-model,
    makes a 5-minute ahead direction prediction, filters it using the meta-model,
    saves it to SQLite and JSONL, and runs the actuals backfill process.
    
    Thresholds default to auto-tuned values from metrics.json if available.
    """
    # Load tuned thresholds if not explicitly provided
    if up_threshold is None or down_threshold is None or meta_trust_threshold is None:
        tuned_up, tuned_down, tuned_trust = load_tuned_thresholds()
        up_threshold = up_threshold if up_threshold is not None else tuned_up
        down_threshold = down_threshold if down_threshold is not None else tuned_down
        meta_trust_threshold = meta_trust_threshold if meta_trust_threshold is not None else tuned_trust
        
    logger.info("Fetching latest cached candles from database...")
    df_raw = get_all_candles()
    
    if len(df_raw) < 100:
        logger.warning(f"Insufficient candle history ({len(df_raw)} candles) to build features. Need at least 100.")
        return None
    
    # Fetch cross-asset data — graceful fallback if unavailable
    df_eurusd = pd.DataFrame()
    df_usdjpy = pd.DataFrame()
    
    try:
        df_eurusd = get_all_candles(table_name="candles_eurusd")
        if not df_eurusd.empty:
            logger.info(f"Loaded {len(df_eurusd)} EURUSD candles for correlation features.")
    except Exception as e:
        logger.warning(f"Failed to load EURUSD candles: {e}. Correlation features will be NaN.")
    
    try:
        df_usdjpy = get_all_candles(table_name="candles_usdjpy")
        if not df_usdjpy.empty:
            logger.info(f"Loaded {len(df_usdjpy)} USDJPY candles for correlation features.")
    except Exception as e:
        logger.warning(f"Failed to load USDJPY candles: {e}. Correlation features will be NaN.")
        
    # Build features (live mode)
    df_features, feature_cols = build_features_df(
        df_raw, is_training=False, flat_threshold_pct=flat_threshold_pct,
        df_eurusd=df_eurusd if not df_eurusd.empty else None,
        df_usdjpy=df_usdjpy if not df_usdjpy.empty else None
    )
    
    if df_features.empty:
        logger.warning("Feature matrix is empty after preprocessing.")
        return None
        
    # Get the latest row for inference
    latest_row = df_features.iloc[-1]
    latest_ts = latest_row["timestamp"]
    
    logger.info(f"Generating prediction for latest candle at timestamp: {latest_ts}")
    
    # Load both binary models and meta model
    try:
        model_up, model_down, model_meta = get_trained_models()
    except Exception as e:
        logger.error(f"Error loading/training models: {e}")
        return None
        
    # Align features with what the model expects from metrics.json to prevent LightGBM shape mismatch
    expected_features = None
    if os.path.exists(METRICS_SAVE_PATH):
        try:
            with open(METRICS_SAVE_PATH, "r") as f:
                metrics_data = json.load(f)
            expected_features = metrics_data.get("feature_columns")
        except Exception as e:
            logger.warning(f"Failed to load expected feature list from metrics: {e}")
            
    if not expected_features:
        logger.warning("Metrics expected features list not found. Falling back to feature builder list.")
        expected_features = feature_cols
    else:
        logger.info(f"Aligning inference vector to expected feature list ({len(expected_features)} columns).")
        
    # Reindex series to expected features: inserts NaN for missing features, drops extra features
    latest_series = latest_row.reindex(expected_features)
    X_inference = latest_series.values.reshape(1, -1)
    
    # Run both models independently
    prob_up = model_up.predict(X_inference)[0]    # P(UP)
    prob_down = model_down.predict(X_inference)[0]  # P(DOWN)
    
    # Apply combination rule
    pred_array, conf_array = combine_binary_predictions(
        np.array([prob_up]), np.array([prob_down]),
        up_threshold=up_threshold, down_threshold=down_threshold
    )
    
    primary_pred_class = int(pred_array[0])
    primary_confidence = float(conf_array[0])
    
    # Run secondary meta-model filter if prediction is directional
    from src.labeling.meta_labeling import build_meta_features, apply_meta_filter
    
    df_features_row = pd.DataFrame([latest_row]).reset_index(drop=True)
    meta_df_row = build_meta_features(
        df_features_row,
        np.array([prob_up]),
        np.array([prob_down]),
        np.array([primary_pred_class])
    )
    
    final_pred_class, meta_confidence = apply_meta_filter(
        primary_pred=primary_pred_class,
        prob_up=prob_up,
        prob_down=prob_down,
        meta_model=model_meta,
        meta_row=meta_df_row.values,
        trust_threshold=meta_trust_threshold
    )
    
    # Derive probability triplet for JSONL schema compatibility.
    prob_flat_synthetic = float(max(0.0, 1.0 - prob_up - prob_down))
    
    # Save the prediction to the database (with meta_confidence passed dynamically)
    probs_tuple = (float(prob_down), float(prob_flat_synthetic), float(prob_up))
    save_prediction(
        latest_ts, final_pred_class, primary_confidence, probs_tuple,
        meta_confidence=meta_confidence
    )
    logger.info(f"Saved prediction for {latest_ts}: Direction={final_pred_class} (Primary={primary_pred_class}), "
                f"Conf={primary_confidence:.2%}, Meta Trust={meta_confidence:.2%}")
    logger.info(f"  P(UP)={prob_up:.3f}, P(DOWN)={prob_down:.3f}, P(FLAT)~={prob_flat_synthetic:.3f}")
    
    # Run backfill of actuals for past predictions in SQLite and JSONL
    backfilled_count = backfill_actuals(db_path=DEFAULT_DB_PATH, flat_threshold_pct=flat_threshold_pct)
    jsonl_resolved = resolve_pending_predictions(db_path=DEFAULT_DB_PATH, flat_threshold_pct=flat_threshold_pct)
    
    if backfilled_count > 0 or jsonl_resolved > 0:
        logger.info(f"Backfilled actual target performance for {backfilled_count} past SQL predictions and {jsonl_resolved} JSONL predictions.")
        
    # Run live regression distribution check
    try:
        check_live_regression(get_log_path())
    except Exception as e:
        logger.warning(f"Error executing live regression checks: {e}")
        
    return {
        "timestamp": latest_ts,
        "prediction": final_pred_class,
        "confidence": primary_confidence,
        "meta_confidence": meta_confidence,
        "prob_down": float(prob_down),
        "prob_flat": float(prob_flat_synthetic),
        "prob_up": float(prob_up),
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
        print(f"Meta Trust Confidence: {res['meta_confidence']:.2%}")
        print(f"Probability Distribution: Down={res['prob_down']:.1%}, Flat={res['prob_flat']:.1%}, Up={res['prob_up']:.1%}")
