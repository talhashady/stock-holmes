import os
import logging
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any

from src.serving.db_utils import get_all_candles
from src.features.builder import build_features_df

logger = logging.getLogger("models.train")

MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.pkl"
)

METRICS_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "metrics.json"
)

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, Any]:
    """Computes prediction metrics for model evaluations."""
    total = len(y_true)
    if total == 0:
        return {}
        
    accuracy = np.mean(y_true == y_pred)
    
    # Naive baseline: predicts FLAT (0) constantly
    naive_flat_acc = np.mean(y_true == 0)
    
    # Naive baseline: predicting direction of the last 1-minute bar return
    # Since y_true is mapped index-wise, we evaluate this separately
    
    # Compute accuracy for non-flat calls (confidence filter)
    # E.g., confidence thresholding: max probability > 0.45
    high_conf_mask = np.max(y_pred_prob, axis=1) > 0.45
    high_conf_total = np.sum(high_conf_mask)
    high_conf_acc = np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask]) if high_conf_total > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "naive_flat_accuracy": float(naive_flat_acc),
        "high_confidence_accuracy": float(high_conf_acc),
        "high_confidence_count": int(high_conf_total),
        "total_test_samples": int(total)
    }

def train_pipeline(flat_threshold_pct: float = 0.0001, test_ratio: float = 0.2, val_ratio: float = 0.1) -> Dict[str, Any]:
    """Runs data loading, feature building, walk-forward splitting, training, and evaluation."""
    logger.info("Loading candles from SQLite database...")
    df_raw = get_all_candles()
    
    if len(df_raw) < 500:
        raise ValueError(f"Insufficient candle data in database ({len(df_raw)} rows). Need at least 500 candles to train.")
        
    logger.info(f"Building features for {len(df_raw)} candles...")
    df_features, feature_cols = build_features_df(df_raw, is_training=True, flat_threshold_pct=flat_threshold_pct)
    
    # Sort chronologically to prevent leak
    df_features = df_features.sort_values(by="timestamp").reset_index(drop=True)
    
    X = df_features[feature_cols]
    y = df_features["target"]
    
    # Shift target from [-1, 0, 1] to [0, 1, 2] for LightGBM multiclass handling
    # -1 (DOWN) -> 0
    # 0 (FLAT) -> 1
    # 1 (UP) -> 2
    y_shifted = y + 1
    
    n_samples = len(df_features)
    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)
    train_size = n_samples - test_size - val_size
    
    logger.info(f"Splitting data chronologically: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Purge 5-candle target lookup boundaries to prevent lookahead leakage
    X_train, y_train = X.iloc[:train_size - 5], y_shifted.iloc[:train_size - 5]
    X_val, y_val = X.iloc[train_size:train_size+val_size - 5], y_shifted.iloc[train_size:train_size+val_size - 5]
    X_test, y_test = X.iloc[train_size+val_size:], y_shifted.iloc[train_size+val_size:]
    
    # Map back to original targets for evaluation
    y_test_orig = y_test.values - 1
    
    # Setup LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.02,
        "max_depth": 4,
        "num_leaves": 15,
        "min_data_in_leaf": 20,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42
    }
    
    logger.info("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
    )
    
    # Predict on test set
    preds_prob = model.predict(X_test)  # (N, 3) matrix of probabilities
    preds_class = np.argmax(preds_prob, axis=1) - 1 # Map back to [-1, 0, 1]
    
    # Evaluate
    metrics = evaluate_predictions(y_test_orig, preds_class, preds_prob)
    logger.info(f"Test Accuracy: {metrics.get('accuracy', 0.0):.4f} (Naive Flat: {metrics.get('naive_flat_accuracy', 0.0):.4f})")
    
    # Naive carry-forward return-sign baseline evaluation
    # Checks if next 5-min return direction matches last 1-min return direction
    last_1m_ret = X_test["return_1m"].values
    naive_sign_preds = np.zeros_like(last_1m_ret)
    naive_sign_preds[last_1m_ret > flat_threshold_pct] = 1
    naive_sign_preds[last_1m_ret < -flat_threshold_pct] = -1
    naive_sign_acc = np.mean(y_test_orig == naive_sign_preds)
    metrics["naive_sign_accuracy"] = float(naive_sign_acc)
    
    logger.info(f"Naive Return-Sign Baseline Accuracy: {naive_sign_acc:.4f}")
    
    # Save the model
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Construct cumulative equity curve metrics for backtest plotting
    # Assume 1 USD traded per trade: profit/loss = target_return * predicted_direction
    test_timestamps = df_features["timestamp"].iloc[train_size+val_size:].values
    target_returns = df_features["target_return"].iloc[train_size+val_size:].values
    
    # Strategy returns:
    # If prediction is 1, long (+1 * return)
    # If prediction is -1, short (-1 * return)
    # If prediction is 0, flat (0 * return)
    strategy_rets = preds_class * target_returns
    
    # Baseline signals (carry forward last sign)
    baseline_rets = naive_sign_preds * target_returns
    
    cumulative_strategy = np.cumsum(strategy_rets)
    cumulative_baseline = np.cumsum(baseline_rets)
    
    metrics["backtest"] = {
        "timestamps": test_timestamps.tolist(),
        "model_cumulative_returns": cumulative_strategy.tolist(),
        "baseline_cumulative_returns": cumulative_baseline.tolist(),
        "model_signals": preds_class.tolist(),
        "actual_returns": target_returns.tolist()
    }
    
    # Save metrics
    import json
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to {METRICS_SAVE_PATH}")
    
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_pipeline()
