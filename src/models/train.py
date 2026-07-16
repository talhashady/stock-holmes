import os
import logging
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, Any, Tuple

from src.serving.db_utils import get_all_candles
from src.features.builder import build_features_df

logger = logging.getLogger("models.train")

# Model save paths — separate files for UP and DOWN binary detectors
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_UP_PATH = os.path.join(MODEL_DIR, "model_up.pkl")
MODEL_DOWN_PATH = os.path.join(MODEL_DIR, "model_down.pkl")

# Legacy single-model path (kept for backward-compatible detection)
MODEL_LEGACY_PATH = os.path.join(MODEL_DIR, "model.pkl")

METRICS_SAVE_PATH = os.path.join(MODEL_DIR, "metrics.json")


# ---------------------------------------------------------------------------
# Binary Prediction Combination Rule
# ---------------------------------------------------------------------------

def combine_binary_predictions(
    prob_up: np.ndarray,
    prob_down: np.ndarray,
    up_threshold: float = 0.50,
    down_threshold: float = 0.50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Combines outputs from two independent binary classifiers (UP-detector and
    DOWN-detector) into a single 3-class prediction.
    
    Combination rule:
    - UP fires (prob > threshold) AND DOWN does not → predict UP (1)
    - DOWN fires AND UP does not → predict DOWN (-1)
    - Neither fires → predict FLAT (0)
    - Both fire (conflict) → pick whichever has higher raw probability.
      Design choice: resolving via higher confidence rather than defaulting
      to FLAT, because when both models independently detect a directional
      signal, the stronger signal is more likely genuine than noise.
      Alternative (FLAT on conflict) is documented but not used.
    
    Args:
        prob_up: Array of UP-detector probabilities for the positive class.
        prob_down: Array of DOWN-detector probabilities for the positive class.
        up_threshold: Decision threshold for UP-detector (configurable, not hardcoded).
        down_threshold: Decision threshold for DOWN-detector (configurable, not hardcoded).
    
    Returns:
        Tuple of (predicted_classes [-1, 0, 1], confidence_scores [0.0-1.0]).
    """
    n = len(prob_up)
    predictions = np.zeros(n, dtype=int)
    confidences = np.zeros(n, dtype=float)
    
    up_fires = prob_up > up_threshold
    down_fires = prob_down > down_threshold
    
    for i in range(n):
        if up_fires[i] and not down_fires[i]:
            # Clear UP signal
            predictions[i] = 1
            confidences[i] = float(prob_up[i])
        elif down_fires[i] and not up_fires[i]:
            # Clear DOWN signal
            predictions[i] = -1
            confidences[i] = float(prob_down[i])
        elif up_fires[i] and down_fires[i]:
            # Conflict: both fire — resolve by higher confidence
            # Alternative: predictions[i] = 0  (treat as genuine uncertainty)
            if prob_up[i] >= prob_down[i]:
                predictions[i] = 1
                confidences[i] = float(prob_up[i])
            else:
                predictions[i] = -1
                confidences[i] = float(prob_down[i])
        else:
            # Neither fires — predict FLAT
            predictions[i] = 0
            # Confidence for FLAT = certainty of "no signal"
            confidences[i] = 1.0 - max(float(prob_up[i]), float(prob_down[i]))
    
    return predictions, confidences


# ---------------------------------------------------------------------------
# Per-Class Metrics
# ---------------------------------------------------------------------------

def compute_binary_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, 
                           threshold: float, class_name: str) -> Dict[str, Any]:
    """Computes precision, recall, F1 for a binary classifier's positive class."""
    y_pred = (y_pred_prob > threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        f"{class_name}_precision": float(precision),
        f"{class_name}_recall": float(recall),
        f"{class_name}_f1": float(f1),
        f"{class_name}_threshold": float(threshold),
        f"{class_name}_positive_rate": float(np.mean(y_pred)),
    }


def evaluate_combined_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    confidences: np.ndarray
) -> Dict[str, Any]:
    """Computes combined prediction metrics using the binary model ensemble."""
    total = len(y_true)
    if total == 0:
        return {}
        
    accuracy = np.mean(y_true == y_pred)
    
    # Naive baseline: predicts FLAT (0) constantly
    naive_flat_acc = np.mean(y_true == 0)
    
    # High-confidence accuracy (confidence > 0.55)
    high_conf_mask = confidences > 0.55
    high_conf_total = np.sum(high_conf_mask)
    high_conf_acc = np.mean(y_true[high_conf_mask] == y_pred[high_conf_mask]) if high_conf_total > 0 else 0.0
    
    return {
        "accuracy": float(accuracy),
        "naive_flat_accuracy": float(naive_flat_acc),
        "high_confidence_accuracy": float(high_conf_acc),
        "high_confidence_count": int(high_conf_total),
        "total_test_samples": int(total)
    }


# ---------------------------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------------------------

def train_pipeline(
    flat_threshold_pct: float = 0.0001, 
    test_ratio: float = 0.2, 
    val_ratio: float = 0.1, 
    up_threshold: float = 0.50,
    down_threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Runs the full training pipeline with two independent binary classifiers:
    - UP-detector: predicts UP vs NOT-UP (DOWN + FLAT combined)
    - DOWN-detector: predicts DOWN vs NOT-DOWN (UP + FLAT combined)
    
    Each model uses balanced class weights to handle class imbalance.
    Walk-forward chronological splitting prevents lookahead bias.
    
    Args:
        flat_threshold_pct: Price change threshold for FLAT classification.
        test_ratio: Fraction of data held out for testing (chronological).
        val_ratio: Fraction of data used for early-stopping validation.
        up_threshold: Decision threshold for UP-detector.
        down_threshold: Decision threshold for DOWN-detector.
    """
    logger.info("Loading candles from SQLite database...")
    df_raw = get_all_candles()
    
    if len(df_raw) < 500:
        raise ValueError(f"Insufficient candle data in database ({len(df_raw)} rows). Need at least 500 candles to train.")
    
    # Load cross-asset data (graceful fallback if tables don't exist)
    df_eurusd = get_all_candles(table_name="candles_eurusd")
    df_usdjpy = get_all_candles(table_name="candles_usdjpy")
    
    logger.info(f"Cross-asset data: EUR/USD={len(df_eurusd)} candles, USDJPY={len(df_usdjpy)} candles")
    
    logger.info(f"Building features for {len(df_raw)} candles with flat threshold {flat_threshold_pct}...")
    df_features, feature_cols = build_features_df(
        df_raw, is_training=True, flat_threshold_pct=flat_threshold_pct,
        df_eurusd=df_eurusd if not df_eurusd.empty else None,
        df_usdjpy=df_usdjpy if not df_usdjpy.empty else None
    )
    
    # Sort chronologically to prevent leak
    df_features = df_features.sort_values(by="timestamp").reset_index(drop=True)
    
    X = df_features[feature_cols]
    y = df_features["target"]
    
    # Create binary targets for each detector
    y_up = (y == 1).astype(int)      # UP vs NOT-UP
    y_down = (y == -1).astype(int)   # DOWN vs NOT-DOWN
    
    n_samples = len(df_features)
    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)
    train_size = n_samples - test_size - val_size
    
    logger.info(f"Splitting data chronologically: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Purge 5-candle target lookup boundaries to prevent lookahead leakage
    X_train = X.iloc[:train_size - 5]
    X_val = X.iloc[train_size:train_size + val_size - 5]
    X_test = X.iloc[train_size + val_size:]
    
    y_up_train = y_up.iloc[:train_size - 5]
    y_up_val = y_up.iloc[train_size:train_size + val_size - 5]
    y_up_test = y_up.iloc[train_size + val_size:]
    
    y_down_train = y_down.iloc[:train_size - 5]
    y_down_val = y_down.iloc[train_size:train_size + val_size - 5]
    y_down_test = y_down.iloc[train_size + val_size:]
    
    # Original 3-class test targets for combined evaluation
    y_test_orig = y.iloc[train_size + val_size:].values
    
    # LightGBM parameters — binary classification
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
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
    
    metrics = {}
    models = {}
    
    # --- Train UP-detector ---
    logger.info("Training UP-detector (UP vs NOT-UP)...")
    from sklearn.utils.class_weight import compute_sample_weight
    
    sw_up = compute_sample_weight(class_weight="balanced", y=y_up_train)
    train_data_up = lgb.Dataset(X_train, label=y_up_train, weight=sw_up)
    val_data_up = lgb.Dataset(X_val, label=y_up_val, reference=train_data_up)
    
    model_up = lgb.train(
        params,
        train_data_up,
        num_boost_round=1000,
        valid_sets=[train_data_up, val_data_up],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    models["up"] = model_up
    
    # --- Train DOWN-detector ---
    logger.info("Training DOWN-detector (DOWN vs NOT-DOWN)...")
    
    sw_down = compute_sample_weight(class_weight="balanced", y=y_down_train)
    train_data_down = lgb.Dataset(X_train, label=y_down_train, weight=sw_down)
    val_data_down = lgb.Dataset(X_val, label=y_down_val, reference=train_data_down)
    
    model_down = lgb.train(
        params,
        train_data_down,
        num_boost_round=1000,
        valid_sets=[train_data_down, val_data_down],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)]
    )
    models["down"] = model_down
    
    # --- Auto-tune thresholds on validation set ---
    # The balanced class weights make both models fire aggressively at 0.5.
    # We jointly optimize both thresholds on the validation set by maximizing
    # combined 3-class accuracy — this naturally balances the class distribution
    # because accuracy peaks when predictions match the true distribution.
    logger.info("Auto-tuning decision thresholds on validation set...")
    
    val_prob_up_raw = model_up.predict(X_val)
    val_prob_down_raw = model_down.predict(X_val)
    y_val_orig = y.iloc[train_size:train_size + val_size - 5].values
    
    best_acc = 0.0
    best_up_thresh = up_threshold
    best_down_thresh = down_threshold
    
    for ut in np.arange(0.40, 0.72, 0.02):
        for dt in np.arange(0.40, 0.72, 0.02):
            val_preds, _ = combine_binary_predictions(
                val_prob_up_raw, val_prob_down_raw,
                up_threshold=ut, down_threshold=dt
            )
            acc = np.mean(val_preds == y_val_orig)
            if acc > best_acc:
                best_acc = acc
                best_up_thresh = float(ut)
                best_down_thresh = float(dt)
    
    up_threshold = best_up_thresh
    down_threshold = best_down_thresh
    logger.info(f"Optimal thresholds — UP: {up_threshold:.3f}, DOWN: {down_threshold:.3f} "
                f"(val accuracy: {best_acc:.4f})")
    
    # --- Predict on test set with tuned thresholds ---
    prob_up_test = model_up.predict(X_test)   # P(UP)
    prob_down_test = model_down.predict(X_test)  # P(DOWN)
    
    preds_class, confidences = combine_binary_predictions(
        prob_up_test, prob_down_test,
        up_threshold=up_threshold, down_threshold=down_threshold
    )
    
    # --- Per-model binary metrics ---
    up_metrics = compute_binary_metrics(y_up_test.values, prob_up_test, up_threshold, "up")
    down_metrics = compute_binary_metrics(y_down_test.values, prob_down_test, down_threshold, "down")
    metrics.update(up_metrics)
    metrics.update(down_metrics)
    
    logger.info(f"UP-detector: P={up_metrics['up_precision']:.3f}, R={up_metrics['up_recall']:.3f}, F1={up_metrics['up_f1']:.3f}")
    logger.info(f"DOWN-detector: P={down_metrics['down_precision']:.3f}, R={down_metrics['down_recall']:.3f}, F1={down_metrics['down_f1']:.3f}")
    
    # --- Combined evaluation ---
    combined_metrics = evaluate_combined_predictions(y_test_orig, preds_class, confidences)
    metrics.update(combined_metrics)
    
    # Naive baselines
    test_counts = pd.Series(y_test_orig).value_counts()
    majority_test_class = test_counts.idxmax() if not test_counts.empty else 0
    naive_majority_acc = np.mean(y_test_orig == majority_test_class)
    metrics["naive_majority_accuracy"] = float(naive_majority_acc)
    
    logger.info(f"Combined Test Accuracy: {metrics.get('accuracy', 0.0):.4f} (Naive Majority: {naive_majority_acc:.4f})")
    
    # --- Class distribution check ---
    pred_counts = pd.Series(preds_class).value_counts(normalize=True)
    
    majority_pred_pct = pred_counts.max() if not pred_counts.empty else 0.0
    majority_pred_class = pred_counts.idxmax() if not pred_counts.empty else None
    
    metrics["test_class_distribution"] = {str(int(k)): float(v) for k, v in pred_counts.items()}
    metrics["model_collapsed"] = bool(majority_pred_pct > 0.90)
    metrics["up_best_iteration"] = int(model_up.best_iteration)
    metrics["down_best_iteration"] = int(model_down.best_iteration)
    metrics["up_num_trees"] = int(model_up.num_trees())
    metrics["down_num_trees"] = int(model_down.num_trees())
    metrics["up_threshold"] = float(up_threshold)
    metrics["down_threshold"] = float(down_threshold)
    
    if metrics["model_collapsed"]:
        logger.warning(f"⚠️ REGRESSION WARNING: Combined prediction distribution collapsed! "
                      f"Class {majority_pred_class} accounts for {majority_pred_pct:.1%}")
    else:
        logger.info(f"Distribution check passed: {dict(pred_counts.items())}")
    
    # --- Naive carry-forward baseline ---
    last_1m_ret = X_test["return_1m"].values
    naive_sign_preds = np.zeros_like(last_1m_ret)
    naive_sign_preds[last_1m_ret > flat_threshold_pct] = 1
    naive_sign_preds[last_1m_ret < -flat_threshold_pct] = -1
    naive_sign_acc = np.mean(y_test_orig == naive_sign_preds)
    metrics["naive_sign_accuracy"] = float(naive_sign_acc)
    
    logger.info(f"Naive Return-Sign Baseline: {naive_sign_acc:.4f}")
    
    # --- Feature importance ---
    for model_name, model in models.items():
        importance = model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
        metrics[f"{model_name}_feature_importance"] = {f: float(v) for f, v in feat_imp}
        logger.info(f"{model_name.upper()}-detector top 5 features: {feat_imp[:5]}")
    
    # --- Save models ---
    with open(MODEL_UP_PATH, "wb") as f:
        pickle.dump(model_up, f)
    logger.info(f"UP-detector saved to {MODEL_UP_PATH}")
    
    with open(MODEL_DOWN_PATH, "wb") as f:
        pickle.dump(model_down, f)
    logger.info(f"DOWN-detector saved to {MODEL_DOWN_PATH}")
    
    # --- Validation set distribution (regression check) ---
    val_prob_up = model_up.predict(X_val)
    val_prob_down = model_down.predict(X_val)
    val_preds, _ = combine_binary_predictions(val_prob_up, val_prob_down, up_threshold, down_threshold)
    val_counts = pd.Series(val_preds).value_counts(normalize=True)
    metrics["validation_class_distribution"] = {str(int(k)): float(v) for k, v in val_counts.items()}
    
    # --- Backtest equity curve ---
    test_timestamps = df_features["timestamp"].iloc[train_size + val_size:].values
    target_returns = df_features["target_return"].iloc[train_size + val_size:].values
    
    strategy_rets = preds_class * target_returns
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
    
    # --- Feature columns list (needed for inference to know what the model expects) ---
    metrics["feature_columns"] = feature_cols
    
    # --- Save metrics ---
    import json
    with open(METRICS_SAVE_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Evaluation metrics saved to {METRICS_SAVE_PATH}")
    
    return metrics

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_pipeline()
