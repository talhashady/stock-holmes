"""
Meta-Labeling Module for Stock Holmes.

Implements secondary meta-labeling (López de Prado, 2018) to filter false positives
and improve precision.

The meta-labeling flow:
  1. The primary binary classifiers (UP/DOWN detectors) are run and combined to produce
     a directional prediction (UP, DOWN, or FLAT).
  2. If the primary prediction is directional (UP or DOWN), we construct a binary target
     for the meta-model:
       - 1: Primary directional prediction was CORRECT (realized triple-barrier target matched).
       - 0: Primary prediction was INCORRECT (realized target was FLAT or the opposite direction).
  3. A secondary meta-model is trained on contextual market features and primary model probabilities
     to predict whether the primary model is likely correct.
  4. At inference time, if the meta-model's probability of being correct is below a trust
     threshold, the directional signal is filtered out and overridden to FLAT.
"""

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("labeling.meta_labeling")


def generate_meta_labels(
    primary_preds: np.ndarray,
    actual_labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates binary meta-labels (1 for correct directional predictions, 0 for incorrect).
    
    Only directional predictions (1 or -1) are evaluated. FLAT predictions
    are excluded from meta-model training as there is no active trade signal to filter.
    
    Args:
        primary_preds: 1D array of primary model predictions (-1, 0, or 1).
        actual_labels: 1D array of actual realized triple-barrier targets (-1, 0, or 1).
        
    Returns:
        Tuple of (meta_targets, mask_of_directional_predictions).
    """
    # Create mask for directional signals (1 or -1)
    directional_mask = (primary_preds == 1) | (primary_preds == -1)
    
    # Meta-label is 1 if prediction matches actual realized label, else 0
    meta_labels = (primary_preds == actual_labels).astype(np.int32)
    
    # Filter to only directional predictions
    meta_labels_filtered = meta_labels[directional_mask]
    
    return meta_labels_filtered, directional_mask


def build_meta_features(
    df_features: pd.DataFrame,
    prob_up: np.ndarray,
    prob_down: np.ndarray,
    primary_preds: np.ndarray
) -> pd.DataFrame:
    """
    Constructs the feature matrix for the meta-model.
    
    Includes:
      - Primary model confidence: max(prob_up, prob_down)
      - Probability spread: |prob_up - prob_down|
      - Current volatility regime (vol_regime_class, atr_pct, volatility_5m)
      - Time-of-day encoding (hour_sin, hour_cos)
      - Session flags (session_overlap)
      - Directional bias: primary prediction class
    
    Args:
        df_features: Primary feature DataFrame.
        prob_up: Array of UP-detector probabilities.
        prob_down: Array of DOWN-detector probabilities.
        primary_preds: Array of combined primary predictions (-1, 0, or 1).
        
    Returns:
        DataFrame of meta-features.
    """
    meta_df = pd.DataFrame(index=df_features.index)
    
    # Primary model confidence features
    meta_df["primary_confidence"] = np.maximum(prob_up, prob_down)
    meta_df["primary_prob_spread"] = np.abs(prob_up - prob_down)
    meta_df["primary_pred"] = primary_preds
    
    # Contextual market features from primary features list
    context_cols = [
        "vol_regime_class", "atr_pct", "volatility_5m", "volatility_60m",
        "hour_sin", "hour_cos", "session_overlap", "session_london", "session_ny"
    ]
    
    for col in context_cols:
        if col in df_features.columns:
            meta_df[col] = df_features[col].values
        else:
            # Graceful fallback to zero if features are absent
            meta_df[col] = 0.0
            
    return meta_df


def train_meta_model(
    X_meta_train: pd.DataFrame,
    y_meta_train: np.ndarray
) -> lgb.Booster:
    """
    Trains a lightweight LightGBM binary classifier as the secondary meta-model.
    
    Uses class balancing to handle imbalance in the correct vs incorrect label ratio,
    and a shallow structure to prevent overfitting.
    
    Args:
        X_meta_train: Meta-features DataFrame.
        y_meta_train: Meta-labels binary array (1=correct, 0=incorrect).
        
    Returns:
        Trained LightGBM Booster object.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Compute balanced sample weights
    sw = compute_sample_weight(class_weight="balanced", y=y_meta_train)
    train_data = lgb.Dataset(X_meta_train, label=y_meta_train, weight=sw)
    
    # Shallow parameters for meta-model to prevent overfitting
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "max_depth": 3,
        "num_leaves": 7,
        "min_data_in_leaf": 10,
        "verbosity": -1,
        "seed": 42
    }
    
    logger.info(f"Training meta-model on {len(X_meta_train)} directional events...")
    
    meta_model = lgb.train(
        params,
        train_data,
        num_boost_round=150
    )
    
    return meta_model


def apply_meta_filter(
    primary_pred: int,
    prob_up: float,
    prob_down: float,
    meta_model: lgb.Booster,
    meta_row: np.ndarray,
    trust_threshold: float = 0.50
) -> Tuple[int, float]:
    """
    Applies the meta-labeling filter to a single prediction.
    
    If the primary model predicts FLAT, it is passed through directly.
    If the primary model predicts UP or DOWN, the meta-model estimates the
    probability of the call being correct. If this probability is below the
    trust_threshold, the prediction is overridden to FLAT.
    
    Args:
        primary_pred: Combined prediction from primary models (-1, 0, 1).
        prob_up: Probability of UP.
        prob_down: Probability of DOWN.
        meta_model: Trained meta-model Booster.
        meta_row: 2D NumPy array of shape (1, num_features) for the meta-model.
        trust_threshold: Threshold probability below which we filter/discard the signal.
        
    Returns:
        Tuple of (filtered_prediction, meta_confidence).
    """
    # Meta-model only filters active directional signals
    if primary_pred == 0:
        return 0, 1.0 - max(prob_up, prob_down)
        
    # Get meta-model confidence (probability of being correct)
    meta_confidence = float(meta_model.predict(meta_row)[0])
    
    if meta_confidence >= trust_threshold:
        # Trust and pass through primary prediction
        return primary_pred, meta_confidence
    else:
        # Override to FLAT (filtered out)
        logger.info(f"Meta-model filtered out primary prediction {primary_pred}: "
                    f"confidence={meta_confidence:.2%} < trust_threshold={trust_threshold:.2%}")
        return 0, meta_confidence
