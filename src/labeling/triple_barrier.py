"""
Triple-Barrier Labeling for Stock Holmes.

Implements the triple-barrier method (López de Prado, 2018) for generating
path-dependent, volatility-scaled labels for financial time-series classification.

Unlike fixed-horizon labeling (which only looks at the price at t+N), triple-barrier
labeling defines three barriers for each observation:
  - Upper barrier (profit-taking): entry_price + pt_multiplier * ATR
  - Lower barrier (stop-loss):     entry_price - sl_multiplier * ATR
  - Vertical barrier (time limit): a fixed number of bars ahead

The label is determined by WHICH barrier is touched FIRST:
  - Upper → UP (1):   price rose enough to hit the profit-taking level
  - Lower → DOWN (-1): price fell enough to hit the stop-loss level
  - Vertical → FLAT (0): neither price barrier was touched within the time limit

This produces labels that are:
  1. Volatility-scaled: barriers widen during high-ATR periods and tighten during
     calm periods, so the same "significance" threshold adapts to market conditions.
  2. Path-dependent: if price spikes up then crashes, the label captures the spike
     (upper barrier hit first), whereas fixed-horizon labeling would only see the crash.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional, NamedTuple

logger = logging.getLogger("labeling.triple_barrier")


class BarrierEvent(NamedTuple):
    """Result of a single triple-barrier label computation.
    
    Attributes:
        label: 1 (UP, upper barrier hit), -1 (DOWN, lower barrier hit), 0 (FLAT, vertical barrier hit).
        t_touch_idx: The absolute DataFrame index at which the barrier was touched.
        holding_bars: Number of bars from entry to barrier touch.
        barrier_type: "upper", "lower", or "vertical".
    """
    label: int
    t_touch_idx: int
    holding_bars: int
    barrier_type: str


def get_triple_barrier_label(
    close_prices: np.ndarray,
    atr_values: np.ndarray,
    t_start: int,
    pt_multiplier: float = 1.0,
    sl_multiplier: float = 1.0,
    max_holding_bars: int = 5
) -> BarrierEvent:
    """
    Computes the triple-barrier label for a single observation at index t_start.
    
    Scans forward bar-by-bar from t_start+1 to t_start+max_holding_bars, checking
    whether price crosses the upper or lower barrier. If neither is crossed by the
    time the vertical barrier (max_holding_bars) is reached, the label is FLAT.
    
    Args:
        close_prices: 1D NumPy array of close prices (full dataset).
        atr_values: 1D NumPy array of ATR values (full dataset, same length as close_prices).
        t_start: Index of the current observation (entry bar).
        pt_multiplier: Profit-taking barrier width in ATR units.
        sl_multiplier: Stop-loss barrier width in ATR units.
        max_holding_bars: Maximum number of bars to hold before the vertical barrier.
    
    Returns:
        BarrierEvent with the label, touch index, holding bars, and barrier type.
    """
    n = len(close_prices)
    entry_price = close_prices[t_start]
    atr_at_entry = atr_values[t_start]
    
    # Guard: if ATR is NaN or zero, we cannot define meaningful barriers.
    # Fall back to FLAT at the vertical barrier.
    if np.isnan(atr_at_entry) or atr_at_entry <= 0:
        t_end = min(t_start + max_holding_bars, n - 1)
        return BarrierEvent(label=0, t_touch_idx=t_end, holding_bars=t_end - t_start, barrier_type="vertical")
    
    upper_barrier = entry_price + pt_multiplier * atr_at_entry
    lower_barrier = entry_price - sl_multiplier * atr_at_entry
    
    # Scan forward bar-by-bar
    for offset in range(1, max_holding_bars + 1):
        t_check = t_start + offset
        if t_check >= n:
            # Ran out of data before reaching the vertical barrier
            return BarrierEvent(label=0, t_touch_idx=n - 1, holding_bars=n - 1 - t_start, barrier_type="vertical")
        
        price = close_prices[t_check]
        
        # Check upper barrier first (profit-taking)
        if price >= upper_barrier:
            return BarrierEvent(label=1, t_touch_idx=t_check, holding_bars=offset, barrier_type="upper")
        
        # Check lower barrier (stop-loss)
        if price <= lower_barrier:
            return BarrierEvent(label=-1, t_touch_idx=t_check, holding_bars=offset, barrier_type="lower")
    
    # Vertical barrier hit — neither price barrier was touched
    t_end = min(t_start + max_holding_bars, n - 1)
    return BarrierEvent(label=0, t_touch_idx=t_end, holding_bars=max_holding_bars, barrier_type="vertical")


def apply_triple_barrier_labels(
    df: pd.DataFrame,
    pt_multiplier: float = 1.0,
    sl_multiplier: float = 1.0,
    max_holding_bars: int = 5,
    atr_column: str = "atr",
    close_column: str = "close"
) -> pd.DataFrame:
    """
    Applies the triple-barrier labeling method across the full historical dataset.
    
    Performance approach:
      The inner scan per observation is O(max_holding_bars) — typically 5 comparisons.
      The outer loop is O(N). With N≈7,000 and max_holding_bars=5, this totals ~35,000
      comparisons, completing in <50ms in pure Python on NumPy arrays. No parallelization
      is needed for this dataset size.
    
    Args:
        df: DataFrame with at minimum 'close' and 'atr' columns.
        pt_multiplier: Profit-taking barrier width in ATR units.
        sl_multiplier: Stop-loss barrier width in ATR units.
        max_holding_bars: Maximum number of bars to hold before the vertical barrier.
        atr_column: Name of the ATR column in df.
        close_column: Name of the close price column in df.
    
    Returns:
        df augmented with columns:
          - 'tb_label': the triple-barrier label (1, -1, or 0)
          - 'tb_t_touch_idx': the absolute index where the barrier was touched
          - 'tb_holding_bars': number of bars from entry to barrier touch
          - 'tb_barrier_type': "upper", "lower", or "vertical"
    """
    if atr_column not in df.columns:
        raise ValueError(f"ATR column '{atr_column}' not found in DataFrame. "
                         f"Compute ATR first using compute_atr().")
    if close_column not in df.columns:
        raise ValueError(f"Close column '{close_column}' not found in DataFrame.")
    
    # Extract raw NumPy arrays for speed — avoids Pandas overhead in the inner loop
    close_arr = df[close_column].values.astype(np.float64)
    atr_arr = df[atr_column].values.astype(np.float64)
    n = len(df)
    
    # Pre-allocate output arrays
    labels = np.zeros(n, dtype=np.int32)
    touch_indices = np.zeros(n, dtype=np.int32)
    holding_bars_arr = np.zeros(n, dtype=np.int32)
    barrier_types = np.empty(n, dtype=object)
    
    # The last max_holding_bars rows cannot have their barriers fully resolved
    # because there isn't enough forward data. We'll mark them as NaN later.
    valid_end = n - max_holding_bars
    
    for i in range(n):
        if i >= valid_end:
            # Not enough forward data to resolve barriers
            labels[i] = 0
            touch_indices[i] = min(i + max_holding_bars, n - 1)
            holding_bars_arr[i] = min(max_holding_bars, n - 1 - i)
            barrier_types[i] = "insufficient_data"
        else:
            event = get_triple_barrier_label(
                close_arr, atr_arr, i,
                pt_multiplier=pt_multiplier,
                sl_multiplier=sl_multiplier,
                max_holding_bars=max_holding_bars
            )
            labels[i] = event.label
            touch_indices[i] = event.t_touch_idx
            holding_bars_arr[i] = event.holding_bars
            barrier_types[i] = event.barrier_type
    
    df = df.copy()
    df["tb_label"] = labels
    df["tb_t_touch_idx"] = touch_indices
    df["tb_holding_bars"] = holding_bars_arr
    df["tb_barrier_type"] = barrier_types
    
    # Mark rows with insufficient forward data as NaN labels (excluded from training)
    df.loc[df["tb_barrier_type"] == "insufficient_data", "tb_label"] = np.nan
    
    logger.info(f"Triple-barrier labels applied: pt={pt_multiplier}, sl={sl_multiplier}, "
                f"max_bars={max_holding_bars}, dataset_size={n}")
    
    return df


def validate_triple_barrier_labels(
    df: pd.DataFrame,
    old_labels: Optional[pd.Series] = None,
    label_column: str = "tb_label"
) -> dict:
    """
    Diagnostic function that reports the triple-barrier label distribution,
    average holding time, and barrier type breakdown.
    
    Args:
        df: DataFrame with triple-barrier columns.
        old_labels: Optional Series of old fixed-horizon labels for comparison.
        label_column: Column name containing the triple-barrier labels.
    
    Returns:
        Dictionary with diagnostic statistics.
    """
    # Filter to valid labels only (exclude insufficient_data rows)
    valid = df[df["tb_barrier_type"] != "insufficient_data"].copy()
    
    if valid.empty:
        logger.warning("No valid triple-barrier labels found.")
        return {}
        
    # Use label_column, or fallback to 'target' if it is not present
    if label_column not in valid.columns and "target" in valid.columns:
        label_column = "target"
        
    if label_column not in valid.columns:
        logger.warning(f"Label column '{label_column}' not found in DataFrame.")
        return {}
    
    # Class distribution
    label_counts = valid[label_column].value_counts()
    label_pcts = valid[label_column].value_counts(normalize=True)
    
    logger.info("=" * 60)
    logger.info("TRIPLE-BARRIER LABEL DISTRIBUTION")
    logger.info("=" * 60)
    
    label_names = {1: "UP", -1: "DOWN", 0: "FLAT"}
    for label_val in [1, -1, 0]:
        count = label_counts.get(label_val, 0)
        pct = label_pcts.get(label_val, 0.0)
        logger.info(f"  {label_names[label_val]:>5}: {count:>5} ({pct:.1%})")
    
    # Average holding time per class
    logger.info("\nAVERAGE HOLDING TIME (bars):")
    for label_val in [1, -1, 0]:
        subset = valid[valid[label_column] == label_val]
        if not subset.empty:
            avg_hold = subset["tb_holding_bars"].mean()
            logger.info(f"  {label_names[label_val]:>5}: {avg_hold:.2f} bars")
    
    overall_avg_hold = valid["tb_holding_bars"].mean()
    logger.info(f"  Overall: {overall_avg_hold:.2f} bars")
    
    # Barrier type breakdown
    barrier_counts = valid["tb_barrier_type"].value_counts()
    barrier_pcts = valid["tb_barrier_type"].value_counts(normalize=True)
    
    logger.info("\nBARRIER TYPE BREAKDOWN:")
    for bt in ["upper", "lower", "vertical"]:
        count = barrier_counts.get(bt, 0)
        pct = barrier_pcts.get(bt, 0.0)
        logger.info(f"  {bt:>8}: {count:>5} ({pct:.1%})")
    
    # Comparison with old labels if provided
    if old_labels is not None:
        old_counts = old_labels.value_counts()
        old_pcts = old_labels.value_counts(normalize=True)
        
        logger.info("\nCOMPARISON WITH OLD FIXED-HORIZON LABELS:")
        for label_val in [1, -1, 0]:
            old_pct = old_pcts.get(label_val, 0.0)
            new_pct = label_pcts.get(label_val, 0.0)
            diff = new_pct - old_pct
            logger.info(f"  {label_names[label_val]:>5}: {old_pct:.1%} → {new_pct:.1%} ({diff:+.1%})")
    
    stats = {
        "label_distribution": {label_names[k]: int(v) for k, v in label_counts.items()},
        "label_percentages": {label_names[k]: float(v) for k, v in label_pcts.items()},
        "avg_holding_bars": float(overall_avg_hold),
        "avg_holding_bars_per_class": {
            label_names[lv]: float(valid[valid[label_column] == lv]["tb_holding_bars"].mean())
            for lv in [1, -1, 0] if not valid[valid[label_column] == lv].empty
        },
        "barrier_type_distribution": {str(k): int(v) for k, v in barrier_counts.items()},
        "barrier_type_percentages": {str(k): float(v) for k, v in barrier_pcts.items()},
        "total_valid_labels": int(len(valid)),
    }
    
    logger.info("=" * 60)
    
    return stats
