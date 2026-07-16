import numpy as np
import pandas as pd
from typing import Tuple, Optional
import logging

logger = logging.getLogger("features.builder")


def compute_log_returns(df: pd.DataFrame, periods: list) -> pd.DataFrame:
    """Computes log returns for specified periods."""
    df = df.copy()
    log_close = np.log(df["close"])
    for p in periods:
        df[f"return_{p}m"] = log_close.diff(p)
    return df

def compute_realized_volatility(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Computes rolling standard deviation of 1-minute log returns over different windows."""
    df = df.copy()
    # 1-minute return as baseline for volatility calculations
    ret_1m = np.log(df["close"]).diff(1)
    for w in windows:
        df[f"volatility_{w}m"] = ret_1m.rolling(window=w).std()
    return df

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Computes the Average True Range (ATR) normalized by close price."""
    df = df.copy()
    high_low = df["high"] - df["low"]
    high_pc = (df["high"] - df["close"].shift(1)).abs()
    low_pc = (df["low"] - df["close"].shift(1)).abs()
    
    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    # Simple rolling mean of True Range
    df["atr"] = tr.rolling(window=period).mean()
    # Normalize ATR to be unitless (percentage of close price)
    df["atr_pct"] = df["atr"] / df["close"]
    return df

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Computes the Relative Strength Index (RSI)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Computes MACD, Signal Line, and MACD Histogram normalized by close."""
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    
    # Normalize by close to ensure stationary features
    df["macd"] = macd_line / df["close"]
    df["macd_signal"] = signal_line / df["close"]
    df["macd_hist"] = macd_hist / df["close"]
    return df

def compute_ema_ratios(df: pd.DataFrame, spans: list) -> pd.DataFrame:
    """Computes close price divided by EMA to measure deviations."""
    df = df.copy()
    for span in spans:
        ema = df["close"].ewm(span=span, adjust=False).mean()
        df[f"ema_ratio_{span}"] = df["close"] / ema
    return df

def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Computes Bollinger Band distance (z-score from rolling mean)."""
    df = df.copy()
    rolling_mean = df["close"].rolling(window=period).mean()
    rolling_std = df["close"].rolling(window=period).std()
    
    # Z-score of price relative to BB
    df["bb_zscore"] = (df["close"] - rolling_mean) / (rolling_std + 1e-9)
    return df

def add_session_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds binary flags for active trading sessions (based on UTC/GMT hour).
    - London open: 08:00 - 16:00 UTC
    - New York open: 13:00 - 21:00 UTC
    - London/NY Overlap: 13:00 - 16:00 UTC
    """
    df = df.copy()
    timestamps = pd.to_datetime(df["timestamp"])
    hours = timestamps.dt.hour
    
    df["session_london"] = ((hours >= 8) & (hours < 16)).astype(int)
    df["session_ny"] = ((hours >= 13) & (hours < 21)).astype(int)
    df["session_overlap"] = ((hours >= 13) & (hours < 16)).astype(int)
    
    # Sine/Cosine hour encoding for cyclic time
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    
    return df


# ---------------------------------------------------------------------------
# Step 1: Cross-Asset Correlation Features
# ---------------------------------------------------------------------------

def compute_cross_asset_correlations(
    df_xau: pd.DataFrame,
    df_eurusd: Optional[pd.DataFrame] = None,
    df_usdjpy: Optional[pd.DataFrame] = None,
    window: int = 60
) -> pd.DataFrame:
    """
    Computes rolling Pearson correlations between XAUUSD returns and
    cross-asset (EUR/USD, USDJPY) returns over a specified window.
    
    Gold typically has an inverse correlation with USD strength (captured
    by EUR/USD inverted), and a variable correlation with USDJPY depending
    on risk-on/risk-off dynamics. These features capture regime-dependent
    cross-asset relationships that a single-asset model would miss.
    
    Warm-up handling: the first `window` bars produce NaN — NOT zero-filled.
    Zero-filling would falsely imply "no correlation" rather than "unknown."
    These NaN rows are excluded from training via dropna(subset=feature_cols).
    
    Args:
        df_xau: XAUUSD candle DataFrame with 'timestamp' and 'close' columns.
        df_eurusd: EUR/USD candle DataFrame (or None/empty if unavailable).
        df_usdjpy: USDJPY candle DataFrame (or None/empty if unavailable).
        window: Rolling correlation window (number of bars).
    
    Returns:
        df_xau with new columns: 'xau_eurusd_corr_60m', 'xau_usdjpy_corr_60m'.
    """
    df = df_xau.copy()
    
    # Compute XAUUSD log returns for correlation calculation
    xau_returns = np.log(df["close"]).diff(1)
    
    # EUR/USD correlation
    if df_eurusd is not None and not df_eurusd.empty and len(df_eurusd) > window:
        try:
            # Merge on timestamp to align bars
            eurusd_aligned = df[["timestamp"]].merge(
                df_eurusd[["timestamp", "close"]].rename(columns={"close": "close_eurusd"}),
                on="timestamp",
                how="left"
            )
            # EUR/USD is inversely related to USD strength (lower EUR/USD = stronger USD)
            eurusd_returns = np.log(eurusd_aligned["close_eurusd"]).diff(1)
            df["xau_eurusd_corr_60m"] = xau_returns.rolling(window=window).corr(eurusd_returns)
            logger.info(f"EUR/USD correlation computed: {df['xau_eurusd_corr_60m'].dropna().describe().to_dict()}")
        except Exception as e:
            logger.warning(f"Failed to compute EUR/USD correlation: {e}. Filling with NaN.")
            df["xau_eurusd_corr_60m"] = np.nan
    else:
        logger.info("EUR/USD data unavailable — xau_eurusd_corr_60m will be NaN.")
        df["xau_eurusd_corr_60m"] = np.nan
    
    # USDJPY correlation
    if df_usdjpy is not None and not df_usdjpy.empty and len(df_usdjpy) > window:
        try:
            usdjpy_aligned = df[["timestamp"]].merge(
                df_usdjpy[["timestamp", "close"]].rename(columns={"close": "close_usdjpy"}),
                on="timestamp",
                how="left"
            )
            usdjpy_returns = np.log(usdjpy_aligned["close_usdjpy"]).diff(1)
            df["xau_usdjpy_corr_60m"] = xau_returns.rolling(window=window).corr(usdjpy_returns)
            logger.info(f"USDJPY correlation computed: {df['xau_usdjpy_corr_60m'].dropna().describe().to_dict()}")
        except Exception as e:
            logger.warning(f"Failed to compute USDJPY correlation: {e}. Filling with NaN.")
            df["xau_usdjpy_corr_60m"] = np.nan
    else:
        logger.info("USDJPY data unavailable — xau_usdjpy_corr_60m will be NaN.")
        df["xau_usdjpy_corr_60m"] = np.nan
    
    return df


# ---------------------------------------------------------------------------
# Step 2: Volatility Regime Features
# ---------------------------------------------------------------------------

def compute_volatility_regime(
    df: pd.DataFrame,
    atr_period: int = 14,
    regime_window: int = 50,
    low_threshold: float = 0.8,
    high_threshold: float = 1.2
) -> pd.DataFrame:
    """
    Classifies the current volatility regime by comparing the ATR to its own
    rolling moving average.
    
    Rationale: Market dynamics differ fundamentally between low-volatility
    consolidation periods and high-volatility breakout/trend periods. A regime
    classifier allows the model to learn different directional patterns for
    each regime rather than averaging across all conditions.
    
    Encoding: Ordinal numeric (0=low, 1=normal, 2=high).
    Rationale for ordinal vs one-hot: LightGBM handles ordinal splits natively
    via its histogram-based splitting. One-hot encoding would add 2 extra sparse
    columns with no benefit for tree-based models, and would increase feature
    dimensionality unnecessarily.
    
    Args:
        df: DataFrame with 'atr' column (must call compute_atr first).
        atr_period: ATR period (for naming the output column).
        regime_window: Rolling window for ATR moving average comparison.
        low_threshold: ATR ratio below this = low volatility regime.
        high_threshold: ATR ratio above this = high volatility regime.
    
    Returns:
        df with new columns: 'atr_14' (renamed raw ATR), 'vol_regime_ratio',
        'vol_regime_class' (int: 0=low, 1=normal, 2=high).
    """
    df = df.copy()
    
    if "atr" not in df.columns:
        raise ValueError("ATR column not found. Call compute_atr() before compute_volatility_regime().")
    
    # Rename raw ATR for explicit feature naming
    df[f"atr_{atr_period}"] = df["atr"]
    
    # ATR relative to its own rolling mean — captures whether current volatility
    # is above or below the recent "normal" level
    atr_rolling_mean = df["atr"].rolling(window=regime_window).mean()
    df["vol_regime_ratio"] = df["atr"] / (atr_rolling_mean + 1e-12)
    
    # Bucket into ordinal regime classes
    df["vol_regime_class"] = 1  # Default: normal
    df.loc[df["vol_regime_ratio"] < low_threshold, "vol_regime_class"] = 0  # Low
    df.loc[df["vol_regime_ratio"] > high_threshold, "vol_regime_class"] = 2  # High
    
    return df


def compute_directional_vol_regime(
    df: pd.DataFrame,
    fast_ema: int = 12,
    slow_ema: int = 50
) -> pd.DataFrame:
    """
    Computes a directional volatility regime feature that combines EMA-based
    trend strength with ATR volatility intensity.
    
    Rationale: A signed feature encoding both trend direction AND volatility
    intensity together — not just volatility alone — because a high-volatility
    uptrend has very different predictive dynamics than a high-volatility
    downtrend. This feature captures that asymmetry in a single column.
    
    Calculation:
        trend_strength = (EMA_fast - EMA_slow) / close  (normalized for stationarity)
        directional_vol_regime = trend_strength × vol_regime_ratio
    
    Positive values → uptrend with elevated volatility.
    Negative values → downtrend with elevated volatility.
    Near-zero → no clear trend or very low volatility.
    
    Args:
        df: DataFrame with 'close' and 'vol_regime_ratio' columns.
        fast_ema: Fast EMA period for trend detection.
        slow_ema: Slow EMA period for trend detection.
    
    Returns:
        df with new column: 'directional_vol_regime' (float).
    """
    df = df.copy()
    
    ema_fast = df["close"].ewm(span=fast_ema, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_ema, adjust=False).mean()
    
    # Normalized trend strength (positive = uptrend, negative = downtrend)
    trend_strength = (ema_fast - ema_slow) / df["close"]
    
    # Multiply by regime ratio to incorporate volatility intensity
    if "vol_regime_ratio" in df.columns:
        df["directional_vol_regime"] = trend_strength * df["vol_regime_ratio"]
    else:
        # Fallback: just trend strength if regime ratio not yet computed
        df["directional_vol_regime"] = trend_strength
    
    return df


# ---------------------------------------------------------------------------
# Main Feature Builder
# ---------------------------------------------------------------------------

def build_features_df(
    df: pd.DataFrame,
    is_training: bool = False,
    flat_threshold_pct: float = 0.0001,
    df_eurusd: Optional[pd.DataFrame] = None,
    df_usdjpy: Optional[pd.DataFrame] = None,
    use_triple_barrier: bool = True,
    pt_multiplier: float = 1.0,
    sl_multiplier: float = 1.0,
    max_holding_bars: int = 5
) -> Tuple[pd.DataFrame, list]:
    """
    Builds the full feature matrix from OHLCV DataFrame.
    If is_training is True, constructs the target classification column 'target'.
    
    Args:
        df: Primary XAUUSD OHLCV DataFrame.
        is_training: If True, compute forward-looking target labels.
        flat_threshold_pct: Price change threshold for UP/DOWN vs FLAT classification.
        df_eurusd: Optional EUR/USD candle DataFrame for cross-asset correlation features.
        df_usdjpy: Optional USDJPY candle DataFrame for cross-asset correlation features.
    
    Returns:
        Tuple of (feature DataFrame, list of feature column names).
    """
    df = df.sort_values(by="timestamp", ascending=True).copy()
    
    # --- Original indicators ---
    df = compute_log_returns(df, periods=[1, 3, 5, 10, 15])
    df = compute_realized_volatility(df, windows=[5, 15, 60])
    df = compute_atr(df, period=14)
    df = compute_rsi(df, period=14)
    df = compute_macd(df, fast=12, slow=26, signal=9)
    df = compute_ema_ratios(df, spans=[9, 21, 50])
    df = compute_bollinger_bands(df, period=20)
    df = add_session_flags(df)
    
    # --- Step 2: Volatility regime features ---
    df = compute_volatility_regime(df, atr_period=14, regime_window=50)
    df = compute_directional_vol_regime(df, fast_ema=12, slow_ema=50)
    
    # --- Step 1: Cross-asset correlation features ---
    df = compute_cross_asset_correlations(df, df_eurusd=df_eurusd, df_usdjpy=df_usdjpy, window=60)
    
    # Define feature columns — original + new features
    feature_cols = [
        # Original features
        "return_1m", "return_3m", "return_5m", "return_10m", "return_15m",
        "volatility_5m", "volatility_15m", "volatility_60m",
        "atr_pct", "rsi", "macd", "macd_signal", "macd_hist",
        "ema_ratio_9", "ema_ratio_21", "ema_ratio_50", "bb_zscore",
        "session_london", "session_ny", "session_overlap",
        "hour_sin", "hour_cos",
        # New: Volatility regime features (Step 2)
        "atr_14", "vol_regime_class", "directional_vol_regime",
        # New: Cross-asset correlation features (Step 1)
        "xau_eurusd_corr_60m", "xau_usdjpy_corr_60m",
    ]
    
    # Filter to only features that actually have data (graceful degradation).
    # If cross-asset data was unavailable, those columns are all-NaN and should
    # be excluded from training to avoid losing all rows to dropna.
    available_features = []
    for col in feature_cols:
        if col in df.columns:
            # Keep the feature if it has at least SOME non-NaN values
            if not df[col].isna().all():
                available_features.append(col)
            else:
                logger.info(f"Feature '{col}' is entirely NaN — excluding from feature matrix.")
        else:
            logger.warning(f"Feature '{col}' not found in DataFrame — excluding.")
    
    feature_cols = available_features
    
    if is_training:
        if use_triple_barrier:
            # Generate labels using the triple-barrier method
            from src.labeling.triple_barrier import apply_triple_barrier_labels
            
            # Create a temporary 'atr' column mapped to 'atr_14' for raw ATR values
            df_temp = df.copy()
            df_temp["atr"] = df_temp["atr_14"]
            
            df_temp = apply_triple_barrier_labels(
                df_temp,
                pt_multiplier=pt_multiplier,
                sl_multiplier=sl_multiplier,
                max_holding_bars=max_holding_bars,
                atr_column="atr",
                close_column="close"
            )
            
            df["target"] = df_temp["tb_label"]
            df["tb_t_touch_idx"] = df_temp["tb_t_touch_idx"]
            df["tb_holding_bars"] = df_temp["tb_holding_bars"]
            df["tb_barrier_type"] = df_temp["tb_barrier_type"]
            
            # For backtesting, calculate the return at the actual touch time.
            t_touch_idx = df["tb_t_touch_idx"].values
            close_vals = df["close"].values
            
            forward_return = np.zeros(len(df))
            for idx in range(len(df)):
                touch_idx = int(t_touch_idx[idx])
                if touch_idx < len(df) and touch_idx > idx:
                    forward_return[idx] = (close_vals[touch_idx] - close_vals[idx]) / close_vals[idx]
            
            df["target_return"] = forward_return
            
            # Drop rows at the beginning that lack history, and rows at the end with insufficient forward data
            # Also drop rows where target is NaN (i.e. insufficient_data)
            df = df.dropna(subset=feature_cols + ["target", "target_return"])
        else:
            # Fallback to the old fixed-horizon labeling logic
            future_close = df["close"].shift(-5)
            forward_return = (future_close - df["close"]) / df["close"]
            
            target = pd.Series(0, index=df.index)
            target[forward_return > flat_threshold_pct] = 1
            target[forward_return < -flat_threshold_pct] = -1
            
            df["target"] = target
            df["target_return"] = forward_return
            df = df.dropna(subset=feature_cols + ["target", "target_return"])
    else:
        # For live inference, we only need to drop rows at the beginning that lack history,
        # but keep the very last row (the current live tick) to predict its future direction.
        df = df.dropna(subset=feature_cols)
        
    return df, feature_cols
