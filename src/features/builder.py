import numpy as np
import pandas as pd
from typing import Tuple

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

def build_features_df(df: pd.DataFrame, is_training: bool = False, flat_threshold_pct: float = 0.0001) -> Tuple[pd.DataFrame, list]:
    """
    Builds the full feature matrix from OHLCV DataFrame.
    If is_training is True, constructs the target classification column 'target'.
    """
    df = df.sort_values(by="timestamp", ascending=True).copy()
    
    # Add indicators
    df = compute_log_returns(df, periods=[1, 3, 5, 10, 15])
    df = compute_realized_volatility(df, windows=[5, 15, 60])
    df = compute_atr(df, period=14)
    df = compute_rsi(df, period=14)
    df = compute_macd(df, fast=12, slow=26, signal=9)
    df = compute_ema_ratios(df, spans=[9, 21, 50])
    df = compute_bollinger_bands(df, period=20)
    df = add_session_flags(df)
    
    feature_cols = [
        "return_1m", "return_3m", "return_5m", "return_10m", "return_15m",
        "volatility_5m", "volatility_15m", "volatility_60m",
        "atr_pct", "rsi", "macd", "macd_signal", "macd_hist",
        "ema_ratio_9", "ema_ratio_21", "ema_ratio_50", "bb_zscore",
        "session_london", "session_ny", "session_overlap",
        "hour_sin", "hour_cos"
    ]
    
    if is_training:
        # Define target: 5-minute ahead close direction.
        # target_t = (close_t+5 - close_t) / close_t
        future_close = df["close"].shift(-5)
        forward_return = (future_close - df["close"]) / df["close"]
        
        # 3-way classification: UP (1), DOWN (-1), FLAT (0)
        target = pd.Series(0, index=df.index)
        target[forward_return > flat_threshold_pct] = 1
        target[forward_return < -flat_threshold_pct] = -1
        
        # We need to align targets and features. Since we shift -5, 
        # the last 5 rows won't have targets. We drop them during training.
        df["target"] = target
        df["target_return"] = forward_return
        # We also drop rows at the beginning that don't have enough history for EMAs/Rolling metrics
        df = df.dropna(subset=feature_cols + ["target", "target_return"])
    else:
        # For live inference, we only need to drop rows at the beginning that lack history,
        # but keep the very last row (the current live tick) to predict its future direction.
        df = df.dropna(subset=feature_cols)
        
    return df, feature_cols
