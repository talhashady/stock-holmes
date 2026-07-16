"""
Feature Validation Script for Stock Holmes v2.

Validates that the new cross-asset correlation and volatility regime features
show real variation and behave sensibly before relying on them in the model.

Usage:
    python -m src.features.validate_features
"""
import os
import sys
import logging
import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.serving.db_utils import get_all_candles
from src.features.builder import build_features_df

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("features.validate")


def validate_features() -> None:
    """
    Runs comprehensive validation on the new feature set:
    1. Cross-asset correlation distribution analysis
    2. Volatility regime time distribution
    3. Directional volatility regime vs forward returns analysis
    """
    logger.info("=" * 60)
    logger.info("STOCK HOLMES v2 — Feature Validation Report")
    logger.info("=" * 60)
    
    # Load data
    df_xau = get_all_candles(table_name="candles")
    df_eurusd = get_all_candles(table_name="candles_eurusd")
    df_usdjpy = get_all_candles(table_name="candles_usdjpy")
    
    logger.info(f"XAUUSD candles: {len(df_xau)}")
    logger.info(f"EURUSD candles: {len(df_eurusd)}")
    logger.info(f"USDJPY candles: {len(df_usdjpy)}")
    
    if len(df_xau) < 200:
        logger.error("Insufficient XAUUSD data for feature validation. Need at least 200 candles.")
        return
    
    # Build features
    df_features, feature_cols = build_features_df(
        df_xau, is_training=True, flat_threshold_pct=0.0001,
        df_eurusd=df_eurusd if not df_eurusd.empty else None,
        df_usdjpy=df_usdjpy if not df_usdjpy.empty else None
    )
    
    logger.info(f"\nFeature matrix: {len(df_features)} rows, {len(feature_cols)} features")
    logger.info(f"Feature columns: {feature_cols}")
    
    # --- 1. Cross-Asset Correlation Distribution ---
    logger.info("\n" + "=" * 60)
    logger.info("1. CROSS-ASSET CORRELATION FEATURES")
    logger.info("=" * 60)
    
    for col in ["xau_eurusd_corr_60m", "xau_usdjpy_corr_60m"]:
        if col in df_features.columns and not df_features[col].isna().all():
            series = df_features[col].dropna()
            logger.info(f"\n  {col}:")
            logger.info(f"    Count:  {len(series)}")
            logger.info(f"    Mean:   {series.mean():.4f}")
            logger.info(f"    Std:    {series.std():.4f}")
            logger.info(f"    Min:    {series.min():.4f}")
            logger.info(f"    25%:    {series.quantile(0.25):.4f}")
            logger.info(f"    50%:    {series.median():.4f}")
            logger.info(f"    75%:    {series.quantile(0.75):.4f}")
            logger.info(f"    Max:    {series.max():.4f}")
            
            if series.std() < 0.05:
                logger.warning(f"    ⚠️ LOW VARIANCE: std={series.std():.4f} — feature may be near-constant!")
            else:
                logger.info(f"    ✅ Healthy variance (std > 0.05)")
        else:
            logger.info(f"\n  {col}: UNAVAILABLE (cross-asset data not loaded)")
    
    # --- 2. Volatility Regime Distribution ---
    logger.info("\n" + "=" * 60)
    logger.info("2. VOLATILITY REGIME DISTRIBUTION")
    logger.info("=" * 60)
    
    if "vol_regime_class" in df_features.columns:
        regime_counts = df_features["vol_regime_class"].value_counts(normalize=True).sort_index()
        regime_labels = {0: "Low", 1: "Normal", 2: "High"}
        
        for regime_val, pct in regime_counts.items():
            label = regime_labels.get(regime_val, f"Unknown({regime_val})")
            logger.info(f"  {label} (class={regime_val}): {pct:.1%} of time ({int(pct * len(df_features))} bars)")
        
        if len(regime_counts) < 3:
            logger.warning("  ⚠️ Not all regime classes represented! Check thresholds.")
        else:
            logger.info("  ✅ All three regime classes present")
    
    if "vol_regime_ratio" in df_features.columns:
        ratio = df_features["vol_regime_ratio"].dropna()
        logger.info(f"\n  vol_regime_ratio stats:")
        logger.info(f"    Mean: {ratio.mean():.4f}, Std: {ratio.std():.4f}")
        logger.info(f"    Range: [{ratio.min():.4f}, {ratio.max():.4f}]")
    
    # --- 3. Directional Vol Regime vs Forward Returns ---
    logger.info("\n" + "=" * 60)
    logger.info("3. DIRECTIONAL VOL REGIME vs FORWARD RETURNS")
    logger.info("=" * 60)
    
    if "directional_vol_regime" in df_features.columns and "target_return" in df_features.columns:
        dvr = df_features["directional_vol_regime"].dropna()
        
        # Bucket directional_vol_regime into quintiles for analysis
        df_features["dvr_bucket"] = pd.qcut(
            df_features["directional_vol_regime"], 
            q=5, labels=["Q1(bearish)", "Q2", "Q3(neutral)", "Q4", "Q5(bullish)"],
            duplicates="drop"
        )
        
        grouped = df_features.groupby("dvr_bucket", observed=True)["target_return"].agg(["mean", "std", "count"])
        logger.info("\n  Average 5-min forward return by directional vol regime quintile:")
        for bucket, row in grouped.iterrows():
            direction = "📈" if row["mean"] > 0 else "📉" if row["mean"] < 0 else "➡️"
            logger.info(f"    {bucket}: mean={row['mean']:.6f} {direction}, std={row['std']:.6f}, n={int(row['count'])}")
        
        # Check if there's a monotonic relationship
        means = grouped["mean"].values
        if len(means) >= 3:
            trend = np.corrcoef(range(len(means)), means)[0, 1]
            logger.info(f"\n  Monotonicity correlation: {trend:.4f}")
            if abs(trend) > 0.5:
                logger.info("  ✅ Directional vol regime shows sensible monotonic relationship with forward returns")
            else:
                logger.info("  ℹ️ Weak monotonic relationship — feature may still be useful non-linearly for tree models")
    
    logger.info("\n" + "=" * 60)
    logger.info("Feature validation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    validate_features()
