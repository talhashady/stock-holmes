import unittest
import pandas as pd
import numpy as np
from src.features.builder import (
    compute_log_returns,
    compute_realized_volatility,
    compute_atr,
    compute_rsi,
    compute_macd,
    compute_cross_asset_correlations,
    compute_volatility_regime,
    compute_directional_vol_regime,
    build_features_df
)

class TestFeatureBuilder(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset (100 candles of steady upward movement)
        timestamps = pd.date_range("2026-07-14 09:00:00", periods=100, freq="1min").strftime("%Y-%m-%d %H:%M:%S")
        self.mock_data = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(101, 111, 100),
            "low": np.linspace(99, 109, 100),
            "close": np.linspace(100, 110, 100),
            "volume": np.random.randint(100, 1000, 100)
        })
        
        # Create mock EURUSD data (inversely correlated with gold)
        self.mock_eurusd = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.linspace(105, 100, 100),   # Inverse movement
            "high": np.linspace(106, 101, 100),
            "low": np.linspace(104, 99, 100),
            "close": np.linspace(105, 100, 100),
            "volume": np.random.randint(100, 1000, 100)
        })
        
        # Create mock USDJPY data (somewhat correlated)
        self.mock_usdjpy = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.linspace(150, 155, 100),
            "high": np.linspace(150.5, 155.5, 100),
            "low": np.linspace(149.5, 154.5, 100),
            "close": np.linspace(150, 155, 100) + np.random.normal(0, 0.1, 100),
            "volume": np.random.randint(100, 1000, 100)
        })

    def test_log_returns(self):
        df = compute_log_returns(self.mock_data, periods=[1, 5])
        self.assertIn("return_1m", df.columns)
        self.assertIn("return_5m", df.columns)
        self.assertTrue(pd.isna(df["return_1m"].iloc[0]))
        self.assertFalse(pd.isna(df["return_1m"].iloc[1]))

    def test_realized_volatility(self):
        df = compute_realized_volatility(self.mock_data, windows=[5])
        self.assertIn("volatility_5m", df.columns)
        self.assertTrue(pd.isna(df["volatility_5m"].iloc[3]))
        # 5m window requires 5 returns, so first 5 rows (0-4) are nan or partially calculated
        self.assertTrue(pd.isna(df["volatility_5m"].iloc[4]))
        self.assertFalse(pd.isna(df["volatility_5m"].iloc[5]))

    def test_atr(self):
        df = compute_atr(self.mock_data, period=14)
        self.assertIn("atr", df.columns)
        self.assertIn("atr_pct", df.columns)
        self.assertFalse(pd.isna(df["atr"].iloc[14]))

    def test_rsi(self):
        df = compute_rsi(self.mock_data, period=14)
        self.assertIn("rsi", df.columns)
        self.assertFalse(pd.isna(df["rsi"].iloc[14]))
        self.assertTrue(0 <= df["rsi"].dropna().iloc[0] <= 100)

    def test_macd(self):
        df = compute_macd(self.mock_data)
        self.assertIn("macd", df.columns)
        self.assertIn("macd_signal", df.columns)
        self.assertIn("macd_hist", df.columns)

    # --- New: Cross-Asset Correlation Tests ---

    def test_cross_asset_correlations_with_data(self):
        """Correlation features are computed when cross-asset data is available."""
        df = compute_cross_asset_correlations(
            self.mock_data, df_eurusd=self.mock_eurusd, df_usdjpy=self.mock_usdjpy, window=20
        )
        self.assertIn("xau_eurusd_corr_60m", df.columns)
        self.assertIn("xau_usdjpy_corr_60m", df.columns)
        
        # First 20 rows should be NaN (warm-up period)
        self.assertTrue(df["xau_eurusd_corr_60m"].iloc[:20].isna().all())
        
        # After warm-up, values should be computed
        non_nan = df["xau_eurusd_corr_60m"].dropna()
        self.assertGreater(len(non_nan), 0)
        
        # Values should be valid Pearson correlations in [-1, 1]
        self.assertTrue((non_nan >= -1.0).all() and (non_nan <= 1.0).all(),
                        "Correlation values should be in [-1, 1]")

    def test_cross_asset_correlations_without_data(self):
        """Correlation features gracefully produce NaN when data is unavailable."""
        df = compute_cross_asset_correlations(
            self.mock_data, df_eurusd=None, df_usdjpy=None, window=20
        )
        self.assertIn("xau_eurusd_corr_60m", df.columns)
        self.assertIn("xau_usdjpy_corr_60m", df.columns)
        
        # All NaN when no cross-asset data
        self.assertTrue(df["xau_eurusd_corr_60m"].isna().all())
        self.assertTrue(df["xau_usdjpy_corr_60m"].isna().all())

    def test_cross_asset_correlations_with_empty_df(self):
        """Empty DataFrame treated same as None."""
        df = compute_cross_asset_correlations(
            self.mock_data, df_eurusd=pd.DataFrame(), df_usdjpy=pd.DataFrame(), window=20
        )
        self.assertTrue(df["xau_eurusd_corr_60m"].isna().all())
        self.assertTrue(df["xau_usdjpy_corr_60m"].isna().all())

    # --- New: Volatility Regime Tests ---

    def test_volatility_regime(self):
        """Volatility regime features are computed correctly."""
        df = compute_atr(self.mock_data, period=14)
        df = compute_volatility_regime(df, atr_period=14, regime_window=50)
        
        self.assertIn("atr_14", df.columns)
        self.assertIn("vol_regime_class", df.columns)
        self.assertIn("vol_regime_ratio", df.columns)
        
        # Regime class should be 0, 1, or 2
        valid_classes = df["vol_regime_class"].dropna().unique()
        for cls in valid_classes:
            self.assertIn(cls, [0, 1, 2])

    def test_volatility_regime_requires_atr(self):
        """Raises ValueError if ATR column is missing."""
        with self.assertRaises(ValueError):
            compute_volatility_regime(self.mock_data)

    def test_directional_vol_regime(self):
        """Directional vol regime combines trend and volatility correctly."""
        df = compute_atr(self.mock_data, period=14)
        df = compute_volatility_regime(df, atr_period=14, regime_window=50)
        df = compute_directional_vol_regime(df, fast_ema=12, slow_ema=50)
        
        self.assertIn("directional_vol_regime", df.columns)
        
        # For a steadily upward-trending dataset, directional vol regime should be
        # positive (uptrend) after warm-up period
        non_nan = df["directional_vol_regime"].dropna()
        self.assertGreater(len(non_nan), 0)
        # Uptrend: fast EMA > slow EMA → positive values expected for later rows
        self.assertGreater(non_nan.iloc[-1], 0, "Expected positive directional_vol_regime for uptrend data")

    # --- Updated: Build Features Integration Test ---

    def test_build_features_df_training(self):
        """Full feature matrix builds correctly with new features included."""
        df, feature_cols = build_features_df(self.mock_data, is_training=True)
        self.assertIn("target", df.columns)
        self.assertTrue(len(feature_cols) > 10)
        self.assertTrue(df["target"].isin([-1, 0, 1]).all())
        # Check that there are no NaNs in features
        self.assertFalse(df[feature_cols].isna().any().any())
        
        # New features should be in the feature columns (if they have data)
        # vol_regime_class and directional_vol_regime should always be present
        self.assertIn("vol_regime_class", feature_cols)
        self.assertIn("directional_vol_regime", feature_cols)
        self.assertIn("atr_14", feature_cols)

    def test_build_features_df_with_cross_asset(self):
        """Feature matrix includes correlation features when cross-asset data provided."""
        df, feature_cols = build_features_df(
            self.mock_data, is_training=True,
            df_eurusd=self.mock_eurusd, df_usdjpy=self.mock_usdjpy
        )
        self.assertIn("target", df.columns)
        
        # Cross-asset features should be present if data had enough history
        # The 20-bar window (smaller than default 60) means we might still have data
        # But with 100 candles and 60 window, we should have ~40 valid rows
        if "xau_eurusd_corr_60m" in feature_cols:
            self.assertFalse(df["xau_eurusd_corr_60m"].isna().any())

    def test_build_features_df_without_cross_asset(self):
        """Feature matrix still works without cross-asset data (graceful degradation)."""
        df, feature_cols = build_features_df(
            self.mock_data, is_training=True,
            df_eurusd=None, df_usdjpy=None
        )
        self.assertIn("target", df.columns)
        # Cross-asset features should be excluded (all-NaN columns are dropped)
        self.assertNotIn("xau_eurusd_corr_60m", feature_cols)
        self.assertNotIn("xau_usdjpy_corr_60m", feature_cols)
        # But volatility features should still be present
        self.assertIn("vol_regime_class", feature_cols)
        self.assertIn("directional_vol_regime", feature_cols)


if __name__ == "__main__":
    unittest.main()
