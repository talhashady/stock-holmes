import unittest
import pandas as pd
import numpy as np
from src.features.builder import (
    compute_log_returns,
    compute_realized_volatility,
    compute_atr,
    compute_rsi,
    compute_macd,
    build_features_df
)

class TestFeatureBuilder(unittest.TestCase):

    def setUp(self):
        # Create a mock dataset (50 candles of steady upward movement followed by consolidation)
        timestamps = pd.date_range("2026-07-14 09:00:00", periods=100, freq="1min").strftime("%Y-%m-%d %H:%M:%S")
        self.mock_data = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.linspace(100, 110, 100),
            "high": np.linspace(101, 111, 100),
            "low": np.linspace(99, 109, 100),
            "close": np.linspace(100, 110, 100),
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

    def test_build_features_df_training(self):
        # We need is_training=True to build target
        df, feature_cols = build_features_df(self.mock_data, is_training=True)
        self.assertIn("target", df.columns)
        self.assertTrue(len(feature_cols) > 10)
        self.assertTrue(df["target"].isin([-1, 0, 1]).all())
        # Check that there are no NaNs in features
        self.assertFalse(df[feature_cols].isna().any().any())

if __name__ == "__main__":
    unittest.main()
