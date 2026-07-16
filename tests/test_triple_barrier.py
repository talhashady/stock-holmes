import unittest
import numpy as np
import pandas as pd
from src.labeling.triple_barrier import (
    get_triple_barrier_label,
    apply_triple_barrier_labels,
    validate_triple_barrier_labels,
    BarrierEvent
)


class TestTripleBarrierLabeling(unittest.TestCase):
    def setUp(self):
        # Create a basic price series
        # Entry price = 100.0, ATR = 2.0
        # Upper barrier (1x) = 102.0, Lower barrier (1x) = 98.0
        self.base_close = 100.0
        self.atr_val = 2.0

    def test_upper_barrier_hit(self):
        # Price path hits upper barrier at bar 2
        close_prices = np.array([100.0, 101.0, 102.5, 101.5, 100.0])
        atr_values = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        
        event = get_triple_barrier_label(
            close_prices, atr_values, t_start=0,
            pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4
        )
        self.assertEqual(event.label, 1)
        self.assertEqual(event.t_touch_idx, 2)
        self.assertEqual(event.holding_bars, 2)
        self.assertEqual(event.barrier_type, "upper")

    def test_lower_barrier_hit(self):
        # Price path hits lower barrier at bar 1
        close_prices = np.array([100.0, 97.5, 96.0, 99.0, 101.0])
        atr_values = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
        
        event = get_triple_barrier_label(
            close_prices, atr_values, t_start=0,
            pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4
        )
        self.assertEqual(event.label, -1)
        self.assertEqual(event.t_touch_idx, 1)
        self.assertEqual(event.holding_bars, 1)
        self.assertEqual(event.barrier_type, "lower")

    def test_vertical_barrier_hit(self):
        # Price stays between 98.0 and 102.0
        close_prices = np.array([100.0, 100.5, 99.5, 101.0, 99.0, 98.5])
        atr_values = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        
        event = get_triple_barrier_label(
            close_prices, atr_values, t_start=0,
            pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=4
        )
        self.assertEqual(event.label, 0)
        self.assertEqual(event.t_touch_idx, 4)
        self.assertEqual(event.holding_bars, 4)
        self.assertEqual(event.barrier_type, "vertical")

    def test_nan_atr_handling(self):
        close_prices = np.array([100.0, 105.0, 106.0])
        atr_values = np.array([np.nan, 2.0, 2.0])
        
        event = get_triple_barrier_label(
            close_prices, atr_values, t_start=0,
            pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=2
        )
        self.assertEqual(event.label, 0)
        self.assertEqual(event.barrier_type, "vertical")

    def test_out_of_bounds_handling(self):
        # Scan reaches end of series before max_holding_bars
        close_prices = np.array([100.0, 101.0])
        atr_values = np.array([2.0, 2.0])
        
        event = get_triple_barrier_label(
            close_prices, atr_values, t_start=0,
            pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=5
        )
        self.assertEqual(event.label, 0)
        self.assertEqual(event.t_touch_idx, 1)
        self.assertEqual(event.holding_bars, 1)

    def test_apply_labels_dataframe(self):
        df = pd.DataFrame({
            "close": [100.0, 101.0, 102.5, 101.5, 100.0],
            "atr": [2.0, 2.0, 2.0, 2.0, 2.0],
            "timestamp": pd.date_range("2026-07-16 12:00:00", periods=5, freq="5min")
        })
        
        df_labeled = apply_triple_barrier_labels(
            df, pt_multiplier=1.0, sl_multiplier=1.0, max_holding_bars=2,
            atr_column="atr", close_column="close"
        )
        
        self.assertIn("tb_label", df_labeled.columns)
        self.assertIn("tb_t_touch_idx", df_labeled.columns)
        self.assertIn("tb_holding_bars", df_labeled.columns)
        self.assertIn("tb_barrier_type", df_labeled.columns)
        
        # Row 0: entry 100, atr 2. At bar 1 (101), no hit. At bar 2 (102.5), hits upper barrier (102.0)
        self.assertEqual(df_labeled["tb_label"].iloc[0], 1)
        self.assertEqual(df_labeled["tb_barrier_type"].iloc[0], "upper")
        self.assertEqual(df_labeled["tb_holding_bars"].iloc[0], 2)
        
        # Row 3 & 4 should be marked as "insufficient_data" and have NaN labels
        # because they are within max_holding_bars (2) of the end of the series.
        self.assertEqual(df_labeled["tb_barrier_type"].iloc[3], "insufficient_data")
        self.assertEqual(df_labeled["tb_barrier_type"].iloc[4], "insufficient_data")
        self.assertTrue(pd.isna(df_labeled["tb_label"].iloc[3]))
        self.assertTrue(pd.isna(df_labeled["tb_label"].iloc[4]))

    def test_validate_labels(self):
        df = pd.DataFrame({
            "close": np.linspace(100, 110, 100),
            "atr": np.ones(100) * 1.0,
            "timestamp": pd.date_range("2026-07-16 12:00:00", periods=100, freq="5min")
        })
        df_labeled = apply_triple_barrier_labels(df, max_holding_bars=5)
        stats = validate_triple_barrier_labels(df_labeled)
        
        self.assertIsInstance(stats, dict)
        self.assertIn("label_distribution", stats)
        self.assertIn("avg_holding_bars", stats)
        self.assertEqual(stats["total_valid_labels"], 95)  # 100 - 5


if __name__ == "__main__":
    unittest.main()
