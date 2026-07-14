import unittest
import os
import pandas as pd
import sqlite3
from src.serving.db_utils import (
    get_connection,
    init_db,
    save_candles,
    get_latest_candle_time,
    get_all_candles,
    save_prediction,
    backfill_actuals,
    get_predictions_history
)

TEST_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests",
    "test_stock_holmes.db"
)

class TestDatabaseUtils(unittest.TestCase):

    def setUp(self):
        # Initialize test db and clear test jsonl
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        test_log_path = os.path.join(os.path.dirname(TEST_DB_PATH), "predictions_log.jsonl")
        if os.path.exists(test_log_path):
            os.remove(test_log_path)
        init_db(TEST_DB_PATH)

    def tearDown(self):
        # Cleanup
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        test_log_path = os.path.join(os.path.dirname(TEST_DB_PATH), "predictions_log.jsonl")
        if os.path.exists(test_log_path):
            os.remove(test_log_path)

    def test_save_and_get_candles(self):
        # Create some test candles
        df_candles = pd.DataFrame({
            "timestamp": ["2026-07-14 10:00:00", "2026-07-14 10:01:00"],
            "open": [100.0, 101.0],
            "high": [102.5, 103.0],
            "low": [99.0, 100.5],
            "close": [101.0, 102.0],
            "volume": [500.0, 600.0]
        })
        
        inserted = save_candles(df_candles, TEST_DB_PATH)
        self.assertEqual(inserted, 2)
        
        # Test duplicates are ignored (INSERT OR IGNORE)
        inserted_dup = save_candles(df_candles, TEST_DB_PATH)
        self.assertEqual(inserted_dup, 0)
        
        # Get latest timestamp
        latest = get_latest_candle_time(TEST_DB_PATH)
        self.assertEqual(latest, "2026-07-14 10:01:00")
        
        # Get all candles
        retrieved_df = get_all_candles(TEST_DB_PATH)
        self.assertEqual(len(retrieved_df), 2)
        self.assertEqual(retrieved_df.iloc[0]["close"], 101.0)

    def test_predictions_and_backfill(self):
        # Save a mock prediction
        timestamp = "2026-07-14 10:00:00"
        save_prediction(
            timestamp=timestamp,
            predicted_direction=1,
            confidence=0.75,
            probs=(0.10, 0.15, 0.75),
            db_path=TEST_DB_PATH
        )
        
        # Verify prediction is stored
        history = get_predictions_history(TEST_DB_PATH)
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["predicted_direction"], 1)
        self.assertTrue(pd.isna(history.iloc[0]["actual_direction"]))
        
        # Populate future actual candle (5 mins later: 10:05:00)
        # We also need the base candle at 10:00:00 to calculate direction
        df_candles = pd.DataFrame({
            "timestamp": ["2026-07-14 10:00:00", "2026-07-14 10:05:00"],
            "open": [100.0, 105.0],
            "high": [101.0, 106.0],
            "low": [99.0, 104.0],
            "close": [100.0, 106.0], # +6.0% return
            "volume": [100.0, 100.0]
        })
        save_candles(df_candles, TEST_DB_PATH)
        
        # Run actuals backfilling
        updated = backfill_actuals(TEST_DB_PATH, flat_threshold_pct=0.0001)
        self.assertEqual(updated, 1)
        
        # Check that actual was populated correctly
        history_updated = get_predictions_history(TEST_DB_PATH)
        self.assertEqual(history_updated.iloc[0]["actual_direction"], 1)
        self.assertEqual(history_updated.iloc[0]["actual_close"], 106.0)

if __name__ == "__main__":
    unittest.main()
