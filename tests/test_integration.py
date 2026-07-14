import unittest
import os
import shutil
import pandas as pd
import numpy as np
import pickle

from src.serving.db_utils import init_db, save_candles, get_all_candles, get_predictions_history, save_prediction
from src.features.builder import build_features_df
from src.models.train import train_pipeline, MODEL_SAVE_PATH, METRICS_SAVE_PATH
from src.models.predict import predict_latest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DB_PATH = os.path.join(TEST_DIR, "test_integration.db")
TEST_MODEL_PATH = os.path.join(TEST_DIR, "test_model.pkl")
TEST_METRICS_PATH = os.path.join(TEST_DIR, "test_metrics.json")

class TestIntegrationPipeline(unittest.TestCase):

    def setUp(self):
        # Create a clean database file
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        init_db(TEST_DB_PATH)
        
        # Backup existing models to prevent overriding user's local files
        self.model_backup = MODEL_SAVE_PATH + ".bak"
        self.metrics_backup = METRICS_SAVE_PATH + ".bak"
        self.log_file = os.path.join(TEST_DIR, "predictions_log.jsonl")
        
        if os.path.exists(MODEL_SAVE_PATH):
            shutil.copy(MODEL_SAVE_PATH, self.model_backup)
        if os.path.exists(METRICS_SAVE_PATH):
            shutil.copy(METRICS_SAVE_PATH, self.metrics_backup)
        if os.path.exists(self.log_file):
            os.remove(self.log_file)

    def tearDown(self):
        # Cleanup integration test db
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except PermissionError:
                pass
                
        # Clean up any test log file generated during execution
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
            
        # Restore backups
        if os.path.exists(self.model_backup):
            if os.path.exists(MODEL_SAVE_PATH):
                os.remove(MODEL_SAVE_PATH)
            shutil.move(self.model_backup, MODEL_SAVE_PATH)
        else:
            if os.path.exists(MODEL_SAVE_PATH):
                os.remove(MODEL_SAVE_PATH)
                
        if os.path.exists(self.metrics_backup):
            if os.path.exists(METRICS_SAVE_PATH):
                os.remove(METRICS_SAVE_PATH)
            shutil.move(self.metrics_backup, METRICS_SAVE_PATH)
        else:
            if os.path.exists(METRICS_SAVE_PATH):
                os.remove(METRICS_SAVE_PATH)

    def test_end_to_end_pipeline(self):
        """Validates ingestion -> features -> train -> inference path in sequence."""
        # 1. Mock Ingesting 600 candles of data to satisfy training constraints
        # Ensure chronological order and slight trend to avoid zero variance in metrics
        timestamps = pd.date_range("2026-07-14 00:00:00", periods=600, freq="1min").strftime("%Y-%m-%d %H:%M:%S")
        
        # Simple trend + minor sine wave
        base_price = 4000.0
        prices = [base_price + i*0.05 + 2.0*np.sin(i/10.0) for i in range(600)]
        
        mock_candles = pd.DataFrame({
            "timestamp": timestamps,
            "open": [p - 0.5 for p in prices],
            "high": [p + 1.0 for p in prices],
            "low": [p - 1.0 for p in prices],
            "close": prices,
            "volume": [0.0] * 600
        })
        
        # Save to integration db
        inserted = save_candles(mock_candles, TEST_DB_PATH)
        self.assertEqual(inserted, 600)
        
        # Verify candles exist
        candles_df = get_all_candles(TEST_DB_PATH)
        self.assertEqual(len(candles_df), 600)
        
        # 2. Run features build locally (dry-run check)
        df_features, feature_cols = build_features_df(candles_df, is_training=True)
        self.assertFalse(df_features.empty)
        self.assertIn("target", df_features.columns)
        
        # 3. Patch db_utils DEFAULT_DB_PATH in train.py and predict.py to point to TEST_DB_PATH
        # We can temporarily patch db paths by modifying the functions internally
        import src.models.train as train_mod
        import src.models.predict as predict_mod
        import src.serving.db_utils as db_mod
        
        original_db_path = db_mod.DEFAULT_DB_PATH
        
        try:
            # Reassign default paths to force test db scope
            db_mod.DEFAULT_DB_PATH = TEST_DB_PATH
            train_mod.get_all_candles = lambda: get_all_candles(TEST_DB_PATH)
            predict_mod.get_all_candles = lambda: get_all_candles(TEST_DB_PATH)
            predict_mod.save_prediction = lambda ts, pred, conf, probs: save_prediction(ts, pred, conf, probs, TEST_DB_PATH)
            
            # Run model training
            metrics = train_mod.train_pipeline(test_ratio=0.1, val_ratio=0.1)
            
            # Assert model was generated and serialized
            self.assertTrue(os.path.exists(MODEL_SAVE_PATH))
            self.assertTrue(os.path.exists(METRICS_SAVE_PATH))
            self.assertIn("accuracy", metrics)
            self.assertIn("backtest", metrics)
            
            # Run inference on latest candle
            inference_res = predict_mod.predict_latest()
            self.assertIsNotNone(inference_res)
            self.assertEqual(inference_res["timestamp"], mock_candles["timestamp"].iloc[-1])
            
            # Confirm prediction was persisted in DB
            history = get_predictions_history(TEST_DB_PATH)
            self.assertEqual(len(history), 1)
            self.assertEqual(history.iloc[0]["predicted_direction"], inference_res["prediction"])
            self.assertAlmostEqual(history.iloc[0]["confidence"], inference_res["confidence"], places=5)
            
        finally:
            # Restore paths
            db_mod.DEFAULT_DB_PATH = original_db_path
            import importlib
            importlib.reload(train_mod)
            importlib.reload(predict_mod)

if __name__ == "__main__":
    unittest.main()
