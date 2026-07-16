import unittest
import os
import shutil
import numpy as np
from src.serving.db_utils import save_prediction, init_db

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DB_PATH = os.path.join(TEST_DIR, "test_validation.db")
TEST_JSONL_PATH = os.path.join(TEST_DIR, "predictions_log.jsonl")

class TestPredictionsValidation(unittest.TestCase):

    def setUp(self):
        # Setup clean test environment
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        if os.path.exists(TEST_JSONL_PATH):
            os.remove(TEST_JSONL_PATH)
        init_db(TEST_DB_PATH)

    def tearDown(self):
        # Cleanup test files
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        if os.path.exists(TEST_JSONL_PATH):
            os.remove(TEST_JSONL_PATH)

    def test_valid_prediction_passes(self):
        # Valid prediction input should run successfully
        save_prediction(
            timestamp="2026-07-16 12:00:00",
            predicted_direction=1,
            confidence=0.75,
            probs=(0.10, 0.15, 0.75),
            db_path=TEST_DB_PATH,
            meta_confidence=0.85
        )
        # Check that JSONL file exists and is populated
        self.assertTrue(os.path.exists(TEST_JSONL_PATH))
        with open(TEST_JSONL_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

    def test_invalid_timestamp_raises_error(self):
        # Incorrect format
        with self.assertRaises(ValueError):
            save_prediction(
                timestamp="2026/07/16 12:00:00",
                predicted_direction=1,
                confidence=0.75,
                probs=(0.10, 0.15, 0.75),
                db_path=TEST_DB_PATH
            )
        # Verify JSONL log was NOT created/written to
        self.assertFalse(os.path.exists(TEST_JSONL_PATH))

    def test_invalid_direction_raises_error(self):
        # Direction 2 is invalid
        with self.assertRaises(ValueError):
            save_prediction(
                timestamp="2026-07-16 12:00:00",
                predicted_direction=2,
                confidence=0.75,
                probs=(0.10, 0.15, 0.75),
                db_path=TEST_DB_PATH
            )
        self.assertFalse(os.path.exists(TEST_JSONL_PATH))

    def test_out_of_bounds_confidence_raises_error(self):
        # Confidence 1.5 is out of bounds
        with self.assertRaises(ValueError):
            save_prediction(
                timestamp="2026-07-16 12:00:00",
                predicted_direction=-1,
                confidence=1.5,
                probs=(0.80, 0.10, 0.10),
                db_path=TEST_DB_PATH
            )
        self.assertFalse(os.path.exists(TEST_JSONL_PATH))

    def test_nan_probabilities_raise_error(self):
        # Probability tuple contains NaN
        with self.assertRaises(ValueError):
            save_prediction(
                timestamp="2026-07-16 12:00:00",
                predicted_direction=0,
                confidence=0.45,
                probs=(np.nan, 0.55, 0.45),
                db_path=TEST_DB_PATH
            )
        self.assertFalse(os.path.exists(TEST_JSONL_PATH))

    def test_wrong_probs_length_raises_error(self):
        # Probs tuple does not have 3 elements
        with self.assertRaises(ValueError):
            save_prediction(
                timestamp="2026-07-16 12:00:00",
                predicted_direction=1,
                confidence=0.80,
                probs=(0.10, 0.90),
                db_path=TEST_DB_PATH
            )
        self.assertFalse(os.path.exists(TEST_JSONL_PATH))
