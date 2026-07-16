"""
Tests for the binary-split directional model combination rule.
Validates that combine_binary_predictions correctly handles all
signal/conflict/threshold scenarios.
"""
import unittest
import numpy as np

from src.models.train import combine_binary_predictions


class TestCombineBinaryPredictions(unittest.TestCase):
    """Tests for the binary model combination rule."""

    def test_clear_up_signal(self):
        """UP fires above threshold, DOWN does not → predict UP."""
        prob_up = np.array([0.7])
        prob_down = np.array([0.3])
        preds, confs = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        self.assertEqual(preds[0], 1)
        self.assertAlmostEqual(confs[0], 0.7)

    def test_clear_down_signal(self):
        """DOWN fires above threshold, UP does not → predict DOWN."""
        prob_up = np.array([0.3])
        prob_down = np.array([0.8])
        preds, confs = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        self.assertEqual(preds[0], -1)
        self.assertAlmostEqual(confs[0], 0.8)

    def test_neither_fires_flat(self):
        """Neither detector fires above threshold → predict FLAT."""
        prob_up = np.array([0.4])
        prob_down = np.array([0.3])
        preds, confs = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        self.assertEqual(preds[0], 0)
        # Confidence for FLAT = 1 - max(prob_up, prob_down)
        self.assertAlmostEqual(confs[0], 1.0 - 0.4)

    def test_conflict_resolution_up_wins(self):
        """Both fire — UP has higher probability → predict UP."""
        prob_up = np.array([0.75])
        prob_down = np.array([0.60])
        preds, confs = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        self.assertEqual(preds[0], 1)
        self.assertAlmostEqual(confs[0], 0.75)

    def test_conflict_resolution_down_wins(self):
        """Both fire — DOWN has higher probability → predict DOWN."""
        prob_up = np.array([0.55])
        prob_down = np.array([0.80])
        preds, confs = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        self.assertEqual(preds[0], -1)
        self.assertAlmostEqual(confs[0], 0.80)

    def test_conflict_equal_probabilities(self):
        """Both fire with equal probability → predict UP (>= comparison)."""
        prob_up = np.array([0.65])
        prob_down = np.array([0.65])
        preds, confs = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        self.assertEqual(preds[0], 1)  # UP wins on tie (>= comparison)

    def test_custom_thresholds(self):
        """Custom thresholds change which models fire."""
        prob_up = np.array([0.55])
        prob_down = np.array([0.45])
        
        # With default 0.5 thresholds: UP fires, DOWN doesn't
        preds1, _ = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        self.assertEqual(preds1[0], 1)
        
        # With UP threshold raised to 0.6: neither fires → FLAT
        preds2, _ = combine_binary_predictions(prob_up, prob_down, 0.6, 0.5)
        self.assertEqual(preds2[0], 0)

    def test_exactly_at_threshold(self):
        """Probability exactly at threshold does not fire (strictly greater)."""
        prob_up = np.array([0.50])
        prob_down = np.array([0.50])
        preds, _ = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        self.assertEqual(preds[0], 0)  # Neither fires at exactly threshold

    def test_batch_predictions(self):
        """Handles multiple samples correctly."""
        prob_up = np.array([0.7, 0.3, 0.6, 0.4])
        prob_down = np.array([0.3, 0.8, 0.7, 0.2])
        preds, confs = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        
        self.assertEqual(len(preds), 4)
        self.assertEqual(preds[0], 1)   # UP fires
        self.assertEqual(preds[1], -1)  # DOWN fires
        self.assertEqual(preds[2], -1)  # Both fire, DOWN wins (0.7 > 0.6)
        self.assertEqual(preds[3], 0)   # Neither fires

    def test_output_types(self):
        """Output arrays have correct dtypes."""
        prob_up = np.array([0.7, 0.3])
        prob_down = np.array([0.3, 0.8])
        preds, confs = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        
        self.assertEqual(preds.dtype, int)
        self.assertEqual(confs.dtype, float)

    def test_all_flat(self):
        """All probabilities below threshold → all FLAT predictions."""
        prob_up = np.array([0.1, 0.2, 0.3])
        prob_down = np.array([0.2, 0.1, 0.4])
        preds, _ = combine_binary_predictions(prob_up, prob_down, 0.5, 0.5)
        
        np.testing.assert_array_equal(preds, [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
