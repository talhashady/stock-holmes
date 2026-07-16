import unittest
import numpy as np
import pandas as pd
import lightgbm as lgb
from src.labeling.meta_labeling import (
    generate_meta_labels,
    build_meta_features,
    train_meta_model,
    apply_meta_filter
)


class TestMetaLabeling(unittest.TestCase):
    def test_generate_meta_labels(self):
        # 1 = UP, -1 = DOWN, 0 = FLAT
        primary_preds = np.array([1, 1, -1, 0, -1])
        actual_labels = np.array([1, -1, -1, 1, 0])
        
        meta_labels, mask = generate_meta_labels(primary_preds, actual_labels)
        
        # Expected mask: True for indices 0, 1, 2, 4 (directional primary predictions)
        # Expected meta labels for these:
        # idx 0: pred 1 vs actual 1 → 1 (correct)
        # idx 1: pred 1 vs actual -1 → 0 (incorrect)
        # idx 2: pred -1 vs actual -1 → 1 (correct)
        # idx 4: pred -1 vs actual 0 → 0 (incorrect)
        # Result should be [1, 0, 1, 0]
        
        np.testing.assert_array_equal(mask, [True, True, True, False, True])
        np.testing.assert_array_equal(meta_labels, [1, 0, 1, 0])

    def test_build_meta_features(self):
        df_features = pd.DataFrame({
            "vol_regime_class": [1, 2],
            "atr_pct": [0.002, 0.003],
            "volatility_5m": [0.001, 0.0015],
            "hour_sin": [0.5, 0.6],
            "hour_cos": [-0.5, -0.6],
            "session_overlap": [1, 0]
        })
        prob_up = np.array([0.6, 0.4])
        prob_down = np.array([0.3, 0.7])
        primary_preds = np.array([1, -1])
        
        meta_df = build_meta_features(df_features, prob_up, prob_down, primary_preds)
        
        self.assertEqual(len(meta_df), 2)
        # Check confidence: max(0.6, 0.3) = 0.6; max(0.4, 0.7) = 0.7
        np.testing.assert_allclose(meta_df["primary_confidence"].values, [0.6, 0.7], rtol=1e-5)
        # Check spread: |0.6 - 0.3| = 0.3; |0.4 - 0.7| = 0.3
        np.testing.assert_allclose(meta_df["primary_prob_spread"].values, [0.3, 0.3], rtol=1e-5)
        self.assertEqual(meta_df["vol_regime_class"].iloc[1], 2)

    def test_apply_meta_filter_flat_bypass(self):
        # FLAT predictions are always bypassed without running the model
        pred, conf = apply_meta_filter(
            primary_pred=0, prob_up=0.3, prob_down=0.2,
            meta_model=None, meta_row=None, trust_threshold=0.5
        )
        self.assertEqual(pred, 0)
        # Synthetic flat confidence = 1.0 - max(0.3, 0.2) = 0.7
        self.assertAlmostEqual(conf, 0.7)

    def test_apply_meta_filter_directional(self):
        # Mock lightgbm Booster
        # We subclass to return a mock predict value
        class MockBooster:
            def __init__(self, prob):
                self.prob = prob
            def predict(self, data):
                return np.array([self.prob])

        # High confidence (0.8) meta-model, trust threshold 0.50 → prediction should PASS
        booster_high = MockBooster(0.8)
        pred, conf = apply_meta_filter(
            primary_pred=1, prob_up=0.6, prob_down=0.3,
            meta_model=booster_high, meta_row=np.array([[0.0]]), trust_threshold=0.5
        )
        self.assertEqual(pred, 1)
        self.assertEqual(conf, 0.8)

        # Low confidence (0.3) meta-model, trust threshold 0.50 → prediction should FILTER to FLAT (0)
        booster_low = MockBooster(0.3)
        pred, conf = apply_meta_filter(
            primary_pred=1, prob_up=0.6, prob_down=0.3,
            meta_model=booster_low, meta_row=np.array([[0.0]]), trust_threshold=0.5
        )
        self.assertEqual(pred, 0)
        self.assertEqual(conf, 0.3)


if __name__ == "__main__":
    unittest.main()
