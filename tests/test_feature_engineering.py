"""
Unit Tests for Feature Scaling & Ratio Engineering
"""

import sys
import os
import unittest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from co2_prediction.data_loader import CO2DataLoader
from co2_prediction.feature_engineering import CO2FeaturePipeline


class TestCO2FeatureEngineering(unittest.TestCase):
    def setUp(self):
        loader = CO2DataLoader(data_path="data/data.csv")
        self.df = loader.load_data()

    def test_baseline_features(self):
        pipeline = CO2FeaturePipeline(include_ratio=False, include_brand=False)
        X, y, names = pipeline.prepare_features(self.df)

        self.assertEqual(X.shape[0], 36)
        self.assertEqual(X.shape[1], 2)
        self.assertEqual(names, ["Volume", "Weight"])

    def test_engineered_ratio_features(self):
        pipeline = CO2FeaturePipeline(include_ratio=True, include_brand=False)
        X, y, names = pipeline.prepare_features(self.df)

        self.assertEqual(X.shape[1], 3)
        self.assertIn("Volume_Weight_Ratio", names)


if __name__ == "__main__":
    unittest.main()
