"""
Unit Tests for Regression Fitting & Cross-Validation
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from co2_prediction.data_loader import CO2DataLoader
from co2_prediction.feature_engineering import CO2FeaturePipeline
from co2_prediction.regression_models import CO2RegressionModels
from co2_prediction.metrics import CO2MetricsCalculator


class TestCO2Models(unittest.TestCase):
    def setUp(self):
        loader = CO2DataLoader(data_path="data/data.csv")
        self.df = loader.load_data()
        pipeline = CO2FeaturePipeline()
        self.X, self.y, self.names = pipeline.prepare_features(self.df)

    def test_regression_evaluations(self):
        model_runner = CO2RegressionModels()
        results = model_runner.train_and_evaluate_all(self.X, self.y)

        self.assertGreater(len(results), 0)
        first = results[0]
        self.assertIn("r2", first)
        self.assertIn("cv_r2_mean", first)
        self.assertIn("cv_r2_std", first)

    def test_metrics_evaluator(self):
        metrics = CO2MetricsCalculator.evaluate(self.y, self.y)
        self.assertEqual(metrics["r2"], 1.0)
        self.assertEqual(metrics["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
