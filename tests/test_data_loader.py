"""
Unit Tests for CO2 Data Loader & Schema Verification
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from co2_prediction.data_loader import CO2DataLoader


class TestCO2DataLoader(unittest.TestCase):
    def test_load_data_structure(self):
        loader = CO2DataLoader(data_path="data/data.csv")
        df = loader.load_data()

        self.assertEqual(len(df), 36)
        self.assertIn("Volume", df.columns)
        self.assertIn("Weight", df.columns)
        self.assertIn("CO2", df.columns)
        self.assertFalse(df["CO2"].isnull().any())


if __name__ == "__main__":
    unittest.main()
