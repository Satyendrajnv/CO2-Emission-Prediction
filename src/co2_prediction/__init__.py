"""
CO2 Emission Prediction & Regression Modeling Package
"""

__version__ = "0.1.0"

from .data_loader import CO2DataLoader
from .feature_engineering import CO2FeaturePipeline
from .metrics import CO2MetricsCalculator
from .regression_models import CO2RegressionModels

__all__ = [
    "CO2DataLoader",
    "CO2FeaturePipeline",
    "CO2MetricsCalculator",
    "CO2RegressionModels",
]
