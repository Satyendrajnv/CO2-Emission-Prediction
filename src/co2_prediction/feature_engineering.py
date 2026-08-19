"""
Feature Scaling & Engineering Pipeline
"""

from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

try:
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class CO2FeaturePipeline:
    """
    Constructs baseline and engineered tabular feature sets for CO2 emission regression.
    """

    def __init__(self, include_ratio: bool = False, include_brand: bool = False):
        self.include_ratio = include_ratio
        self.include_brand = include_brand
        self.scaler = StandardScaler() if HAS_SKLEARN else None

    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Returns feature matrix X, target vector y, and list of feature names.
        """
        data = df.copy()
        feature_names = ["Volume", "Weight"]

        if self.include_ratio:
            data["Volume_Weight_Ratio"] = data["Volume"] / np.maximum(1, data["Weight"])
            feature_names.append("Volume_Weight_Ratio")

        if self.include_brand and "Car" in data.columns:
            brand_dummies = pd.get_dummies(data["Car"], prefix="brand", drop_first=True)
            data = pd.concat([data, brand_dummies], axis=1)
            feature_names.extend(brand_dummies.columns.tolist())

        X_raw = data[feature_names].values.astype(float)
        y = data["CO2"].values.astype(float)

        if HAS_SKLEARN and self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X_raw)
        else:
            # Manual z-score standardization fallback
            mean = np.mean(X_raw, axis=0)
            std = np.std(X_raw, axis=0)
            std[std == 0] = 1.0
            X_scaled = (X_raw - mean) / std

        return X_scaled, y, feature_names
