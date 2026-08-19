# 🚗 Reproducible Small-Sample CO₂ Emission Regression Study

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](#)

A reproducible small-sample regression study comparing multiple linear, regularized (Ridge/Lasso), and ensemble models for vehicle CO₂ emission prediction, with explicit attention to feature scaling, cross-validation variance, and model limitations.

---

## 🎯 Executive Summary & Methodological Focus

Predicting vehicle CO₂ tailpipe emissions (g/km) from physical characteristics (engine volume, curb weight) presents a classical tabular regression problem. With a small 36-observation dataset, this repository prioritizes **reproducible ML pipeline design**, **rigorous cross-validation ($\text{Mean} \pm \text{Std}$)**, and **honest reporting of model limitations**.

```text
Vehicle Specifications (Volume & Weight)
          ↓
StandardScaler & Feature Engineering Validation
          ↓
[Baseline] OLS Multiple Linear Regression (Historical R² ≈ 0.33)
[Regularized] Ridge (L2) & Lasso (L1) Regression
[Ensemble] Random Forest & Gradient Boosting Regressors
          ↓
5-Fold Cross-Validation Evaluation Matrix (R², MAE, RMSE)
```

---

## 🛑 Scope & Methodological Limitations

- **Dataset Boundary**: 36 vehicle observations.
- **Historical Baseline**: OLS Multiple Linear Regression achieves $R^2 \approx 0.33$.
- **What This Study Demonstrates**:
  - Reproducible ML pipeline design with scikit-learn.
  - Empirical evaluation of feature engineering (Volume/Weight ratios).
  - Regularization ($L_1$/$L_2$) and cross-validation variance tracking.
  - Automated testing for tabular regression pipelines.
- **What This Study Does NOT Claim**:
  - Production-level deployment readiness.
  - Generalization across modern hybrid/electric vehicle populations.
  - Causal relationships between vehicle mass and tailpipe emissions.

---

## 🏗️ Repository Architecture

```text
co2-emission-prediction/
├── README.md                                  # Small-sample regression overview & limits
├── LICENSE                                    # MIT License
├── .gitignore                                 # Exclusion rules for caches & virtualenvs
├── requirements.txt                           # Dependency manifest
├── data/
│   ├── README.md                              # Dataset schema & column specifications
│   └── data.csv                               # Clean 36-sample vehicle emission dataset
├── docs/
│   └── methodology.md                        # Mathematical specs for regression & 5-fold CV
├── notebooks/
│   └── co2_emissions_eda.ipynb                # Feature correlation & diagnostic plots
├── src/
│   └── co2_prediction/
│       ├── __init__.py                        # Package exports
│       ├── data_loader.py                     # Ingestion & schema validation
│       ├── feature_engineering.py             # Feature scaling & ratio transformations
│       ├── metrics.py                         # R², MAE, RMSE & 5-fold CV score engine
│       └── regression_models.py              # Linear, Ridge, Lasso & Ensemble Regressors
├── tests/
│   ├── test_data_loader.py                    # Unit tests for data loading
│   ├── test_feature_engineering.py            # Unit tests for transformations
│   └── test_models.py                         # Unit tests for model fitting & prediction
└── examples/
    └── train_and_evaluate.py                  # Executable training & cross-validation script
```

---

## 🚀 Quickstart & Setup

### 1. Installation
```bash
git clone https://github.com/Satyendrajnv/co2-emission-prediction.git
cd co2-emission-prediction
pip install -r requirements.txt
```

### 2. Dataset Location
The 36-sample dataset is located at `data/data.csv`. Refer to [data/README.md](data/README.md) for column descriptions.

---

## 🛡️ License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
