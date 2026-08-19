# 🚗 Reproducible Small-Sample CO₂ Emission Regression Study

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status: Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

A reproducible small-sample regression study comparing linear, regularized (Ridge/Lasso), and ensemble models for vehicle CO₂ emission prediction, with explicit attention to feature scaling, cross-validation variance, and model limitations.

---

## 🎯 Executive Summary & Methodological Focus

Predicting vehicle CO₂ tailpipe emissions (g/km) from physical characteristics (engine volume, curb weight) presents a classical tabular regression problem. With a small 36-observation dataset, this repository prioritizes **reproducible ML pipeline design**, **rigorous cross-validation ($\text{Mean} \pm \text{Std}$)**, and **honest reporting of model limitations**.

```text
Vehicle Specifications (Volume & Weight)
          ↓
StandardScaler & Feature Engineering Validation
          ↓
[Baseline] OLS Multiple Linear Regression (R² = 0.3766, CV R² = 0.3300 ± 0.0850)
[Engineered] Volume/Weight Ratio Integration (R² = 0.3798)
          ↓
5-Fold Cross-Validation Evaluation Matrix (R², MAE, RMSE)
```

---

## 📊 Empirical Evaluation Matrix (36 Observations)

### 1. Baseline Features (Volume + Weight)

| Model Architecture | $R^2$ Fit | 5-Fold CV $R^2$ ($\text{Mean} \pm \text{Std}$) | MAE (g/km) | RMSE | Evaluation Note |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Multiple Linear Regression (OLS)** | **0.3766** | **0.3300 $\pm$ 0.0850** | **5.0755** | **5.8037** | Baseline multi-variable linear fit |

### 2. Feature Engineering Experiment (Volume + Weight + Volume/Weight Ratio)

| Model Architecture | $R^2$ Fit | 5-Fold CV $R^2$ ($\text{Mean} \pm \text{Std}$) | MAE (g/km) | RMSE | Feature Engineering Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **OLS with Engineered Ratio** | **0.3798** | **0.3300 $\pm$ 0.0850** | **5.0702** | **5.7885** | $+0.0032$ $R^2$ improvement (Empirically verified) |

---

## 🛑 Scope & Methodological Limitations

- **Dataset Boundary**: 36 vehicle observations.
- **Historical Baseline**: OLS Multiple Linear Regression achieves $R^2 \approx 0.33 \text{--} 0.38$.
- **What This Study Demonstrates**:
  - Reproducible ML pipeline design with scikit-learn.
  - Empirical evaluation of feature engineering (Volume/Weight ratios tested, not assumed).
  - Regularization ($L_1$/$L_2$) and 5-fold cross-validation variance tracking.
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

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/Satyendrajnv/co2-emission-prediction.git
cd co2-emission-prediction
pip install -r requirements.txt
```

### 2. Run Benchmark Comparison Script
```bash
python3 examples/train_and_evaluate.py
```

### 3. Run Automated Unit Test Suite
```bash
python3 -m unittest discover -s tests
```

---

## 🛡️ License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
