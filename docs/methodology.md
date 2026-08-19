# Small-Sample Regression Methodology & Evaluation Specifications

This document specifies the mathematical formulations for feature scaling, regression algorithms, regularizers, and cross-validation procedures.

---

## 1. Regression Formulations

### 1.1 Multiple Linear Regression (OLS Baseline)
Given feature matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ ($n=36$) and target vector $\mathbf{y} \in \mathbb{R}^n$:
$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b$$
Objective function:
$$\min_{\mathbf{w}, b} \frac{1}{2n} \|\mathbf{y} - (\mathbf{X} \mathbf{w} + b)\|_2^2$$

### 1.2 Ridge Regression ($L_2$ Regularization)
$$\min_{\mathbf{w}, b} \frac{1}{2n} \|\mathbf{y} - (\mathbf{X} \mathbf{w} + b)\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$

### 1.3 Lasso Regression ($L_1$ Regularization)
$$\min_{\mathbf{w}, b} \frac{1}{2n} \|\mathbf{y} - (\mathbf{X} \mathbf{w} + b)\|_2^2 + \alpha \|\mathbf{w}\|_1$$

---

## 2. Feature Transformations

- **Standardization ($z$-score)**: $\mathbf{x}_{\text{scaled}} = \frac{\mathbf{x} - \mu}{\sigma}$
- **Volume-to-Weight Ratio**: $\text{Ratio} = \frac{\text{Volume}}{\text{Weight}}$

---

## 3. Evaluation Metrics & 5-Fold Cross-Validation

- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

- **Mean Absolute Error (MAE)**:
  $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

- **Root Mean Squared Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

- **5-Fold Cross-Validation ($\mu \pm \sigma$)**:
  Reports mean $R^2$ across 5 validation folds alongside standard deviation ($\sigma$).
