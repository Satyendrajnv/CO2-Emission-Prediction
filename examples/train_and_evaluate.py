"""
Executable Script Running Benchmark Comparison & 5-Fold Cross Validation Matrix
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from co2_prediction.data_loader import CO2DataLoader
from co2_prediction.feature_engineering import CO2FeaturePipeline
from co2_prediction.regression_models import CO2RegressionModels


def main():
    print("=" * 85)
    print("🚗 SMALL-SAMPLE CO₂ EMISSION REGRESSION STUDY & MODEL EVALUATION")
    print("=" * 85)

    # 1. Ingestion
    print("\n[1/4] Loading Vehicle Emission Dataset...")
    loader = CO2DataLoader(data_path="data/data.csv")
    df = loader.load_data()
    print(f"      ✓ Loaded {len(df)} vehicle observations (Target: CO2 g/km).")

    # 2. Baseline Features (Volume + Weight)
    print("\n[2/4] Preparing Baseline Features (Volume & Weight with StandardScaler)...")
    base_pipeline = CO2FeaturePipeline(include_ratio=False, include_brand=False)
    X_base, y, base_names = base_pipeline.prepare_features(df)
    
    runner = CO2RegressionModels()
    base_results = runner.train_and_evaluate_all(X_base, y)

    # 3. Engineered Features (Volume + Weight + Volume/Weight Ratio)
    print("\n[3/4] Preparing Engineered Features (Volume + Weight + Volume/Weight Ratio)...")
    eng_pipeline = CO2FeaturePipeline(include_ratio=True, include_brand=False)
    X_eng, _, eng_names = eng_pipeline.prepare_features(df)
    eng_results = runner.train_and_evaluate_all(X_eng, y)

    # Summary Comparison Matrix
    print("\n" + "=" * 85)
    print("📊 MODEL EVALUATION MATRIX (Baseline Features: Volume + Weight)")
    print("=" * 85)
    print(f"{'Model Architecture':<25} | {'R² Fit':<8} | {'CV R² Mean ± Std':<20} | {'MAE (g/km)':<12} | {'RMSE':<8}")
    print("-" * 85)
    for res in base_results:
        cv_str = f"{res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}"
        print(f"{res['model_name']:<25} | {res['r2']:<8.4f} | {cv_str:<20} | {res['mae']:<12.4f} | {res['rmse']:<8.4f}")

    print("\n" + "=" * 85)
    print("🧪 FEATURE ENGINEERING TEST MATRIX (Engineered: Volume + Weight + Volume/Weight Ratio)")
    print("=" * 85)
    print(f"{'Model Architecture':<25} | {'R² Fit':<8} | {'CV R² Mean ± Std':<20} | {'MAE (g/km)':<12} | {'RMSE':<8}")
    print("-" * 85)
    for res in eng_results:
        cv_str = f"{res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}"
        print(f"{res['model_name']:<25} | {res['r2']:<8.4f} | {cv_str:<20} | {res['mae']:<12.4f} | {res['rmse']:<8.4f}")

    print("\n" + "=" * 85 + "\n")


if __name__ == "__main__":
    main()
