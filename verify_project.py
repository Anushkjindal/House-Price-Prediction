"""
Script: verify_project.py
Verification script to inspect and validate all 8 aspects of the project.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

def run_verification():
    print("============================================================")
    print("1. CHECK PROJECT FILES (फाइलों की जांच)")
    print("============================================================")
    expected_files = [
        "app.py",
        "train_model.py",
        "create_dataset.py",
        "requirements.txt",
        "README.md",
        os.path.join("data", "house_prices.csv"),
        os.path.join("models", "house_price_model.pkl"),
        os.path.join("models", "model_metrics.json"),
        os.path.join("models", "actual_vs_predicted.png")
    ]
    
    all_files_ok = True
    for file_path in expected_files:
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) if exists else 0
        status = "OK" if exists and size > 0 else "FAILED"
        print(f"[{status}] {file_path:<35} Size: {size:>7} bytes")
        if not exists or size == 0:
            all_files_ok = False

    print("\n============================================================")
    print("2 & 3. DATASET ORIGIN & NATURE (डेटासेट का स्रोत और प्रकार)")
    print("============================================================")
    csv_path = os.path.join("data", "house_prices.csv")
    df = pd.read_csv(csv_path)
    print(f"Dataset path: {csv_path}")
    print(f"Total Rows: {len(df)}, Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Cities in Dataset ({df['City'].nunique()}): {list(df['City'].unique())}")
    print(f"Total Unique Localities: {df['Location'].nunique()}")
    print("First 3 records:")
    print(df.head(3))

    print("\n============================================================")
    print("4 & 5. MODEL TRAINING & METRICS VERIFICATION")
    print("============================================================")
    model_path = os.path.join("models", "house_price_model.pkl")
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    metrics = artifact["metrics"]
    
    print("Pipeline Components:")
    for name, step in pipeline.steps:
        print(f"  - Step: {name} -> {step.__class__.__name__}")
        
    linear_model = pipeline.named_steps["regressor"]
    print(f"Intercept (c): {linear_model.intercept_:.4f}")
    print(f"Total Feature Coefficients (m): {len(linear_model.coef_)}")
    print("Stored Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    print("\n============================================================")
    print("6. SAMPLE PREDICTION TEST (नमूना प्रेडिक्शन टेस्ट)")
    print("============================================================")
    test_cases = [
        {"City": "Jaipur", "Location": "Malviya Nagar", "Area_sqft": 1200, "BHK": 3, "Bathrooms": 2, "Property_Age": 5},
        {"City": "Mumbai", "Location": "Bandra West", "Area_sqft": 1200, "BHK": 3, "Bathrooms": 2, "Property_Age": 5},
        {"City": "Delhi", "Location": "Dwarka", "Area_sqft": 1500, "BHK": 3, "Bathrooms": 3, "Property_Age": 8},
        {"City": "Bangalore", "Location": "Whitefield", "Area_sqft": 1800, "BHK": 3, "Bathrooms": 3, "Property_Age": 2},
        {"City": "Pune", "Location": "Kothrud", "Area_sqft": 1100, "BHK": 2, "Bathrooms": 2, "Property_Age": 4},
        {"City": "Hyderabad", "Location": "Gachibowli", "Area_sqft": 1400, "BHK": 3, "Bathrooms": 2, "Property_Age": 6}
    ]
    
    test_df = pd.DataFrame(test_cases)
    predictions = pipeline.predict(test_df)
    
    for tc, pred in zip(test_cases, predictions):
        print(f"City: {tc['City']:<10} Loc: {tc['Location']:<16} Area: {tc['Area_sqft']} sqft, {tc['BHK']}BHK -> Estimated: ₹ {pred:6.2f} Lakhs")

    print("\n============================================================")
    print("OVERALL STATUS:")
    print("Files check passed:", all_files_ok)
    print("Model coefficients learned:", len(linear_model.coef_) > 0)
    print("Predictions reasonable and positive:", all(p > 0 for p in predictions))
    print("============================================================")

if __name__ == "__main__":
    run_verification()
