"""
=============================================================================
Project: Indian House Price Prediction System
File: train_model.py
Description:
    Yeh script dataset ko load karti hai, data preprocessing aur categorical
    encoding karti hai, Linear Regression model train karti hai,
    aur MAE, RMSE, R2 Score jaise metrics calculate karke model ko
    Joblib ke through save karti hai.
=============================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def main():
    print("=" * 60)
    print("STEP 1: Dataset ko Load karna (Loading Dataset)")
    print("=" * 60)
    
    data_path = os.path.join("data", "house_prices.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset nahi mila: {data_path}. Pehle create_dataset.py chalayein.")
    
    # Pandas DataFrame me data read karna
    df = pd.read_csv(data_path)
    print(f"Dataset successfully load ho gaya! Shape: {df.shape[0]} rows aur {df.shape[1]} columns.")
    print("\nDataset ke pehle 5 rows:")
    print(df.head())

    print("\n" + "=" * 60)
    print("STEP 2: Data Cleaning & Preprocessing")
    print("=" * 60)
    
    # Missing values check karna
    missing_vals = df.isnull().sum()
    print("Missing values count per column:")
    print(missing_vals)
    
    # Agar koi null values ho to handle karna
    if missing_vals.sum() > 0:
        df = df.dropna()
        print("Missing values wali rows drop kar di gayi hain.")
    else:
        print("Dataset me koi missing values nahi hain. Data bilkul clean hai!")

    # Unique cities aur locations nikalna UI dropdown ke liye
    city_locations = {}
    for city in df["City"].unique():
        locations = sorted(df[df["City"] == city]["Location"].unique().tolist())
        city_locations[city] = locations
        
    print(f"\nSupported Cities ({len(city_locations)}): {list(city_locations.keys())}")

    print("\n" + "=" * 60)
    print("STEP 3: Features (X) aur Target (y) ko Separate karna")
    print("=" * 60)
    
    # Features (Inputs): Jin cheezon ke basis par prediction karni hai
    feature_cols = ["City", "Location", "Area_sqft", "BHK", "Bathrooms", "Property_Age"]
    target_col = "Price_in_Lakhs"
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Input Features (X): {feature_cols}")
    print(f"Target Column (y): {target_col}")

    print("\n" + "=" * 60)
    print("STEP 4: Train-Test Split (80% Training, 20% Testing)")
    print("=" * 60)
    
    # Data ko 80% training aur 20% testing me divide karna
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")

    print("\n" + "=" * 60)
    print("STEP 5: Preprocessing & Pipeline Building")
    print("=" * 60)
    
    # Categorical features ko One-Hot Encode karna
    # City aur Location string format me hain, Machine Learning model ko numbers chahiye
    categorical_features = ["City", "Location"]
    numerical_features = ["Area_sqft", "BHK", "Bathrooms", "Property_Age"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
            ("num", "passthrough", numerical_features)
        ]
    )
    
    # Scikit-Learn Pipeline create karna
    # Pipeline data preprocessing aur model training ko ek single reusable object me pack karta hai
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    print("\n" + "=" * 60)
    print("STEP 6: Linear Regression Model Training")
    print("=" * 60)
    
    print("Model train ho raha hai... (Fitting Linear Regression)")
    pipeline.fit(X_train, y_train)
    print("Model training complete ho gayi successfully!")

    print("\n" + "=" * 60)
    print("STEP 7: Model Evaluation (Testing Set par Performance)")
    print("=" * 60)
    
    # Predictions on test data
    y_pred = pipeline.predict(X_test)
    
    # Metrics Calculation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Absolute Error (MAE)   : {mae:.2f} Lakhs")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} Lakhs")
    print(f"R² Score (Accuracy)         : {r2 * 100:.2f}% ({r2:.4f})")
    
    # Matplotlib Visualization: Actual vs Predicted Prices
    os.makedirs("models", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color="#1f77b4", alpha=0.7, edgecolors="k", label="Predicted Houses")
    
    # 45-degree reference line (Perfect Prediction)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Fit (y = x)")
    
    plt.title("Indian House Price: Actual vs Predicted Prices (in Lakhs)", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Price (₹ Lakhs)", fontsize=11)
    plt.ylabel("Predicted Price (₹ Lakhs)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join("models", "actual_vs_predicted.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Evaluation Plot save ho gaya: {plot_path}")

    print("\n" + "=" * 60)
    print("STEP 8: Model aur Metadata ko Joblib se Save karna")
    print("=" * 60)
    
    # Package model along with all required metadata for Streamlit UI
    model_artifact = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "city_locations": city_locations,
        "metrics": {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
            "Accuracy_Percentage": round(float(r2 * 100), 2),
            "Total_Samples": len(df),
            "Train_Samples": len(X_train),
            "Test_Samples": len(X_test)
        }
    }
    
    model_save_path = os.path.join("models", "house_price_model.pkl")
    joblib.dump(model_artifact, model_save_path)
    print(f"Trained Pipeline & Metadata save ho gaya: {model_save_path}")
    
    # Also save metrics as JSON for easy reading
    metrics_json_path = os.path.join("models", "model_metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(model_artifact["metrics"], f, indent=4)
    print(f"Metrics JSON save ho gaya: {metrics_json_path}")
    
    print("\n" + "=" * 60)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
