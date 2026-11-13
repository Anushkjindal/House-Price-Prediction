# 🏠 Indian House Price Prediction System

An end-to-end beginner-friendly Machine Learning web application built using **Python, Scikit-Learn, and Streamlit** to predict residential property prices across major Indian cities (Mumbai, Delhi, Bangalore, Hyderabad, Pune, and Jaipur).

---

## 📌 Project Overview
Buying a house is a major financial decision. Property prices in India vary significantly depending on the city, locality, carpet area, BHK count, number of bathrooms, and the age of the property. 

This project demonstrates a complete machine learning lifecycle:
1. **Data Preparation**: A structured, realistic dataset representing Indian real estate dynamics.
2. **Preprocessing**: Handling missing values and encoding categorical attributes (`OneHotEncoder`).
3. **Model Training**: Multiple Linear Regression with a reusable Scikit-Learn `Pipeline`.
4. **Evaluation**: Measuring accuracy using **R² Score**, **MAE** (Mean Absolute Error), and **RMSE** (Root Mean Squared Error).
5. **Model Serialization**: Saving and persisting the trained pipeline using **Joblib**.
6. **Web Dashboard**: An interactive, modern web application built with **Streamlit** for real-time inference.

---

## 🛠️ Technology Stack & Library Purpose

| Library / Tool | Purpose & Kyun Use Kiya gaya hai? |
| :--- | :--- |
| **Python** | Primary programming language (Simple, readable, and standard for AI/ML). |
| **Pandas** | Tabular data reading (`read_csv`), cleaning, and dataframe manipulation. |
| **NumPy** | High-performance numerical computations and array operations. |
| **Scikit-learn** | Machine Learning algorithms (`LinearRegression`), `train_test_split`, `Pipeline`, and metrics (`r2_score`, `mean_absolute_error`). |
| **Matplotlib** | Data visualization for generating the Actual vs. Predicted price scatter plot. |
| **Joblib** | Lightweight serialization to save (`dump`) and load (`load`) trained Python models fast. |
| **Streamlit** | Rapid frontend framework to build modern, interactive ML web apps without HTML/CSS/JS expertise. |

---

## 📂 Project Structure

```text
House_price_prediction/
│
├── data/
│   └── house_prices.csv           # Indian house prices dataset (550+ records)
│
├── models/
│   ├── house_price_model.pkl      # Saved trained Pipeline + Metadata (Joblib)
│   ├── model_metrics.json         # Performance metrics (R², MAE, RMSE)
│   └── actual_vs_predicted.png    # Matplotlib evaluation chart
│
├── app.py                         # Streamlit interactive web application
├── train_model.py                 # ML training script with evaluation
├── create_dataset.py              # Script to generate sample Indian real estate data
├── requirements.txt               # List of Python dependencies
└── README.md                      # Complete project documentation & guide
```

---

## ⚙️ How to Run the Project Locally

Follow these simple steps in your terminal or command prompt:

### Step 1: Open Terminal / Command Prompt
Navigate to your project directory:
```bash
cd c:\Users\asdfghj\Downloads\House_price_prediction
```

### Step 2: Install Required Libraries
```bash
python -m pip install -r requirements.txt
```

### Step 3: Train the Machine Learning Model
Run the training script. This will train the Linear Regression model, compute metrics, generate the evaluation plot, and save the model into the `models/` folder:
```bash
python train_model.py
```

### Step 4: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
After running this command, your browser will automatically open:
```text
http://localhost:8501
```

---

## 🎯 Input Features & Prediction Logic
The application accepts 6 primary features:
1. **City**: Mumbai, Delhi, Bangalore, Hyderabad, Pune, Jaipur.
2. **Location / Locality**: Specific popular residential locations within each city (e.g., Bandra West in Mumbai, Indiranagar in Bangalore, Malviya Nagar in Jaipur).
3. **Area in Square Feet**: Built-up / Carpet area (300 sq.ft to 10,000 sq.ft).
4. **Bedrooms (BHK)**: 1, 2, 3, 4, or 5 BHK.
5. **Bathrooms**: 1 to 5 bathrooms.
6. **Property Age**: 0 (Brand new / Under construction) to 30 years old.

### Output:
- **Estimated Price in Lakhs / Crores** (e.g., ₹ 85.50 Lakhs or ₹ 2.20 Crore).
- **Exact Indian Rupee value** (e.g., ₹ 85,50,000).
- **Estimated Rate per Square Foot** (e.g., ₹ 7,125 / sq.ft).
- **Confidence Range (±5%)**.

---

## 🎓 Internship & Viva Interview Q&A Guide

### Q1: Why did you choose Linear Regression for this project?
> **Answer**: Real estate pricing has an inherent linear relationship with primary drivers—as square footage increases, base price increases proportionally. Linear Regression provides an interpretable baseline where every feature's coefficient has clear economic meaning (e.g., additional area directly adds value, whereas property age slightly depreciates value).

### Q2: How did you handle Categorical variables like City and Location?
> **Answer**: Machine learning models only understand numerical data. We used Scikit-Learn's `OneHotEncoder` wrapped inside a `ColumnTransformer`. It converts string categories (like `Jaipur` or `Mumbai`) into binary column flags (0 or 1), without imposing an arbitrary mathematical ordering.

### Q3: What is the benefit of using `sklearn.pipeline.Pipeline`?
> **Answer**: A Pipeline bundles the preprocessing steps (`OneHotEncoder`) and the estimator (`LinearRegression`) together into one object. This prevents data leakage during cross-validation/testing and allows the Streamlit app to pass raw user inputs directly to `pipeline.predict()` without having to manually construct dummy columns in the frontend code.

### Q4: What do MAE, RMSE, and R² Score mean in your project?
> **Answer**:
> - **R² Score**: Measures the percentage of variance in house prices explained by our model (higher is better, typically > 0.90).
> - **MAE (Mean Absolute Error)**: The average difference between predicted and actual prices in Lakhs.
> - **RMSE (Root Mean Squared Error)**: Similar to MAE but penalizes larger prediction errors more heavily because errors are squared before taking the root.

---

## 📜 Disclaimer
The sample dataset included in `data/house_prices.csv` is a clearly labeled representative academic demo dataset calibrated to typical Indian real estate price levels across tier-1 and tier-2 cities.
