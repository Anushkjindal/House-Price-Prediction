"""
=============================================================================
Project: Indian House Price Prediction System
File: app.py
Description:
    Streamlit Web Application for estimating Indian House Prices.
    Includes an interactive form, live prediction in Lakhs/Crores,
    model performance metrics, and project details.
=============================================================================
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------------------------------------------------------
# 1. Page Configuration (Title, Icon, Layout)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Custom CSS for Modern, Responsive UI
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    /* Metric Card */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-title {
        color: #64748B;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-val {
        color: #1E293B;
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 4px;
    }
    /* Result Box */
    .result-box {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 2px solid #3B82F6;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        margin-top: 15px;
    }
    .result-title {
        color: #1E40AF;
        font-size: 1rem;
        font-weight: 600;
    }
    .result-price {
        color: #1E3A8A;
        font-size: 2.3rem;
        font-weight: 800;
        margin: 6px 0;
    }
    .result-sub {
        color: #374151;
        font-size: 0.95rem;
    }
    /* Footer */
    .footer-text {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.85rem;
        margin-top: 40px;
        padding-top: 15px;
        border-top: 1px solid #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Model Loading Helper Function (Cached for fast performance)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_trained_model():
    """Trained pipeline aur metadata ko models folder se load karta hai."""
    model_path = os.path.join("models", "house_price_model.pkl")
    if not os.path.exists(model_path):
        return None
    try:
        artifact = joblib.load(model_path)
        return artifact
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
artifact = load_trained_model()

# -----------------------------------------------------------------------------
# 4. Sidebar: Navigation & Project Information
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/609/609803.png", width=70)
    st.title("🏠 Navigation")
    menu = st.radio("Go to:", ["Price Predictor", "Model Performance", "Dataset Explorer", "About Project"])
    
    st.markdown("---")
    st.subheader("💡 Tech Stack")
    st.markdown("""
    - **Language**: Python 🐍
    - **Data**: Pandas & NumPy
    - **ML Model**: Scikit-Learn (Linear Regression)
    - **Serialization**: Joblib
    - **Web App**: Streamlit
    """)
    
    st.markdown("---")
    st.caption("Developed by AI & Data Science B.Tech Student")

# -----------------------------------------------------------------------------
# 5. Fallback if Model is not yet trained
# -----------------------------------------------------------------------------
if artifact is None:
    st.error("⚠️ Trained model file `models/house_price_model.pkl` nahi mila!")
    st.info("Pehle terminal me `python train_model.py` command run karke model train karein.")
    st.stop()

pipeline = artifact["pipeline"]
city_locations = artifact["city_locations"]
metrics = artifact["metrics"]

# -----------------------------------------------------------------------------
# PAGE 1: PRICE PREDICTOR
# -----------------------------------------------------------------------------
if menu == "Price Predictor":
    st.markdown('<div class="main-header">Indian House Price Prediction System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Predict estimated property prices across major Indian cities using Machine Learning.</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        st.subheader("📝 Enter Property Details")
        st.write("Niche diye gaye form me property ki details bharein:")
        
        with st.form("prediction_form"):
            # City Selection
            cities = list(city_locations.keys())
            selected_city = st.selectbox("🏙️ Select City:", cities, index=cities.index("Jaipur") if "Jaipur" in cities else 0)
            
            # Location Selection (Filtered according to City)
            locations = city_locations.get(selected_city, ["Other"])
            selected_location = st.selectbox("📍 Select Locality / Area:", locations)
            
            # Area in Sq Ft
            area_sqft = st.number_input(
                "📐 Built-up Area (in Square Feet):",
                min_value=300,
                max_value=10000,
                value=1200,
                step=50,
                help="Ghar ka carpet/built-up area square feet me."
            )
            
            # BHK & Bathrooms in columns
            c1, c2 = st.columns(2)
            with c1:
                bhk = st.selectbox("🛏️ Bedrooms (BHK):", [1, 2, 3, 4, 5], index=1)
            with c2:
                bathrooms = st.selectbox("🚿 Bathrooms:", [1, 2, 3, 4, 5], index=1)
                
            # Property Age
            property_age = st.slider(
                "⏳ Property Age (Years):",
                min_value=0,
                max_value=30,
                value=3,
                help="Property kitni saal purani hai (0 = Brand New/Under Construction)."
            )
            
            # Submit Button
            submitted = st.form_submit_button("💰 Predict Estimated Price", use_container_width=True)

    with col_right:
        st.subheader("📊 Price Estimation")
        
        if submitted:
            # Prepare Input DataFrame matching model features
            input_data = pd.DataFrame([{
                "City": selected_city,
                "Location": selected_location,
                "Area_sqft": area_sqft,
                "BHK": bhk,
                "Bathrooms": bathrooms,
                "Property_Age": property_age
            }])
            
            # Predict using the loaded pipeline
            prediction = pipeline.predict(input_data)[0]
            price_lakhs = max(1.0, round(prediction, 2))  # Ensure positive reasonable price
            
            # Format price nicely
            if price_lakhs >= 100:
                crores = price_lakhs / 100
                price_display = f"₹ {crores:.2f} Crore"
            else:
                price_display = f"₹ {price_lakhs:.2f} Lakh"
                
            inr_full = int(price_lakhs * 100000)
            formatted_inr = f"₹ {inr_full:,}"
            
            # Calculate estimated rate per sqft
            rate_per_sqft = int(inr_full / area_sqft)
            
            # Render attractive result box
            st.markdown(f"""
            <div class="result-box">
                <div class="result-title">ESTIMATED MARKET VALUE</div>
                <div class="result-price">{price_display}</div>
                <div class="result-sub">({formatted_inr})</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📌 Summary of Selection")
            summary_df = pd.DataFrame({
                "Parameter": ["City", "Location", "Area", "BHK", "Bathrooms", "Property Age", "Est. Rate / Sq.Ft"],
                "Value": [
                    selected_city,
                    selected_location,
                    f"{area_sqft} sq.ft.",
                    f"{bhk} BHK",
                    f"{bathrooms} Bath",
                    f"{property_age} Years",
                    f"₹ {rate_per_sqft:,} / sq.ft."
                ]
            })
            st.table(summary_df)
            
            # Confidence Range Box
            lower_bound = max(1.0, round(price_lakhs * 0.95, 2))
            upper_bound = round(price_lakhs * 1.05, 2)
            st.info(f"💡 **Estimated Price Range (±5%):** ₹ {lower_bound:.2f} Lakhs - ₹ {upper_bound:.2f} Lakhs")
            
        else:
            st.info("👈 Left side ke form me property ki details bharein aur **Predict Estimated Price** button par click karein.")
            
            # Display Quick City Price Guidance
            st.markdown("#### 🌆 Average Property Rates in India (Guidance)")
            city_guide = pd.DataFrame({
                "City": ["Mumbai", "Delhi", "Bangalore", "Pune", "Hyderabad", "Jaipur"],
                "Typical Avg Rate": ["₹ 14,000 - 28,000/sqft", "₹ 8,500 - 22,000/sqft", "₹ 6,000 - 14,000/sqft", "₹ 6,500 - 10,500/sqft", "₹ 6,000 - 13,000/sqft", "₹ 3,800 - 7,500/sqft"]
            })
            st.dataframe(city_guide, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# PAGE 2: MODEL PERFORMANCE METRICS
# -----------------------------------------------------------------------------
elif menu == "Model Performance":
    st.markdown('<div class="main-header">📈 Model Performance & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Evaluation metrics on the test dataset using Scikit-Learn.</div>', unsafe_allow_html=True)

    # Metric Cards Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">R² Score (Accuracy)</div>
            <div class="metric-val">{metrics.get('Accuracy_Percentage', 0)}%</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">MAE (Mean Absolute Error)</div>
            <div class="metric-val">₹ {metrics.get('MAE', 0)} L</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE</div>
            <div class="metric-val">₹ {metrics.get('RMSE', 0)} L</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Test Samples</div>
            <div class="metric-val">{metrics.get('Test_Samples', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    col_img, col_explain = st.columns([1.2, 0.8], gap="large")
    
    with col_img:
        st.subheader("Actual vs Predicted House Prices Plot")
        plot_path = os.path.join("models", "actual_vs_predicted.png")
        if os.path.exists(plot_path):
            st.image(plot_path, caption="Matplotlib: Actual Prices vs Predicted Prices (Red dashed line indicates perfect fit)")
        else:
            st.warning("Evaluation plot image nahi mili.")
            
    with col_explain:
        st.subheader("🧠 Metric Explanation (Hindi)")
        st.markdown("""
        - **R² Score (Coefficient of Determination)**:
          Yeh batata hai ki hamara model kitna variance explain kar pa raha hai. High score (jaise > 90%) ka matlab model actual patterns ko accurately capture kar raha hai.
        
        - **MAE (Mean Absolute Error)**:
          Model ke predicted price aur actual price ke beech ka average difference (in Lakhs).
          
        - **RMSE (Root Mean Squared Error)**:
          Badi ghaltiyon (large errors) par zyada penalty lagata hai. Lower is better.
          
        - **Algorithm**:
          **Multiple Linear Regression** with One-Hot Categorical Encoding & Scikit-Learn Pipeline.
        """)

# -----------------------------------------------------------------------------
# PAGE 3: DATASET EXPLORER
# -----------------------------------------------------------------------------
elif menu == "Dataset Explorer":
    st.markdown('<div class="main-header">📁 Indian House Prices Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Inspect the dataset used for training the model.</div>', unsafe_allow_html=True)
    
    csv_path = os.path.join("data", "house_prices.csv")
    if os.path.exists(csv_path):
        df_display = pd.read_csv(csv_path)
        
        # Filter by City
        all_cities = ["All"] + sorted(df_display["City"].unique().tolist())
        filter_city = st.selectbox("Filter records by City:", all_cities)
        
        if filter_city != "All":
            filtered_df = df_display[df_display["City"] == filter_city]
        else:
            filtered_df = df_display
            
        st.write(f"Showing **{len(filtered_df)}** records:")
        st.dataframe(filtered_df, use_container_width=True)
        
        st.markdown("### 📊 Quick Statistical Summary")
        st.dataframe(filtered_df.describe(), use_container_width=True)
    else:
        st.warning("Dataset file `data/house_prices.csv` nahi mili.")

# -----------------------------------------------------------------------------
# PAGE 4: ABOUT PROJECT
# -----------------------------------------------------------------------------
elif menu == "About Project":
    st.markdown('<div class="main-header">ℹ️ About This Project</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 🎯 Project Overview
    The **Indian House Price Prediction System** is a full-stack Machine Learning application developed to estimate residential real estate prices across major Indian metropolitan areas including **Mumbai, Delhi, Bangalore, Hyderabad, Pune, and Jaipur**.
    
    ### 🏗️ Architecture & Workflow
    1. **Data Ingestion**: Dataset with realistic Indian property features (City, Location, Area, BHK, Bathrooms, Age, Price).
    2. **Data Preprocessing**: Handling missing values, categorical encoding via Scikit-Learn `OneHotEncoder`.
    3. **Model Training**: Multiple Linear Regression with `Pipeline`.
    4. **Model Evaluation**: Metrics including R² score, Mean Absolute Error (MAE), and RMSE.
    5. **Model Serialization**: Saving the pipeline with `joblib`.
    6. **Interactive Web App**: Modern Streamlit frontend with responsive layout and real-time inference.
    
    ### 🎓 Interview / Viva Talking Points
    - **Why Linear Regression?** 
      Linear Regression is interpretable, efficient, and serves as an ideal baseline for real-estate pricing where price scales directly with square footage and locality multipliers.
    - **Why Scikit-Learn Pipeline?**
      Pipelines prevent data leakage between training and testing splits and allow the Streamlit app to pass raw categorical inputs directly without manual dummy encoding.
    """)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer-text">
    Indian House Price Prediction System • Built with Streamlit & Scikit-Learn • Academic ML Project
</div>
""", unsafe_allow_html=True)
