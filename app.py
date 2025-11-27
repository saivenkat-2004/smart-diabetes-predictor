import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------- LOAD MODEL ----------------------------
model = tf.keras.models.load_model("diabetes_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# ---------------------------- LOAD DATA ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes1.csv")
    return df

df = load_data()

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home","🔮 Prediction", "ℹ About"])

st.sidebar.markdown("---")
st.sidebar.write("Created by **Banu Prakash ,Sai venkat**")

# ---------------------------- HOME PAGE ----------------------------
if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction App")
    st.write("""
    Welcome to the **Diabetes Prediction App**.  
    This application uses a trained **Artificial Neural Network (ANN)** model  
    to predict whether an individual is likely to have diabetes based on  
    their health and lifestyle patterns.

    ### 🔍 Features Inside the App:
    - 📊 **Interactive EDA**  
    - 🔮 **Diabetes Prediction**  
    - 📈 **Health Pattern Visualizations**  
    - 🤖 **Deployed ANN Model**  

    Navigate using the sidebar on the left.
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966481.png", width=200)
# ---------------------------- PREDICTION PAGE ----------------------------
elif page == "🔮 Prediction":
    st.title("🔮 Diabetes Prediction")

    st.write("Fill the form below to get a prediction from the ANN model.")

    # Input fields
    col1, col2 = st.columns(2)

    with col1:
        high_bp = st.selectbox("High Blood Pressure", [0, 1])
        high_chol = st.selectbox("High Cholesterol", [0, 1])
        chol_check = st.selectbox("Cholesterol Check in 5 Years", [0, 1])
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        smoker = st.selectbox("Smoker (100+ cigarettes)", [0, 1])
        stroke = st.selectbox("Ever Had Stroke", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        phys_act = st.selectbox("Physical Activity", [0, 1])
        fruits = st.selectbox("Daily Fruit Intake", [0, 1])

    with col2:
        veggies = st.selectbox("Daily Vegetable Intake", [0, 1])
        heavy_alcohol = st.selectbox("Heavy Alcohol Consumption", [0, 1])
        healthcare = st.selectbox("Any Healthcare Coverage", [0, 1])
        no_doctor_cost = st.selectbox("Couldn't Afford Doctor", [0, 1])
        gen_health = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
        mental_health = st.slider("Poor Mental Health Days (0–30)", 0, 30, 5)
        physical_health = st.slider("Poor Physical Health Days (0–30)", 0, 30, 5)
        diff_walk = st.selectbox("Difficulty Walking", [0, 1])
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        age = st.slider("Age Category (1–13)", 1, 13, 5)
        education = st.slider("Education Level (1–6)", 1, 6, 4)
        income = st.slider("Income Level (1–8)", 1, 8, 4)

    data = np.array([
        high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
        phys_act, fruits, veggies, heavy_alcohol, healthcare, no_doctor_cost,
        gen_health, mental_health, physical_health, diff_walk, sex, age,
        education, income
    ]).reshape(1, -1)

    if st.button("Predict"):
        scaled_data = scaler.transform(data)
        prediction = (model.predict(scaled_data) > 0.5).astype("int32")[0][0]

        if prediction == 1:
            st.error("⚠ **Prediction: Diabetes Detected**")
        else:
            st.success("✅ **Prediction: No Diabetes**")

# ---------------------------- ABOUT PAGE ----------------------------
elif page == "ℹ About":
    st.title("ℹ About This Project")
    st.markdown(""" ## 🩺 Diabetes Prediction Using Artificial Neural Networks (ANN)

    This project is an end-to-end **Machine Learning + Deep Learning** system  
    built to predict the likelihood of diabetes using health and lifestyle factors.

    ### 🎯 Project Objectives
    - Build an **ANN-based binary classification model** to predict diabetes.
    - Analyze health risk factors using **Exploratory Data Analysis (EDA)**.
    - Handle data imbalance using **SMOTE**.
    - Apply **Standard Scaling** for improved model convergence.
    - Deploy the trained ANN model using **Streamlit Web Application**.

    ### 📊 Dataset Information
    - Source: BRFSS 2015 Survey (CDC)
    - Original target classes:
        - 0 → No Diabetes  
        - 1 → Prediabetes  
        - 2 → Diabetes  
    - Modified for binary classification:
        - Removed class 1  
        - Converted class 2 → 1 (Diabetes)  
    - Final: **0 = No Diabetes, 1 = Diabetes**

    ### 🔧 Technologies Used:
    - Streamlit
    - TensorFlow
    - Keras
    - Scikit-learn
    - Pandas, NumPy
    - Matplotlib, Seaborn

    ### 🧠 Model:
    - ANN (Artificial Neural Network)
    - Binary Classification
    - Trained with SMOTE + Scaling
    - Accuracy: **85%**

    ### 👨‍💻 Developer:
    - Sai venkat
    - Banu Prakash
    """)

    st.info("This app is for educational purposes and not a medical diagnosis tool.")







