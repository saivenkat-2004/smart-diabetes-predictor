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






