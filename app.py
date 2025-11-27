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
page = st.sidebar.radio("Go to:", ["🏠 Home", "📊 EDA", "🔮 Prediction", "ℹ About"])

st.sidebar.markdown("---")
st.sidebar.write("Created by **Your Name**")

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

# ---------------------------- EDA PAGE ----------------------------
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")
    st.write("Here are key visual insights from the dataset.")

    # Show raw data
    if st.checkbox("Show Dataset"):
        st.dataframe(df)

    st.subheader("🔹 Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=df["diabetes"], ax=ax)
    st.pyplot(fig)

    st.subheader("🔹 BMI Distribution by Diabetes Status")
    fig, ax = plt.subplots()
    sns.kdeplot(data=df, x="bmi", hue="diabetes", fill=True, ax=ax)
    st.pyplot(fig)

    st.subheader("🔹 Age Category Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="age", hue="diabetes", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("🔹 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    st.success("EDA Visualizations Completed ✔")

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
    st.write("""
    This diabetes prediction system was developed using:

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
    - Your Name  
    """)

    st.info("This app is for educational purposes and not a medical diagnosis tool.")

