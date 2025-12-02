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


# ---------------------------- REAL WATERMARK BACKGROUND ----------------------------
def set_bg():
    st.markdown("""
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1581091012184-5c7b2baf8bff");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            position: relative;
        }

        /* Soft white overlay for watermark look */
        .stApp::before {
            content: "";
            position: absolute;
            inset: 0;
            background: rgba(255, 255, 255, 0.78);  /* LIGHTER WATERMARK EFFECT */
            backdrop-filter: blur(2px);
            z-index: -1;
        }

        /* Clean White Container */
        .block-container {
            background: rgba(255, 255, 255, 0.95);
            padding: 2rem 3rem;
            border-radius: 18px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
            margin-top: 30px !important;
        }

        /* Make headings darker for clarity */
        h1, h2, h3, h4 {
            color: #222 !important;
            font-weight: 700 !important;
        }

        p, label {
            color: #333 !important;
            font-size: 1rem !important;
        }

        /* Remove Scrollbars */
        ::-webkit-scrollbar { width: 0px; }
        html, body { overflow: hidden !important; }

        /* Sidebar Light Style */
        section[data-testid="stSidebar"] {
            background-color: #f7f7f7 !important;
        }
    </style>
    """, unsafe_allow_html=True)

set_bg()



# ---------------------------- LOAD MODEL ----------------------------
model = tf.keras.models.load_model("diabetes_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# ---------------------------- LOAD DATA ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes1.csv")

df = load_data()

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔮 Prediction", "ℹ About"])

st.sidebar.markdown("---")
st.sidebar.write("Created by **Banu Prakash - Sai Venkat**")


# ============================ HOME PAGE =============================
if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction App")

    st.write("""
    Welcome to the **Diabetes Prediction App**.  
    This app uses a trained **Artificial Neural Network (ANN)** model to predict diabetes based on health and lifestyle data.
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966481.png", width=220)


# ============================ PREDICTION PAGE =============================
elif page == "🔮 Prediction":
    st.title("🔮 Diabetes Prediction")

    st.write("Fill the details below:")

    col1, col2 = st.columns(2)

    with col1:
        high_bp = st.selectbox("High Blood Pressure", [0, 1])
        high_chol = st.selectbox("High Cholesterol", [0, 1])
        chol_check = st.selectbox("Cholesterol Check in 5 Years", [0, 1])
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        smoker = st.selectbox("Smoker", [0, 1])
        stroke = st.selectbox("Stroke", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        phys_act = st.selectbox("Physical Activity", [0, 1])
        fruits = st.selectbox("Fruit Intake", [0, 1])

    with col2:
        veggies = st.selectbox("Vegetable Intake", [0, 1])
        heavy_alcohol = st.selectbox("Heavy Alcohol", [0, 1])
        healthcare = st.selectbox("Healthcare Coverage", [0, 1])
        no_doctor_cost = st.selectbox("Can't Afford Doctor", [0, 1])
        gen_health = st.selectbox("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
        mental_health = st.selecctbox("Poor Mental Health Days", 0, 30, 5)
        physical_health = st.selectbox("Poor Physical Health Days", 0, 30, 5)
        diff_walk = st.selectbox("Difficulty Walking", [0, 1])
        sex = st.selectbox("Sex (0=Female, 1=Male)", [0, 1])
        age = st.selectbox("Age Category (1–13)", 1, 13, 5)
        education = st.selectbox("Education (1–6)", 1, 6, 4)
        income = st.selectbox("Income (1–8)", 1, 8, 4)


    user_data = np.array([
        high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
        phys_act, fruits, veggies, heavy_alcohol, healthcare, no_doctor_cost,
        gen_health, mental_health, physical_health, diff_walk, sex, age,
        education, income
    ]).reshape(1, -1)

    if st.button("Predict"):
        scaled = scaler.transform(user_data)
        pred = (model.predict(scaled) > 0.5).astype(int)[0][0]

        if pred == 1:
            st.error("⚠ **Prediction: Diabetes Detected**")
        else:
            st.success("✅ **Prediction: No Diabetes**")


# ============================ ABOUT PAGE =============================
elif page == "ℹ About":
    st.title("ℹ About This Project")

    st.write("""
    ### 🔧 Technologies Used  
    - Streamlit  
    - TensorFlow / Keras  
    - Scikit-learn  
    - Pandas, NumPy  
    - Matplotlib, Seaborn  

    ### 🧠 Model  
    - ANN (Artificial Neural Network)  
    - Binary Classification  
    - Accuracy: **85%**  

    ### 👨‍💻 Developer  
    - Your Name  

    **This app is for educational purposes only**.
    """)



