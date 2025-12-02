import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

# ---------------------------- PAGE CONFIG ----------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------- DARK THEME ----------------------------
st.markdown(
    """
    <style>
        /* Full dark background */
        .stApp {
            background-color: #0d0d0d !important;
        }

        /* Main content card */
        .block-container {
            background: #1a1a1a !important;
            padding: 2rem 3rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.7);
            margin-top: 30px;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #ffffff !important;
            font-weight: 700;
        }

        /* Regular text */
        p, label, span {
            color: #dcdcdc !important;
        }

        /* Sidebar dark */
        section[data-testid="stSidebar"] {
            background-color: #141414 !important;
        }

        /* Sidebar text */
        .css-1d391kg, .css-qri22k, .css-1kyxreq {
            color: #ffffff !important;
        }

        /* Remove scrollbars */
        ::-webkit-scrollbar { width: 0px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------- LOAD MODEL ----------------------------
model = tf.keras.models.load_model("diabetes_ann_model.h5")
scaler = joblib.load("scaler.pkl")

# ---------------------------- SIDEBAR ----------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to:", ["🏠 Home", "🔮 Prediction", "ℹ About"])

st.sidebar.markdown("---")
st.sidebar.write("Created by **Banu Prakash – Sai Venkat**")

# =======================================================================================
#                                        HOME PAGE
# =======================================================================================
if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction App")

    st.write("""
    Welcome to the **Diabetes Prediction App**.  
    This application uses a trained **Artificial Neural Network (ANN)**  
    to predict diabetes based on health and lifestyle inputs.
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966481.png", width=220)

# =======================================================================================
#                                   PREDICTION PAGE
# =======================================================================================
elif page == "🔮 Prediction":
    st.title("🔮 Diabetes Prediction")

    st.write("Fill in your details below:")

    col1, col2 = st.columns(2)

    # ----------------------- LEFT COLUMN -----------------------
    with col1:
        high_bp = st.selectbox("High Blood Pressure", [0, 1])
        high_chol = st.selectbox("High Cholesterol", [0, 1])
        chol_check = st.selectbox("Cholesterol Check (5 yrs)", [0, 1])
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
        smoker = st.selectbox("Smoker (100+ cigarettes)", [0, 1])
        stroke = st.selectbox("Stroke", [0, 1])
        heart_disease = st.selectbox("Heart Disease", [0, 1])
        phys_act = st.selectbox("Physical Activity", [0, 1])
        fruits = st.selectbox("Daily Fruit Intake", [0, 1])

    # ----------------------- RIGHT COLUMN -----------------------
    with col2:
        veggies = st.selectbox("Daily Vegetable Intake", [0, 1])
        heavy_alcohol = st.selectbox("Heavy Alcohol Usage", [0, 1])
        healthcare = st.selectbox("Healthcare Coverage", [0, 1])
        no_doctor_cost = st.selectbox("Couldn't Afford Doctor", [0, 1])

        gen_health = st.selectbox("General Health (1=Excellent,5=Poor)", [1,2,3,4,5])
        mental_health = st.selectbox("Poor Mental Health Days", list(range(0,31)))
        physical_health = st.selectbox("Poor Physical Health Days", list(range(0,31)))

        diff_walk = st.selectbox("Difficulty Walking", [0, 1])
        sex = st.selectbox("Sex (0=Female,1=Male)", [0, 1])

        age = st.selectbox("Age Category (1–13)", list(range(1, 14)))
        education = st.selectbox("Education Level (1–6)", list(range(1, 7)))
        income = st.selectbox("Income Level (1–8)", list(range(1, 9)))

    # ----------------------- PREPARE DATA -----------------------
    user_data = np.array([
        high_bp, high_chol, chol_check, bmi, smoker, stroke, heart_disease,
        phys_act, fruits, veggies, heavy_alcohol, healthcare, no_doctor_cost,
        gen_health, mental_health, physical_health, diff_walk, sex, age,
        education, income
    ]).reshape(1, -1)

    # ----------------------- PREDICT -----------------------
    if st.button("Predict"):
        scaled = scaler.transform(user_data)
        pred = (model.predict(scaled) > 0.5).astype(int)[0][0]

        if pred == 1:
            st.error("⚠ **Prediction: Diabetes Likely Detected**")
        else:
            st.success("✅ **Prediction: No Diabetes**")

# =======================================================================================
#                                      ABOUT PAGE
# =======================================================================================
elif page == "ℹ About":
    st.title("ℹ About This Project")

    st.write("""
    ### 🔧 Technologies
    - Streamlit  
    - TensorFlow / Keras  
    - Scikit-learn  
    - Pandas  
    - NumPy  

    ### 🧠 Model  
    - Artificial Neural Network (ANN)  
    - Accuracy: **85%**

    ### 👨‍💻 Developers  
    **Banu Prakash**  
    **Sai Venkat**

    This app is created for educational purposes.
    """)
