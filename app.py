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
        .stApp { background-color: #0d0d0d !important; }
        .block-container {
            background: #1a1a1a !important;
            padding: 2rem 3rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.7);
            margin-top: 30px;
        }
        h1,h2,h3,h4 { color:#ffffff !important; font-weight:700; }
        p, label, span { color:#dcdcdc !important; }
        section[data-testid="stSidebar"] {
            background-color:#141414 !important;
        }
        ::-webkit-scrollbar { width:0; }
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

    ### 🔍 Features Inside the App:
    - 📊 **Interactive EDA**
    - 🔮 **Diabetes Prediction**
    - 🤖 **Deployed ANN Model**

    **Navigate using the sidebar on the left.**
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2966/2966481.png", width=220)

# =======================================================================================
#                                   PREDICTION PAGE
# =======================================================================================
elif page == "🔮 Prediction":

    # ---------------------------- Prediction Page UI CSS ----------------------------
    st.markdown(
        """
        <style>
        :root{
            --accent-1: #2F80ED;
            --accent-2: #56CCF2;
            --card-bg: rgba(255,255,255,0.08);
            --glass-border: rgba(255,255,255,0.28);
            --muted: #6b7280;
        }

        .hero h1{
            font-size: 48px;
            color: var(--accent-2);
            text-shadow: 0 6px 18px rgba(47,128,237,0.35);
        }
        .hero{
            text-align:center;
            margin-bottom:15px;
        }

        .card{
            background: var(--card-bg);
            padding: 20px;
            border-radius: 16px;
            border: 1px solid var(--glass-border);
            box-shadow: 0 10px 28px rgba(0,0,0,0.35);
            margin-bottom: 18px;
        }

        .stButton>button {
            width: 100%;
            font-size: 18px;
            background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
            color: white;
            border: none;
            padding: 12px;
            border-radius: 10px;
            box-shadow: 0 6px 20px rgba(47,128,237,0.25);
        }

        .result-circle {
            width: 170px;
            height: 170px;
            font-size: 26px;
            font-weight: bold;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: auto;
        }
        .bad-circle {
            background: linear-gradient(130deg, #ff4d4d, #d10000);
        }
        .good-circle {
            background: linear-gradient(130deg, #14cc7a, #009e52);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------- Header ----------------------------
    st.markdown(
        """
        <div class='hero'>
            <h1>🧬 Diabetes Prediction System</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---------------------------- Feature Mapping ----------------------------
    feature_defs = {
        "HighBP": ("High Blood Pressure", "binary"),
        "HighChol": ("High Cholesterol", "binary"),
        "CholCheck": ("Cholesterol Check Recently", "binary"),
        "BMI": ("Body Mass Index", "float", 10.0, 60.0),
        "Smoker": ("Smoker", "binary"),
        "Stroke": ("Stroke History", "binary"),
        "HeartDiseaseorAttack": ("Heart Disease / Attack", "binary"),
        "PhysActivity": ("Physical Activity", "binary"),
        "Fruits": ("Fruit Consumption", "binary"),
        "Veggies": ("Vegetable Consumption", "binary"),
        "HvyAlcoholConsump": ("Heavy Alcohol Use", "binary"),
        "AnyHealthcare": ("Has Healthcare", "binary"),
        "NoDocbcCost": ("Avoided Doctor (Cost)", "binary"),
        "GenHlth": ("General Health (1=Best → 5=Worst)", "int", 1, 5),
        "MentHlth": ("Bad Mental Health Days (0–30)", "int", 0, 30),
        "PhysHlth": ("Bad Physical Health Days (0–30)", "int", 0, 30),
        "DiffWalk": ("Difficulty Walking", "binary"),
        "Sex": ("Gender (1=Male,0=Female)", "binary"),
        "Age": ("Age Category (1–13)", "int", 1, 13),
        "Education": ("Education Level (1–6)", "int", 1, 6),
        "Income": ("Income Level (1–8)", "int", 1, 8),
    }

    ordered_keys = list(feature_defs.keys())

    # ---------------------------- Input Form ----------------------------
    left, right = st.columns(2)
    inputs = {}

    with left:
        st.markdown("<div class='card'><h4>🩺 Patient Details</h4>", unsafe_allow_html=True)
        for key in ordered_keys[:11]:
            label, kind, *r = feature_defs[key]
            if kind == "binary":
                inputs[key] = st.radio(label, [0,1], horizontal=True)
            else:
                inputs[key] = st.slider(label, min_value=r[0], max_value=r[1], step=0.1 if kind=="float" else 1)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'><h4>🧑‍⚕️ Lifestyle & Health</h4>", unsafe_allow_html=True)
        for key in ordered_keys[11:]:
            label, kind, *r = feature_defs[key]
            if kind == "binary":
                inputs[key] = st.radio(label, [0,1], horizontal=True)
            else:
                inputs[key] = st.slider(label, min_value=r[0], max_value=r[1], step=0.1 if kind=="float" else 1)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------- Predict Button ----------------------------
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    colA, colB = st.columns([2,1])

    with colA:
        predict_click = st.button("🔍 Predict Diabetes Risk")

    with colB:
        result_box = st.empty()

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------- Prediction Logic ----------------------------
    if predict_click:
        try:
            x = np.array([[inputs[k] for k in ordered_keys]])
            x_scaled = scaler.transform(x)
            prob = float(model.predict(x_scaled)[0][0])
            pct = int(prob * 100)

            if prob > 0.5:
                # High risk
                html = f"""
                <div class='result-circle bad-circle'>{pct}%</div>
                <div style='text-align:center; font-size:20px; color:#ff3333; font-weight:700;'>High Diabetes Risk</div>
                """
            else:
                # Low risk
                html = f"""
                <div class='result-circle good-circle'>{pct}%</div>
                <div style='text-align:center; font-size:20px; color:#00cc66; font-weight:700;'>Low Diabetes Risk</div>
                """

            result_box.markdown(html, unsafe_allow_html=True)
            st.progress(prob)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

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
    **Sai Venkat**  
    **Banu Prakash**

    This app is created for educational purposes.
    """)
