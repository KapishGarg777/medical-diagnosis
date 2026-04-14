import streamlit as st
import numpy as np
import joblib
 
# --- Page Config ---
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺")
 
st.title("🩺 Diabetes Prediction")
st.write("Fill in the patient's medical details below and click **Predict**.")
st.divider()
 
# --- Load saved model, scaler, selector ---
@st.cache_resource
def load_artifacts():
    model    = joblib.load("best_model.pkl")
    scaler   = joblib.load("scaler.pkl")
    selector = joblib.load("selector.pkl")
    return model, scaler, selector
 
try:
    model, scaler, selector = load_artifacts()
except FileNotFoundError:
    st.error("❌ Model files not found. Please run `diabetes_prediction.py` first to train and save the model.")
    st.stop()
 
# --- Input Form ---
with st.form("prediction_form"):
 
    col1, col2 = st.columns(2)
 
    with col1:
        pregnancies = st.number_input("Pregnancies",            min_value=0,   max_value=20,  value=1,    step=1)
        glucose     = st.number_input("Glucose (mg/dL)",        min_value=0,   max_value=300, value=110,  step=1)
        blood_pres  = st.number_input("Blood Pressure (mm Hg)", min_value=0,   max_value=180, value=72,   step=1)
        skin_thick  = st.number_input("Skin Thickness (mm)",    min_value=0,   max_value=100, value=20,   step=1)
 
    with col2:
        insulin     = st.number_input("Insulin (μU/mL)",        min_value=0,   max_value=900, value=80,   step=1)
        bmi         = st.number_input("BMI",                    min_value=0.0, max_value=70.0,value=25.0, step=0.1, format="%.1f")
        dpf         = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.001, format="%.3f")
        age         = st.number_input("Age",                    min_value=1,   max_value=120, value=30,   step=1)
 
    submitted = st.form_submit_button("🔍 Predict", use_container_width=True)
 
# --- Prediction ---
if submitted:
    input_data = np.array([[pregnancies, glucose, blood_pres, skin_thick,
                            insulin, bmi, dpf, age]])
 
    input_scaled   = scaler.transform(input_data)
    input_selected = selector.transform(input_scaled)
    prediction     = model.predict(input_selected)[0]
 
    st.divider()
 
    if prediction == 1:
        st.error("⚠️ Result: **Diabetic**")
        st.write("The model predicts this patient is likely **diabetic**. Please consult a doctor.")
    else:
        st.success("✅ Result: **Not Diabetic**")
        st.write("The model predicts this patient is likely **not diabetic**.")
 
    # Show input summary
    with st.expander("View entered values"):
        st.write({
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "Blood Pressure": blood_pres,
            "Skin Thickness": skin_thick,
            "Insulin": insulin,
            "BMI": bmi,
            "Diabetes Pedigree Function": dpf,
            "Age": age
        })