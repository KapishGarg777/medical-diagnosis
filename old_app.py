import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ============================================================
# LOAD SAVED FILES
# ============================================================

model    = joblib.load("best_model.pkl")
scaler   = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict diabetes")

# ============================================================
# USER INPUTS
# ============================================================

Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
Glucose     = st.number_input("Glucose", 0, 200, 100)
BP          = st.number_input("Blood Pressure", 0, 140, 70)
Skin        = st.number_input("Skin Thickness", 0, 100, 20)
Insulin     = st.number_input("Insulin", 0, 900, 80)
BMI         = st.number_input("BMI", 0.0, 70.0, 25.0)
DPF         = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age         = st.number_input("Age", 1, 120, 30)

# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button("Predict"):

    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_df = pd.DataFrame([[
        Pregnancies, Glucose, BP, Skin,
        Insulin, BMI, DPF, Age
    ]], columns=[
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ])

    # ========================================================
    # 🔥 APPLY SAME FEATURE ENGINEERING (VERY IMPORTANT)
    # ========================================================

    input_df["Glucose_BMI"]     = input_df["Glucose"] * input_df["BMI"]
    input_df["BMI_Age"]         = input_df["BMI"] * input_df["Age"]
    input_df["Glucose_Age"]     = input_df["Glucose"] * input_df["Age"]
    input_df["Insulin_Glucose"] = input_df["Insulin"] / (input_df["Glucose"] + 1)
    input_df["BP_per_Age"]      = input_df["BloodPressure"] / (input_df["Age"] + 1)

    # ========================================================
    # SCALING
    # ========================================================

    input_scaled = scaler.transform(input_df)

    # ========================================================
    # FEATURE SELECTION
    # ========================================================

    input_selected = selector.transform(input_scaled)

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(input_selected)[0]

    # ========================================================
    # OUTPUT
    # ========================================================

    if prediction == 1:
        st.error("⚠️ The model predicts: **Diabetic**")
    else:
        st.success("✅ The model predicts: **Not Diabetic**")