from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
from pymongo import MongoClient
import bcrypt
import jwt
import datetime

app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

# ============================================================
# SECRET KEY
# ============================================================

SECRET_KEY = "medical_ai_secret"

# ============================================================
# MONGODB
# ============================================================

client = MongoClient("mongodb+srv://Kapish:Kapish%40174@cluster0.ezvzmzl.mongodb.net/medicaldiagnosis?retryWrites=true&w=majority")

db = client["medical_ai"]

users_collection = db["users"]
reports_collection = db["reports"]

# ============================================================
# LOAD MODEL FILES
# ============================================================

model    = joblib.load("best_model.pkl")
scaler   = joblib.load("scaler.pkl")
selector = joblib.load("selector.pkl")

# ============================================================
# ZERO IMPUTATION MEDIANS
# Calculated from YOUR diabetes.csv using the exact same
# cleaning + outlier removal steps as the training notebook.
# The model was never trained on 0s in these columns —
# they were replaced with per-class medians before scaling.
# We use the overall median (avg of both classes) at
# prediction time since we don't know the class yet.
#
#   Non-Diabetic medians: Glucose=107, BP=70, Skin=27, Insulin=102.5, BMI=30.1
#   Diabetic medians:     Glucose=138, BP=74.5, Skin=32, Insulin=169.5, BMI=34.3
# ============================================================

IMPUTE_MEDIANS = {
    "Glucose":       122.5,
    "BloodPressure":  72.25,
    "SkinThickness":  29.5,
    "Insulin":       136.0,
    "BMI":            32.2,
}

# ============================================================
# REGISTER
# ============================================================

@app.route("/register", methods=["POST"])
def register():

    data = request.json

    existing_user = users_collection.find_one({
        "email": data["email"]
    })

    if existing_user:
        return jsonify({
            "message": "User already exists"
        }), 400

    hashed_password = bcrypt.hashpw(
        data["password"].encode("utf-8"),
        bcrypt.gensalt()
    )

    users_collection.insert_one({
        "name": data["name"],
        "email": data["email"],
        "password": hashed_password
    })

    return jsonify({
        "message": "Registration successful"
    })

# ============================================================
# LOGIN
# ============================================================

@app.route("/login", methods=["POST"])
def login():

    data = request.json

    user = users_collection.find_one({
        "email": data["email"]
    })

    if not user:
        return jsonify({
            "message": "Invalid email"
        }), 401

    password_correct = bcrypt.checkpw(
        data["password"].encode("utf-8"),
        user["password"]
    )

    if not password_correct:
        return jsonify({
            "message": "Invalid password"
        }), 401

    token = jwt.encode({
        "email": user["email"],
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }, SECRET_KEY, algorithm="HS256")

    return jsonify({
        "token": token,
        "name": user["name"]
    })

# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    # ========================================================
    # READ RAW VALUES
    # ========================================================

    pregnancies    = float(data["pregnancies"])
    glucose        = float(data["glucose"])
    blood_pressure = float(data["bloodPressure"])
    skin_thickness = float(data["skinThickness"])
    insulin        = float(data["insulin"])
    bmi            = float(data["bmi"])
    dpf            = float(data["dpf"])
    age            = float(data["age"])

    # ========================================================
    # CRITICAL FIX: Replace 0s with training medians
    # The training notebook replaced 0s with per-class medians
    # BEFORE scaling. Without this, the RobustScaler receives
    # values it was never trained on, producing extreme scaled
    # values that push every prediction to Non-Diabetic.
    # ========================================================

    if glucose        == 0: glucose        = IMPUTE_MEDIANS["Glucose"]
    if blood_pressure == 0: blood_pressure = IMPUTE_MEDIANS["BloodPressure"]
    if skin_thickness == 0: skin_thickness = IMPUTE_MEDIANS["SkinThickness"]
    if insulin        == 0: insulin        = IMPUTE_MEDIANS["Insulin"]
    if bmi            == 0: bmi            = IMPUTE_MEDIANS["BMI"]

    # ========================================================
    # FEATURE ENGINEERING (identical to training notebook)
    # ========================================================

    glucose_bmi     = glucose * bmi
    bmi_age         = bmi * age
    glucose_age     = glucose * age
    insulin_glucose = insulin / (glucose + 1)
    bp_per_age      = blood_pressure / (age + 1)

    # ========================================================
    # BUILD NUMPY ARRAY in exact training column order
    # Using numpy array (not DataFrame) matches how the scaler
    # was fit — on numpy output from SMOTE, not a DataFrame.
    # ========================================================

    input_array = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age,
        glucose_bmi,
        bmi_age,
        glucose_age,
        insulin_glucose,
        bp_per_age
    ]])

    # ========================================================
    # SCALE → SELECT → PREDICT
    # ========================================================

    input_scaled   = scaler.transform(input_array)
    input_selected = selector.transform(input_scaled)

    prediction  = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0][1]

    # ========================================================
    # RISK LEVEL
    # ========================================================

    if probability < 0.30:
        risk = "Low"
    elif probability < 0.70:
        risk = "Moderate"
    else:
        risk = "High"

    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    # ========================================================
    # SAVE REPORT (probability stored as 0-100)
    # ========================================================

    reports_collection.insert_one({
        "email":       data["email"],
        "result":      result,
        "risk":        risk,
        "probability": round(float(probability) * 100, 2)
    })

    return jsonify({
        "result":      result,
        "risk":        risk,
        "probability": round(float(probability) * 100, 2)
    })

# ============================================================
# REPORTS
# ============================================================

@app.route("/reports/<email>", methods=["GET"])
def reports(email):

    reports = list(
        reports_collection.find(
            {"email": email},
            {"_id": 0}
        )
    )

    return jsonify(reports)

# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return "Backend Working"

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)