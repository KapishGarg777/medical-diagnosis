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
CORS(app)

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
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_df = pd.DataFrame([[

        float(data["pregnancies"]),
        float(data["glucose"]),
        float(data["bloodPressure"]),
        float(data["skinThickness"]),
        float(data["insulin"]),
        float(data["bmi"]),
        float(data["dpf"]),
        float(data["age"])

    ]], columns=[

        "Pregnancies",
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
        "DiabetesPedigreeFunction",
        "Age"
    ])

    # ========================================================
    # SAME FEATURE ENGINEERING
    # ========================================================

    input_df["Glucose_BMI"]     = input_df["Glucose"] * input_df["BMI"]
    input_df["BMI_Age"]         = input_df["BMI"] * input_df["Age"]
    input_df["Glucose_Age"]     = input_df["Glucose"] * input_df["Age"]
    input_df["Insulin_Glucose"] = input_df["Insulin"] / (input_df["Glucose"] + 1)
    input_df["BP_per_Age"]      = input_df["BloodPressure"] / (input_df["Age"] + 1)

    # ========================================================
    # SAME SCALING
    # ========================================================

    input_scaled = scaler.transform(input_df)

    # ========================================================
    # SAME FEATURE SELECTION
    # ========================================================

    input_selected = selector.transform(input_scaled)

    # ========================================================
    # SAME PREDICTION
    # ========================================================

    prediction = model.predict(input_selected)[0]

    # ========================================================
    # PROBABILITY
    # ========================================================

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
    # SAVE REPORT
    # ========================================================

    reports_collection.insert_one({
        "email": data["email"],
        "result": result,
        "risk": risk,
        "probability": float(probability)
    })

    return jsonify({
        "result": result,
        "risk": risk,
        "probability": round(probability * 100, 2)
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
# TEST ROUTE
# ============================================================

@app.route("/")
def home():
    return "Backend Working"

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)