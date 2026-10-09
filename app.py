
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "random_forest_churn_model.pkl"
FEATURES_PATH = PROJECT_ROOT / "churn_model_features.pkl"
THRESHOLD_PATH = PROJECT_ROOT / "churn_threshold.json"


# Load saved model, feature names, and threshold
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
        threshold = json.load(file)["threshold"]

    return model, features, threshold


st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="centered",
)

st.title("📊 Customer Churn Intelligence")
st.write(
    "Estimate customer churn risk using a trained machine learning model."
)

try:
    model, features, threshold = load_artifacts()
except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()


st.subheader("Enter Customer Details")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0, value=840.0, step=10.0
        )

    submitted = st.form_submit_button("Predict Churn Risk")


if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }])

    # Apply the same one-hot encoding convention as model training
    customer_encoded = pd.get_dummies(
        customer, drop_first=True, dtype=int
    )
    customer_encoded = customer_encoded.reindex(
        columns=features, fill_value=0
    )

    probability = float(model.predict_proba(customer_encoded)[0, 1])
    prediction = int(probability >= threshold)

    st.subheader("Prediction Result")
    st.metric("Estimated Churn Probability", f"{probability:.1%}")

    if prediction == 1:
        st.warning("Higher churn risk — consider reviewing this customer's needs.")
    else:
        st.success("Lower churn risk at the selected threshold.")

    st.caption(
        f"Decision threshold: {threshold:.0%}. "
        "This estimate is not a guarantee of future customer behavior."
    )
