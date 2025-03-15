import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("./diabetes_symptoms_model.pkl")

# Streamlit App
st.title("🏥 Diabetes Staging Predictor")
st.write("Enter your details below to predict your diabetes stage.")

# User Input Fields
age = st.number_input("Enter Age:", min_value=1, max_value=120, step=1, format="%d")

gender = st.selectbox("Select Gender:", ["", "Male", "Female"])  # Placeholder for empty selection
polyuria = st.selectbox("Do you experience Polyuria (Excessive Urination)?", ["", "Yes", "No"])
polydipsia = st.selectbox("Do you experience Polydipsia (Excessive Thirst)?", ["", "Yes", "No"])
sudden_weight_loss = st.selectbox("Have you had Sudden Weight Loss?", ["", "Yes", "No"])
polyphagia = st.selectbox("Do you experience Polyphagia (Excessive Hunger)?", ["", "Yes", "No"])
irritability = st.selectbox("Do you feel Irritability often?", ["", "Yes", "No"])
partial_paresis = st.selectbox("Do you have Partial Paresis (Muscle Weakness)?", ["", "Yes", "No"])

# Function to preprocess inputs
def preprocess_inputs():
    if not age or not gender or not polyuria or not polydipsia or not sudden_weight_loss or not polyphagia or not irritability or not partial_paresis:
        st.warning("⚠️ Please fill in all fields before predicting.")
        return None

    gender_val = 1 if gender == "Male" else 0
    polyuria_val = 1 if polyuria == "Yes" else 0
    polydipsia_val = 1 if polydipsia == "Yes" else 0
    sudden_weight_loss_val = 1 if sudden_weight_loss == "Yes" else 0
    polyphagia_val = 1 if polyphagia == "Yes" else 0
    irritability_val = 1 if irritability == "Yes" else 0
    partial_paresis_val = 1 if partial_paresis == "Yes" else 0

    return np.array([[age / 120, gender_val, polyuria_val, polydipsia_val, sudden_weight_loss_val, 
                      polyphagia_val, irritability_val, partial_paresis_val]])

# Prediction Button
if st.button("🔍 Predict Diabetes Stage"):
    input_data = preprocess_inputs()
    
    if input_data is not None:
        # Get Prediction Probability
        prediction_proba = model.predict_proba(input_data)
        diabetes_prob = prediction_proba[0][1]  # Probability of having diabetes

        # Decision Making with 3 Categories
        if diabetes_prob < 0.4:
            st.success("✅ No Diabetes Detected!")
        elif 0.4 <= diabetes_prob < 0.7:
            st.warning("⚠️ Mild Diabetes (Borderline Case) - Monitor Your Health")
        else:
            st.error("🚨 Diabetes Detected - Consult a Doctor")

        # Show Probability to User
        st.write(f"📊 **Model Confidence:** {diabetes_prob*100:.2f}%")
