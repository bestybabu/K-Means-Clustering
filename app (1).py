import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set Page configuration
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("🩺 Breast Cancer Co-relation Predictor")
st.write("""
This app uses a Machine Learning model (KNN + K-Means Clustering features) to predict the likelihood of breast cancer based on metabolic and clinical indicators.
""")

# Load artifacts
@st.cache_resource
def load_artifacts():
    with open('model_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

try:
    artifacts = load_artifacts()
    scaler = artifacts['scaler']
    kmeans_k2 = artifacts['kmeans_k2']
    classifier = artifacts['classifier']
    feature_names = artifacts['feature_names']
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# User input form
st.sidebar.header("Input Patient Biometric Data")

def get_user_inputs():
    inputs = {}
    # Setting default values close to dataset averages
    inputs['age'] = st.sidebar.slider("Age", 10, 100, 57)
    inputs['bmi'] = st.sidebar.slider("BMI", 10.0, 50.0, 27.5, step=0.1)
    inputs['glucose'] = st.sidebar.slider("Glucose", 40, 300, 97)
    inputs['insulin'] = st.sidebar.slider("Insulin", 1.0, 100.0, 10.0, step=0.1)
    inputs['homa'] = st.sidebar.slider("HOMA", 0.1, 30.0, 2.6, step=0.1)
    inputs['leptin'] = st.sidebar.slider("Leptin", 1.0, 100.0, 26.5, step=0.1)
    inputs['adiponectin'] = st.sidebar.slider("Adiponectin", 1.0, 40.0, 10.0, step=0.1)
    inputs['resistin'] = st.sidebar.slider("Resistin", 1.0, 100.0, 14.7, step=0.1)
    inputs['mcp_1'] = st.sidebar.slider("MCP-1", 10.0, 2000.0, 534.0, step=10.0)
    return pd.DataFrame([inputs], columns=feature_names)

input_df = get_user_inputs()

# Display the inputted features
st.subheader("Patient Biometric Summary")
st.dataframe(input_df)

if st.button("Predict Diagnostic Status"):
    # 1. Scale input features
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
    
    # 2. Assign K-Means cluster
    cluster_label = kmeans_k2.predict(input_scaled)[0]
    
    # 3. Create dummy column matching dummy layout: 'cluster_2' (representing cluster 1 as drop_first=True was used)
    # Since our training dummy variable was 'cluster_True' or 'cluster_1' depending on pd.get_dummies naming scheme,
    # let's replicate the columns expected by the classifier:
    # In training, X_with_k2 columns were original scaled features + 'cluster_1' (if drop_first dropped cluster 0)
    # Let's assign cluster dummy column explicitly based on model expectation
    expected_features = classifier.feature_names_in_
    
    # Re-build final row for prediction
    for col in expected_features:
        if col in input_scaled_df.columns:
            continue
        else:
            # This is the cluster dummy column (e.g., 'cluster_1' or 'cluster_1.0')
            # If assigned cluster is 1, dummy variable is 1 (True), otherwise 0 (False)
            # We match the value based on string checking
            val = True if str(cluster_label) in col else False
            input_scaled_df[col] = val
            
    final_input = input_scaled_df[expected_features]
    
    # Make prediction
    prediction = classifier.predict(final_input)[0]
    probabilities = classifier.predict_proba(final_input)[0]
    
    st.markdown("--- ")
    if prediction == 1:
        st.success("🎉 Prediction: **Healthy Control**")
    else:
        st.warning("⚠️ Prediction: **Patient (Breast Cancer Diagnostic Group)**")
        
    # Show probability details
    st.write(f"Confidence: **{probabilities[prediction-1]:.2%}**")
