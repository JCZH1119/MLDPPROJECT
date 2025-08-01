import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load("random_forest_model.pkl")

# Page layout
st.title("Random Forest Predictor")
st.write("Enter the feature values below to get a prediction from the pre-trained model.")

# Define the feature names (replace with actual feature names from your training data)
feature_names = ['feature1', 'feature2', 'feature3']  # 🔁 CHANGE THIS

# Create input widgets for each feature
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"Enter value for {feature}:", value=0.0)

# When button is clicked, make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([user_input])  # Convert to DataFrame with one row
    prediction = model.predict(input_df)
    st.success(f"Predicted value: {prediction[0]:.2f}")
