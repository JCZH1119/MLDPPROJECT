import streamlit as st
import numpy as np
import joblib

model = joblib.load("random_forest_model.pkl")

st.title("HDB Resale Price Prediction")

st.write("Enter the property details below:")

floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=10.0, max_value=300.0, value=70.0)
cbd_dist = st.number_input("Distance to CBD (km)", min_value=0.0, max_value=50.0, value=10.0)
year = st.number_input("Transaction Year", min_value=1990, max_value=2030, value=2023)
lease_commence_date = st.number_input("Lease Commencement Year", min_value=1960, max_value=2030, value=2000)
latitude = st.number_input("Latitude", min_value=1.0, max_value=2.0, value=1.35)
longitude = st.number_input("Longitude", min_value=103.0, max_value=104.0, value=103.8)
closest_mrt_dist = st.number_input("Distance to Closest MRT (km)", min_value=0.0, max_value=10.0, value=0.5)
remaining_lease_years = st.number_input("Remaining Lease (years)", min_value=0.0, max_value=99.0, value=80.0)
storey_mid = st.selectbox("Is Storey Mid-Level?", options=[0, 1])  # 1 for Yes, 0 for No

input_data = np.array([[
    latitude,
    longitude,
    closest_mrt_dist,
    cbd_dist,
    floor_area_sqm,
    lease_commence_date,
    year,
    0,  
    remaining_lease_years,
    storey_mid
]])

# Prediction
if st.button("Predict Resale Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Resale Price: ${prediction[0]:,.2f}")
