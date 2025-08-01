import streamlit as st
import numpy as np
import joblib

model = joblib.load("random_forest_model.pkl")

st.set_page_config(page_title="HDB Resale Price Predictor", layout="centered")

st.title("🏠 HDB Resale Price Predictor")
st.markdown("Fill in the property details below to get an estimated resale price.")


st.header("🏘️ Property Info")

floor_area_sqm = st.number_input(
    "Floor Area (sqm)", 
    min_value=10.0, max_value=300.0, value=85.0,
    help="Typical HDB flats range from 30 to 150 sqm"
)

storey_mid = st.radio(
    "Storey Level",
    options=[1, 0],
    format_func=lambda x: "Mid Floor (4–6)" if x == 1 else "Other",
    help="Mid floors are often more desirable"
)


st.header("📍 Location Info")

cbd_dist = st.slider(
    "Distance to CBD (km)", 
    min_value=0.0, max_value=30.0, value=9.5,
    help="Distance to Central Business District (e.g., Raffles Place)"
)

closest_mrt_dist = st.slider(
    "Distance to Closest MRT (km)", 
    min_value=0.0, max_value=5.0, value=0.3,
    help="Proximity to public transport increases value"
)


st.header("📅 Lease & Transaction")

lease_commence_date = st.number_input(
    "Lease Commencement Year", 
    min_value=1960, max_value=2030, value=1999
)

year = st.number_input(
    "Transaction Year", 
    min_value=1990, max_value=2030, value=2023
)

remaining_lease_years = st.slider(
    "Remaining Lease (years)", 
    min_value=0.0, max_value=99.0, value=74.0,
    help="HDB flats have a 99-year lease"
)

if st.button("🔍 Predict Resale Price"):


    input_data = np.array([[
        0.0,  
        0.0,  
        closest_mrt_dist,
        cbd_dist,
        floor_area_sqm,
        lease_commence_date,
        year,
        0,  
        remaining_lease_years,
        storey_mid
    ]])

    try:
        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Resale Price: **${prediction[0]:,.2f}**")
    except Exception as e:
        st.error("Prediction failed. Check that your model input size matches.")
        st.exception(e)
