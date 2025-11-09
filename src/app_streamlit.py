# app_streamlit.py
import streamlit as st
from predict import predict_co2, recommend_actions

st.set_page_config(page_title="AI Carbon Footprint Estimator", page_icon="🌱")

st.title("🌍 AI-Powered Carbon Footprint Estimator")
st.write("Estimate your monthly carbon emissions and get tips to reduce your impact.")

# --- Input fields ---
electricity_kwh = st.number_input("🔌 Monthly electricity use (kWh)", min_value=0.0, max_value=10000.0, value=250.0, step=10.0)
vehicle_km = st.number_input("🚗 Monthly vehicle distance (km)", min_value=0.0, max_value=20000.0, value=800.0, step=10.0)
vehicle_type = st.selectbox("Vehicle type", ['petrol', 'diesel', 'ev', 'public'])
flights_hours = st.number_input("✈️ Annual flight hours", min_value=0.0, max_value=500.0, value=2.0, step=1.0)
food_type = st.selectbox("🥗 Diet type", ['meat_heavy', 'mixed', 'veg'])
waste_kg = st.number_input("🗑️ Monthly waste generated (kg)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

# --- Predict button ---
if st.button("Estimate My Carbon Footprint"):
    result = predict_co2(electricity_kwh, vehicle_km, vehicle_type, flights_hours, food_type, waste_kg)
    st.subheader(f"🌡️ Estimated CO₂ Emission: {result:.2f} kg/month")

    st.write("### 💡 Recommendations to Reduce Your Footprint:")
    tips = recommend_actions(electricity_kwh, vehicle_km, vehicle_type, flights_hours, food_type, waste_kg, result)
    for t in tips:
        st.write("- ", t)

st.markdown("---")
st.caption("Built by Shruti Goyal | Powered by Python, Streamlit & Machine Learning")
