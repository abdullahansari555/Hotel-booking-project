import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("hotel_booking_model.pkl", "rb"))

st.set_page_config(page_title="Hotel Booking Prediction")
st.title("🏨 Hotel Booking Cancellation Predictor")

st.write("Enter booking details:")

# Inputs
lead_time = st.number_input("Lead Time", 0, 500)
adr = st.number_input("ADR (Price)", 0.0, 1000.0)
adults = st.number_input("Adults", 1, 10)
prev_cancel = st.number_input("Previous Cancellations", 0, 10)

# Predict button
if st.button("Predict"):
    data = np.array([[lead_time, adr, adults, prev_cancel]])
    result = model.predict(data)

    if result[0] == 1:
        st.error("❌ Booking WILL be Canceled")
    else:
        st.success("✔ Booking will NOT be Canceled")