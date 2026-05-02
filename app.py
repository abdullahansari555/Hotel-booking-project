import streamlit as st
import pickle
import numpy as np


model = pickle.load(open("hotel_booking_model.pkl", "rb"))


st.set_page_config(
    page_title="Hotel Booking AI",
    page_icon="🏨",
    layout="centered"
)


st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 40px;
        color: #1f77b4;
        font-weight: bold;
    }
    .sub-title {
        text-align: center;
        font-size: 18px;
        color: gray;
    }
    .box {
        padding: 15px;
        border-radius: 10px;
        background-color: #f5f5f5;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🏨 Hotel Booking Cancellation Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>AI based prediction system for hotel booking cancellation</div>", unsafe_allow_html=True)

st.write("")


st.markdown("### 📊 Enter Booking Details")

col1, col2 = st.columns(2)

with col1:
    lead_time = st.number_input("⏳ Lead Time", min_value=0, max_value=500, value=50)
    adults = st.number_input("👨 Adults", min_value=1, max_value=10, value=2)

with col2:
    adr = st.number_input("💰 Average Daily Rate (ADR)", min_value=0.0, value=100.0)
    prev_cancel = st.number_input("❌ Previous Cancellations", min_value=0, max_value=10, value=0)

st.write("")


if st.button("🔍 Predict Now"):

    input_data = np.array([[lead_time, adr, adults, prev_cancel]])
    result = model.predict(input_data)

    st.write("")

  
    if result[0] == 1:
        st.error("❌ Booking WILL be CANCELLED")
        st.markdown("<div class='box'>⚠ Recommendation: Take advance payment or verify customer history.</div>", unsafe_allow_html=True)
    else:
        st.success("✔ Booking will NOT be cancelled")
        st.markdown("<div class='box'>✅ Safe booking confirmed.</div>", unsafe_allow_html=True)


st.write("")
st.markdown("<p style='text-align:center; color:gray;'>Made with ❤️ using Machine Learning</p>", unsafe_allow_html=True)
