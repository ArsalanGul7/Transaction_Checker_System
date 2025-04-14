import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
from geopy.distance import geodesic

# Load model and encoder
model = joblib.load("Credit_Card_Fraud_Detection.jb")
encoder = joblib.load("label_encoder.jb")

# Helper function
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Custom CSS to change the background color
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;  /* Light blue background */
    }
    .stButton>button {
        background-color: #0073e6; /* Button color */
        color: white;
        border-radius: 5px;
        font-size: 16px;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        background-color: #f7f7f7;
    }
    .stSelectbox>div>div>input {
        border-radius: 5px;
        background-color: #f7f7f7;
    }
    .stSlider>div>div>div {
        background-color: #f7f7f7;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App Title
st.markdown("<h1 style='text-align: center;'>🔍 Credit Card Transaction Checker</h1>", unsafe_allow_html=True)
st.markdown("Use the form below to check if a transaction is legitimate or fraudulent.")

st.markdown("---")
st.header("🧾 Transaction Info")

col1, col2 = st.columns(2)

with col1:
    merchant = st.text_input("Merchant Name")
    category = st.text_input("Category")
    city = st.text_input("City")
    state = st.text_input("State")
    gender = st.selectbox("Gender", ["Male", "Female"])
    
with col2:
    amt = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
    hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)
    day = st.slider("Transaction Day", 1, 31, 15)
    month = st.slider("Transaction Month", 1, 12, 6)
    cc_num = st.text_input("Credit Card Number")

st.markdown("---")
st.header("📍 Location Info")

col3, col4 = st.columns(2)

with col3:
    lat = st.number_input("User Latitude", format="%.6f")
    long = st.number_input("User Longitude", format="%.6f")

with col4:
    merch_lat = st.number_input("Merchant Latitude", format="%.6f")
    merch_long = st.number_input("Merchant Longitude", format="%.6f")

# Calculate distance
distance = haversine(lat, long, merch_lat, merch_long)

# Prediction Button
st.markdown("---")
if st.button("🔎 Check Transaction"):
    if merchant and category and cc_num:
        input_data = pd.DataFrame([[merchant, category, city, state, amt, distance, hour, day, month, gender, cc_num]],
                                  columns=['merchant', 'category', 'city', 'state', 'amt', 'distance',
                                           'hour', 'day', 'month', 'gender', 'cc_num'])

        categorical_col = ['merchant', 'category', 'gender', 'city', 'state']
        for col in categorical_col:
            try:
                input_data[col] = encoder[col].transform(input_data[col])
            except ValueError:
                input_data[col] = -1

        input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        prediction = model.predict(input_data)[0]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"
        st.success(f"**Prediction:** {result}")
    else:
        st.error("Please fill in all the required fields.")