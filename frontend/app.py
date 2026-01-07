import streamlit as st
import requests

st.set_page_config(page_title="AI Health Intelligence System", layout="centered")

st.title("🍎 AI Food Recognition & Calorie Estimator")

st.success("Frontend is running successfully 🚀")

API_URL = "http://127.0.0.1:8000/predict-food"

uploaded_file = st.file_uploader(
    "Upload a food image",
    type=["jpg", "jpeg", "png"]
)

grams = st.number_input(
    "Enter food weight (grams)",
    min_value=1,
    value=100
)

if uploaded_file is not None and st.button("Predict"):
    with st.spinner("Analyzing image..."):
        response = requests.post(
            API_URL,
            files={"file": uploaded_file},
            data={"grams": grams}
        )

        if response.status_code == 200:
            result = response.json()
            st.success("Prediction Successful ✅")
            st.write(f"**Food:** {result['food']}")
            st.write(f"**Confidence:** {result['confidence']}")
            st.write(f"**Estimated Calories:** {result['estimated_calories']} kcal")
        else:
            st.error("Backend error. Is FastAPI running?")
