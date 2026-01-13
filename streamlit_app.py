import streamlit as st
import requests
from PIL import Image
import io

BACKEND_URL = "https://ai-health-intelligence-system.onrender.com"

st.set_page_config(page_title="Health AI System", layout="centered")

st.title("🥗 AI-Powered Health Intelligence System")

st.markdown("Upload a food image to predict calories")

uploaded_file = st.file_uploader(
    "Upload food image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Food"):
        with st.spinner("Analyzing image..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            try:
                response = requests.post(
                    f"{BACKEND_URL}/predict-food",
                    files=files,
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()

                    st.success("Prediction successful 🎉")
                    st.write("### 🍎 Food:", data["food"])
                    st.write("### 📊 Confidence:", data["confidence"])
                    st.write("### 🔥 Estimated Calories:", data["estimated_calories"])

                else:
                    st.error("Backend error")
                    st.code(response.text)

            except Exception as e:
                st.error("Backend error")
                st.code(str(e))
