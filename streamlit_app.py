import streamlit as st
import requests
from PIL import Image

# ---------------- CONFIG ----------------
BACKEND_URL = "http://127.0.0.1:8000/predict-food"  # will change after backend deploy

st.set_page_config(
    page_title="AI Health Intelligence System",
    page_icon="🥗",
    layout="centered"
)

# ---------------- UI ----------------
st.title("🥗 AI-Powered Health Intelligence System")
st.markdown(
    """
    Upload a food image to **identify the food item** and  
    **estimate calories** using a deep learning model.
    """
)

uploaded_file = st.file_uploader(
    "Upload a food image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict Food & Calories"):
        with st.spinner("Analyzing image..."):
            try:
                response = requests.post(
                    BACKEND_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()

                    st.subheader("✅ Prediction Result")
                    st.write(f"**Food Item:** {result['food']}")
                    st.write(f"**Confidence:** {result['confidence']:.2f}")
                    st.write(f"**Estimated Calories:** {result['estimated_calories']} kcal")

                else:
                    st.error("Backend error. Please try again later.")

            except Exception:
                st.warning(
                    "⚠ Backend is not reachable.\n\n"
                    "This frontend is deployed successfully.\n"
                    "Backend deployment will be connected next."
                )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption(
    "Streamlit Frontend • FastAPI Backend • TensorFlow Model\n"
    "Deployed architecture-ready AI system"
)
