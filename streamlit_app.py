import streamlit as st
import tensorflow as tf
import numpy as np
import json
import cv2
from PIL import Image

st.set_page_config(page_title="AI Food Calorie Estimator", layout="centered")
st.title("🍎 AI Food Recognition & Calorie Estimator")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("backend/model/food_classifier")

@st.cache_resource
def load_classes():
    with open("backend/model/class_indices.json", "r") as f:
        return {int(v): k for k, v in json.load(f).items()}

model = load_model()
class_map = load_classes()

def preprocess(img):
    img = np.array(img)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

uploaded = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
grams = st.number_input("Enter food weight (grams)", min_value=1, value=100)

if uploaded and st.button("Predict"):
    image = Image.open(uploaded).convert("RGB")
    x = preprocess(image)

    preds = model.predict(x)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))

    food = class_map.get(idx, "Unknown")
    calories = round(0.96 * grams, 2)  # your logic

    st.success("Prediction Successful ✅")
    st.write(f"**Food:** {food}")
    st.write(f"**Confidence:** {conf:.2f}")
    st.write(f"**Estimated Calories:** {calories} kcal")
