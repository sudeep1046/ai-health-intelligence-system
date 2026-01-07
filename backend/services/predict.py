import json
import numpy as np
import tensorflow as tf
from backend.services.preprocess import preprocess_image_bytes

# Correct paths (matching your files)
MODEL_PATH = "backend/model/food_classifier.h5"
CLASS_PATH = "backend/model/class_indices.json"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class indices
with open(CLASS_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → food name
idx_to_class = {v: k for k, v in class_indices.items()}

# Simple calorie database (per 100g)
calorie_map = {
    "apple": 52,
    "banana": 96,
    "carrot": 41,
    "tomato": 18
}

def predict_food(image_bytes, grams=100):
    img = preprocess_image_bytes(image_bytes)

    preds = model.predict(img)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    food = idx_to_class[idx]
    calories = (calorie_map.get(food, 0) * grams) / 100

    return food, confidence, calories
