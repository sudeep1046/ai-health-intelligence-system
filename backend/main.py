from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io

app = FastAPI(title="Health AI Backend")

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- PATHS ----------------
MODEL_PATH = "backend/model/food_classifier.h5"
CLASS_INDEX_PATH = "backend/model/class_indices.json"

model = None
index_to_class = {}

# ---------------- LAZY MODEL LOADING ----------------
def get_model():
    global model, index_to_class

    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)

        with open(CLASS_INDEX_PATH, "r") as f:
            class_indices = json.load(f)

        index_to_class = {v: k for k, v in class_indices.items()}
        print("Model loaded successfully")

    return model


# ---------------- IMAGE PREPROCESS ----------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# ---------------- ROUTES ----------------
@app.get("/")
def root():
    return {"status": "Health AI Backend Running"}


@app.post("/predict-food")
async def predict_food(file: UploadFile = File(...)):
    try:
        model = get_model()

        image_bytes = await file.read()
        image = preprocess_image(image_bytes)

        preds = model.predict(image)
        confidence = float(np.max(preds))
        idx = int(np.argmax(preds))

        food_name = index_to_class.get(idx, "unknown")
        estimated_calories = round(confidence * 100, 2)

        return {
            "food": food_name,
            "confidence": round(confidence, 3),
            "estimated_calories": estimated_calories
        }

    except Exception as e:
        return {"error": str(e)}
