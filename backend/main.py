from fastapi import FastAPI, UploadFile, File
from backend.services.predict import predict_food

app = FastAPI(title="Health AI Food Recognition API")

@app.get("/")
def root():
    return {"status": "Health AI Backend Running"}

@app.post("/predict-food")
async def predict(file: UploadFile = File(...), grams: int = 100):
    image_bytes = await file.read()
    food, confidence, calories = predict_food(image_bytes, grams)

    return {
        "food": food,
        "confidence": round(confidence, 3),
        "estimated_calories": calories
    }

