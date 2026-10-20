
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model_pipeline.joblib"
HTML_PATH = BASE_DIR / "index.html"

app = FastAPI(
    title="QuickBite Delivery Time Predictor",
    description="Predicts estimated food delivery time using a trained ML pipeline.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
model_error = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model_error = "Model file not found."
except Exception as exc:
    model_error = str(exc)


class PredictionInput(BaseModel):
    Distance_km: float = Field(gt=0, le=200)
    Preparation_Time_min: int = Field(ge=0, le=300)
    Courier_Experience_yrs: float = Field(ge=0, le=60)
    Weather_Condition: str
    Traffic_Level: str
    Time_of_Day: str


@app.get("/", response_class=HTMLResponse)
def read_root():
    if HTML_PATH.exists():
        return HTML_PATH.read_text(encoding="utf-8")

    return """
    <!DOCTYPE html>
    <html lang="en">
    <head><meta charset="UTF-8"><title>QuickBite</title></head>
    <body>
        <h1>QuickBite Delivery Time Predictor</h1>
        <p>API is online. Open <a href="/docs">API documentation</a>.</p>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "error",
        "model_loaded": model is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "model_error": model_error,
    }


@app.post("/predict")
def predict(data: PredictionInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction model is unavailable: {model_error}",
        )

    try:
        input_data = pd.DataFrame([data.model_dump()])
        prediction = model.predict(input_data)[0]

        return {
            "predicted_time_min": round(float(prediction), 2),
            "unit": "minutes",
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc
