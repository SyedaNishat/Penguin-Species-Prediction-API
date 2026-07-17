from pathlib import Path
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(title="Penguin Species Prediction API")

# -----------------------------
# Load Model
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "penguin_model.pkl"

model = joblib.load(MODEL_PATH)

# -----------------------------
# Request Model
# -----------------------------
class PenguinRequest(BaseModel):
    bill_length_mm: float = Field(..., gt=0)
    bill_depth_mm: float = Field(..., gt=0)
    flipper_length_mm: float = Field(..., gt=0)
    body_mass_g: float = Field(..., gt=0)


# -----------------------------
# Home Endpoint
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "Penguin Species Prediction API"
    }


# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/predict")
def predict(data: PenguinRequest):

    input_df = pd.DataFrame(
        [[
            data.bill_length_mm,
            data.bill_depth_mm,
            data.flipper_length_mm,
            data.body_mass_g
        ]],
        columns=[
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g"
        ]
    )

    prediction = model.predict(input_df)[0]

    return {
        "predicted_species": prediction
    }