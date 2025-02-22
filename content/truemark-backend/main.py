from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pickle
import os

app = FastAPI()

# ✅ Load the ML model from root directory
model_path = os.path.join(os.path.dirname(__file__), "content/truemark-backend/fakeproduct_tm_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "TrueMark API is live on Render 🚀"})

@app.post("/predict/")
def predict(data: dict):
    try:
        features = data.get("features", [])
        prediction = model.predict([features])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
