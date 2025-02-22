from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib  # ✅ Use joblib for compressed model
import os

app = FastAPI()

# ✅ Load the compressed ML model
model_path = os.path.join(os.path.dirname(__file__), "content/truemark-backend/fakeproduct_tm_model.pkl")
with open(model_path, "rb") as f:
    model = joblib.load(f)

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "TrueMark API is live on Vercel 🚀"})

@app.post("/predict/")
def predict(data: dict):
    try:
        # Example: Assuming your model expects a list of features
        features = data.get("features", [])
        prediction = model.predict([features])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
