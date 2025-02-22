from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import requests
from io import BytesIO

app = FastAPI()

# Load model from Google Drive
def load_model():
    file_id = "10rcpRSAQwkaBV8cAkVXS9MQpCYPZASZp"  # Replace with your Google Drive file ID
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    model = joblib.load(BytesIO(response.content))
    return model

model = load_model()

@app.get("/")
def read_root():
    return JSONResponse(content={"message": "TrueMark API is live on Vercel 🚀"})

@app.post("/predict/")
def predict(data: dict):
    try:
        features = data.get("features", [])
        prediction = model.predict([features])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
