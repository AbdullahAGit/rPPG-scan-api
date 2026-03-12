from fastapi import FastAPI
import numpy as np

app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):

    signal = np.array(data["signal"])

    # dummy prediction for now
    test1 = float(signal.mean() * 100)
    test2 = float(signal.mean() * 60)

    return {
        "systolic": test1,
        "diastolic": test2
    }
