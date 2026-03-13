from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np

from model import PPGtoBPNet

app = FastAPI()

device = torch.device("cpu")

# load model once when server starts
model = PPGtoBPNet()
model.load_state_dict(torch.load("ppg_to_bp.pth", map_location=device))
model.eval()


class SignalInput(BaseModel):
    signal: list[float]


@app.get("/")
def root():
    return {"message": "API running"}


@app.post("/predict")
def predict(data: SignalInput):

    signal = np.array(data.signal)

    # normalize like training
    signal = (signal - signal.mean()) / (signal.std()+1e-6)

    tensor = torch.tensor(signal).float().unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor).numpy()[0]

    systolic = float(pred[0])
    diastolic = float(pred[1])

    return {
        "systolic": systolic,
        "diastolic": diastolic
    }
