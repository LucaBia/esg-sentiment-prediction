# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import uvicorn


class DeepMLP(nn.Module):
    def __init__(self, input_size):
        super(DeepMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)


app = FastAPI()


class PredictionRequest(BaseModel):
    features: list[float]


model = DeepMLP(input_size=5)
model.load_state_dict(torch.load("mlp_model.pth"))
model.eval()


@app.get("/")
def read_root():
    return {"message": "API is running!"}


@app.post("/predict/")
async def predict(data: PredictionRequest):
    try:
        input_tensor = torch.tensor([data.features], dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor).item()

        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al hacer la predicción: {str(e)}")
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)