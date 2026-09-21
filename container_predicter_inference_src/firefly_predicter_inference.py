"""FastAPI inference service for the Firefly predictor saved by
firefly_encoder_predictor_train.py.

Loads model.safetensors and config.json at startup and serves a POST endpoint
that predicts the next latent state given the current latent state and an
LED action.

Example:
    POST /predict {"embedding": [...], "action": [1]}
    -> {"predicted_embedding": [...], "embedding_dim": 8}
"""

import argparse

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from container_encoder_predictor_train_src.firefly_encoder_predictor_train import (
    PREDICTOR_CONFIG_PATH,
    PREDICTOR_WEIGHTS_PATH,
    Predictor,
)

app = FastAPI(title="Firefly Predictor Inference")
model: Predictor = None


class PredictRequest(BaseModel):
    embedding: list[float]
    action: list[float] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    predicted_embedding: list[float]
    embedding_dim: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    latent_dim = model.config["latent_dim"]
    action_dim = model.config["action_dim"]
    action_values = request.action or [0.0] * action_dim
    if len(request.embedding) != latent_dim:
        raise HTTPException(status_code=422, detail=f"embedding must have {latent_dim} values")
    if len(action_values) != action_dim:
        raise HTTPException(status_code=422, detail=f"action must have {action_dim} values")

    with torch.no_grad():
        embedding = torch.tensor([request.embedding], dtype=torch.float32)
        action = torch.tensor([action_values], dtype=torch.float32)
        predicted_embedding = model(embedding, action)[0].tolist()
    return PredictionResponse(predicted_embedding=predicted_embedding, embedding_dim=len(predicted_embedding))


def main():
    parser = argparse.ArgumentParser(description="Serve FastAPI inference for the trained Firefly predictor.")
    parser.add_argument("--weights-path", default=str(PREDICTOR_WEIGHTS_PATH))
    parser.add_argument("--config-path", default=str(PREDICTOR_CONFIG_PATH))
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8007, help="Port to bind (default: 8007)")
    args = parser.parse_args()

    global model
    model = Predictor.from_pretrained(args.config_path, args.weights_path)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
