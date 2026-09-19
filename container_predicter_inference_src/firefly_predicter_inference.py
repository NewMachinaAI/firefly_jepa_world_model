"""FastAPI inference service for a Firefly predicter saved by
firefly_predicter_train.py.

Loads model.safetensors + config.json from a model directory and serves a
POST endpoint that predicts the next latent embedding given a current
embedding (and optional action).

Example:
    POST /predict {"embedding": [...], "action": [...]}
    -> {"predicted_embedding": [...], "embedding_dim": 16}
"""

import argparse
import json
import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from safetensors.torch import load_file

from container_predicter_train_src.firefly_predicter_train import Predicter, OUT_DIR

app = FastAPI(title="Firefly Predicter Inference")
model: Predicter = None
action_dim: int = 0


class PredictRequest(BaseModel):
    embedding: list[float]
    action: list[float] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    predicted_embedding: list[float]
    embedding_dim: int


def load_model(model_dir: str) -> Predicter:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    global action_dim
    action_dim = config["action_dim"]

    loaded = Predicter(embedding_dim=config["embedding_dim"], action_dim=action_dim)
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    loaded.load_state_dict(state_dict)
    loaded.eval()
    return loaded


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest):
    action_values = request.action if request.action else [0.0] * action_dim
    with torch.no_grad():
        embedding = torch.tensor([request.embedding], dtype=torch.float32)
        action = torch.tensor([action_values], dtype=torch.float32)
        predicted_embedding = model(embedding, action)[0].tolist()
    return PredictionResponse(predicted_embedding=predicted_embedding, embedding_dim=len(predicted_embedding))


def main():
    parser = argparse.ArgumentParser(description="Serve FastAPI inference for a trained Firefly predicter.")
    parser.add_argument(
        "--model-dir",
        default=OUT_DIR,
        help=f"Directory containing model.safetensors and config.json (default: {OUT_DIR})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8002, help="Port to bind (default: 8002)")
    args = parser.parse_args()

    global model
    model = load_model(args.model_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
