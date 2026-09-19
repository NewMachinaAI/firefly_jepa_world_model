"""FastAPI inference service for a Firefly encoder saved by
firefly_encoder_train.py.

Loads model.safetensors + config.json from a model directory and serves a
POST endpoint that encodes an observation vector into a latent embedding.

Example:
    POST /encode {"observation": [0.1, 0.2, ...]}
    -> {"embedding": [...], "embedding_dim": 16}
"""

import argparse
import json
import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from safetensors.torch import load_file

from container_encoder_train_src.firefly_encoder_train import Encoder, OUT_DIR

app = FastAPI(title="Firefly Encoder Inference")
model: Encoder = None


class ObservationRequest(BaseModel):
    observation: list[float] = Field(default_factory=list)


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    embedding_dim: int


def load_model(model_dir: str) -> Encoder:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    loaded = Encoder(observation_dim=config["observation_dim"], embedding_dim=config["embedding_dim"])
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    loaded.load_state_dict(state_dict)
    loaded.eval()
    return loaded


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/encode", response_model=EmbeddingResponse)
def encode(request: ObservationRequest):
    with torch.no_grad():
        x = torch.tensor([request.observation], dtype=torch.float32)
        embedding = model(x)[0].tolist()
    return EmbeddingResponse(embedding=embedding, embedding_dim=len(embedding))


def main():
    parser = argparse.ArgumentParser(description="Serve FastAPI inference for a trained Firefly encoder.")
    parser.add_argument(
        "--model-dir",
        default=OUT_DIR,
        help=f"Directory containing model.safetensors and config.json (default: {OUT_DIR})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind (default: 8001)")
    args = parser.parse_args()

    global model
    model = load_model(args.model_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
