"""FastAPI inference service for the Firefly context encoder saved by
firefly_encoder_predictor_train.py.

Loads model.safetensors and config.json at startup and serves a POST endpoint
that encodes a 16-sample light history into a latent state.

Example:
    POST /encode {"observation": [0, 0, 1, ...]}    (16 values, 0 = dark, 1 = light)
    -> {"embedding": [...], "embedding_dim": 8}
"""

import argparse

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from container_encoder_predictor_train_src.firefly_encoder_predictor_train import (
    ENCODER_CONFIG_PATH,
    ENCODER_WEIGHTS_PATH,
    Encoder,
)

app = FastAPI(title="Firefly Encoder Inference")
model: Encoder = None


class ObservationRequest(BaseModel):
    observation: list[float] = Field(default_factory=list)


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    embedding_dim: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/encode", response_model=EmbeddingResponse)
def encode(request: ObservationRequest):
    expected = model.config["input_length"]
    if len(request.observation) != expected:
        raise HTTPException(status_code=422, detail=f"observation must have {expected} values")

    with torch.no_grad():
        x = torch.tensor([request.observation], dtype=torch.float32).unsqueeze(1)  # (1, 1, L)
        embedding = model(x)[0].tolist()
    return EmbeddingResponse(embedding=embedding, embedding_dim=len(embedding))


def main():
    parser = argparse.ArgumentParser(description="Serve FastAPI inference for the trained Firefly encoder.")
    parser.add_argument("--weights-path", default=str(ENCODER_WEIGHTS_PATH))
    parser.add_argument("--config-path", default=str(ENCODER_CONFIG_PATH))
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8006, help="Port to bind (default: 8006)")
    args = parser.parse_args()

    global model
    model = Encoder.from_pretrained(args.config_path, args.weights_path)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
