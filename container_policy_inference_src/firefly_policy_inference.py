"""FastAPI inference service for a Firefly policy saved by
firefly_policy_train.py.

Loads model.safetensors + config.json from a model directory and serves a
POST endpoint that chooses an action given a latent embedding.

Example:
    POST /act {"embedding": [...]}
    -> {"action": [...], "action_dim": 4}
"""

import argparse
import json
import os

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from safetensors.torch import load_file

from container_policy_train_src.firefly_policy_train import Policy, OUT_DIR

app = FastAPI(title="Firefly Policy Inference")
model: Policy = None


class ActRequest(BaseModel):
    embedding: list[float]


class ActionResponse(BaseModel):
    action: list[float]
    action_dim: int


def load_model(model_dir: str) -> Policy:
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    loaded = Policy(embedding_dim=config["embedding_dim"], action_dim=config["action_dim"])
    state_dict = load_file(os.path.join(model_dir, "model.safetensors"))
    loaded.load_state_dict(state_dict)
    loaded.eval()
    return loaded


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/act", response_model=ActionResponse)
def act(request: ActRequest):
    with torch.no_grad():
        x = torch.tensor([request.embedding], dtype=torch.float32)
        action = model(x)[0].tolist()
    return ActionResponse(action=action, action_dim=len(action))


def main():
    parser = argparse.ArgumentParser(description="Serve FastAPI inference for a trained Firefly policy.")
    parser.add_argument(
        "--model-dir",
        default=OUT_DIR,
        help=f"Directory containing model.safetensors and config.json (default: {OUT_DIR})",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind (default: 8003)")
    args = parser.parse_args()

    global model
    model = load_model(args.model_dir)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
