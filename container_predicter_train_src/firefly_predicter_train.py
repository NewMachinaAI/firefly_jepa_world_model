"""Trains a stub linear predicter for the Firefly JEPA world model, saves it
(safetensors + config.json), and reports loss.

This is a stub: it trains on random synthetic (embedding, action) ->
next-embedding triples purely to exercise the train -> save -> load -> serve
pipeline. Replace the synthetic data and Predicter architecture with a real
JEPA predictor trained on real embedding/action/next-embedding transitions.
"""

import json
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

MODEL_TYPE = "firefly_predicter"
OUT_DIR = "predicter_model"

EMBEDDING_DIM = 16
ACTION_DIM = 4


class Predicter(nn.Module):
    def __init__(self, embedding_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim + action_dim, embedding_dim)

    def forward(self, embedding, action):
        return self.linear(torch.cat([embedding, action], dim=-1))


def train():
    torch.manual_seed(0)

    # TODO: replace with real (embedding, action, next_embedding)
    # transitions captured from the environment. Stub regresses toward
    # random targets purely to exercise the pipeline.
    embeddings = torch.randn(256, EMBEDDING_DIM)
    actions = torch.randn(256, ACTION_DIM)
    next_embeddings = torch.randn(256, EMBEDDING_DIM)

    model = Predicter(EMBEDDING_DIM, ACTION_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted = model(embeddings, actions)
        loss = criterion(predicted, next_embeddings)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model


def save_pretrained(model: Predicter, out_dir: str, embedding_dim: int, action_dim: int):
    os.makedirs(out_dir, exist_ok=True)

    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(out_dir, "model.safetensors"))

    config = {
        "model_type": MODEL_TYPE,
        "architectures": ["Predicter"],
        "embedding_dim": embedding_dim,
        "action_dim": action_dim,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved model.safetensors and config.json to {out_dir}/")


if __name__ == "__main__":
    model = train()
    save_pretrained(model, OUT_DIR, embedding_dim=EMBEDDING_DIM, action_dim=ACTION_DIM)
