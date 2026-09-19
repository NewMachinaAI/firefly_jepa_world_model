"""Trains a stub linear policy for the Firefly JEPA world model, saves it
(safetensors + config.json), and reports loss.

This is a stub: it trains on random synthetic embedding -> action pairs
purely to exercise the train -> save -> load -> serve pipeline. Replace the
synthetic data and Policy architecture with a real policy trained via RL or
imitation learning against the encoder/predicter.
"""

import json
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

MODEL_TYPE = "firefly_policy"
OUT_DIR = "policy_model"

EMBEDDING_DIM = 16
ACTION_DIM = 4


class Policy(nn.Module):
    def __init__(self, embedding_dim: int, action_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, action_dim)
        self.activation = nn.Tanh()

    def forward(self, embedding):
        return self.activation(self.linear(embedding))


def train():
    torch.manual_seed(0)

    # TODO: replace with real training (RL rollouts or imitation targets).
    # Stub regresses toward random targets purely to exercise the pipeline.
    embeddings = torch.randn(256, EMBEDDING_DIM)
    target_actions = torch.empty(256, ACTION_DIM).uniform_(-1.0, 1.0)

    model = Policy(EMBEDDING_DIM, ACTION_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        predicted_actions = model(embeddings)
        loss = criterion(predicted_actions, target_actions)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model


def save_pretrained(model: Policy, out_dir: str, embedding_dim: int, action_dim: int):
    os.makedirs(out_dir, exist_ok=True)

    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(out_dir, "model.safetensors"))

    config = {
        "model_type": MODEL_TYPE,
        "architectures": ["Policy"],
        "embedding_dim": embedding_dim,
        "action_dim": action_dim,
        "activation": "tanh",
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved model.safetensors and config.json to {out_dir}/")


if __name__ == "__main__":
    model = train()
    save_pretrained(model, OUT_DIR, embedding_dim=EMBEDDING_DIM, action_dim=ACTION_DIM)
