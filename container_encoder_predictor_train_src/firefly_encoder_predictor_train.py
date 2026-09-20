"""Trains a stub linear autoencoder for the Firefly JEPA encoder, saves it
(safetensors + config.json), and reports reconstruction loss.

This is a stub: it trains on random synthetic observations purely to
exercise the train -> save -> load -> serve pipeline. Replace the synthetic
data and Encoder architecture with a real JEPA encoder trained on real
observations.
"""

import json
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

MODEL_TYPE = "firefly_encoder"
OUT_DIR = "encoder_model"

OBSERVATION_DIM = 16
EMBEDDING_DIM = 16


class Encoder(nn.Module):
    def __init__(self, observation_dim: int, embedding_dim: int):
        super().__init__()
        self.linear = nn.Linear(observation_dim, embedding_dim)

    def forward(self, x):
        return self.linear(x)


class _Decoder(nn.Module):
    """Reconstruction head used only during stub training; not saved."""

    def __init__(self, embedding_dim: int, observation_dim: int):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, observation_dim)

    def forward(self, z):
        return self.linear(z)


def train():
    torch.manual_seed(0)

    # TODO: replace with real observations. Stub trains a linear
    # autoencoder on random data purely to exercise the pipeline.
    X = torch.randn(256, OBSERVATION_DIM)

    encoder = Encoder(OBSERVATION_DIM, EMBEDDING_DIM)
    decoder = _Decoder(EMBEDDING_DIM, OBSERVATION_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)

    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstruction = decoder(encoder(X))
        loss = criterion(reconstruction, X)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return encoder


def save_pretrained(model: Encoder, out_dir: str, observation_dim: int, embedding_dim: int):
    """Save weights as safetensors and architecture as config.json, Hugging Face style."""
    os.makedirs(out_dir, exist_ok=True)

    state_dict = model.state_dict()
    save_file(state_dict, os.path.join(out_dir, "model.safetensors"))

    config = {
        "model_type": MODEL_TYPE,
        "architectures": ["Encoder"],
        "observation_dim": observation_dim,
        "embedding_dim": embedding_dim,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved model.safetensors and config.json to {out_dir}/")


if __name__ == "__main__":
    model = train()
    save_pretrained(model, OUT_DIR, observation_dim=OBSERVATION_DIM, embedding_dim=EMBEDDING_DIM)
