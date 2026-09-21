"""Trains the Firefly JEPA context encoder and predictor together.

Data: the CSVs written by rules_based_firefly.py (100 ms samples). For each
step t we build:
    x_t      = ldr_raw[t-15 .. t]      16-sample history (context window)
    x_{t+1}  = ldr_raw[t-14 .. t+1]    the same window advanced one step
    a_t      = led_state[t]            action applied at step t

JEPA objective (latent-space prediction, no pixel/sample reconstruction):
    s_t        = ContextEncoder(x_t)
    s*_{t+1}   = TargetEncoder(x_{t+1})         (EMA copy of the context encoder, no grad)
    s^_{t+1}   = Predictor(s_t, a_t)
    loss       = MSE(s^_{t+1}, s*_{t+1}) + var_weight * variance hinge on s_t
The variance term stops the encoder from collapsing to a constant embedding.

Outputs (safetensors weights + config.json architecture, per PRD):
    data/model_data/encoder/weights/model.safetensors
    data/model_data/encoder/architecture/config.json
    data/model_data/predictor/weights/model.safetensors
    data/model_data/predictor/architecture/config.json
    data/model_data/policy/config.json   (latent threshold calibrated from the training data)
"""

import argparse
import copy
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from safetensors.torch import load_file, save_file

# <repo root>/data, which is /app/data inside the containers
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TRAINING_DATA_DIR = DATA_DIR / "training_data"
MODEL_DIR = DATA_DIR / "model_data"

ENCODER_WEIGHTS_PATH = MODEL_DIR / "encoder" / "weights" / "model.safetensors"
ENCODER_CONFIG_PATH = MODEL_DIR / "encoder" / "architecture" / "config.json"
PREDICTOR_WEIGHTS_PATH = MODEL_DIR / "predictor" / "weights" / "model.safetensors"
PREDICTOR_CONFIG_PATH = MODEL_DIR / "predictor" / "architecture" / "config.json"
POLICY_CONFIG_PATH = MODEL_DIR / "policy" / "config.json"

TRAIN_CSV = TRAINING_DATA_DIR / "Firefly_Raw_Training_Samples.csv"
VAL_CSV = TRAINING_DATA_DIR / "Firefly_Raw_Validation_Samples.csv"

# ~1,500 parameters
DEFAULT_ENCODER_CONFIG = {
    "model_type": "firefly_encoder",
    "architectures": ["Encoder"],
    "input_length": 16,
    "in_channels": 1,
    "conv_channels": [8, 16],
    "strides": [1, 2],
    "kernel_size": 3,
    "latent_dim": 8,
}

# ~600 parameters
DEFAULT_PREDICTOR_CONFIG = {
    "model_type": "firefly_predictor",
    "architectures": ["Predictor"],
    "latent_dim": 8,
    "action_dim": 1,
    "hidden_dim": 32,
}


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class Encoder(nn.Module):
    """1D conv encoder: (B, 1, 16) binary light history -> (B, latent_dim)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        kernel = config["kernel_size"]
        padding = kernel // 2

        layers = []
        channels = config["in_channels"]
        length = config["input_length"]
        for out_channels, stride in zip(config["conv_channels"], config["strides"]):
            layers += [nn.Conv1d(channels, out_channels, kernel, stride=stride, padding=padding), nn.ReLU()]
            channels = out_channels
            length = (length + 2 * padding - kernel) // stride + 1

        self.conv = nn.Sequential(*layers, nn.Flatten())
        self.proj = nn.Linear(channels * length, config["latent_dim"])

    def forward(self, x):
        return self.proj(self.conv(x))

    @classmethod
    def from_pretrained(cls, config_path, weights_path) -> "Encoder":
        model = cls(_read_json(config_path))
        model.load_state_dict(load_file(str(weights_path)))
        return model.eval()


class Predictor(nn.Module):
    """(latent s(t), action a(t)) -> predicted latent s(t+1)."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config["latent_dim"] + config["action_dim"], config["hidden_dim"])
        self.norm = nn.LayerNorm(config["hidden_dim"])
        self.fc2 = nn.Linear(config["hidden_dim"], config["latent_dim"])

    def forward(self, latent, action):
        h = F.relu(self.norm(self.fc1(torch.cat([latent, action], dim=-1))))
        return self.fc2(h)

    @classmethod
    def from_pretrained(cls, config_path, weights_path) -> "Predictor":
        model = cls(_read_json(config_path))
        model.load_state_dict(load_file(str(weights_path)))
        return model.eval()


def save_pretrained(model: nn.Module, weights_path, config_path):
    """Weights as safetensors, architecture as config.json (Hugging Face style)."""
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    save_file(model.state_dict(), str(weights_path))
    _write_json(config_path, model.config)
    print(f"Saved {weights_path}\nSaved {config_path}")


def load_samples(csv_path):
    """Returns (ldr_raw, led_state) as float32 arrays, in file order."""
    ldr, led = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ldr.append(float(row["ldr_raw"]))
            led.append(float(row["led_state"]))
    return np.asarray(ldr, dtype=np.float32), np.asarray(led, dtype=np.float32)


def make_windows(ldr, led, window):
    """Returns x_t (N, 1, W), a_t (N, 1), x_{t+1} (N, 1, W) as tensors."""
    n = len(ldr)
    if n < window + 1:
        raise ValueError(f"Need at least {window + 1} samples, got {n}")
    windows = sliding_window_view(ldr, window)  # windows[i] ends at t = i + window - 1
    x = torch.from_numpy(np.ascontiguousarray(windows[:-1])).unsqueeze(1)
    x_next = torch.from_numpy(np.ascontiguousarray(windows[1:])).unsqueeze(1)
    a = torch.from_numpy(led[window - 1 : n - 1].copy()).unsqueeze(1)
    return x, a, x_next


def variance_hinge(latents, min_std=1.0):
    return F.relu(min_std - latents.std(dim=0)).mean()


@torch.no_grad()
def ema_update(target: nn.Module, source: nn.Module, momentum: float):
    for t, s in zip(target.parameters(), source.parameters()):
        t.mul_(momentum).add_(s, alpha=1 - momentum)


@torch.no_grad()
def evaluate(encoder, target_encoder, predictor, data):
    x, a, x_next = data
    s = encoder(x)
    loss = F.mse_loss(predictor(s, a), target_encoder(x_next)).item()
    return loss, s.std(dim=0).mean().item()


def train(args):
    torch.manual_seed(args.seed)

    encoder_config = _read_json(args.encoder_config) if args.encoder_config else DEFAULT_ENCODER_CONFIG
    predictor_config = _read_json(args.predictor_config) if args.predictor_config else DEFAULT_PREDICTOR_CONFIG
    window = encoder_config["input_length"]

    train_ldr, train_led = load_samples(args.train_csv)
    train_data = make_windows(train_ldr, train_led, window)
    val_data = make_windows(*load_samples(args.val_csv), window)
    print(f"Training windows: {len(train_data[0])}, validation windows: {len(val_data[0])}")

    encoder = Encoder(encoder_config)
    target_encoder = copy.deepcopy(encoder).requires_grad_(False)
    predictor = Predictor(predictor_config)
    n_params = sum(p.numel() for p in encoder.parameters()), sum(p.numel() for p in predictor.parameters())
    print(f"Encoder parameters: {n_params[0]}, predictor parameters: {n_params[1]}")

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=args.lr)

    x, a, x_next = train_data
    n = len(x)
    for epoch in range(1, args.epochs + 1):
        encoder.train()
        predictor.train()
        perm = torch.randperm(n)
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            s = encoder(x[idx])
            with torch.no_grad():
                s_target = target_encoder(x_next[idx])
            loss = F.mse_loss(predictor(s, a[idx]), s_target) + args.var_weight * variance_hinge(s)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_update(target_encoder, encoder, args.ema)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            encoder.eval()
            predictor.eval()
            train_loss, train_std = evaluate(encoder, target_encoder, predictor, train_data)
            val_loss, val_std = evaluate(encoder, target_encoder, predictor, val_data)
            print(
                f"Epoch [{epoch}/{args.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"latent_std(train/val)={train_std:.3f}/{val_std:.3f}"
            )

    encoder.eval()
    predictor.eval()
    return encoder, predictor, train_ldr


@torch.no_grad()
def calibrate_policy(encoder: Encoder, ldr, window) -> dict | None:
    """Finds the mean-latent threshold separating steady-light from steady-dark windows."""
    windows = torch.from_numpy(np.ascontiguousarray(sliding_window_view(ldr, window))).unsqueeze(1)
    means = encoder(windows).mean(dim=1)
    all_light = windows.amin(dim=(1, 2)) == 1
    all_dark = windows.amax(dim=(1, 2)) == 0
    if not all_light.any() or not all_dark.any():
        return None
    light_mean = means[all_light].mean().item()
    dark_mean = means[all_dark].mean().item()
    return {
        "model_type": "firefly_policy",
        "threshold": (light_mean + dark_mean) / 2,
        "light_above_threshold": light_mean > dark_mean,
        "flash_probability": 0.1,
        "light_mean_latent": light_mean,
        "dark_mean_latent": dark_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the Firefly JEPA encoder and predictor.")
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--val-csv", default=str(VAL_CSV))
    parser.add_argument("--encoder-config", help="Optional architecture JSON overriding the default encoder")
    parser.add_argument("--predictor-config", help="Optional architecture JSON overriding the default predictor")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ema", type=float, default=0.99, help="Target encoder EMA momentum")
    parser.add_argument("--var-weight", type=float, default=1.0, help="Weight of the anti-collapse variance term")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    encoder, predictor, train_ldr = train(args)

    print()
    save_pretrained(encoder, ENCODER_WEIGHTS_PATH, ENCODER_CONFIG_PATH)
    save_pretrained(predictor, PREDICTOR_WEIGHTS_PATH, PREDICTOR_CONFIG_PATH)

    policy_config = calibrate_policy(encoder, train_ldr, encoder.config["input_length"])
    if policy_config is None:
        print("WARNING: training data has no fully-dark and fully-light windows; policy threshold not calibrated.")
    else:
        _write_json(POLICY_CONFIG_PATH, policy_config)
        print(f"Saved {POLICY_CONFIG_PATH} (threshold={policy_config['threshold']:.4f})")


if __name__ == "__main__":
    main()
