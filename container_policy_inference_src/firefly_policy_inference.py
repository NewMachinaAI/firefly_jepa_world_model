"""FastAPI inference service for the Firefly policy.

The policy has no neural network: it is an implicit policy using one-step
latent MPC. The harness sends the predictor's next-latent prediction for each
candidate LED action; the policy assigns each candidate a scalar cost from the
mean of its predicted latent and returns the cheapest action.

    Light (mean latent on the light side of the threshold):
        cost(a=0) = 0, cost(a=1) = 1                  -> LED stays off
    Dark:
        cost(a=0) = 0
        cost(a=1) = -1 with probability flash_probability, else +1
                                                       -> random firefly pulses

The threshold and its orientation are calibrated by the training script and
read from data/model_data/policy/config.json at startup.

Example:
    POST /act {"candidates": [{"action": 0, "predicted_latent": [...]},
                              {"action": 1, "predicted_latent": [...]}]}
    -> {"action": 1, "costs": [0.0, -1.0]}
"""

import argparse
import json
import random
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "data" / "model_data" / "policy" / "config.json"

app = FastAPI(title="Firefly Policy Inference")
policy_config = {"threshold": 0.0, "light_above_threshold": True, "flash_probability": 0.1}


class Candidate(BaseModel):
    action: int
    predicted_latent: list[float] = Field(min_length=1)


class ActRequest(BaseModel):
    candidates: list[Candidate] = Field(min_length=1)


class ActionResponse(BaseModel):
    action: int
    costs: list[float]


def is_light(latent: list[float]) -> bool:
    mean = sum(latent) / len(latent)
    if policy_config["light_above_threshold"]:
        return mean >= policy_config["threshold"]
    return mean < policy_config["threshold"]


def candidate_cost(candidate: Candidate) -> float:
    if candidate.action == 0:
        return 0.0
    if is_light(candidate.predicted_latent):
        return 1.0
    return -1.0 if random.random() < policy_config["flash_probability"] else 1.0


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/act", response_model=ActionResponse)
def act(request: ActRequest):
    costs = [candidate_cost(c) for c in request.candidates]
    best = min(range(len(costs)), key=lambda i: (costs[i], request.candidates[i].action))
    return ActionResponse(action=request.candidates[best].action, costs=costs)


def main():
    parser = argparse.ArgumentParser(description="Serve FastAPI inference for the Firefly policy.")
    parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--threshold", type=float, help="Override the calibrated mean-latent threshold")
    parser.add_argument("--flash-probability", type=float, help="Override the dark-condition flash probability")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8003, help="Port to bind (default: 8003)")
    args = parser.parse_args()

    config_path = Path(args.config_path)
    if config_path.exists():
        policy_config.update(json.loads(config_path.read_text()))
    else:
        print(f"WARNING: {config_path} not found; using uncalibrated defaults ({policy_config}).")
    if args.threshold is not None:
        policy_config["threshold"] = args.threshold
    if args.flash_probability is not None:
        policy_config["flash_probability"] = args.flash_probability

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
