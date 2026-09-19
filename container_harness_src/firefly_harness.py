"""Automated harness for the Firefly JEPA world model.

Every --interval seconds (default 60), calls the encoder, predicter, and
policy REST inference services in sequence: encode a stub observation,
predict the next embedding from it, then choose an action from that
prediction.
"""

import argparse
import sys
import time

import requests

DEFAULT_OBSERVATION_DIM = 16


def call_encoder(base_url: str, observation: list[float]) -> list[float]:
    response = requests.post(f"{base_url}/encode", json={"observation": observation}, timeout=10)
    print(f"POST {base_url}/encode -> {response.status_code}")
    response.raise_for_status()
    return response.json()["embedding"]


def call_predicter(base_url: str, embedding: list[float]) -> list[float]:
    response = requests.post(f"{base_url}/predict", json={"embedding": embedding}, timeout=10)
    print(f"POST {base_url}/predict -> {response.status_code}")
    response.raise_for_status()
    return response.json()["predicted_embedding"]


def call_policy(base_url: str, embedding: list[float]) -> list[float]:
    response = requests.post(f"{base_url}/act", json={"embedding": embedding}, timeout=10)
    print(f"POST {base_url}/act -> {response.status_code}")
    response.raise_for_status()
    return response.json()["action"]


def run_cycle(encoder_url: str, predicter_url: str, policy_url: str) -> None:
    # TODO: replace with a real observation source.
    observation = [0.0] * DEFAULT_OBSERVATION_DIM

    embedding = call_encoder(encoder_url, observation)
    predicted_embedding = call_predicter(predicter_url, embedding)
    action = call_policy(policy_url, predicted_embedding)

    print(f"embedding[:4]={embedding[:4]} predicted[:4]={predicted_embedding[:4]} action={action}")


def main():
    parser = argparse.ArgumentParser(
        description="Poll the Firefly encoder, predicter, and policy inference services on a fixed interval."
    )
    parser.add_argument("--encoder-url", default="http://127.0.0.1:8001", help="Base URL of the encoder inference service")
    parser.add_argument("--predicter-url", default="http://127.0.0.1:8002", help="Base URL of the predicter inference service")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8003", help="Base URL of the policy inference service")
    parser.add_argument("--interval", type=float, default=60, help="Seconds between cycles (default: 60)")
    args = parser.parse_args()

    encoder_url = args.encoder_url.rstrip("/")
    predicter_url = args.predicter_url.rstrip("/")
    policy_url = args.policy_url.rstrip("/")

    print(f"Polling every {args.interval}s. Press Ctrl+C to quit.\n")
    try:
        while True:
            try:
                run_cycle(encoder_url, predicter_url, policy_url)
            except requests.exceptions.RequestException as e:
                print(f"Cycle failed: {e}", file=sys.stderr)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
