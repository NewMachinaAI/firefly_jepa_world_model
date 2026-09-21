"""Decider harness for the Firefly JEPA world model.

Every --interval-ms (default 100) it:
  1. reads the light sensor into a rolling 16-sample history,
  2. calls the encoder service to get the current latent state,
  3. calls the predictor service once per candidate action (0 = LED off, 1 = LED on),
  4. sends the predicted latents to the policy service, which returns the action,
  5. drives the LED with that action.

Use --dry-run to run without GPIO: it simulates a light sensor that alternates
between 10 s of light and 10 s of dark, and only prints the LED state.
"""

import argparse
import sys
import time
from collections import deque

import requests

WINDOW = 16
CANDIDATE_ACTIONS = [0, 1]

# Same wiring/polarity as rules_based_firefly.py: LM393 outputs HIGH (1) in dark.
DARK_LEVEL = 1


class GpioIO:
    def __init__(self):
        from gpiozero import DigitalInputDevice, LED

        self.sensor = DigitalInputDevice(17, pull_up=False)
        self.led = LED(18)

    def read_ldr(self) -> int:
        """0 = dark, 1 = light."""
        return 0 if self.sensor.value == DARK_LEVEL else 1

    def set_led(self, state: int) -> None:
        self.led.on() if state else self.led.off()


class SimulatedIO:
    def __init__(self):
        self.start = time.monotonic()

    def read_ldr(self) -> int:
        return 1 if int((time.monotonic() - self.start) // 10) % 2 == 0 else 0

    def set_led(self, state: int) -> None:
        pass


def call_encoder(session: requests.Session, base_url: str, observation: list[float]) -> list[float]:
    response = session.post(f"{base_url}/encode", json={"observation": observation}, timeout=2)
    response.raise_for_status()
    return response.json()["embedding"]


def call_predictor(session: requests.Session, base_url: str, embedding: list[float], action: int) -> list[float]:
    response = session.post(f"{base_url}/predict", json={"embedding": embedding, "action": [float(action)]}, timeout=2)
    response.raise_for_status()
    return response.json()["predicted_embedding"]


def call_policy(session: requests.Session, base_url: str, candidates: list[dict]) -> dict:
    response = session.post(f"{base_url}/act", json={"candidates": candidates}, timeout=2)
    response.raise_for_status()
    return response.json()


def run_cycle(session: requests.Session, history: deque, encoder_url: str, predictor_url: str, policy_url: str) -> dict:
    embedding = call_encoder(session, encoder_url, list(history))
    candidates = [
        {"action": a, "predicted_latent": call_predictor(session, predictor_url, embedding, a)} for a in CANDIDATE_ACTIONS
    ]
    return call_policy(session, policy_url, candidates)


def main():
    parser = argparse.ArgumentParser(description="Run the Firefly encoder -> predictor -> policy control loop.")
    parser.add_argument("--encoder-url", default="http://127.0.0.1:8001", help="Base URL of the encoder service")
    parser.add_argument("--predictor-url", "--predicter-url", dest="predictor_url", default="http://127.0.0.1:8002",
                        help="Base URL of the predictor service")
    parser.add_argument("--policy-url", default="http://127.0.0.1:8005", help="Base URL of the policy service")
    parser.add_argument("--interval-ms", type=float, default=100.0, help="Loop interval in ms (default 100)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate the sensor and LED instead of using GPIO")
    args = parser.parse_args()

    encoder_url = args.encoder_url.rstrip("/")
    predictor_url = args.predictor_url.rstrip("/")
    policy_url = args.policy_url.rstrip("/")
    period = args.interval_ms / 1000.0

    io = SimulatedIO() if args.dry_run else GpioIO()
    history = deque(maxlen=WINDOW)
    session = requests.Session()  # keep-alive connections; per-request setup is too slow for a 100 ms loop

    print(f"Harness running every {args.interval_ms:g} ms{' (dry run)' if args.dry_run else ''}. Press Ctrl+C to quit.")
    index = 0
    next_tick = time.monotonic()
    try:
        while True:
            ldr = io.read_ldr()
            if not history:
                history.extend([ldr] * WINDOW)  # warm-up: assume the first reading held for the whole window
            else:
                history.append(ldr)

            try:
                result = run_cycle(session, history, encoder_url, predictor_url, policy_url)
                action = result["action"]
                print(f"[{index}] ldr_raw={ldr} action={action} costs={result['costs']}")
            except requests.exceptions.RequestException as e:
                action = 0  # fail safe: LED off
                print(f"[{index}] cycle failed, LED off: {e}", file=sys.stderr)

            io.set_led(action)
            index += 1

            next_tick += period
            now = time.monotonic()
            if next_tick < now:  # overran the interval; skip missed ticks rather than bursting to catch up
                next_tick = now
            time.sleep(next_tick - now)
    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        io.set_led(0)


if __name__ == "__main__":
    main()
