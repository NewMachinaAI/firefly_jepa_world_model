import argparse
import csv
import random
import time
from pathlib import Path

from gpiozero import DigitalInputDevice, LED

# Rules-based firefly controller that also logs one row per fixed-rate timestep,
# to be used as raw training data for a JEPA world model.
#
# Columns:
#   timestamp_ms : elapsed ms since start (measured, so timing jitter is visible)
#   sample_index : 0, 1, 2, ... one per tick
#   ldr_raw      : 0 = dark, 1 = light  (observation)
#   led_state    : 0 = off, 1 = on      (action a_t applied on this tick)
#   active_mode  : 1 = flashing cycle active, 0 = standard/off
#   scenario_tag : steady_dark | steady_light | transition_light_to_dark |
#                  transition_dark_to_light
FIELDS = ["timestamp_ms", "sample_index", "ldr_raw", "led_state", "active_mode", "scenario_tag"]

FLASH_ON_S = 1.0
PAUSE_RANGE_S = (2.0, 5.0)

# Controller phases
IDLE, FLASH, PAUSE = "idle", "flash", "pause"


def scenario_tag(prev_ldr, ldr):
    if prev_ldr is None or prev_ldr == ldr:
        return "steady_light" if ldr == 1 else "steady_dark"
    return "transition_dark_to_light" if ldr == 1 else "transition_light_to_dark"


def main():
    parser = argparse.ArgumentParser(description="Rules-based firefly with data logging.")
    parser.add_argument("--interval-ms", type=float, default=100.0, help="Sampling interval in ms (default 100)")
    parser.add_argument("--out-dir", default="data/training_data", help="Directory for CSV output")
    args = parser.parse_args()

    period = args.interval_ms / 1000.0
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fixed filename, opened with "w" below: each run overwrites the previous run's samples
    out_path = out_dir / "Firefly_Raw_Samples.csv"

    # Default LM393 modules output HIGH (1) in dark, LOW (0) in light.
    # Adjust DARK_LEVEL if your module reads HIGH during daylight.
    DARK_LEVEL = 1
    sensor = DigitalInputDevice(17, pull_up=False)
    firefly_led = LED(18)

    print(f"Firefly simulation running, sampling every {args.interval_ms:g} ms. Logging to {out_path}. Press Ctrl+C to exit.")

    phase = IDLE
    phase_ends_at = 0.0
    prev_ldr = None
    index = 0

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(FIELDS)

        start = time.monotonic()
        next_tick = start
        try:
            while True:
                now = time.monotonic()

                # Observation: 0 = dark, 1 = light
                is_dark = sensor.value == DARK_LEVEL
                ldr_raw = 0 if is_dark else 1

                # Rules-based controller (non-blocking state machine)
                if phase == IDLE and is_dark:
                    phase, phase_ends_at = FLASH, now + FLASH_ON_S
                elif phase == FLASH and now >= phase_ends_at:
                    phase, phase_ends_at = PAUSE, now + random.uniform(*PAUSE_RANGE_S)
                elif phase == PAUSE and now >= phase_ends_at:
                    phase = IDLE
                    if is_dark:
                        phase, phase_ends_at = FLASH, now + FLASH_ON_S

                led_state = 1 if phase == FLASH else 0
                active_mode = 0 if phase == IDLE else 1
                firefly_led.on() if led_state else firefly_led.off()

                writer.writerow([
                    int((now - start) * 1000),
                    index,
                    ldr_raw,
                    led_state,
                    active_mode,
                    scenario_tag(prev_ldr, ldr_raw),
                ])
                print(f"[{index}] ldr_raw = {ldr_raw} ({'light' if ldr_raw else 'dark'})")
                if index % 50 == 0:
                    f.flush()

                prev_ldr = ldr_raw
                index += 1

                # Fixed-rate scheduling: sleep until the next deadline, not for a fixed duration
                next_tick += period
                time.sleep(max(0.0, next_tick - time.monotonic()))

        except KeyboardInterrupt:
            print(f"\nExiting script and cleaning up GPIO. Saved {index} samples to {out_path}.")
            firefly_led.off()


if __name__ == "__main__":
    main()
