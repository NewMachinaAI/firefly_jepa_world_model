# Product Requirements Document: Firefly JEPA World Model

| | |
|---|---|
| **Status** | Version 1 |
| **Owner** | Enrique Perez |
| **Last updated** | 2026-09-20 |

## 1. Overview

This project implements a small JEPA based World model that runs on a Raspberry Pi that reproduces a firefly behaviour.  This are 2 implementations of this Firefly system in this project. The first with a hand-written rules-based controller, that is used to generate training and validation data. The second is a JEPA (Joint-Embedding Predictive Architecture) world model, trained and validated with the data samples recorded from the rules-based version.

The Firefly system includes electrical components, on a breadboard, connected to the Raspberry Pi GPIO pins.  The first is light detector. The second is LED.

So this is intended to be a simple, almost like "Hello World" of a JEPA based World Models. The Raspberry Pi, light detection circuit to detect the amoutn of light in the room. When there is no light, the Green LED should flash on and off randomly just like a firefly.  If there is light detected, the LED will be turned off.


## 2. Problem Statement

We want to learn whether a JEPA world model, trained only on raw sensor and actuator data, can reproduce the behaviour of a simple rules-based controller. The rules-based controller provides both the training data and the baseline to compare against.

## 3. Goals

- Record clean, fixed-rate raw data from the rules-based firefly.
= Rules based Firefly capturing sampling data at 10Hz, to be used for training and validation JEPA based implementation.
- Train a JEPA world model (context encoder + predictor) on that data.
- Train a POLICY model to help determine the cost of all possible sequences of actions.
- Build an inference service, accessible via REST endpoint for Encoder to get current latent state.
- Build an inference service, accessible via REST endpoint for Predictor, to get next latest state predictions, given possible action.
- Build an inference service, accessible via REST endpoint for Policy model to provide cost of action.
- Build a decider, ie a harness, which runs continously, and calls the Encoder, Predictor and Policy service, and takes the appropriate actions
- Use the trained model to drive the LED and reproduce the rules-based behaviour.

## 4. Non-Goals

- Supporting more than one light sensor or LED.
- General-purpose robotics or multi-environment generalisation.
- Real-time performance beyond the 10 Hz sampling rate.

## 5. Hardware

| Component | Pin | Role |
|---|---|---|
| LM393 light sensor module | GPIO 17 | Observation input |
| Green LED | GPIO 18 | Action output |

## 6. Rules-Based Behaviour (Baseline)

- **Dark:** flash the LED on for 1 second, then pause a random 2–5 seconds, then check the sensor again.
- **Light:** keep the LED off and check the sensor continuously.

## 7. Data Requirements

Script: `rules_based_firefly_src/rules_based_firefly.py`, run as `python rules_based_firefly.py <training|validation>`.

- **Sampling interval:** 100 ms (10 Hz), fixed-rate.
- **Output:** CSV in `data/training_data`, overwritten on each run.
  - `Firefly_Raw_Training_Samples.csv`
  - `Firefly_Raw_Validation_Samples.csv`

| Attribute | Type | Description |
|---|---|---|
| `timestamp_ms` | int | Measured elapsed time since start, for checking timing jitter |
| `sample_index` | int | Sequential step index, for building sliding windows |
| `ldr_raw` | int (0/1) | Light sensor reading, 0 = dark, 1 = light. Observation for the context encoder |
| `led_state` | int (0/1) | LED state, 0 = off, 1 = on. Action input to the predictor |
| `active_mode` | int (0/1) | 1 = flashing cycle active, 0 = idle. Used for policy targets and cost functions |
| `scenario_tag` | string | `steady_dark`, `steady_light`, `transition_light_to_dark` or `transition_dark_to_light` |

## 8. Model Requirements

Neural Network Architectures:
- Context & Target Encoder model architectures
The encoders should be a 1D Convolutional. The encoder will input 1-bit binary sensing, feeding a short time-series history (16 samples of timeseries data) to provide termporal information.
Input Layer: Tensor shape (B,1,16) — batch size, 1 channel, sequence length 16.
Conv1D Layer: 8 filters, kernel size = 3, stride = 1, padding = 1, followed by ReLU.
Conv1D Layer: 16 filters, kernel size = 3, stride = 2 (downsamples sequence to 8), followed by ReLU.
Flatten & Dense Layer: Projects flattened features down to Latent Dimension = 8.
Total Parameters: ~1,500 parameters.

- Predictor model architecture.
Input Layer: Latent vector s(t) concatenated with action a(t) → Input size = 8+1=9.
Hidden Layer: Dense layer (9→32), LayerNorm / BatchNorm, ReLU activation.
Output Layer: Dense layer (32→8), producing predicted latent state s(t+1) ∈R 8
Total Parameters: ~600 parameters.

- Policy model architecture



## 9. Model Training
- The Context Encoder should be trained simulatanously with the Predictor and Target Encoder. The training data can be found in '/Documents/GitHub/firefly_jepa_world_model/data/training_data/Firefly_Raw_Training_Samples.csv'.  The validation data can be found in '/Documents/GitHub/firefly_jepa_world_model/data/training_data/Firefly_Raw_Validation_Samples.csv'.
- 


## 10. Inference Services


## 9. Success Criteria

TBD. Candidates:
- Prediction error on the validation set below a chosen threshold.
- The model-driven LED matches the rules-based `led_state` on at least X% of validation steps.
- Correct behaviour on both light-to-dark and dark-to-light transitions.

## 10. Milestones

- [x] Rules-based controller with raw data logging
- [ ] Collect training and validation datasets
- [ ] Train the JEPA model
- [ ] Deploy the model on the Pi
- [ ] Evaluate against the baseline

## 11. Open Questions

- How much training data is needed, and under what lighting conditions?
- Should manual scenario labels (for example a flashlight pulse) be added?
- What is the pass/fail criterion for "reproduces the use case"?
