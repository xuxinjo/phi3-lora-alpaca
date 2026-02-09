"""Global configuration for training and demo mode."""

from __future__ import annotations

# When True, the project runs in a very lightweight demo mode.
# This can be toggled at runtime via the CLI flag ``--demo``.
DEMO_MODE: bool = False

# Base models
PHI3_MODEL_ID: str = "microsoft/Phi-3-mini-4k-instruct"
# Demo model: small, actually trained, and served via safetensors on the Hub.
# Using safetensors avoids torch.load restrictions that apply to legacy .bin
# files on torch 2.5.x.
TINY_DEMO_MODEL_ID: str = "gpt2"

# Demo-mode hyperparameters
DEMO_MAX_LENGTH: int = 64
DEMO_TRAIN_BATCH_SIZE: int = 1
DEMO_VAL_BATCH_SIZE: int = 1
DEMO_NUM_EPOCHS: int = 1
DEMO_GRAD_ACCUMULATION_STEPS: int = 1
DEMO_DATASET_MAX_SAMPLES: int = 50

