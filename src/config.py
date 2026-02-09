"""Global configuration for training and demo mode."""

from __future__ import annotations

# When True, the project runs in a very lightweight demo mode.
# This can be toggled at runtime via the CLI flag ``--demo``.
DEMO_MODE: bool = False

# Base models
PHI3_MODEL_ID: str = "microsoft/Phi-3-mini-4k-instruct"
# Tiny demo model uses safetensors (no .bin load), so it avoids torch.load
# safety checks on older torch versions.
TINY_DEMO_MODEL_ID: str = "hf-internal-testing/tiny-random-gpt2"

# Demo-mode hyperparameters
DEMO_MAX_LENGTH: int = 64
DEMO_TRAIN_BATCH_SIZE: int = 1
DEMO_VAL_BATCH_SIZE: int = 1
DEMO_NUM_EPOCHS: int = 1
DEMO_GRAD_ACCUMULATION_STEPS: int = 1
DEMO_DATASET_MAX_SAMPLES: int = 50

