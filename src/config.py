"""Global configuration for training and demo mode."""

from __future__ import annotations

DEMO_MODE: bool = False

PHI3_MODEL_ID: str = "microsoft/Phi-3-mini-4k-instruct"
TINY_DEMO_MODEL_ID: str = "gpt2"
DEMO_MAX_LENGTH: int = 64
DEMO_TRAIN_BATCH_SIZE: int = 1
DEMO_VAL_BATCH_SIZE: int = 1
DEMO_NUM_EPOCHS: int = 1
DEMO_GRAD_ACCUMULATION_STEPS: int = 1
DEMO_DATASET_MAX_SAMPLES: int = 50

