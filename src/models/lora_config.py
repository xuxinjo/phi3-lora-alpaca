"""LoRA configuration for Phi-3 Mini using PEFT."""

from peft import LoraConfig, TaskType


def get_phi3_lora_config() -> LoraConfig:
    """
    Return a LoraConfig for Phi-3 Mini fine-tuning.

    Targets the attention projection layers (q, k, v, o) for efficient adaptation.
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
