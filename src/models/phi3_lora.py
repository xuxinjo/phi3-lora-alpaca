"""Load Phi-3 Mini and apply LoRA via PEFT with optional 4-bit quantization."""

from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PHI3_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


def load_phi3_with_lora(
    model_id: str = PHI3_MODEL_ID,
    use_4bit: bool = True,
    bnb_4bit_compute_dtype: Optional[torch.dtype] = None,
    bnb_4bit_quant_type: str = "nf4",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list[str]] = None,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = True,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Phi-3 Mini and apply LoRA adapters for efficient fine-tuning.

    Args:
        model_id: HuggingFace model ID for Phi-3.
        use_4bit: Use 4-bit quantization via bitsandbytes to reduce VRAM.
        bnb_4bit_compute_dtype: Compute dtype for 4-bit (e.g. torch.bfloat16).
        bnb_4bit_quant_type: Quantization type for 4-bit ('nf4' or 'fp4').
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha (scaling).
        lora_dropout: LoRA dropout.
        lora_target_modules: Module names to apply LoRA to; default for Phi-3.
        device_map: Device map for model ('auto' for multi-GPU/CPU offload).
        trust_remote_code: Allow custom model code.

    Returns:
        (model, tokenizer) ready for training.
    """
    if bnb_4bit_compute_dtype is None:
        bnb_4bit_compute_dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if use_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        torch_dtype=bnb_4bit_compute_dtype if use_4bit else torch.bfloat16,
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    if lora_target_modules is None:
        lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer
